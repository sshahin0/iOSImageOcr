import UIKit
import AVFoundation
import CoreImage
import Vision

// MARK: - Ticket OCR
struct TicketRow {
    let numbers: [Int]   // left-to-right regular numbers
    let special: Int?    // last (red) number if present
}

final class TicketNumberOCR {
    
    func parseTicket(from image: UIImage, completion: @escaping (Result<[TicketRow], Error>) -> Void) {
        guard let cg = image.cgImage else {
            completion(.failure(NSError(domain: "TicketOCR", code: -1)))
            return
        }
        
        let req = VNRecognizeTextRequest { req, err in
            if let err = err { completion(.failure(err)); return }
            let obs = (req.results as? [VNRecognizedTextObservation]) ?? []
            
            // best candidates → lines
            var lines: [(text: String, box: CGRect)] = obs.compactMap {
                guard let t = $0.topCandidates(1).first?.string else { return nil }
                return (t, $0.boundingBox)
            }
            
            // sort visual order: top→bottom, left→right
            lines.sort {
                let dy = abs($0.box.midY - $1.box.midY)
                return dy < 0.01 ? $0.box.minX < $1.box.minX : $0.box.midY > $1.box.midY
            }
            
            // keep lines that are mostly digits/spaces
            let digitish = lines
                .map { (text: $0.text.trimmingCharacters(in: .whitespacesAndNewlines), box: $0.box) }
                .filter { $0.text.range(of: #"^[0-9\s]+$"#, options: .regularExpression) != nil }
            
            // group into rows by Y proximity
            let rows = Self.groupByRows(digitish, yTol: 0.018)
            
            // parse each row: numbers separated by spaces; last = special
            let parsed: [TicketRow] = rows.compactMap { row -> TicketRow? in
                let merged = row.map(\.text).joined(separator: " ").replacingOccurrences(of: "  ", with: " ")
                let tokens = merged.split(separator: " ").compactMap { Int($0) }
                
                print("OCR Debug - Processing row text: '\(merged)' -> tokens: \(tokens)")
                
                // Only process rows that have at least 2 numbers (1 regular + 1 special minimum)
                guard tokens.count >= 2 else { 
                    print("OCR Debug - Skipping row (insufficient numbers): \(tokens)")
                    return nil 
                }
                
                // Separate regular numbers and special number
                let specials = tokens.last
                let normals = Array(tokens.dropLast())
                
                // Apply sanity filters but keep invalid numbers as 0 (empty circles)
                let saneNormals = normals.map { num in
                    (1...70).contains(num) ? num : 0  // 0 means empty circle
                }
                let saneSpecial = specials.flatMap { num in
                    (1...99).contains(num) ? num : 0  // 0 means empty circle
                }
                
                print("OCR Debug - Parsed row: \(saneNormals) PB: \(saneSpecial ?? 0)")
                
                // Always return a row, even if some numbers are 0
                return TicketRow(numbers: saneNormals, special: saneSpecial)
            }
            
            completion(.success(parsed))
        }
        
        req.recognitionLevel = .accurate
        req.usesLanguageCorrection = false
        req.minimumTextHeight = 0.02
        req.recognitionLanguages = ["en-US"]
        
        let handler = VNImageRequestHandler(cgImage: cg, options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            do { try handler.perform([req]) } catch { completion(.failure(error)) }
        }
    }
    
    // MARK: - Row grouping
    private static func groupByRows(_ items: [(text: String, box: CGRect)], yTol: CGFloat) -> [[(text: String, box: CGRect)]] {
        var rows: [[(text: String, box: CGRect)]] = []
        for it in items {
            var placed = false
            for i in rows.indices {
                if let ref = rows[i].first, abs(ref.box.midY - it.box.midY) < yTol {
                    rows[i].append(it); placed = true; break
                }
            }
            if !placed { rows.append([it]) }
        }
        // sort each row left→right; rows top→bottom
        rows = rows.map { $0.sorted { $0.box.minX < $1.box.minX } }
        rows.sort { $0.first!.box.midY > $1.first!.box.midY }
        return rows
    }
}

// MARK: - QRScannerDelegate Protocol
protocol QRScannerDelegate: AnyObject {
    func didScanQRCode(_ code: String, image: UIImage?)
    func didFailToScan()
}

// MARK: - QRScannerViewController
class QRScannerViewController: UIViewController {
    
    weak var delegate: QRScannerDelegate?
    
    private var captureSession: AVCaptureSession!
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private var photoOutput: AVCapturePhotoOutput!
    private var closeButton: UIButton!
    private var captureButton: UIButton!
    private var capturedImage: UIImage?
    private var scanningAreaView: UIView!
    private var enableCropping: Bool = true
    private var useSimpleCropping: Bool = true
    private var useAlternativeCropping: Bool = false
    private var useZeroPadding: Bool = false
    private var usePreviewLayerConversion: Bool = false
    private var useFixedCenterCrop: Bool = false
    private var usePercentageCrop: Bool = false
    private var useTicketDetection: Bool = false
    private var usePreciseGreenBoxCrop: Bool = true // New default - crops exactly to green box
    private var previewImageView: UIImageView!
    private var previewContainerView: UIView!
    private var retakeButton: UIButton!
    private var processButton: UIButton!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupCamera()
        setupUI()
        
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        if !captureSession.isRunning {
            DispatchQueue.global(qos: .background).async {
                self.captureSession.startRunning()
            }
        }
        
        // Ensure buttons are enabled
        closeButton?.isEnabled = true
        closeButton?.isUserInteractionEnabled = true
        captureButton?.isEnabled = true
        captureButton?.isUserInteractionEnabled = true
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        if captureSession.isRunning {
            DispatchQueue.global(qos: .background).async {
                self.captureSession.stopRunning()
            }
        }
    }
    
    private func setupCamera() {
        captureSession = AVCaptureSession()
        
        guard let videoCaptureDevice = AVCaptureDevice.default(for: .video) else { 
            print("Failed to get video capture device")
            return 
        }
        let videoInput: AVCaptureDeviceInput
        
        do {
            videoInput = try AVCaptureDeviceInput(device: videoCaptureDevice)
        } catch {
            print("Failed to create video input: \(error)")
            return
        }
        
        if captureSession.canAddInput(videoInput) {
            captureSession.addInput(videoInput)
        } else {
            print("Cannot add video input to capture session")
            return
        }
        
        let metadataOutput = AVCaptureMetadataOutput()
        
        if captureSession.canAddOutput(metadataOutput) {
            captureSession.addOutput(metadataOutput)
            
            metadataOutput.setMetadataObjectsDelegate(self, queue: DispatchQueue.main)
            metadataOutput.metadataObjectTypes = [.qr, .ean13, .ean8, .pdf417, .code128, .code39, .code93, .aztec, .dataMatrix]
            
            // Set the scanning area to the full screen
            DispatchQueue.main.async {
                metadataOutput.rectOfInterest = CGRect(x: 0, y: 0, width: 1, height: 1)
            }
        } else {
            print("Cannot add metadata output to capture session")
            return
        }
        
        // Add photo output for capture functionality
        photoOutput = AVCapturePhotoOutput()
        if captureSession.canAddOutput(photoOutput) {
            captureSession.addOutput(photoOutput)
        } else {
            print("Cannot add photo output to capture session")
        }
        
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.frame = view.layer.bounds
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        
        // Start the session
        DispatchQueue.global(qos: .background).async {
            self.captureSession.startRunning()
        }
    }
    
    private func setupUI() {
        view.backgroundColor = .black
        
        // Close button
        closeButton = UIButton(type: .system)
        closeButton.setTitle("Close", for: .normal)
        closeButton.titleLabel?.font = UIFont.boldSystemFont(ofSize: 18)
        closeButton.backgroundColor = .systemBlue
        closeButton.setTitleColor(.white, for: .normal)
        closeButton.layer.cornerRadius = 12
        closeButton.translatesAutoresizingMaskIntoConstraints = false
        closeButton.isEnabled = true
        closeButton.isUserInteractionEnabled = true
        closeButton.addTarget(self, action: #selector(closeButtonTapped), for: .touchUpInside)
        
        // Capture button
        captureButton = UIButton(type: .system)
        captureButton.setTitle("Capture", for: .normal)
        captureButton.titleLabel?.font = UIFont.boldSystemFont(ofSize: 18)
        captureButton.backgroundColor = .systemGreen
        captureButton.setTitleColor(.white, for: .normal)
        captureButton.layer.cornerRadius = 12
        captureButton.translatesAutoresizingMaskIntoConstraints = false
        captureButton.isEnabled = true
        captureButton.isUserInteractionEnabled = true
        captureButton.addTarget(self, action: #selector(captureButtonTapped), for: .touchUpInside)
        
        // Add long press gesture to toggle cropping
        let longPressGesture = UILongPressGestureRecognizer(target: self, action: #selector(captureButtonLongPressed(_:)))
        captureButton.addGestureRecognizer(longPressGesture)
        
        // Add double tap gesture to toggle zero padding
        let doubleTapGesture = UITapGestureRecognizer(target: self, action: #selector(captureButtonDoubleTapped))
        doubleTapGesture.numberOfTapsRequired = 2
        captureButton.addGestureRecognizer(doubleTapGesture)
        
        // Add buttons to view
        view.addSubview(closeButton)
        view.addSubview(captureButton)
        
        NSLayoutConstraint.activate([
            closeButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            closeButton.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            closeButton.widthAnchor.constraint(equalToConstant: 80),
            closeButton.heightAnchor.constraint(equalToConstant: 40),
            
            captureButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -30),
            captureButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            captureButton.widthAnchor.constraint(equalToConstant: 120),
            captureButton.heightAnchor.constraint(equalToConstant: 50)
        ])
        
        // Scanning overlay
        let overlayView = UIView()
        overlayView.backgroundColor = UIColor.black.withAlphaComponent(0.3)
        overlayView.translatesAutoresizingMaskIntoConstraints = false
        overlayView.isUserInteractionEnabled = false  // Allow touches to pass through
        view.addSubview(overlayView)
        
        // Create a scanning area with a cutout
        let scanningArea = UIView()
        scanningArea.backgroundColor = .clear
        scanningArea.layer.borderColor = UIColor.green.cgColor
        scanningArea.layer.borderWidth = 3
        scanningArea.layer.cornerRadius = 15
        scanningArea.translatesAutoresizingMaskIntoConstraints = false
        overlayView.addSubview(scanningArea)
        
        // Store reference to scanning area for cropping
        self.scanningAreaView = scanningArea
        
        // Add corner indicators
        addCornerIndicators(to: scanningArea)
        
        // Setup preview container (initially hidden)
        setupPreviewContainer()
        
        let instructionLabel = UILabel()
        instructionLabel.text = "Position lottery ticket within the green frame (350x350)\nOnly the area inside the green box will be captured\n\nLong press 'Capture' to toggle cropping mode\nDouble tap 'Capture' to toggle padding\n\nSupports QR codes, barcodes, and printed numbers"
        instructionLabel.textColor = .white
        instructionLabel.textAlignment = .center
        instructionLabel.font = UIFont.boldSystemFont(ofSize: 16)
        instructionLabel.numberOfLines = 0
        instructionLabel.translatesAutoresizingMaskIntoConstraints = false
        overlayView.addSubview(instructionLabel)
        
        NSLayoutConstraint.activate([
            overlayView.topAnchor.constraint(equalTo: view.topAnchor),
            overlayView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            overlayView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            overlayView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            
            scanningArea.centerXAnchor.constraint(equalTo: overlayView.centerXAnchor),
            scanningArea.centerYAnchor.constraint(equalTo: overlayView.centerYAnchor),
            scanningArea.widthAnchor.constraint(equalToConstant: 350),
            scanningArea.heightAnchor.constraint(equalToConstant: 350),
            
            instructionLabel.topAnchor.constraint(equalTo: scanningArea.bottomAnchor, constant: 20),
            instructionLabel.leadingAnchor.constraint(equalTo: overlayView.leadingAnchor, constant: 20),
            instructionLabel.trailingAnchor.constraint(equalTo: overlayView.trailingAnchor, constant: -20)
        ])
        
        // Ensure buttons are on top of overlay
        view.bringSubviewToFront(closeButton)
        view.bringSubviewToFront(captureButton)
    }
    
    @objc private func closeButtonTapped() {
        dismiss(animated: true)
    }
    
    @objc private func captureButtonTapped() {
        guard photoOutput != nil else { return }
        
        let settings = AVCapturePhotoSettings()
        photoOutput.capturePhoto(with: settings, delegate: self)
    }
    
    @objc private func captureButtonLongPressed(_ gesture: UILongPressGestureRecognizer) {
        if gesture.state == .began {
            // Cycle through cropping modes
            if usePreciseGreenBoxCrop {
                usePreciseGreenBoxCrop = false
                useTicketDetection = true
            } else if useTicketDetection {
                useTicketDetection = false
                usePercentageCrop = true
            } else if usePercentageCrop {
                usePercentageCrop = false
                useFixedCenterCrop = true
            } else if useFixedCenterCrop {
                useFixedCenterCrop = false
                usePreviewLayerConversion = true
            } else if usePreviewLayerConversion {
                usePreviewLayerConversion = false
                useSimpleCropping = true
            } else if useSimpleCropping {
                useSimpleCropping = false
                useAlternativeCropping = true
            } else if useAlternativeCropping {
                useAlternativeCropping = false
                usePreciseGreenBoxCrop = true
            } else {
                usePreciseGreenBoxCrop = true
                useTicketDetection = false
                usePercentageCrop = false
                useFixedCenterCrop = false
                usePreviewLayerConversion = false
                useSimpleCropping = false
                useAlternativeCropping = false
            }
            
            // Toggle zero padding
            useZeroPadding.toggle()
            
            let message: String
            if usePreciseGreenBoxCrop {
                message = "Precise Green Box - crops exactly to green box area (350x350)"
            } else if useTicketDetection {
                message = "Ticket detection - uses AI to find lottery ticket area"
            } else if usePercentageCrop {
                message = "Percentage crop - crops 60% from center (no padding)"
            } else if useFixedCenterCrop {
                message = "Fixed center crop - crops center of image" + (useZeroPadding ? " (zero padding)" : " (10px padding)")
            } else if usePreviewLayerConversion {
                message = "Preview layer conversion - Apple's built-in method" + (useZeroPadding ? " (zero padding)" : " (2px padding)")
            } else if useSimpleCropping {
                message = "Simple cropping - aspect ratio aware" + (useZeroPadding ? " (zero padding)" : " (5px padding)")
            } else if useAlternativeCropping {
                message = "Alternative cropping - normalized coordinates" + (useZeroPadding ? " (zero padding)" : " (5px padding)")
            } else {
                message = "Advanced cropping - complex calculation" + (useZeroPadding ? " (zero padding)" : " (5px padding)")
            }
            
            let alert = UIAlertController(title: "Cropping Mode", message: message, preferredStyle: .alert)
            alert.addAction(UIAlertAction(title: "OK", style: .default))
            present(alert, animated: true)
        }
    }
    
    @objc private func captureButtonDoubleTapped() {
        // Toggle zero padding
        useZeroPadding.toggle()
        
        let message = useZeroPadding ? "Zero padding enabled - exact green box cropping" : "5px padding enabled - slight margin around green box"
        
        let alert = UIAlertController(title: "Padding Mode", message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
    
    @objc private func retakeButtonTapped() {
        // Hide preview and restart camera
        previewContainerView.isHidden = true
        captureSession.startRunning()
    }
    
    @objc private func processButtonTapped() {
        // Process the cropped image
        guard let croppedImage = previewImageView.image else { return }
        
        print("Process Button - Using cropped image: \(croppedImage.size)")
        
        // Update capturedImage to the cropped image for results screen
        self.capturedImage = croppedImage
        
        // Convert to CIImage for processing
        guard let ciImage = CIImage(image: croppedImage) else { return }
        
        // Process with OCR
        processImageWithOCR(ciImage)
    }
    
    private func showCroppedImagePreview(_ croppedImage: UIImage) {
        // Stop the camera
        captureSession.stopRunning()
        
        // Show the cropped image in preview
        previewImageView.image = croppedImage
        
        // Show the preview container
        previewContainerView.isHidden = false
        
        // Add a subtle animation
        previewContainerView.alpha = 0
        UIView.animate(withDuration: 0.3) {
            self.previewContainerView.alpha = 1
        }
    }
    
    private func detectQRCodeInImageWithFallback(_ croppedImage: UIImage, fullImage: UIImage) {
        // First try with the cropped image
        if detectQRCodeInImage(croppedImage) {
            return
        }
        
        // If cropped image fails, try with the full image
        detectQRCodeInImage(fullImage)
    }
    
    @discardableResult
    private func detectQRCodeInImage(_ image: UIImage) -> Bool {
        guard let ciImage = CIImage(image: image) else { return false }
        
        let context = CIContext()
        
        // Try different detection options for QR codes
        let options: [String: Any] = [
            CIDetectorAccuracy: CIDetectorAccuracyHigh,
            CIDetectorMinFeatureSize: 0.05  // Reduced minimum feature size
        ]
        
        // Detect QR codes
        let qrDetector = CIDetector(ofType: CIDetectorTypeQRCode, context: context, options: options)
        
        if let qrDetector = qrDetector {
            let qrFeatures = qrDetector.features(in: ciImage)
            
            for feature in qrFeatures {
            if let qrFeature = feature as? CIQRCodeFeature {
                if let messageString = qrFeature.messageString {
                    AudioServicesPlaySystemSound(SystemSoundID(kSystemSoundID_Vibrate))
                    captureSession.stopRunning()
                        print("QR Detection - Passing image to delegate: \(self.capturedImage?.size ?? CGSize.zero)")
                        delegate?.didScanQRCode(messageString, image: self.capturedImage)
                        return true
                    }
                }
            }
        }
        
        // Detect barcodes (Code 128, EAN13, etc.)
        let barcodeDetector = CIDetector(ofType: CIDetectorTypeRectangle, context: context, options: options)
        
        if let barcodeDetector = barcodeDetector {
            let barcodeFeatures = barcodeDetector.features(in: ciImage)
            
            for feature in barcodeFeatures {
                if let barcodeFeature = feature as? CIRectangleFeature {
                    if let croppedImage = cropImage(ciImage, to: barcodeFeature.bounds) {
                        processBarcodeWithOCR(croppedImage)
                        return true
                    }
                }
            }
        }
        
        // If no barcodes found, try OCR on the entire image
        processImageWithOCR(ciImage)
        return false
    }
    
    private func cropImage(_ ciImage: CIImage, to bounds: CGRect) -> CIImage? {
        let croppedImage = ciImage.cropped(to: bounds)
        return croppedImage
    }
    
    private func cropImageToScanningArea(_ image: UIImage) -> UIImage {
        guard let scanningArea = scanningAreaView else { return image }
        
        if usePreciseGreenBoxCrop {
            return cropImagePreciseGreenBox(image, scanningArea: scanningArea)
        } else if useTicketDetection {
            return cropImageWithTicketDetection(image, scanningArea: scanningArea)
        } else if usePercentageCrop {
            return cropImagePercentage(image, scanningArea: scanningArea)
        } else if useFixedCenterCrop {
            return cropImageFixedCenter(image, scanningArea: scanningArea)
        } else if usePreviewLayerConversion {
            return cropImageUsingPreviewLayer(image, scanningArea: scanningArea)
        } else if useAlternativeCropping {
            return cropImageAlternative(image, scanningArea: scanningArea)
        } else if useSimpleCropping {
            return cropImageSimple(image, scanningArea: scanningArea)
        } else {
            return cropImageAdvanced(image, scanningArea: scanningArea)
        }
    }
    
    private func cropImagePreciseGreenBox(_ image: UIImage, scanningArea: UIView) -> UIImage {
        // Crop exactly to the green box area using precise coordinate mapping
        let imageSize = image.size
        let previewSize = previewLayer.bounds.size
        let scanningFrame = scanningArea.frame
        
        print("Precise Green Box Crop:")
        print("  Image size: \(imageSize)")
        print("  Preview size: \(previewSize)")
        print("  Scanning frame: \(scanningFrame)")
        
        // Calculate the aspect ratio and video gravity
        let imageAspect = imageSize.width / imageSize.height
        let previewAspect = previewSize.width / previewSize.height
        
        var cropRect: CGRect
        
        if imageAspect > previewAspect {
            // Image is wider than preview - letterboxed
            let scale = previewSize.height / imageSize.height
            let scaledWidth = imageSize.width * scale
            let xOffset = (previewSize.width - scaledWidth) / 2
            
            // Convert scanning area coordinates to image coordinates
            let relativeX = (scanningFrame.origin.x - xOffset) / scaledWidth
            let relativeY = scanningFrame.origin.y / previewSize.height
            let relativeWidth = scanningFrame.width / scaledWidth
            let relativeHeight = scanningFrame.height / previewSize.height
            
            cropRect = CGRect(
                x: relativeX * imageSize.width,
                y: relativeY * imageSize.height,
                width: relativeWidth * imageSize.width,
                height: relativeHeight * imageSize.height
            )
        } else {
            // Image is taller than preview - pillarboxed
            let scale = previewSize.width / imageSize.width
            let scaledHeight = imageSize.height * scale
            let yOffset = (previewSize.height - scaledHeight) / 2
            
            // Convert scanning area coordinates to image coordinates
            let relativeX = scanningFrame.origin.x / previewSize.width
            let relativeY = (scanningFrame.origin.y - yOffset) / scaledHeight
            let relativeWidth = scanningFrame.width / previewSize.width
            let relativeHeight = scanningFrame.height / scaledHeight
            
            cropRect = CGRect(
                x: relativeX * imageSize.width,
                y: relativeY * imageSize.height,
                width: relativeWidth * imageSize.width,
                height: relativeHeight * imageSize.height
            )
        }
        
        // Ensure crop rect is within image bounds
        cropRect = CGRect(
            x: max(0, min(cropRect.origin.x, imageSize.width - cropRect.width)),
            y: max(0, min(cropRect.origin.y, imageSize.height - cropRect.height)),
            width: min(cropRect.width, imageSize.width),
            height: min(cropRect.height, imageSize.height)
        )
        
        print("  Calculated crop rect: \(cropRect)")
        
        // Crop the image
        guard let cgImage = image.cgImage?.cropping(to: cropRect) else {
            print("Failed to crop image")
            return image
        }
        
        let croppedImage = UIImage(cgImage: cgImage)
        print("  Cropped image size: \(croppedImage.size)")
        
        return croppedImage
    }
    
    private func cropImageWithTicketDetection(_ image: UIImage, scanningArea: UIView) -> UIImage {
        // Use computer vision to detect the lottery ticket area
        let imageSize = image.size
        let scanningFrame = scanningArea.frame
        let previewSize = previewLayer.bounds.size
        
        // Convert to CIImage for processing
        guard let ciImage = CIImage(image: image) else { return image }
        
        // First try to detect rectangles
        let rectangleRequest = VNDetectRectanglesRequest { request, error in
            guard error == nil,
                  let observations = request.results as? [VNRectangleObservation] else {
                print("Rectangle detection failed: \(error?.localizedDescription ?? "Unknown error")")
                    return
            }
            
            // Find the largest rectangle (likely the lottery ticket)
            let largestRectangle = observations.max { $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height }
            
            if let rectangle = largestRectangle {
                print("Detected rectangle: \(rectangle.boundingBox)")
                
                // Convert normalized coordinates to image coordinates
                let cropRect = CGRect(
                    x: rectangle.boundingBox.origin.x * imageSize.width,
                    y: (1 - rectangle.boundingBox.origin.y - rectangle.boundingBox.height) * imageSize.height, // Flip Y coordinate
                    width: rectangle.boundingBox.width * imageSize.width,
                    height: rectangle.boundingBox.height * imageSize.height
                )
                
                print("Crop rect: \(cropRect)")
                
                // Crop the image to the detected rectangle
                DispatchQueue.main.async {
                    self.cropImageToRect(image, rect: cropRect, scanningFrame: scanningFrame, previewSize: previewSize)
                }
                } else {
                print("No rectangles detected, trying text detection")
                // Fallback to text detection
                self.detectTicketByText(image, scanningFrame: scanningFrame, previewSize: previewSize)
            }
        }
        
        rectangleRequest.minimumAspectRatio = 0.3
        rectangleRequest.maximumAspectRatio = 3.0
        rectangleRequest.minimumSize = 0.1
        rectangleRequest.minimumConfidence = 0.5
        
        let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        try? handler.perform([rectangleRequest])
        
        // Return original image for now, will be replaced by async result
        return image
    }
    
    private func cropImageToRect(_ image: UIImage, rect: CGRect, scanningFrame: CGRect, previewSize: CGSize) {
        // Ensure crop rect is within image bounds
        let imageSize = image.size
        let clampedRect = CGRect(
            x: max(0, min(rect.origin.x, imageSize.width - rect.width)),
            y: max(0, min(rect.origin.y, imageSize.height - rect.height)),
            width: min(rect.width, imageSize.width),
            height: min(rect.height, imageSize.height)
        )
        
        // Add small padding
        let padding: CGFloat = 20
        let paddedRect = CGRect(
            x: max(0, clampedRect.origin.x - padding),
            y: max(0, clampedRect.origin.y - padding),
            width: min(clampedRect.width + (padding * 2), imageSize.width - max(0, clampedRect.origin.x - padding)),
            height: min(clampedRect.height + (padding * 2), imageSize.height - max(0, clampedRect.origin.y - padding))
        )
        
        print("Ticket Detection Crop:")
        print("  Image size: \(imageSize)")
        print("  Detected rect: \(rect)")
        print("  Clamped rect: \(clampedRect)")
        print("  Padded rect: \(paddedRect)")
        
        // Crop the image
        guard let cgImage = image.cgImage?.cropping(to: paddedRect) else { return }
        let croppedImage = UIImage(cgImage: cgImage)
        
        // Show the cropped image preview
        showCroppedImagePreview(croppedImage)
    }
    
    private func detectTicketByText(_ image: UIImage, scanningFrame: CGRect, previewSize: CGSize) {
        // Use text detection to find lottery ticket area
        guard let cgImage = image.cgImage else { return }
        
        let request = VNRecognizeTextRequest { request, error in
            guard error == nil,
                  let observations = request.results as? [VNRecognizedTextObservation] else {
                print("Text detection failed: \(error?.localizedDescription ?? "Unknown error")")
                        return
                    }
            
            // Look for "LOTTERY" text to identify the ticket area
            var ticketBounds: CGRect?
            
            for observation in observations {
                if let candidate = observation.topCandidates(1).first,
                   candidate.string.uppercased().contains("LOTTERY") {
                    ticketBounds = observation.boundingBox
                    print("Found LOTTERY text at: \(ticketBounds!)")
                    break
                }
            }
            
            if let bounds = ticketBounds {
                // Convert normalized coordinates to image coordinates
                let imageSize = image.size
                let cropRect = CGRect(
                    x: bounds.origin.x * imageSize.width,
                    y: (1 - bounds.origin.y - bounds.height) * imageSize.height, // Flip Y coordinate
                    width: bounds.width * imageSize.width,
                    height: bounds.height * imageSize.height
                )
                
                // Expand the bounds to include more of the ticket
                let expandedRect = CGRect(
                    x: max(0, cropRect.origin.x - cropRect.width * 0.5),
                    y: max(0, cropRect.origin.y - cropRect.height * 0.2),
                    width: min(cropRect.width * 2, imageSize.width - max(0, cropRect.origin.x - cropRect.width * 0.5)),
                    height: min(cropRect.height * 1.5, imageSize.height - max(0, cropRect.origin.y - cropRect.height * 0.2))
                )
                
                print("Expanded ticket bounds: \(expandedRect)")
                
                DispatchQueue.main.async {
                    self.cropImageToRect(image, rect: expandedRect, scanningFrame: scanningFrame, previewSize: previewSize)
                }
            } else {
                print("No LOTTERY text found, using center crop")
                DispatchQueue.main.async {
                    let imageSize = image.size
                    let centerRect = CGRect(
                        x: imageSize.width * 0.2,
                        y: imageSize.height * 0.2,
                        width: imageSize.width * 0.6,
                        height: imageSize.height * 0.6
                    )
                    self.cropImageToRect(image, rect: centerRect, scanningFrame: scanningFrame, previewSize: previewSize)
                }
            }
        }
        
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = true
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        try? handler.perform([request])
    }
    
    private func cropImagePercentage(_ image: UIImage, scanningArea: UIView) -> UIImage {
        // Simple percentage-based crop from center
        let imageSize = image.size
        let scanningFrame = scanningArea.frame
        let previewSize = previewLayer.bounds.size
        
        // Crop 60% of the image from the center
        let cropPercentage: CGFloat = 0.6
        let cropWidth = imageSize.width * cropPercentage
        let cropHeight = imageSize.height * cropPercentage
        
        // Center the crop
        let cropRect = CGRect(
            x: (imageSize.width - cropWidth) / 2,
            y: (imageSize.height - cropHeight) / 2,
            width: cropWidth,
            height: cropHeight
        )
        
        print("Percentage Crop:")
        print("  Image size: \(imageSize)")
        print("  Crop percentage: \(cropPercentage)")
        print("  Crop size: \(cropWidth) x \(cropHeight)")
        print("  Crop rect: \(cropRect)")
        
        // Show visual feedback
        DispatchQueue.main.async {
            self.showCropPreview(scanningFrame: scanningFrame, cropRect: cropRect, imageSize: imageSize, previewSize: previewSize)
        }
        
        guard let cgImage = image.cgImage?.cropping(to: cropRect) else { return image }
        return UIImage(cgImage: cgImage)
    }
    
    private func cropImageFixedCenter(_ image: UIImage, scanningArea: UIView) -> UIImage {
        // Fixed center crop - crop a fixed area from the center of the image
        let imageSize = image.size
        let scanningFrame = scanningArea.frame
        let previewSize = previewLayer.bounds.size
        
        // Calculate the center of the image
        let imageCenter = CGPoint(x: imageSize.width / 2, y: imageSize.height / 2)
        
        // Use a fixed crop size based on the scanning area size
        // Scale the scanning area size to image coordinates
        let scaleX = imageSize.width / previewSize.width
        let scaleY = imageSize.height / previewSize.height
        let scale = min(scaleX, scaleY)
        
        let cropWidth = scanningFrame.width * scale
        let cropHeight = scanningFrame.height * scale
        
        // Create crop rect centered on the image
        let cropRect = CGRect(
            x: imageCenter.x - cropWidth / 2,
            y: imageCenter.y - cropHeight / 2,
            width: cropWidth,
            height: cropHeight
        )
        
        // Ensure crop rect is within image bounds
        let clampedRect = CGRect(
            x: max(0, min(cropRect.origin.x, imageSize.width - cropRect.width)),
            y: max(0, min(cropRect.origin.y, imageSize.height - cropRect.height)),
            width: min(cropRect.width, imageSize.width),
            height: min(cropRect.height, imageSize.height)
        )
        
        // Add minimal padding
        let padding: CGFloat = useZeroPadding ? 0 : 10
        let paddedRect = CGRect(
            x: max(0, clampedRect.origin.x - padding),
            y: max(0, clampedRect.origin.y - padding),
            width: min(clampedRect.width + (padding * 2), imageSize.width - max(0, clampedRect.origin.x - padding)),
            height: min(clampedRect.height + (padding * 2), imageSize.height - max(0, clampedRect.origin.y - padding))
        )
        
        print("Fixed Center Crop:")
        print("  Image size: \(imageSize)")
        print("  Preview size: \(previewSize)")
        print("  Scale: \(scale)")
        print("  Crop width: \(cropWidth), height: \(cropHeight)")
        print("  Image center: \(imageCenter)")
        print("  Crop rect: \(cropRect)")
        print("  Clamped rect: \(clampedRect)")
        print("  Padded rect: \(paddedRect)")
        
        // Show visual feedback
        DispatchQueue.main.async {
            self.showCropPreview(scanningFrame: scanningFrame, cropRect: paddedRect, imageSize: imageSize, previewSize: previewSize)
        }
        
        guard let cgImage = image.cgImage?.cropping(to: paddedRect) else { return image }
        return UIImage(cgImage: cgImage)
    }
    
    private func cropImageUsingPreviewLayer(_ image: UIImage, scanningArea: UIView) -> UIImage {
        // Use preview layer's built-in coordinate conversion
        let scanningFrame = scanningArea.frame
        let imageSize = image.size
        let previewSize = previewLayer.bounds.size
        
        // Convert preview coordinates to normalized coordinates (0-1)
        let normalizedRect = previewLayer.metadataOutputRectConverted(fromLayerRect: scanningFrame)
        
        // Convert normalized coordinates to image coordinates
        let cropRect = CGRect(
            x: normalizedRect.origin.x * imageSize.width,
            y: normalizedRect.origin.y * imageSize.height,
            width: normalizedRect.width * imageSize.width,
            height: normalizedRect.height * imageSize.height
        )
        
        // Ensure crop rect is within image bounds
        let clampedRect = CGRect(
            x: max(0, min(cropRect.origin.x, imageSize.width - cropRect.width)),
            y: max(0, min(cropRect.origin.y, imageSize.height - cropRect.height)),
            width: min(cropRect.width, imageSize.width),
            height: min(cropRect.height, imageSize.height)
        )
        
        // Add minimal padding
        let padding: CGFloat = useZeroPadding ? 0 : 2
        let paddedRect = CGRect(
            x: max(0, clampedRect.origin.x - padding),
            y: max(0, clampedRect.origin.y - padding),
            width: min(clampedRect.width + (padding * 2), imageSize.width - max(0, clampedRect.origin.x - padding)),
            height: min(clampedRect.height + (padding * 2), imageSize.height - max(0, clampedRect.origin.y - padding))
        )
        
        print("Preview Layer Conversion:")
        print("  Scanning frame: \(scanningFrame)")
        print("  Image size: \(imageSize)")
        print("  Preview size: \(previewSize)")
        print("  Normalized rect: \(normalizedRect)")
        print("  Crop rect: \(cropRect)")
        print("  Clamped rect: \(clampedRect)")
        print("  Padded rect: \(paddedRect)")
        
        // Show visual feedback
        DispatchQueue.main.async {
            self.showCropPreview(scanningFrame: scanningFrame, cropRect: paddedRect, imageSize: imageSize, previewSize: previewSize)
            self.showDebugOverlay(scanningFrame: scanningFrame, normalizedRect: normalizedRect)
        }
        
        guard let cgImage = image.cgImage?.cropping(to: paddedRect) else { return image }
        return UIImage(cgImage: cgImage)
    }
    
    private func cropImageSimple(_ image: UIImage, scanningArea: UIView) -> UIImage {
        // More accurate approach using preview layer's video gravity
        let scanningFrame = scanningArea.frame
        let imageSize = image.size
        let previewSize = previewLayer.bounds.size
        
        // Get the video gravity to understand how the image is displayed
        let videoGravity = previewLayer.videoGravity
        
        var cropRect: CGRect
        
        if videoGravity == .resizeAspectFill {
            // Image fills the preview layer, maintaining aspect ratio
            let imageAspectRatio = imageSize.width / imageSize.height
            let previewAspectRatio = previewSize.width / previewSize.height
            
            var scaleX: CGFloat
            var scaleY: CGFloat
            var offsetX: CGFloat = 0
            var offsetY: CGFloat = 0
            
            if imageAspectRatio > previewAspectRatio {
                // Image is wider than preview - scaled by height
                scaleY = imageSize.height / previewSize.height
                scaleX = scaleY
                offsetX = (imageSize.width - previewSize.width * scaleX) / 2
            } else {
                // Image is taller than preview - scaled by width
                scaleX = imageSize.width / previewSize.width
                scaleY = scaleX
                offsetY = (imageSize.height - previewSize.height * scaleY) / 2
            }
            
            // Convert scanning area to image coordinates
            cropRect = CGRect(
                x: (scanningFrame.origin.x * scaleX) + offsetX,
                y: (scanningFrame.origin.y * scaleY) + offsetY,
                width: scanningFrame.width * scaleX,
                height: scanningFrame.height * scaleY
            )
        } else {
            // For other video gravity modes, use simple scaling
            let scaleX = imageSize.width / previewSize.width
            let scaleY = imageSize.height / previewSize.height
            
            cropRect = CGRect(
                x: scanningFrame.origin.x * scaleX,
                y: scanningFrame.origin.y * scaleY,
                width: scanningFrame.width * scaleX,
                height: scanningFrame.height * scaleY
            )
        }
        
        // Ensure crop rect is within image bounds
        let clampedRect = CGRect(
            x: max(0, min(cropRect.origin.x, imageSize.width - cropRect.width)),
            y: max(0, min(cropRect.origin.y, imageSize.height - cropRect.height)),
            width: min(cropRect.width, imageSize.width),
            height: min(cropRect.height, imageSize.height)
        )
        
        // Add minimal padding for precise cropping
        let padding: CGFloat = useZeroPadding ? 0 : 5
        let paddedRect = CGRect(
            x: max(0, clampedRect.origin.x - padding),
            y: max(0, clampedRect.origin.y - padding),
            width: min(clampedRect.width + (padding * 2), imageSize.width - max(0, clampedRect.origin.x - padding)),
            height: min(clampedRect.height + (padding * 2), imageSize.height - max(0, clampedRect.origin.y - padding))
        )
        
        print("Simple Cropping:")
        print("  Scanning frame: \(scanningFrame)")
        print("  Image size: \(imageSize)")
        print("  Preview size: \(previewSize)")
        print("  Video gravity: \(videoGravity)")
        print("  Image aspect ratio: \(imageSize.width / imageSize.height)")
        print("  Preview aspect ratio: \(previewSize.width / previewSize.height)")
        print("  Crop rect: \(cropRect)")
        print("  Clamped rect: \(clampedRect)")
        print("  Padded rect: \(paddedRect)")
        
        // Show visual feedback
        DispatchQueue.main.async {
            self.showCropPreview(scanningFrame: scanningFrame, cropRect: paddedRect, imageSize: imageSize, previewSize: previewSize)
        }
        
        guard let cgImage = image.cgImage?.cropping(to: paddedRect) else { return image }
        return UIImage(cgImage: cgImage)
    }
    
    private func cropImageAlternative(_ image: UIImage, scanningArea: UIView) -> UIImage {
        // Alternative approach using preview layer's coordinate conversion
        let scanningFrame = scanningArea.frame
        let imageSize = image.size
        let previewSize = previewLayer.bounds.size
        
        // Convert preview coordinates to normalized coordinates
        let normalizedRect = CGRect(
            x: scanningFrame.origin.x / previewSize.width,
            y: scanningFrame.origin.y / previewSize.height,
            width: scanningFrame.width / previewSize.width,
            height: scanningFrame.height / previewSize.height
        )
        
        // Convert normalized coordinates to image coordinates
        let cropRect = CGRect(
            x: normalizedRect.origin.x * imageSize.width,
            y: normalizedRect.origin.y * imageSize.height,
            width: normalizedRect.width * imageSize.width,
            height: normalizedRect.height * imageSize.height
        )
        
        // Ensure crop rect is within image bounds
        let clampedRect = CGRect(
            x: max(0, min(cropRect.origin.x, imageSize.width - cropRect.width)),
            y: max(0, min(cropRect.origin.y, imageSize.height - cropRect.height)),
            width: min(cropRect.width, imageSize.width),
            height: min(cropRect.height, imageSize.height)
        )
        
        // Add minimal padding for precise cropping
        let padding: CGFloat = useZeroPadding ? 0 : 5
        let paddedRect = CGRect(
            x: max(0, clampedRect.origin.x - padding),
            y: max(0, clampedRect.origin.y - padding),
            width: min(clampedRect.width + (padding * 2), imageSize.width - max(0, clampedRect.origin.x - padding)),
            height: min(clampedRect.height + (padding * 2), imageSize.height - max(0, clampedRect.origin.y - padding))
        )
        
        print("Alternative Cropping:")
        print("  Scanning frame: \(scanningFrame)")
        print("  Image size: \(imageSize)")
        print("  Preview size: \(previewSize)")
        print("  Normalized rect: \(normalizedRect)")
        print("  Crop rect: \(cropRect)")
        print("  Clamped rect: \(clampedRect)")
        print("  Padded rect: \(paddedRect)")
        
        // Show visual feedback
        DispatchQueue.main.async {
            self.showCropPreview(scanningFrame: scanningFrame, cropRect: paddedRect, imageSize: imageSize, previewSize: previewSize)
        }
        
        guard let cgImage = image.cgImage?.cropping(to: paddedRect) else { return image }
        return UIImage(cgImage: cgImage)
    }
    
    private func cropImageAdvanced(_ image: UIImage, scanningArea: UIView) -> UIImage {
        // Get the scanning area frame in the preview layer coordinates
        let scanningFrame = scanningArea.frame
        
        // Calculate the crop rectangle in image coordinates
        let imageSize = image.size
        let previewLayerSize = previewLayer.bounds.size
        
        // Calculate scale factors based on how the image is displayed in the preview layer
        let imageAspectRatio = imageSize.width / imageSize.height
        let previewAspectRatio = previewLayerSize.width / previewLayerSize.height
        
        var scaleX: CGFloat
        var scaleY: CGFloat
        var offsetX: CGFloat = 0
        var offsetY: CGFloat = 0
        
        // Determine how the image is scaled in the preview layer
        if imageAspectRatio > previewAspectRatio {
            // Image is wider than preview - scaled by height, may be cropped horizontally
            scaleY = imageSize.height / previewLayerSize.height
            scaleX = scaleY
            offsetX = (imageSize.width - previewLayerSize.width * scaleX) / 2
        } else {
            // Image is taller than preview - scaled by width, may be cropped vertically
            scaleX = imageSize.width / previewLayerSize.width
            scaleY = scaleX
            offsetY = (imageSize.height - previewLayerSize.height * scaleY) / 2
        }
        
        // Convert scanning area frame to image coordinates
        let cropRect = CGRect(
            x: (scanningFrame.origin.x * scaleX) + offsetX,
            y: (scanningFrame.origin.y * scaleY) + offsetY,
            width: scanningFrame.width * scaleX,
            height: scanningFrame.height * scaleY
        )
        
        // Ensure crop rect is within image bounds
        let clampedRect = CGRect(
            x: max(0, min(cropRect.origin.x, imageSize.width - cropRect.width)),
            y: max(0, min(cropRect.origin.y, imageSize.height - cropRect.height)),
            width: min(cropRect.width, imageSize.width),
            height: min(cropRect.height, imageSize.height)
        )
        
        // Add minimal padding for precise cropping
        let padding: CGFloat = useZeroPadding ? 0 : 5
        let paddedRect = CGRect(
            x: max(0, clampedRect.origin.x - padding),
            y: max(0, clampedRect.origin.y - padding),
            width: min(clampedRect.width + (padding * 2), imageSize.width - max(0, clampedRect.origin.x - padding)),
            height: min(clampedRect.height + (padding * 2), imageSize.height - max(0, clampedRect.origin.y - padding))
        )
        
        // Debug: Print cropping information
        print("Advanced Cropping:")
        print("  Scanning frame: \(scanningFrame)")
        print("  Image size: \(imageSize)")
        print("  Preview size: \(previewLayerSize)")
        print("  Image aspect ratio: \(imageAspectRatio)")
        print("  Preview aspect ratio: \(previewAspectRatio)")
        print("  Scale X: \(scaleX), Scale Y: \(scaleY)")
        print("  Offset X: \(offsetX), Offset Y: \(offsetY)")
        print("  Crop rect: \(cropRect)")
        print("  Clamped rect: \(clampedRect)")
        print("  Padded rect: \(paddedRect)")
        
        // Show visual feedback of what will be cropped
        DispatchQueue.main.async {
            self.showCropPreview(scanningFrame: scanningFrame, cropRect: paddedRect, imageSize: imageSize, previewSize: previewLayerSize)
        }
        
        // Crop the image
        guard let cgImage = image.cgImage?.cropping(to: paddedRect) else { return image }
        return UIImage(cgImage: cgImage)
    }
    
    private func showCropPreview(scanningFrame: CGRect, cropRect: CGRect, imageSize: CGSize, previewSize: CGSize) {
        // Remove any existing crop preview
        view.subviews.forEach { subview in
            if subview.tag == 999 {
                subview.removeFromSuperview()
            }
        }
        
        // Calculate the crop area in preview coordinates for visual feedback
        // This is the reverse calculation of what we did for cropping
        let videoGravity = previewLayer.videoGravity
        var previewCropRect: CGRect
        
        if videoGravity == .resizeAspectFill {
            let imageAspectRatio = imageSize.width / imageSize.height
            let previewAspectRatio = previewSize.width / previewSize.height
            
            var scaleX: CGFloat
            var scaleY: CGFloat
            var offsetX: CGFloat = 0
            var offsetY: CGFloat = 0
            
            if imageAspectRatio > previewAspectRatio {
                scaleY = imageSize.height / previewSize.height
                scaleX = scaleY
                offsetX = (imageSize.width - previewSize.width * scaleX) / 2
            } else {
                scaleX = imageSize.width / previewSize.width
                scaleY = scaleX
                offsetY = (imageSize.height - previewSize.height * scaleY) / 2
            }
            
            // Convert image coordinates back to preview coordinates
            previewCropRect = CGRect(
                x: (cropRect.origin.x - offsetX) / scaleX,
                y: (cropRect.origin.y - offsetY) / scaleY,
                width: cropRect.width / scaleX,
                height: cropRect.height / scaleY
            )
        } else {
            // Simple scaling
            let scaleX = previewSize.width / imageSize.width
            let scaleY = previewSize.height / imageSize.height
            
            previewCropRect = CGRect(
                x: cropRect.origin.x * scaleX,
                y: cropRect.origin.y * scaleY,
                width: cropRect.width * scaleX,
                height: cropRect.height * scaleY
            )
        }
        
        // Create a red overlay to show what will be cropped
        let cropOverlay = UIView(frame: previewCropRect)
        cropOverlay.backgroundColor = UIColor.red.withAlphaComponent(0.3)
        cropOverlay.layer.borderColor = UIColor.red.cgColor
        cropOverlay.layer.borderWidth = 3
        cropOverlay.tag = 999
        
        view.addSubview(cropOverlay)
        
        // Remove the overlay after 3 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
            cropOverlay.removeFromSuperview()
        }
    }
    
    private func showDebugOverlay(scanningFrame: CGRect, normalizedRect: CGRect) {
        // Remove any existing debug overlay
        view.subviews.forEach { subview in
            if subview.tag == 888 {
                subview.removeFromSuperview()
            }
        }
        
        // Create a blue overlay to show the normalized coordinates
        let debugOverlay = UIView(frame: scanningFrame)
        debugOverlay.backgroundColor = UIColor.blue.withAlphaComponent(0.2)
        debugOverlay.layer.borderColor = UIColor.blue.cgColor
        debugOverlay.layer.borderWidth = 2
        debugOverlay.tag = 888
        
        // Add text label showing normalized coordinates
        let label = UILabel()
        label.text = String(format: "Norm: %.2f,%.2f %.2fx%.2f", 
                           normalizedRect.origin.x, normalizedRect.origin.y,
                           normalizedRect.width, normalizedRect.height)
        label.textColor = .white
        label.backgroundColor = UIColor.black.withAlphaComponent(0.7)
        label.font = UIFont.systemFont(ofSize: 12)
        label.textAlignment = .center
        label.translatesAutoresizingMaskIntoConstraints = false
        debugOverlay.addSubview(label)
        
        NSLayoutConstraint.activate([
            label.centerXAnchor.constraint(equalTo: debugOverlay.centerXAnchor),
            label.topAnchor.constraint(equalTo: debugOverlay.topAnchor, constant: 5)
        ])
        
        view.addSubview(debugOverlay)
        
        // Remove the overlay after 5 seconds
        DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
            debugOverlay.removeFromSuperview()
        }
    }
    
    private func fixImageOrientation(_ image: UIImage) -> UIImage {
        // If the image is already in the correct orientation, return it
        if image.imageOrientation == .up {
            return image
        }
        
        // Calculate the transform needed to correct the orientation
        var transform = CGAffineTransform.identity
        
        switch image.imageOrientation {
        case .down, .downMirrored:
            transform = transform.translatedBy(x: image.size.width, y: image.size.height)
            transform = transform.rotated(by: .pi)
        case .left, .leftMirrored:
            transform = transform.translatedBy(x: image.size.width, y: 0)
            transform = transform.rotated(by: .pi / 2)
        case .right, .rightMirrored:
            transform = transform.translatedBy(x: 0, y: image.size.height)
            transform = transform.rotated(by: -.pi / 2)
        default:
            break
        }
        
        switch image.imageOrientation {
        case .upMirrored, .downMirrored:
            transform = transform.translatedBy(x: image.size.width, y: 0)
            transform = transform.scaledBy(x: -1, y: 1)
        case .leftMirrored, .rightMirrored:
            transform = transform.translatedBy(x: image.size.height, y: 0)
            transform = transform.scaledBy(x: -1, y: 1)
        default:
            break
        }
        
        // Create a new image with corrected orientation
        guard let cgImage = image.cgImage else { return image }
        let context = CGContext(data: nil,
                               width: Int(image.size.width),
                               height: Int(image.size.height),
                               bitsPerComponent: cgImage.bitsPerComponent,
                               bytesPerRow: 0,
                               space: cgImage.colorSpace ?? CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: cgImage.bitmapInfo.rawValue)
        
        context?.concatenate(transform)
        
        switch image.imageOrientation {
        case .left, .leftMirrored, .right, .rightMirrored:
            context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: image.size.height, height: image.size.width))
        default:
            context?.draw(cgImage, in: CGRect(x: 0, y: 0, width: image.size.width, height: image.size.height))
        }
        
        guard let correctedCGImage = context?.makeImage() else { return image }
        return UIImage(cgImage: correctedCGImage)
    }
    
    private func processBarcodeWithOCR(_ ciImage: CIImage) {
        // Use the new TicketNumberOCR for barcode processing too
        processImageWithOCR(ciImage)
    }
    
    private func processImageWithOCR(_ ciImage: CIImage) {
        // Convert CIImage to UIImage for the new OCR
        guard let cgImage = CIContext().createCGImage(ciImage, from: ciImage.extent) else {
            DispatchQueue.main.async {
                self.showNoDataFoundAlert()
            }
                        return
                    }
        
        let image = UIImage(cgImage: cgImage)
        let ocr = TicketNumberOCR()
        
        ocr.parseTicket(from: image) { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let rows):
                    print("Ticket OCR Success - Found \(rows.count) rows:")
                    for (index, row) in rows.enumerated() {
                        let regularStr = row.numbers.map { $0 == 0 ? "empty" : String($0) }.joined(separator: ", ")
                        let powerballStr = (row.special ?? 0) == 0 ? "empty" : String(row.special!)
                        print("  Row \(index): [\(regularStr)] PB: \(powerballStr)")
                    }
                    
                    // Convert to lottery data format
                    let lotteryData = self.convertTicketRowsToLotteryData(rows)
                    print("OCR Processing - Generated lottery data: '\(lotteryData)'")
                    
                    if !lotteryData.isEmpty {
                        AudioServicesPlaySystemSound(SystemSoundID(kSystemSoundID_Vibrate))
                        self.captureSession.stopRunning()
                        print("OCR Processing - Passing image to delegate: \(self.capturedImage?.size ?? CGSize.zero)")
                        self.delegate?.didScanQRCode(lotteryData, image: self.capturedImage)
                    } else {
                        print("No valid lottery data found")
                        self.showNoDataFoundAlert()
                    }
                    
                case .failure(let error):
                    print("Ticket OCR Error: \(error.localizedDescription)")
                    self.showNoDataFoundAlert()
                }
            }
        }
    }
    
    private func convertTicketRowsToLotteryData(_ rows: [TicketRow]) -> String {
        var allRows: [String] = []
        
        print("ConvertTicketRows - Input rows count: \(rows.count)")
        for (index, row) in rows.enumerated() {
            print("  Input row \(index): \(row.numbers) PB: \(row.special ?? 0)")
        }
        
        // Filter out completely empty rows (all 0s)
        let validRows = rows.filter { row in
            let hasValidNumbers = row.numbers.contains { $0 > 0 }
            let hasValidSpecial = (row.special ?? 0) > 0
            return hasValidNumbers || hasValidSpecial
        }
        
        print("ConvertTicketRows - Valid rows count: \(validRows.count)")
        
        // Process up to 5 rows, filling with empty rows if needed
        for i in 0..<5 {
            if i < validRows.count {
                let row = validRows[i]
                // Ensure we have exactly 5 regular numbers and 1 powerball
                let regularNumbers = Array(row.numbers.prefix(5))
                let paddedRegulars = regularNumbers + Array(repeating: 0, count: max(0, 5 - regularNumbers.count))
                let powerball = row.special ?? 0
                
                let regularNumbersStr = paddedRegulars.map { String($0) }.joined(separator: " ")
                let powerballStr = String(powerball)
                let rowData = "\(regularNumbersStr) \(powerballStr)"
                allRows.append(rowData)
                print("  Converted row \(i): '\(rowData)'")
            } else {
                // Add empty row (all 0s)
                let emptyRow = "0 0 0 0 0 0"
                allRows.append(emptyRow)
                print("  Empty row \(i): '\(emptyRow)'")
            }
        }
        
        let result = "Lottery: \(allRows.joined(separator: "|")) Ticket:OCR"
        print("ConvertTicketRows - Final result: '\(result)'")
        return result
    }
    
    
    
    private func isValidLotteryRow(_ numbers: [String]) -> Bool {
        guard numbers.count == 6 else { return false }
        
        // Check regular numbers (first 5) are between 1-69
        for i in 0..<5 {
            guard let num = Int(numbers[i]), num >= 1 && num <= 69 else { return false }
        }
        
        // Check powerball (last number) is between 1-26
        guard let powerball = Int(numbers[5]), powerball >= 1 && powerball <= 26 else { return false }
        
        // Check for duplicates in regular numbers
        let regularNumbers = Array(numbers.prefix(5))
        let uniqueNumbers = Set(regularNumbers)
        if uniqueNumbers.count != regularNumbers.count { return false }
        
        return true
    }
    
    private func showNoDataFoundAlert() {
        let alert = UIAlertController(title: "No Data Found", message: "No QR code, barcode, or lottery numbers were detected in the captured image. Please try again.", preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
    
    private func setupPreviewContainer() {
        // Create preview container
        previewContainerView = UIView()
        previewContainerView.backgroundColor = UIColor.black.withAlphaComponent(0.9)
        previewContainerView.isHidden = true
        previewContainerView.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(previewContainerView)
        
        // Create preview image view
        previewImageView = UIImageView()
        previewImageView.contentMode = .scaleAspectFit
        previewImageView.backgroundColor = .white
        previewImageView.layer.cornerRadius = 12
        previewImageView.layer.shadowColor = UIColor.black.cgColor
        previewImageView.layer.shadowOffset = CGSize(width: 0, height: 2)
        previewImageView.layer.shadowOpacity = 0.3
        previewImageView.layer.shadowRadius = 8
        previewImageView.translatesAutoresizingMaskIntoConstraints = false
        previewContainerView.addSubview(previewImageView)
        
        // Create retake button
        retakeButton = UIButton(type: .system)
        retakeButton.setTitle("Retake", for: .normal)
        retakeButton.setTitleColor(.white, for: .normal)
        retakeButton.backgroundColor = UIColor.systemRed
        retakeButton.layer.cornerRadius = 25
        retakeButton.titleLabel?.font = UIFont.boldSystemFont(ofSize: 18)
        retakeButton.translatesAutoresizingMaskIntoConstraints = false
        retakeButton.addTarget(self, action: #selector(retakeButtonTapped), for: .touchUpInside)
        previewContainerView.addSubview(retakeButton)
        
        // Create process button
        processButton = UIButton(type: .system)
        processButton.setTitle("Process", for: .normal)
        processButton.setTitleColor(.white, for: .normal)
        processButton.backgroundColor = UIColor.systemGreen
        processButton.layer.cornerRadius = 25
        processButton.titleLabel?.font = UIFont.boldSystemFont(ofSize: 18)
        processButton.translatesAutoresizingMaskIntoConstraints = false
        processButton.addTarget(self, action: #selector(processButtonTapped), for: .touchUpInside)
        previewContainerView.addSubview(processButton)
        
        // Add title label
        let titleLabel = UILabel()
        titleLabel.text = "Cropped Image Preview"
        titleLabel.textColor = .white
        titleLabel.textAlignment = .center
        titleLabel.font = UIFont.boldSystemFont(ofSize: 20)
        titleLabel.translatesAutoresizingMaskIntoConstraints = false
        previewContainerView.addSubview(titleLabel)
        
        // Setup constraints
        NSLayoutConstraint.activate([
            // Preview container
            previewContainerView.topAnchor.constraint(equalTo: view.topAnchor),
            previewContainerView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            previewContainerView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            previewContainerView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            
            // Title label
            titleLabel.topAnchor.constraint(equalTo: previewContainerView.safeAreaLayoutGuide.topAnchor, constant: 20),
            titleLabel.leadingAnchor.constraint(equalTo: previewContainerView.leadingAnchor, constant: 20),
            titleLabel.trailingAnchor.constraint(equalTo: previewContainerView.trailingAnchor, constant: -20),
            
            // Preview image view
            previewImageView.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 20),
            previewImageView.leadingAnchor.constraint(equalTo: previewContainerView.leadingAnchor, constant: 20),
            previewImageView.trailingAnchor.constraint(equalTo: previewContainerView.trailingAnchor, constant: -20),
            previewImageView.heightAnchor.constraint(equalToConstant: 300),
            
            // Retake button
            retakeButton.topAnchor.constraint(equalTo: previewImageView.bottomAnchor, constant: 30),
            retakeButton.leadingAnchor.constraint(equalTo: previewContainerView.leadingAnchor, constant: 40),
            retakeButton.widthAnchor.constraint(equalToConstant: 120),
            retakeButton.heightAnchor.constraint(equalToConstant: 50),
            
            // Process button
            processButton.topAnchor.constraint(equalTo: previewImageView.bottomAnchor, constant: 30),
            processButton.trailingAnchor.constraint(equalTo: previewContainerView.trailingAnchor, constant: -40),
            processButton.widthAnchor.constraint(equalToConstant: 120),
            processButton.heightAnchor.constraint(equalToConstant: 50)
        ])
    }
    
    private func addCornerIndicators(to view: UIView) {
        let cornerLength: CGFloat = 20
        let cornerWidth: CGFloat = 3
        
        // Top-left corner
        let topLeft = UIView()
        topLeft.backgroundColor = .green
        topLeft.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(topLeft)
        
        // Top-right corner
        let topRight = UIView()
        topRight.backgroundColor = .green
        topRight.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(topRight)
        
        // Bottom-left corner
        let bottomLeft = UIView()
        bottomLeft.backgroundColor = .green
        bottomLeft.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(bottomLeft)
        
        // Bottom-right corner
        let bottomRight = UIView()
        bottomRight.backgroundColor = .green
        bottomRight.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(bottomRight)
        
        NSLayoutConstraint.activate([
            // Top-left corner
            topLeft.topAnchor.constraint(equalTo: view.topAnchor),
            topLeft.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            topLeft.widthAnchor.constraint(equalToConstant: cornerLength),
            topLeft.heightAnchor.constraint(equalToConstant: cornerWidth),
            
            // Top-right corner
            topRight.topAnchor.constraint(equalTo: view.topAnchor),
            topRight.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            topRight.widthAnchor.constraint(equalToConstant: cornerLength),
            topRight.heightAnchor.constraint(equalToConstant: cornerWidth),
            
            // Bottom-left corner
            bottomLeft.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            bottomLeft.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            bottomLeft.widthAnchor.constraint(equalToConstant: cornerLength),
            bottomLeft.heightAnchor.constraint(equalToConstant: cornerWidth),
            
            // Bottom-right corner
            bottomRight.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            bottomRight.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            bottomRight.widthAnchor.constraint(equalToConstant: cornerLength),
            bottomRight.heightAnchor.constraint(equalToConstant: cornerWidth)
        ])
    }
}

// MARK: - AVCaptureMetadataOutputObjectsDelegate
extension QRScannerViewController: AVCaptureMetadataOutputObjectsDelegate {
    func metadataOutput(_ output: AVCaptureMetadataOutput, didOutput metadataObjects: [AVMetadataObject], from connection: AVCaptureConnection) {
        guard let metadataObject = metadataObjects.first,
              let readableObject = metadataObject as? AVMetadataMachineReadableCodeObject,
              let stringValue = readableObject.stringValue else { return }
        
            AudioServicesPlaySystemSound(SystemSoundID(kSystemSoundID_Vibrate))
            captureSession.stopRunning()
        delegate?.didScanQRCode(stringValue, image: nil)
    }
}

// MARK: - AVCapturePhotoCaptureDelegate
extension QRScannerViewController: AVCapturePhotoCaptureDelegate {
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        guard error == nil,
              let imageData = photo.fileDataRepresentation(),
              let image = UIImage(data: imageData) else { return }
        
        // Fix image orientation
        print("Original image orientation: \(image.imageOrientation.rawValue)")
        let correctedImage = fixImageOrientation(image)
        print("Corrected image orientation: \(correctedImage.imageOrientation.rawValue)")
        
        // Store the corrected image
        self.capturedImage = correctedImage
        
        if enableCropping {
            // Crop image to scanning area and show preview
            let croppedImage = cropImageToScanningArea(correctedImage)
            print("Photo Capture - Cropped image size: \(croppedImage.size)")
            // Store the cropped image for use in results screen and OCR
            self.capturedImage = croppedImage
            showCroppedImagePreview(croppedImage)
        } else {
            // Use full image without cropping
            detectQRCodeInImage(correctedImage)
        }
    }
}

// MARK: - Main ViewController
class ViewController: UIViewController {
    
    // MARK: - UI Elements
    private let scanButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle("Scan QR", for: .normal)
        button.titleLabel?.font = UIFont.boldSystemFont(ofSize: 20)
        button.backgroundColor = .systemGreen
        button.setTitleColor(.white, for: .normal)
        button.layer.cornerRadius = 12
        button.layer.borderWidth = 2
        button.layer.borderColor = UIColor.systemGreen.cgColor
        button.translatesAutoresizingMaskIntoConstraints = false
        button.isUserInteractionEnabled = true
        button.isEnabled = true
        return button
    }()
    
    private let instructionLabel: UILabel = {
        let label = UILabel()
        label.text = "Scan lottery ticket QR codes to view results"
        label.textAlignment = .center
        label.font = UIFont.systemFont(ofSize: 16)
        label.textColor = .systemGray
        label.numberOfLines = 0
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    // MARK: - Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupConstraints()
        generateInitialTicket()
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
    }
    
    // MARK: - UI Setup
    private func setupUI() {
        view.backgroundColor = .systemBackground
        title = "Lottery QR Scanner"
        
        view.addSubview(scanButton)
        view.addSubview(instructionLabel)
        
        scanButton.addTarget(self, action: #selector(scanButtonTapped), for: .touchUpInside)
    }
    
    private func setupConstraints() {
        NSLayoutConstraint.activate([
            // Scan Button
            scanButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            scanButton.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            scanButton.widthAnchor.constraint(equalToConstant: 200),
            scanButton.heightAnchor.constraint(equalToConstant: 50),
            
            // Instruction Label
            instructionLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            instructionLabel.topAnchor.constraint(equalTo: scanButton.bottomAnchor, constant: 30),
            instructionLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            instructionLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20)
        ])
    }
    
    private func generateInitialTicket() {
        // Don't generate ticket automatically on load
        instructionLabel.text = "Scan lottery ticket QR codes to view results"
        instructionLabel.textColor = .systemGray
    }
    
    private func showScanQRAlert() {
        // Check camera permission
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            presentQRScanner()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    if granted {
                        self.presentQRScanner()
                    } else {
                        self.showAlert(title: "Camera Access Required", message: "Please enable camera access to scan QR codes.")
                    }
                }
            }
        case .denied, .restricted:
            showAlert(title: "Camera Access Required", message: "Please enable camera access in Settings to scan QR codes.")
        @unknown default:
            showAlert(title: "Error", message: "Unable to access camera.")
        }
    }
    
    private func presentQRScanner() {
        let scannerVC = QRScannerViewController()
        scannerVC.delegate = self
        scannerVC.modalPresentationStyle = .fullScreen
        present(scannerVC, animated: true)
    }
    
    
    private func showAlert(title: String, message: String) {
        let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
}

// MARK: - QRScanResultViewController
class QRScanResultViewController: UIViewController {
    
    private let scrollView: UIScrollView = {
        let scrollView = UIScrollView()
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        scrollView.backgroundColor = .systemBackground
        return scrollView
    }()
    
    private let contentView: UIView = {
        let view = UIView()
        view.translatesAutoresizingMaskIntoConstraints = false
        return view
    }()
    
    
    private let scannedImageView: UIImageView = {
        let imageView = UIImageView()
        imageView.contentMode = .scaleAspectFit
        imageView.backgroundColor = .white
        imageView.layer.cornerRadius = 12
        imageView.layer.shadowColor = UIColor.black.cgColor
        imageView.layer.shadowOffset = CGSize(width: 0, height: 2)
        imageView.layer.shadowOpacity = 0.1
        imageView.layer.shadowRadius = 8
        imageView.translatesAutoresizingMaskIntoConstraints = false
        return imageView
    }()
    
    private let resultsCardView: UIView = {
        let view = UIView()
        view.backgroundColor = .white
        view.layer.cornerRadius = 16
        view.layer.shadowColor = UIColor.black.cgColor
        view.layer.shadowOffset = CGSize(width: 0, height: 2)
        view.layer.shadowOpacity = 0.1
        view.layer.shadowRadius = 8
        view.translatesAutoresizingMaskIntoConstraints = false
        return view
    }()
    
    private let titleLabel: UILabel = {
        let label = UILabel()
        label.text = "Powerball Scan Results"
        label.font = UIFont.boldSystemFont(ofSize: 20)
        label.textAlignment = .center
        label.textColor = .label
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    private let drawingDateLabel: UILabel = {
        let label = UILabel()
        label.text = "Drawing Date"
        label.font = UIFont.systemFont(ofSize: 14)
        label.textColor = .systemGray
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    private let dateValueLabel: UILabel = {
        let label = UILabel()
        let formatter = DateFormatter()
        formatter.dateFormat = "dd MMM, yyyy"
        label.text = formatter.string(from: Date())
        label.font = UIFont.boldSystemFont(ofSize: 16)
        label.textColor = .label
        label.backgroundColor = .systemGray6
        label.layer.cornerRadius = 8
        label.textAlignment = .center
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    private let numbersStackView: UIStackView = {
        let stackView = UIStackView()
        stackView.axis = .vertical
        stackView.spacing = 12
        stackView.translatesAutoresizingMaskIntoConstraints = false
        return stackView
    }()
    
    private let actionButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle("Check For Winners", for: .normal)
        button.titleLabel?.font = UIFont.boldSystemFont(ofSize: 18)
        button.backgroundColor = .systemBlue
        button.setTitleColor(.white, for: .normal)
        button.layer.cornerRadius = 12
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()
    
    private let closeButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle("Close", for: .normal)
        button.titleLabel?.font = UIFont.boldSystemFont(ofSize: 18)
        button.backgroundColor = .systemBlue
        button.setTitleColor(.white, for: .normal)
        button.layer.cornerRadius = 12
        button.translatesAutoresizingMaskIntoConstraints = false
        return button
    }()
    
    var scannedCode: String = ""
    var isLotteryTicket: Bool = false
    var lotteryNumbers: [[Int]] = []
    var powerballNumbers: [Int] = []
    var scannedImage: UIImage?
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupConstraints()
        displayResult()
        
        closeButton.addTarget(self, action: #selector(closeButtonTapped), for: .touchUpInside)
        actionButton.addTarget(self, action: #selector(actionButtonTapped), for: .touchUpInside)
        
        // Add long press gesture to make numbers editable
        let longPressGesture = UILongPressGestureRecognizer(target: self, action: #selector(handleLongPress(_:)))
        view.addGestureRecognizer(longPressGesture)
    }
    
    @objc private func handleLongPress(_ gesture: UILongPressGestureRecognizer) {
        if gesture.state == .began {
            makeNumbersEditable()
        }
    }
    
    private func setupUI() {
        view.backgroundColor = .systemBackground
        title = "Scan Result"
        
        view.addSubview(scrollView)
        scrollView.addSubview(contentView)
        contentView.addSubview(scannedImageView)
        contentView.addSubview(resultsCardView)
        
        resultsCardView.addSubview(titleLabel)
        resultsCardView.addSubview(drawingDateLabel)
        resultsCardView.addSubview(dateValueLabel)
        resultsCardView.addSubview(numbersStackView)
        resultsCardView.addSubview(actionButton)
        
        contentView.addSubview(closeButton)
        
        // Set the scanned image if available
        if let image = scannedImage {
            scannedImageView.image = image
            print("Results Screen - Displaying scanned image: \(image.size)")
            print("Results Screen - Image orientation: \(image.imageOrientation.rawValue)")
        } else {
            print("Results Screen - No scanned image provided")
        }
    }
    
    private func setupConstraints() {
        NSLayoutConstraint.activate([
            // Scroll View
            scrollView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            scrollView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            scrollView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            scrollView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            
            // Content View
            contentView.topAnchor.constraint(equalTo: scrollView.topAnchor),
            contentView.leadingAnchor.constraint(equalTo: scrollView.leadingAnchor),
            contentView.trailingAnchor.constraint(equalTo: scrollView.trailingAnchor),
            contentView.bottomAnchor.constraint(equalTo: scrollView.bottomAnchor),
            contentView.widthAnchor.constraint(equalTo: scrollView.widthAnchor),
            
            // Scanned Image View
            scannedImageView.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 20),
            scannedImageView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            scannedImageView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            scannedImageView.heightAnchor.constraint(equalToConstant: 200),
            
            // Results Card View
            resultsCardView.topAnchor.constraint(equalTo: scannedImageView.bottomAnchor, constant: 20),
            resultsCardView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            resultsCardView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            
            // Title Label
            titleLabel.topAnchor.constraint(equalTo: resultsCardView.topAnchor, constant: 20),
            titleLabel.leadingAnchor.constraint(equalTo: resultsCardView.leadingAnchor, constant: 20),
            titleLabel.trailingAnchor.constraint(equalTo: resultsCardView.trailingAnchor, constant: -20),
            
            // Drawing Date Label
            drawingDateLabel.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 15),
            drawingDateLabel.leadingAnchor.constraint(equalTo: resultsCardView.leadingAnchor, constant: 20),
            
            // Date Value Label
            dateValueLabel.centerYAnchor.constraint(equalTo: drawingDateLabel.centerYAnchor),
            dateValueLabel.trailingAnchor.constraint(equalTo: resultsCardView.trailingAnchor, constant: -20),
            dateValueLabel.widthAnchor.constraint(equalToConstant: 120),
            dateValueLabel.heightAnchor.constraint(equalToConstant: 30),
            
            // Numbers Stack View
            numbersStackView.topAnchor.constraint(equalTo: drawingDateLabel.bottomAnchor, constant: 20),
            numbersStackView.leadingAnchor.constraint(equalTo: resultsCardView.leadingAnchor, constant: 20),
            numbersStackView.trailingAnchor.constraint(equalTo: resultsCardView.trailingAnchor, constant: -20),
            
            // Action Button
            actionButton.topAnchor.constraint(equalTo: numbersStackView.bottomAnchor, constant: 20),
            actionButton.leadingAnchor.constraint(equalTo: resultsCardView.leadingAnchor, constant: 20),
            actionButton.trailingAnchor.constraint(equalTo: resultsCardView.trailingAnchor, constant: -20),
            actionButton.heightAnchor.constraint(equalToConstant: 50),
            actionButton.bottomAnchor.constraint(equalTo: resultsCardView.bottomAnchor, constant: -20),
            
            // Close Button
            closeButton.centerXAnchor.constraint(equalTo: contentView.centerXAnchor),
            closeButton.topAnchor.constraint(equalTo: resultsCardView.bottomAnchor, constant: 30),
            closeButton.widthAnchor.constraint(equalToConstant: 120),
            closeButton.heightAnchor.constraint(equalToConstant: 50),
            
            // Content View Bottom
            contentView.bottomAnchor.constraint(equalTo: closeButton.bottomAnchor, constant: 20)
        ])
    }
    
    private func displayResult() {
        if isLotteryTicket {
            titleLabel.text = "Lottery Scan Results"
            titleLabel.textColor = .label
            
            // Parse lottery numbers
            parseLotteryNumbers()
            setupNumbersGrid()
        } else {
            titleLabel.text = "QR Code Scanned"
            titleLabel.textColor = .systemBlue
            
            // Show raw QR code data
            setupRawDataDisplay()
        }
    }
    
    private func parseLotteryNumbers() {
        lotteryNumbers = []
        powerballNumbers = []
        
        print("Results Screen - Parsing lottery numbers from: \(scannedCode)")
        
        // Parse the scanned code to extract lottery numbers
        if scannedCode.contains("Lottery:") {
            let components = scannedCode.components(separatedBy: "Lottery: ")
            if components.count > 1 {
                let lotteryData = components[1].components(separatedBy: " Ticket:")[0]
                
                // Check if it's multi-row data (separated by |)
                if lotteryData.contains("|") {
                    let rows = lotteryData.components(separatedBy: "|")
                    for row in rows {
                        let numbers = row.trimmingCharacters(in: .whitespacesAndNewlines).components(separatedBy: " ")
                        print("Results Screen - Processing row: '\(row)' -> numbers: \(numbers)")
                        
                        // Always process the row, even if some numbers are missing
                        var regularNumbers: [Int] = []
                        for i in 0..<5 {
                            if i < numbers.count, let num = Int(numbers[i]) {
                                regularNumbers.append(num)
                            } else {
                                regularNumbers.append(0) // Empty circle for missing numbers
                            }
                        }
                        
                        let powerball: Int
                        if numbers.count >= 6, let pb = Int(numbers[5]) {
                            powerball = pb
                        } else {
                            powerball = 0 // Empty circle for missing powerball
                        }
                        
                        lotteryNumbers.append(regularNumbers)
                        powerballNumbers.append(powerball)
                        
                        print("Results Screen - Added row: \(regularNumbers) PB: \(powerball)")
                    }
                } else {
                    // Single row format
                    let numbers = lotteryData.components(separatedBy: ",")
                    if numbers.count >= 6 {
                        var regularNumbers: [Int] = []
                        for i in 0..<5 {
                            if let num = Int(numbers[i]) {
                                regularNumbers.append(num)
                            }
                        }
                        if let powerball = Int(numbers[5].replacingOccurrences(of: " PB:", with: "")) {
                            lotteryNumbers.append(regularNumbers)
                            powerballNumbers.append(powerball)
                        }
                    }
                }
            }
        }
        
        // If no numbers parsed or less than 5 rows, show empty circles
        while lotteryNumbers.count < 5 {
            // Add empty rows (0 means empty circle)
            lotteryNumbers.append([0, 0, 0, 0, 0])
            powerballNumbers.append(0)
        }
        
        print("Results Screen - Parsed lottery numbers:")
        for (index, row) in lotteryNumbers.enumerated() {
            print("  Row \(index): \(row) PB: \(powerballNumbers[index])")
        }
    }
    
    private func setupNumbersGrid() {
        // Clear existing views
        numbersStackView.arrangedSubviews.forEach { $0.removeFromSuperview() }
        
        let rowLabels = ["A", "B", "C", "D", "E"]
        
        for (index, row) in lotteryNumbers.enumerated() {
            let rowView = createNumberRowView(
                label: rowLabels[index],
                numbers: row,
                powerball: powerballNumbers[index]
            )
            numbersStackView.addArrangedSubview(rowView)
        }
    }
    
    private func createNumberRowView(label: String, numbers: [Int], powerball: Int) -> UIView {
        let rowView = UIView()
        rowView.translatesAutoresizingMaskIntoConstraints = false
        
        let labelView = UILabel()
        labelView.text = label
        labelView.font = UIFont.boldSystemFont(ofSize: 16)
        labelView.textColor = .label
        labelView.translatesAutoresizingMaskIntoConstraints = false
        rowView.addSubview(labelView)
        
        let numbersStackView = UIStackView()
        numbersStackView.axis = .horizontal
        numbersStackView.spacing = 8
        numbersStackView.translatesAutoresizingMaskIntoConstraints = false
        rowView.addSubview(numbersStackView)
        
        // Add number buttons
        for number in numbers {
            let button = createNumberButton(number: number)
            numbersStackView.addArrangedSubview(button)
        }
        
        // Add "PB" label
        let pbLabel = UILabel()
        pbLabel.text = "PB"
        pbLabel.font = UIFont.boldSystemFont(ofSize: 14)
        pbLabel.textColor = .systemRed
        pbLabel.translatesAutoresizingMaskIntoConstraints = false
        numbersStackView.addArrangedSubview(pbLabel)
        
        // Add powerball button
        let powerballButton = createPowerballButton(number: powerball)
        numbersStackView.addArrangedSubview(powerballButton)
        
        NSLayoutConstraint.activate([
            rowView.heightAnchor.constraint(equalToConstant: 50),
            
            labelView.leadingAnchor.constraint(equalTo: rowView.leadingAnchor),
            labelView.centerYAnchor.constraint(equalTo: rowView.centerYAnchor),
            labelView.widthAnchor.constraint(equalToConstant: 20),
            
            numbersStackView.leadingAnchor.constraint(equalTo: labelView.trailingAnchor, constant: 10),
            numbersStackView.trailingAnchor.constraint(equalTo: rowView.trailingAnchor),
            numbersStackView.centerYAnchor.constraint(equalTo: rowView.centerYAnchor)
        ])
        
        return rowView
    }
    
    private func createNumberButton(number: Int) -> UIButton {
        let button = UIButton(type: .system)
        
        if number == 0 {
            // Empty circle
            button.setTitle("", for: .normal)
            button.backgroundColor = .systemGray6
            button.setTitleColor(.clear, for: .normal)
        } else {
            // Number circle
        button.setTitle(String(number), for: .normal)
        button.backgroundColor = .white
        button.setTitleColor(.label, for: .normal)
        }
        
        button.titleLabel?.font = UIFont.boldSystemFont(ofSize: 16)
        button.layer.cornerRadius = 20
        button.layer.borderWidth = 1
        button.layer.borderColor = UIColor.systemGray4.cgColor
        button.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            button.widthAnchor.constraint(equalToConstant: 40),
            button.heightAnchor.constraint(equalToConstant: 40)
        ])
        
        return button
    }
    
    private func createPowerballButton(number: Int) -> UIButton {
        let button = UIButton(type: .system)
        
        if number == 0 {
            // Empty circle
            button.setTitle("", for: .normal)
            button.backgroundColor = .systemGray6
            button.setTitleColor(.clear, for: .normal)
        } else {
            // Number circle
            button.setTitle(String(number), for: .normal)
            button.backgroundColor = .white
            button.setTitleColor(.label, for: .normal)
        }
        
        button.titleLabel?.font = UIFont.boldSystemFont(ofSize: 16)
        button.layer.cornerRadius = 20
        button.layer.borderWidth = 1
        button.layer.borderColor = UIColor.systemGray4.cgColor
        button.translatesAutoresizingMaskIntoConstraints = false
        
        NSLayoutConstraint.activate([
            button.widthAnchor.constraint(equalToConstant: 40),
            button.heightAnchor.constraint(equalToConstant: 40)
        ])
        
        return button
    }
    
    private func setupRawDataDisplay() {
        // Clear existing views
        numbersStackView.arrangedSubviews.forEach { $0.removeFromSuperview() }
        
        let textView = UITextView()
        textView.text = scannedCode
        textView.font = UIFont.systemFont(ofSize: 16)
        textView.backgroundColor = .systemGray6
        textView.layer.cornerRadius = 8
        textView.isEditable = false
        textView.translatesAutoresizingMaskIntoConstraints = false
        
        numbersStackView.addArrangedSubview(textView)
        
        NSLayoutConstraint.activate([
            textView.heightAnchor.constraint(equalToConstant: 150)
        ])
    }
    
    @objc private func closeButtonTapped() {
        dismiss(animated: true)
    }
    
    @objc private func actionButtonTapped() {
        if isLotteryTicket {
            // Check for winners functionality
            checkForWinners()
        } else {
            // Copy to clipboard
            UIPasteboard.general.string = scannedCode
            showCopyConfirmation()
        }
    }
    
    private func checkForWinners() {
        // Simulate checking for winners
        let alert = UIAlertController(title: "Checking Winners", message: "Checking your lottery numbers against winning numbers...", preferredStyle: .alert)
        present(alert, animated: true)
        
        // Simulate delay
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            alert.dismiss(animated: true) {
                self.showWinningResults()
            }
        }
    }
    
    private func showWinningResults() {
        let alert = UIAlertController(title: "Winning Results", message: "No winning numbers found. Better luck next time!", preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
    
    private func makeNumbersEditable() {
        // Make all number buttons editable
        for arrangedSubview in numbersStackView.arrangedSubviews {
            if let rowView = arrangedSubview as? UIView {
                for subview in rowView.subviews {
                    if let stackView = subview as? UIStackView {
                        for button in stackView.arrangedSubviews {
                            if let numberButton = button as? UIButton {
                                numberButton.addTarget(self, action: #selector(numberButtonTapped(_:)), for: .touchUpInside)
                                numberButton.layer.borderColor = UIColor.systemBlue.cgColor
                                numberButton.layer.borderWidth = 2
                            }
                        }
                    }
                }
            }
        }
    }
    
    @objc private func numberButtonTapped(_ sender: UIButton) {
        // Determine if this is a Powerball number (last button in each row)
        var isPowerball = false
        var maxValue = 69
        
        // Check if this button is the last one in its row (Powerball)
        if let stackView = sender.superview as? UIStackView,
           let lastButton = stackView.arrangedSubviews.last as? UIButton,
           lastButton == sender {
            isPowerball = true
            maxValue = 26
        }
        
        let message = isPowerball ? "Enter new Powerball number (1-26)" : "Enter new number (1-69)"
        let alert = UIAlertController(title: "Edit Number", message: message, preferredStyle: .alert)
        
        alert.addTextField { textField in
            textField.placeholder = "Number"
            textField.keyboardType = .numberPad
            textField.text = sender.title(for: .normal)
        }
        
        let saveAction = UIAlertAction(title: "Save", style: .default) { _ in
            if let textField = alert.textFields?.first,
               let text = textField.text,
               let number = Int(text),
               number >= 1 && number <= maxValue {
                sender.setTitle(String(number), for: .normal)
            }
        }
        
        let cancelAction = UIAlertAction(title: "Cancel", style: .cancel)
        
        alert.addAction(saveAction)
        alert.addAction(cancelAction)
        
        present(alert, animated: true)
    }
    
    private func showCopyConfirmation() {
        let alert = UIAlertController(title: "Copied!", message: "QR code data has been copied to clipboard.", preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
}

// MARK: - QRScannerDelegate
extension ViewController: QRScannerDelegate {
    func didScanQRCode(_ code: String, image: UIImage?) {
        instructionLabel.text = "QR Code scanned successfully!"
        instructionLabel.textColor = .systemGreen
        
        // Dismiss the scanner first
        dismiss(animated: true) {
            // Present scan result screen after scanner is dismissed
            let resultVC = QRScanResultViewController()
            resultVC.scannedCode = code
            resultVC.isLotteryTicket = code.contains("Lottery:")
            resultVC.scannedImage = image
            resultVC.modalPresentationStyle = .fullScreen
            
            self.present(resultVC, animated: true)
        }
    }
    
    func didFailToScan() {
        instructionLabel.text = "Failed to scan QR code. Please try again."
        instructionLabel.textColor = .systemRed
    }
}

// MARK: - Actions
extension ViewController {
    @objc private func scanButtonTapped() {
        showScanQRAlert()
    }
}


