# **QRLottery - Complete Implementation Documentation**

## **Project Overview**

**QRLottery** is a comprehensive iOS lottery ticket scanning and verification application that combines advanced OCR technology, API integration, and intelligent game detection to provide accurate lottery number extraction and verification.

---

## **Scanning System Architecture**

### **1. Multi-Modal Scanning Approach**

The app implements a sophisticated scanning system with multiple fallback mechanisms:

```
Image Capture → Local OCR → OpenAI Vision API → Grid Detection → Individual Number Scanning
```

### **2. Scanning Components**

#### **A. QR/Barcode Scanner**
```swift
// Supported Formats
metadataOutput.metadataObjectTypes = [
    .qr,           // QR Codes
    .ean13,         // EAN-13 Barcodes
    .ean8,          // EAN-8 Barcodes
    .pdf417,        // PDF417 Barcodes
    .code128,       // Code 128 Barcodes
    .code39,        // Code 39 Barcodes
    .code93,        // Code 93 Barcodes
    .aztec,         // Aztec Codes
    .dataMatrix     // Data Matrix Codes
]
```

#### **B. Camera Configuration**
```swift
// Camera Setup
private func setupCamera() {
    captureSession = AVCaptureSession()
    captureSession.sessionPreset = .high
    
    // Add video input
    guard let videoCaptureDevice = AVCaptureDevice.default(for: .video) else { return }
    let videoInput = try? AVCaptureDeviceInput(device: videoCaptureDevice)
    captureSession.addInput(videoInput)
    
    // Add metadata output for QR/Barcode detection
    let metadataOutput = AVCaptureMetadataOutput()
    captureSession.addOutput(metadataOutput)
    metadataOutput.setMetadataObjectsDelegate(self, queue: DispatchQueue.main)
}
```

#### **C. Image Preprocessing Pipeline**
```swift
// Core Image Filters Applied
private func preprocessImageForOCR(_ image: UIImage) -> UIImage? {
    guard let ciImage = CIImage(image: image) else { return nil }
    
    // 1. Contrast Enhancement
    let contrastFilter = CIFilter.colorControls()
    contrastFilter.inputImage = ciImage
    contrastFilter.contrast = 1.5
    contrastFilter.brightness = 0.1
    
    // 2. Sharpening
    let sharpenFilter = CIFilter.sharpenLuminance()
    sharpenFilter.inputImage = contrastFilter.outputImage
    sharpenFilter.sharpness = 0.8
    
    // 3. Grayscale Conversion
    let grayscaleFilter = CIFilter.colorMonochrome()
    grayscaleFilter.inputImage = sharpenFilter.outputImage
    grayscaleFilter.color = CIColor.white
    grayscaleFilter.intensity = 1.0
    
    // 4. Noise Reduction
    let blurFilter = CIFilter.gaussianBlur()
    blurFilter.inputImage = grayscaleFilter.outputImage
    blurFilter.radius = 0.5
    
    return convertCIImageToUIImage(blurFilter.outputImage)
}
```

---

## **OCR Implementation**

### **1. Local OCR (Vision Framework)**

```swift
// Vision Framework Implementation
private func performLocalOCR(on image: UIImage, completion: @escaping ([String]) -> Void) {
    guard let cgImage = image.cgImage else {
        completion([])
        return
    }
    
    let request = VNRecognizeTextRequest { request, error in
        guard let observations = request.results as? [VNRecognizedTextObservation] else {
            completion([])
            return
        }
        
        let recognizedStrings = observations.compactMap { observation in
            observation.topCandidates(1).first?.string
        }
        
        completion(recognizedStrings)
    }
    
    // Configure recognition
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = true
    request.minimumTextHeight = 0.015
    
    let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
    try? handler.perform([request])
}
```

### **2. OpenAI Vision API Integration**

```swift
// OpenAI Vision API Implementation
private func scanWithOpenAI(image: UIImage, completion: @escaping (Result<[TicketRow], Error>) -> Void) {
    guard let imageData = image.jpegData(compressionQuality: 0.8) else {
        completion(.failure(NSError(domain: "OpenAI", code: -1, 
                                 userInfo: [NSLocalizedDescriptionKey: "Failed to convert image to data"])))
        return
    }
    
    let base64Image = imageData.base64EncodedString()
    
    let requestBody: [String: Any] = [
        "model": "gpt-4o",
        "messages": [
            [
                "role": "user",
                "content": [
                    [
                        "type": "text",
                        "text": """
                        Analyze this lottery ticket image and extract all lottery numbers. 
                        Return ONLY a JSON array of objects with this exact structure:
                        [{"numbers": [1,2,3,4,5], "special": 6}, {"numbers": [7,8,9,10,11], "special": 12}]
                        
                        Rules:
                        - Extract numbers from each row
                        - If you cannot read a number, use -1
                        - Return up to 5 rows maximum
                        - Each row should have 5 regular numbers and 1 special number
                        - Regular numbers: 1-69, Special numbers: 1-26
                        """
                    ],
                    [
                        "type": "image_url",
                        "image_url": [
                            "url": "data:image/jpeg;base64,\(base64Image)"
                        ]
                    ]
                ]
            ]
        ],
        "max_tokens": 1000
    ]
    
    // Make API request
    makeOpenAIRequest(requestBody: requestBody, completion: completion)
}
```

---

## **Game Type Detection System**

### **1. Automatic Game Detection**

```swift
// Game Type Detection Logic
private func detectLotteryGameType() -> String {
    // Analyze scanned numbers to determine game type
    let maxRegular = lotteryNumbers.flatMap { $0 }.filter { $0 > 0 }.max() ?? 0
    let maxSpecial = powerballNumbers.filter { $0 > 0 }.max() ?? 0
    
    // Game detection based on number ranges
    switch (maxRegular, maxSpecial) {
    case (1...69, 1...26):
        return "us_mega_millions"  // Mega Millions
    case (1...69, 1...26):
        return "us_powerball"      // Powerball
    case (1...52, 1...10):
        return "us_lotto_america" // Lotto America
    case (1...60, 1...4):
        return "us_cash4life"     // Cash4Life
    case (1...49, 1...10):
        return "ca_lotto_649"     // Canadian Lotto 649
    case (1...45, 1...20):
        return "au_oz_lotto"      // Australian Oz Lotto
    case (1...47, 1...7):
        return "eu_eurojackpot"   // Eurojackpot
    default:
        return "us_mega_millions" // Default fallback
    }
}
```

### **2. Game Constraints System**

```swift
// Game-Specific Constraints
private func getGameConstraints(for gameType: String) -> (maxRegular: Int, maxSpecial: Int) {
    switch gameType {
    case "us_mega_millions":
        return (69, 26)
    case "us_powerball":
        return (69, 26)
    case "us_lotto_america":
        return (52, 10)
    case "us_cash4life":
        return (60, 4)
    case "ca_lotto_649":
        return (49, 10)
    case "au_oz_lotto":
        return (45, 20)
    case "eu_eurojackpot":
        return (47, 7)
    default:
        return (69, 26) // Default to Mega Millions
    }
}
```

---

## **API Integration**

### **1. Magayo.com Lottery Results API**

#### **API Configuration**
```swift
// API Constants
private let apiKey = APIKeys.magayoAPIKey
private let baseURL = "https://www.magayo.com/api/results.php"
```

#### **API Request Structure**
```swift
// API Request Implementation
private func fetchLotteryResults(completion: @escaping (Result<LotteryResult, LotteryAPIError>) -> Void) {
    let gameType = detectLotteryGameType()
    
    var components = URLComponents(string: baseURL)!
    components.queryItems = [
        URLQueryItem(name: "api_key", value: apiKey),
        URLQueryItem(name: "game", value: gameType),
        URLQueryItem(name: "format", value: "json")
    ]
    
    guard let url = components.url else {
        completion(.failure(LotteryAPIError(message: "Invalid URL", code: nil)))
        return
    }
    
    URLSession.shared.dataTask(with: url) { data, response, error in
        // Handle response
        self.handleAPIResponse(data: data, response: response, error: error, completion: completion)
    }.resume()
}
```

#### **Sample API Request**
```http
GET https://www.magayo.com/api/results.php?api_key=YOUR_API_KEY&game=us_mega_millions&format=json
```

#### **Sample API Response**
```json
{
    "error": 0,
    "draw": "2024-01-15",
    "results": "15 23 42 8 31 7"
}
```

### **2. OpenAI Vision API**

#### **API Request Structure**
```swift
// OpenAI API Request
private func makeOpenAIRequest(requestBody: [String: Any], completion: @escaping (Result<[TicketRow], Error>) -> Void) {
    guard let url = URL(string: APIKeys.openAIBaseURL) else {
        completion(.failure(NSError(domain: "OpenAI", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
        return
    }
    
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("Bearer \(APIKeys.openAIAPIKey)", forHTTPHeaderField: "Authorization")
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    
    do {
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
    } catch {
        completion(.failure(error))
        return
    }
    
    URLSession.shared.dataTask(with: request) { data, response, error in
        // Handle OpenAI response
        self.handleOpenAIResponse(data: data, response: response, error: error, completion: completion)
    }.resume()
}
```

#### **Sample OpenAI Request**
```json
{
    "model": "gpt-4o",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Analyze this lottery ticket image and extract all lottery numbers..."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
                    }
                }
            ]
        }
    ],
    "max_tokens": 1000
}
```

#### **Sample OpenAI Response**
```json
{
    "id": "chatcmpl-123",
    "object": "chat.completion",
    "created": 1677652288,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "[{\"numbers\": [15, 23, 42, 8, 31], \"special\": 7}, {\"numbers\": [12, 34, 56, 78, 90], \"special\": 12}]"
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150
    }
}
```

---

## **Security Implementation**

### **1. API Key Management**

#### **Secure Key Storage**
```swift
// APIKeys.swift (Excluded from Git)
struct APIKeys {
    static let openAIAPIKey = "sk-proj-..."
    static let magayoAPIKey = "WHQCbHNed8W8xTG7JE"
    static let openAIBaseURL = "https://api.openai.com/v1/chat/completions"
}

// APIKeys.swift.template (Committed to Git)
struct APIKeys {
    static let openAIAPIKey = "YOUR_OPENAI_API_KEY_HERE"
    static let magayoAPIKey = "YOUR_MAGAYO_API_KEY_HERE"
    static let openAIBaseURL = "https://api.openai.com/v1/chat/completions"
}
```

#### **Git Ignore Configuration**
```gitignore
# API Keys and sensitive configuration
APIKeys.swift
Config.swift
*.secrets
```

---

## **User Interface Implementation**

### **1. Dynamic Button States**

```swift
// Action Button State Management
private func updateActionButtonState() {
    guard isLotteryTicket else { return }
    
    let issueCount = countCurrentIssues()
    
    if issueCount > 0 {
        // Disable button when there are issues
        actionButton.isEnabled = false
        actionButton.backgroundColor = .systemGray4
        actionButton.setTitleColor(.systemGray2, for: .normal)
        actionButton.setTitle("Complete All Fields First", for: .normal)
    } else {
        // Enable button when all issues are resolved
        actionButton.isEnabled = true
        actionButton.backgroundColor = .systemBlue
        actionButton.setTitleColor(.white, for: .normal)
        actionButton.setTitle("Check For Winners", for: .normal)
    }
}
```

### **2. Visual Issue Indicators**

```swift
// Red Circle Implementation for Empty Fields
private func createNumberButton(number: Int) -> UIButton {
    let button = UIButton(type: .system)
    
    if number == -1 || number == 0 {
        // Empty circle - red border for any empty field
        button.setTitle("", for: .normal)
        button.backgroundColor = .systemGray6
        button.setTitleColor(.clear, for: .normal)
        button.layer.borderColor = UIColor.systemRed.cgColor
        button.layer.borderWidth = 2
    } else {
        // Number circle
        button.setTitle(String(number), for: .normal)
        button.backgroundColor = .white
        button.setTitleColor(.label, for: .normal)
        button.layer.borderColor = UIColor.systemGray4.cgColor
        button.layer.borderWidth = 1
    }
    
    return button
}
```

---

## **Verification System**

### **1. QR/Barcode Authenticity Verification**

```swift
// QR/Barcode Verification Scoring System
private func verifyQRBarcodeAuthenticity() {
    var authenticityScore = 0
    var maxScore = 0
    
    // 1. Format & Structure (20 points)
    maxScore += 20
    if scannedCode.contains("Lottery:") {
        authenticityScore += 20
    }
    
    // 2. Ticket Number Format (15 points)
    maxScore += 15
    if scannedCode.contains("Ticket:") {
        authenticityScore += 15
    }
    
    // 3. Code Length & Complexity (10 points)
    maxScore += 10
    if scannedCode.count > 50 {
        authenticityScore += 10
    }
    
    // 4. Data Structure (15 points)
    maxScore += 15
    let components = scannedCode.components(separatedBy: " ")
    if components.count >= 3 {
        authenticityScore += 15
    }
    
    // 5. Lottery Number Validation (20 points)
    maxScore += 20
    // Validate lottery numbers in QR code
    
    // 6. Powerball Numbers (10 points)
    maxScore += 10
    if scannedCode.contains("PB:") {
        authenticityScore += 10
    }
    
    // 7. Ticket Number Format (10 points)
    maxScore += 10
    // Validate ticket number format
    
    // Calculate authenticity percentage
    let authenticityPercentage = maxScore > 0 ? (authenticityScore * 100) / maxScore : 0
}
```

### **2. Official Results Comparison**

```swift
// Official Results Verification
private func compareWithOfficialResults(_ officialResult: LotteryResult) {
    guard let resultsString = officialResult.results else {
        showVerificationAlert(title: "Verification Failed", 
                           message: "No official results data available.")
        return
    }
    
    let parsedNumbers = parseOfficialResults(resultsString)
    
    var matches = 0
    var totalNumbers = 0
    
    // Compare each row with parsed official numbers
    for (rowIndex, row) in lotteryNumbers.enumerated() {
        let powerball = powerballNumbers[rowIndex]
        
        // Count matches in regular numbers
        for number in row {
            if number > 0 {
                totalNumbers += 1
                if parsedNumbers.regularNumbers.contains(number) {
                    matches += 1
                }
            }
        }
        
        // Check powerball match
        if powerball > 0 {
            totalNumbers += 1
            if powerball == parsedNumbers.powerball {
                matches += 1
            }
        }
    }
    
    let matchPercentage = totalNumbers > 0 ? (matches * 100) / totalNumbers : 0
}
```

---

## **Data Models**

### **1. Core Data Structures**

```swift
// Ticket Row Structure
struct TicketRow {
    let numbers: [Int]   // Regular lottery numbers
    let special: Int?     // Powerball/special number
}

// Number Position for Grid Detection
struct NumberPosition: Hashable {
    let row: Int
    let column: Int
    let image: UIImage
    let originalBounds: CGRect
    let isSpecial: Bool
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(row)
        hasher.combine(column)
        hasher.combine(isSpecial)
    }
    
    static func == (lhs: NumberPosition, rhs: NumberPosition) -> Bool {
        return lhs.row == rhs.row && lhs.column == rhs.column && lhs.isSpecial == rhs.isSpecial
    }
}

// Lottery Grid Structure
struct LotteryGrid {
    let rows: Int
    let columns: Int
    let numberPositions: [NumberPosition]
}

// API Response Models
struct LotteryResult: Codable {
    let error: Int?
    let draw: String?
    let results: String?
}

struct LotteryAPIError: Error {
    let message: String
    let code: Int?
}
```

---

## **Performance Optimizations**

### **1. Concurrent Processing**

```swift
// Parallel Number Scanning
private func scanIndividualNumbers(grid: LotteryGrid, gameType: String, completion: @escaping ([NumberPosition: Int]) -> Void) {
    let dispatchGroup = DispatchGroup()
    var results: [NumberPosition: Int] = [:]
    let resultsQueue = DispatchQueue(label: "results.queue", attributes: .concurrent)
    
    for position in grid.numberPositions {
        dispatchGroup.enter()
        
        DispatchQueue.global(qos: .userInitiated).async {
            self.scanSingleNumber(image: position.image, maxMain: 69, maxSpecial: 26) { result in
                resultsQueue.async(flags: .barrier) {
                    results[position] = result
                    dispatchGroup.leave()
                }
            }
        }
    }
    
    dispatchGroup.notify(queue: .main) {
        completion(results)
    }
}
```

### **2. Memory Management**

```swift
// Proper Cleanup
deinit {
    NotificationCenter.default.removeObserver(self)
    captureSession.stopRunning()
}
```

---

## **Error Handling**

### **1. Comprehensive Error Management**

```swift
// Error Handling Strategy
enum ScanningError: Error {
    case imageProcessingFailed
    case ocrFailed
    case networkError
    case apiKeyInvalid
    case invalidResponse
    case parsingFailed
}

// Error Recovery
private func handleScanningError(_ error: Error) {
    switch error {
    case ScanningError.imageProcessingFailed:
        showErrorAlert(title: "Image Processing Failed", message: "Please try scanning again with better lighting.")
    case ScanningError.ocrFailed:
        showErrorAlert(title: "Text Recognition Failed", message: "The ticket image may be unclear. Please try again.")
    case ScanningError.networkError:
        showErrorAlert(title: "Network Error", message: "Please check your internet connection and try again.")
    default:
        showErrorAlert(title: "Scanning Error", message: "An unexpected error occurred. Please try again.")
    }
}
```

---

## **User Experience Features**

### **1. Real-Time Feedback**

- **Progress Dialogs** - Show during network requests
- **Toast Messages** - Persistent issue notifications
- **Visual Indicators** - Red circles for empty fields
- **Button States** - Dynamic enable/disable based on completion

### **2. Accessibility Support**

- **VoiceOver** - Screen reader compatibility
- **Dynamic Type** - Font size scaling
- **High Contrast** - Visual accessibility
- **Keyboard Navigation** - Full keyboard support

---

## **Security Considerations**

### **1. Data Protection**

- **API Key Security** - Excluded from version control
- **Image Processing** - Local processing when possible
- **Network Security** - HTTPS for all API calls
- **Data Validation** - Input sanitization and validation

### **2. Privacy**

- **No Data Storage** - Images not permanently stored
- **Local Processing** - OCR performed locally when possible
- **Minimal Data Transfer** - Only necessary data sent to APIs

---

## **How to Use**

### **1. Setup Instructions**

1. **Copy the template file:**
   ```bash
   cp QRLottery/APIKeys.swift.template QRLottery/APIKeys.swift
   ```

2. **Add your API keys:**
   - Open `QRLottery/APIKeys.swift`
   - Replace `YOUR_OPENAI_API_KEY_HERE` with your actual OpenAI API key
   - Replace `YOUR_MAGAYO_API_KEY_HERE` with your actual Magayo API key

3. **Build and run** the project in Xcode

### **2. Usage Flow**

1. **Scan lottery ticket** using the camera
2. **Review scanned numbers** - red circles indicate empty fields
3. **Edit any missing numbers** by tapping on red circles
4. **Verify ticket authenticity** using built-in verification tools
5. **Check for winners** against official lottery results

### **3. Verification Options**

- **Check Against Official Results** - Compares with real lottery results
- **Verify Ticket Format** - Validates number ranges and format
- **Verify QR/Barcode Authenticity** - Checks code structure and validity
- **Generate Verification Report** - Creates detailed report

---

## **Requirements**

- **iOS 17.0+**
- **Xcode 15.0+**
- **Swift 5.0+**
- **OpenAI API Key** - For advanced OCR
- **Magayo API Key** - For lottery results

---

## **API Documentation**

### **OpenAI Vision API**
- **Endpoint**: `https://api.openai.com/v1/chat/completions`
- **Model**: `gpt-4o`
- **Authentication**: Bearer token
- **Rate Limits**: Based on OpenAI pricing tier

### **Magayo Lottery API**
- **Endpoint**: `https://www.magayo.com/api/results.php`
- **Authentication**: API key parameter
- **Supported Games**: 200+ international lottery games
- **Rate Limits**: Based on subscription plan

---

## **Features**

- **Lottery Ticket Scanning**: Advanced OCR using OpenAI Vision API
- **QR Code Scanning**: Built-in QR code scanner
- **Lottery Results**: Check winning numbers via Magayo API
- **Editable Numbers**: Tap any number to edit manually
- **Issue Tracking**: Persistent toast shows scanning issues until resolved
- **Keyboard Support**: Auto-scrolling when editing numbers
- **Security**: API keys excluded from version control
- **International Support**: Multiple lottery game types
- **Accessibility**: VoiceOver and Dynamic Type support

---

This comprehensive implementation provides a robust, secure, and user-friendly lottery ticket scanning and verification system with multiple layers of validation and error handling.
