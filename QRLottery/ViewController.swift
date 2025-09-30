//
//  ViewController.swift
//  QRLottery
//
//  Created by Shahin Noor on 22/9/25.
//

import UIKit
import AVFoundation
import CoreImage
import Vision
import Network

// MARK: - Network Connectivity Utility
class NetworkConnectivityManager {
    static let shared = NetworkConnectivityManager()
    private let monitor = NWPathMonitor()
    private let queue = DispatchQueue(label: "NetworkMonitor")
    private var isConnected = false
    
    private init() {
        startMonitoring()
    }
    
    private func startMonitoring() {
        monitor.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async {
                self?.isConnected = path.status == .satisfied
            }
        }
        monitor.start(queue: queue)
    }
    
    /// Check if device has internet connectivity
    func isInternetAvailable() -> Bool {
        return isConnected
    }
    
    /// Check internet connectivity with completion handler
    func checkInternetConnectivity(completion: @escaping (Bool) -> Void) {
        DispatchQueue.main.async {
            completion(self.isConnected)
        }
    }
    
    deinit {
        monitor.cancel()
    }
}

// MARK: - Image Resizing Configuration
struct ImageResizeConfig {
    static let maxWidth: CGFloat = 1024
    static let maxHeight: CGFloat = 1024
    static let compressionQuality: CGFloat = 0.8
    static let maxFileSizeKB: Int = 500 // Maximum file size in KB
}

// MARK: - Image Resizing Utilities
extension UIImage {
    /// Resizes the image to fit within the specified dimensions while maintaining aspect ratio
    func resizedForUpload() -> UIImage? {
        return self.resized(to: CGSize(width: ImageResizeConfig.maxWidth, height: ImageResizeConfig.maxHeight))
    }
    
    /// Resizes image to fit within specified size while maintaining aspect ratio
    func resized(to size: CGSize) -> UIImage? {
        let aspectRatio = self.size.width / self.size.height
        var newSize = size
        
        // Calculate new size maintaining aspect ratio
        if aspectRatio > 1 {
            // Landscape: width is the limiting factor
            newSize.height = size.width / aspectRatio
        } else {
            // Portrait: height is the limiting factor
            newSize.width = size.height * aspectRatio
        }
        
        // Ensure we don't upscale
        if newSize.width > self.size.width || newSize.height > self.size.height {
            return self
        }
        
        UIGraphicsBeginImageContextWithOptions(newSize, false, 0.0)
        defer { UIGraphicsEndImageContext() }
        
        self.draw(in: CGRect(origin: .zero, size: newSize))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
    
    /// Converts image to JPEG data with size optimization
    func optimizedJPEGData() -> Data? {
        var compressionQuality = ImageResizeConfig.compressionQuality
        var imageData = self.jpegData(compressionQuality: compressionQuality)
        
        // Reduce quality if file size is too large
        while let data = imageData, data.count > ImageResizeConfig.maxFileSizeKB * 1024 && compressionQuality > 0.1 {
            compressionQuality -= 0.1
            imageData = self.jpegData(compressionQuality: compressionQuality)
        }
        
        return imageData
    }
    
    /// Gets file size in KB
    func fileSizeKB() -> Int {
        guard let data = self.jpegData(compressionQuality: ImageResizeConfig.compressionQuality) else { return 0 }
        return data.count / 1024
    }
}

// MARK: - Ticket OCR
struct TicketRow {
    let numbers: [Int]   // left-to-right regular numbers
    let special: Int?    // last (red) number if present
}

struct NumberPosition: Hashable {
    let row: Int          // Row index (0-4 for A-E)
    let column: Int       // Column index (0-4 for main numbers, 5 for special)
    let image: UIImage    // Cropped image of the number
    let originalBounds: CGRect // Position in original image
    let isSpecial: Bool  // Whether this is a special number (Mega Ball, Powerball, etc.)
    
    // Hashable conformance - use row and column for hashing since they uniquely identify position
    func hash(into hasher: inout Hasher) {
        hasher.combine(row)
        hasher.combine(column)
        hasher.combine(isSpecial)
    }
    
    // Equatable conformance
    static func == (lhs: NumberPosition, rhs: NumberPosition) -> Bool {
        return lhs.row == rhs.row && 
               lhs.column == rhs.column && 
               lhs.isSpecial == rhs.isSpecial
    }
}

struct LotteryGrid {
    let rows: Int
    let columns: Int
    let numberPositions: [NumberPosition]
}

final class TicketNumberOCR {
    
    // MARK: - OpenAI Integration
    
    private let openAIAPIKey = APIKeys.openAIAPIKey
    private let openAIBaseURL = APIKeys.openAIBaseURL
    
    // MARK: - Row Count Detection
    
    private func detectRowCount(from image: UIImage, completion: @escaping (Result<Int, Error>) -> Void) {
        // Check internet connectivity first
        guard NetworkConnectivityManager.shared.isInternetAvailable() else {
            completion(.failure(NSError(domain: "OpenAI", code: -1000, userInfo: [NSLocalizedDescriptionKey: "No internet connection. Please check your network and try again."])))
            return
        }
        
        // Resize image for optimal upload
        guard let resizedImage = image.resizedForUpload() else {
            completion(.failure(NSError(domain: "OpenAI", code: -1, userInfo: [NSLocalizedDescriptionKey: "Could not resize image"])))
            return
        }
        
        // Convert to optimized JPEG data
        guard let imageData = resizedImage.optimizedJPEGData() else {
            completion(.failure(NSError(domain: "OpenAI", code: -1, userInfo: [NSLocalizedDescriptionKey: "Could not convert image to JPEG"])))
            return
        }
        
        let base64Image = imageData.base64EncodedString()
        
        // Log image size information
        
        let requestBody: [String: Any] = [
            "model": "gpt-4o",
            "messages": [
                [
                    "role": "user",
                    "content": [
                        [
                            "type": "text",
                            "text": """
                               I need to count the number of lottery number rows on this ticket.
                               Look for rows labeled A, B, C, D, E, F, G, H, I, J (or similar).
                               Count ONLY the rows that contain lottery numbers (5 regular numbers + 1 special number).
                               
                               Return ONLY a single number representing the total count of lottery rows.
                               Example responses: 1, 5, 10, etc.
                               
                               Rules:
                               - Count from top to bottom of the ticket
                               - Look for row labels (A, B, C, D, E, F, G, H, I, J)
                               - Only count rows with lottery numbers
                               - Return just the number, no other text
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
            "max_tokens": 50
        ]
        
        guard let url = URL(string: openAIBaseURL) else {
            completion(.failure(NSError(domain: "OpenAI", code: -2, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(openAIAPIKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            completion(.failure(error))
            return
        }
        
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let data = data else {
                completion(.failure(NSError(domain: "OpenAI", code: -3, userInfo: [NSLocalizedDescriptionKey: "No data received"])))
                return
            }
            
            do {
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let choices = json["choices"] as? [[String: Any]],
                   let firstChoice = choices.first,
                   let message = firstChoice["message"] as? [String: Any],
                   let content = message["content"] as? String {
                    
                    
                    // Extract number from response
                    let trimmedContent = content.trimmingCharacters(in: .whitespacesAndNewlines)
                    if let rowCount = Int(trimmedContent) {
                        completion(.success(rowCount))
                    } else {
                        completion(.failure(NSError(domain: "OpenAI", code: -4, userInfo: [NSLocalizedDescriptionKey: "Could not parse row count"])))
                    }
                } else {
                    completion(.failure(NSError(domain: "OpenAI", code: -4, userInfo: [NSLocalizedDescriptionKey: "Invalid response format"])))
                }
            } catch {
                completion(.failure(error))
            }
        }.resume()
    }
    
    private func scanWithSpecificRowCount(image: UIImage, expectedRowCount: Int, completion: @escaping (Result<[TicketRow], Error>) -> Void) {
        // Check internet connectivity first
        guard NetworkConnectivityManager.shared.isInternetAvailable() else {
            completion(.failure(NSError(domain: "OpenAI", code: -1000, userInfo: [NSLocalizedDescriptionKey: "No internet connection. Please check your network and try again."])))
            return
        }
        
        // Resize image for optimal upload
        guard let resizedImage = image.resizedForUpload() else {
            completion(.failure(NSError(domain: "OpenAI", code: -1, userInfo: [NSLocalizedDescriptionKey: "Could not resize image"])))
            return
        }
        
        // Convert to optimized JPEG data
        guard let imageData = resizedImage.optimizedJPEGData() else {
            completion(.failure(NSError(domain: "OpenAI", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to convert image to data"])))
            return
        }
        
        let base64Image = imageData.base64EncodedString()
        
        // Log image size information
        
        // Debug logging
        let key = APIKeys.openAIAPIKey
        precondition(!key.contains("YOUR_"), "APIKeys.swift still has placeholder")
        
        let requestBody: [String: Any] = [
            "model": "gpt-4o",
            "response_format": ["type": "json_object"],
            "messages": [
                [
                    "role": "user",
                    "content": [
                        [
                            "type": "text",
                            "text": """
                               I need to extract lottery numbers from a legitimate lottery ticket for verification purposes.
                               This is for a personal lottery ticket checking app to verify winning numbers.
                               
                               IMPORTANT: This ticket has exactly \(expectedRowCount) rows of lottery numbers.
                               You must extract ALL \(expectedRowCount) rows, no more, no less.
                               
                               Please analyze this lottery ticket image and extract ALL lottery numbers from EVERY row.
                               This ticket has \(expectedRowCount) rows labeled A, B, C, D, E, F, G, H, I, J (or similar).
                               You must scan the ENTIRE ticket image and return exactly \(expectedRowCount) rows.
                               
                               Return ONLY a JSON array of rows with this exact schema:
                               [{"numbers":[n,n,n,n,n], "special": n}] with no prose, no code fences.
                               
                               Example for a 10-row ticket:
                               [
                                 {"numbers":[10,36,47,63,74], "special": 4},
                                 {"numbers":[11,23,50,60,71], "special": 9},
                                 {"numbers":[44,47,53,62,74], "special": 8},
                                 {"numbers":[4,24,30,31,49], "special": 4},
                                 {"numbers":[1,10,29,49,53], "special": 14},
                                 {"numbers":[12,20,21,24,65], "special": 12},
                                 {"numbers":[15,17,43,51,61], "special": 11},
                                 {"numbers":[8,10,15,63,69], "special": 1},
                                 {"numbers":[5,7,12,55,63], "special": 9},
                                 {"numbers":[41,42,44,50,69], "special": 1}
                               ]
                               
                               Rules:
                               - Extract numbers from ALL \(expectedRowCount) rows visible in the image
                               - Look for rows labeled A, B, C, D, E, F, G, H, I, J or similar
                               - If you cannot read a number, use -1
                               - Return exactly \(expectedRowCount) rows, no more, no less
                               - Each row should have 5 regular numbers and 1 special number
                               - Regular numbers: 1-69, Special numbers: 1-26
                               - IMPORTANT: Scan the entire ticket image, not just the first row
                               - CRITICAL: Look for ALL \(expectedRowCount) rows from top to bottom of the ticket
                               - The ticket has exactly \(expectedRowCount) rows - extract EVERY single row you can see
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
            "max_tokens": 3000
        ]
        
        guard let url = URL(string: openAIBaseURL) else {
            completion(.failure(NSError(domain: "OpenAI", code: -2, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(openAIAPIKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            completion(.failure(error))
            return
        }
        
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 60
        let session = URLSession(configuration: config)
        session.dataTask(with: request) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let data = data else {
                completion(.failure(NSError(domain: "OpenAI", code: -3, userInfo: [NSLocalizedDescriptionKey: "No data received"])))
                return
            }
            
            // Debug logging
            if let http = response as? HTTPURLResponse {
            }
            if let body = String(data: data, encoding: .utf8) {
            }
            
            do {
                let top = try JSONSerialization.jsonObject(with: data)
                if let json = top as? [String: Any],
                   let choices = json["choices"] as? [[String: Any]],
                   let firstChoice = choices.first,
                   let message = firstChoice["message"] as? [String: Any] {
                    
                    // Check for refusal first
                    if let refusal = message["refusal"] as? String, !refusal.isEmpty {
                        completion(.failure(NSError(domain: "OpenAI", code: -7, userInfo: [NSLocalizedDescriptionKey: "OpenAI refused to process the image: \(refusal)"])))
                        return
                    }
                    
                    if let content = message["content"] as? String {
                        // Parse the OpenAI response
                        self.parseOpenAIResponse(content: content, completion: completion)
                    } else {
                        completion(.failure(NSError(domain: "OpenAI", code: -4, userInfo: [NSLocalizedDescriptionKey: "Invalid response format"])))
                    }
                } else {
                    completion(.failure(NSError(domain: "OpenAI", code: -4, userInfo: [NSLocalizedDescriptionKey: "Invalid response format"])))
                }
            } catch {
                completion(.failure(error))
            }
        }.resume()
    }
    
    func scanWithOpenAI(image: UIImage, completion: @escaping (Result<[TicketRow], Error>) -> Void) {
        // Check internet connectivity first
        guard NetworkConnectivityManager.shared.isInternetAvailable() else {
            completion(.failure(NSError(domain: "OpenAI", code: -1000, userInfo: [NSLocalizedDescriptionKey: "No internet connection. Please check your network and try again."])))
            return
        }
        
        // Resize image for optimal upload
        guard let resizedImage = image.resizedForUpload() else {
            completion(.failure(NSError(domain: "OpenAI", code: -1, userInfo: [NSLocalizedDescriptionKey: "Could not resize image"])))
            return
        }
        
        // Convert to optimized JPEG data
        guard let imageData = resizedImage.optimizedJPEGData() else {
            completion(.failure(NSError(domain: "OpenAI", code: -1, userInfo: [NSLocalizedDescriptionKey: "Failed to convert image to data"])))
            return
        }
        
        let base64Image = imageData.base64EncodedString()
        
        // Log image size information
        
        // Debug logging
        let key = APIKeys.openAIAPIKey
        precondition(!key.contains("YOUR_"), "APIKeys.swift still has placeholder")
        
        let requestBody: [String: Any] = [
            "model": "gpt-4o",
            "response_format": ["type": "json_object"],
            "messages": [
                [
                    "role": "user",
                    "content": [
                        [
                            "type": "text",
                            "text": """
                               I need to extract lottery numbers from a legitimate lottery ticket for verification purposes.
                               This is for a personal lottery ticket checking app to verify winning numbers.
                               
                               IMPORTANT: This ticket has multiple rows of lottery numbers. You must scan the ENTIRE ticket image from top to bottom.
                               Look for ALL rows labeled A, B, C, D, E, F, G, H, I, J (or similar) that contain lottery numbers.
                               
                               Return ONLY a JSON object with this exact schema:
                               {"rows": [{"numbers":[n,n,n,n,n], "special": n}]} with no prose, no code fences.
                               
                               Rules:
                               - Scan the ENTIRE ticket image from top to bottom
                               - Extract numbers from ALL rows visible in the image
                               - Look for rows labeled A, B, C, D, E, F, G, H, I, J or similar
                               - If you cannot read a number, use -1
                               - Each row should have 5 regular numbers and 1 special number
                               - Regular numbers: 1-69, Special numbers: 1-26
                               - CRITICAL: Look for ALL rows from top to bottom of the ticket
                               - The ticket has multiple rows - extract EVERY single row you can see
                               
                               Example response format:
                               {"rows": [
                                 {"numbers":[10,36,47,63,74], "special": 4},
                                 {"numbers":[11,23,50,60,71], "special": 9},
                                 {"numbers":[44,47,53,62,74], "special": 8},
                                 {"numbers":[4,24,30,31,49], "special": 4},
                                 {"numbers":[1,10,29,49,53], "special": 14},
                                 {"numbers":[12,20,21,24,65], "special": 12},
                                 {"numbers":[15,17,43,51,61], "special": 11},
                                 {"numbers":[8,10,15,63,69], "special": 1},
                                 {"numbers":[5,7,12,55,63], "special": 9},
                                 {"numbers":[41,42,44,50,69], "special": 1}
                               ]}
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
            "max_tokens": 3000
        ]
        
        guard let url = URL(string: openAIBaseURL) else {
            completion(.failure(NSError(domain: "OpenAI", code: -2, userInfo: [NSLocalizedDescriptionKey: "Invalid URL"])))
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("Bearer \(openAIAPIKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        } catch {
            completion(.failure(error))
            return
        }
        
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                completion(.failure(error))
                return
            }
            
            guard let data = data else {
                completion(.failure(NSError(domain: "OpenAI", code: -3, userInfo: [NSLocalizedDescriptionKey: "No data received"])))
                return
            }
            
            do {
                if let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                   let choices = json["choices"] as? [[String: Any]],
                   let firstChoice = choices.first,
                   let message = firstChoice["message"] as? [String: Any] {
                    
                    // Check for refusal first
                    if let refusal = message["refusal"] as? String, !refusal.isEmpty {
                        completion(.failure(NSError(domain: "OpenAI", code: -7, userInfo: [NSLocalizedDescriptionKey: "OpenAI refused to process the image: \(refusal)"])))
                        return
                    }
                    
                    if let content = message["content"] as? String {
                        // Parse the OpenAI response
                        self.parseOpenAIResponse(content: content, completion: completion)
                    } else {
                        completion(.failure(NSError(domain: "OpenAI", code: -4, userInfo: [NSLocalizedDescriptionKey: "Invalid response format"])))
                    }
                } else {
                    completion(.failure(NSError(domain: "OpenAI", code: -5, userInfo: [NSLocalizedDescriptionKey: "Invalid JSON structure"])))
                }
            } catch {
                completion(.failure(NSError(domain: "OpenAI", code: -5, userInfo: [NSLocalizedDescriptionKey: "JSON decode error: \(error.localizedDescription)"])))
            }
        }.resume()
    }
    
    private func parseOpenAIResponse(content: String, completion: @escaping (Result<[TicketRow], Error>) -> Void) {
        // Extract JSON (may be array or object; may be wrapped in code fences)
        let fencedPattern = #"```[a-zA-Z]*\s*([\s\S]*?)\s*```"#
        
        var jsonString: String?
        
        if let range = content.range(of: fencedPattern, options: .regularExpression) {
            let fenced = String(content[range])
            jsonString = fenced.replacingOccurrences(of: "```json", with: "")
                               .replacingOccurrences(of: "```", with: "")
                               .trimmingCharacters(in: .whitespacesAndNewlines)
        } else {
            jsonString = content.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        
        guard let json = jsonString,
              let data = json.data(using: .utf8) else {
            completion(.failure(NSError(domain: "OpenAI", code: -5, userInfo: [NSLocalizedDescriptionKey: "Could not extract JSON from response"])))
            return
        }
        
        do {
            let top = try JSONSerialization.jsonObject(with: data)
            var rowsJSON: [[String: Any]] = []
            
            // Handle different response formats
            if let array = top as? [[String: Any]] {
                // Direct array format: [{"numbers":[...], "special": ...}]
                rowsJSON = array
            } else if let obj = top as? [String: Any], let rows = obj["rows"] as? [[String: Any]] {
                // Nested rows format: {"rows": [{"numbers":[...], "special": ...}]}
                rowsJSON = rows
            } else if let singleRow = top as? [String: Any] {
                // Single row format: {"numbers":[...], "special": ...}
                rowsJSON = [singleRow]
            } else {
                completion(.failure(NSError(domain: "OpenAI", code: -6, userInfo: [NSLocalizedDescriptionKey: "Invalid JSON structure"])))
                return
            }
                
                var ticketRows: [TicketRow] = []
                
                for rowData in rowsJSON {
                    // Handle both old format (regular_numbers) and new format (numbers)
                    let regularNumbersRaw = (rowData["regular_numbers"] as? [Any]) ?? (rowData["numbers"] as? [Any]) ?? []
                    // Coerce each element to Int; use -1 if missing/invalid
                    let coercedRegulars: [Int] = regularNumbersRaw.map { elem in
                        if let v = elem as? Int { return v }
                        if let s = elem as? String, let v = Int(s) { return v }
                        return -1
                    }
                    let specialRaw = rowData["special_number"] ?? rowData["special"]
                    let specialNumber: Int? = {
                        if let v = specialRaw as? Int { return v }
                        if let s = specialRaw as? String, let v = Int(s) { return v }
                        return -1
                    }()
                    
                    // Pad to 5 numbers if needed with -1 for unknowns
                    let paddedNumbers = coercedRegulars + Array(repeating: -1, count: max(0, 5 - coercedRegulars.count))
                    
                    let ticketRow = TicketRow(numbers: paddedNumbers, special: specialNumber)
                    ticketRows.append(ticketRow)
                }
                
                completion(.success(ticketRows))
        } catch {
            completion(.failure(error))
        }
    }
    
    // MARK: - Grid Detection and Individual Number Scanning
    
    func parseTicketWithGridDetection(from image: UIImage, gameType: String = "us_mega_millions", completion: @escaping (Result<[TicketRow], Error>) -> Void) {
        // Step 1: Preprocess the image
        let processedImage = preprocessImageForOCR(image)
        
        // Step 2: Detect the lottery grid structure
        detectLotteryGrid(from: processedImage) { [weak self] result in
            switch result {
            case .success(let grid):
                
                // Step 3: Scan each individual number with initial game type
                self?.scanIndividualNumbers(grid: grid, gameType: gameType) { scanResult in
                    switch scanResult {
                    case .success(let rows):
                        // Check if we got meaningful results
                        let totalNumbers = rows.flatMap { $0.numbers }.filter { $0 > 0 }.count
                        let totalSpecials = rows.compactMap { $0.special }.filter { $0 > 0 }.count
                        
                        if totalNumbers >= 5 || totalSpecials >= 1 {
                            // Step 4: Detect actual game type from scanned numbers and re-scan if needed
                            let detectedGameType = self?.detectGameTypeFromRows(rows) ?? gameType
                            if detectedGameType != gameType {
                                self?.scanIndividualNumbers(grid: grid, gameType: detectedGameType, completion: completion)
                            } else {
                                completion(.success(rows))
                            }
                        } else {
                            // Fallback to OpenAI Vision API
                            self?.scanWithOpenAI(image: image, completion: completion)
                        }
                    case .failure(let error):
                        // Fallback to OpenAI Vision API
                        self?.scanWithOpenAI(image: image, completion: completion)
                    }
                }
                
            case .failure(let error):
                // Fallback to OpenAI Vision API
                self?.scanWithOpenAI(image: image, completion: completion)
            }
        }
    }
    
    private func detectLotteryGrid(from image: UIImage, completion: @escaping (Result<LotteryGrid, Error>) -> Void) {
        guard let cgImage = image.cgImage else {
            completion(.failure(NSError(domain: "GridDetection", code: -1, userInfo: [NSLocalizedDescriptionKey: "Invalid image"])))
            return
        }
        
        // Use Vision to detect text regions
        let request = VNRecognizeTextRequest { request, error in
            guard error == nil,
                  let observations = request.results as? [VNRecognizedTextObservation] else {
                completion(.failure(error ?? NSError(domain: "GridDetection", code: -2)))
                return
            }
            
            // Extract number regions
            let numberRegions = self.extractNumberRegions(from: observations, imageSize: image.size)
            
            // Detect grid structure
            let grid = self.detectGridStructure(numberRegions: numberRegions, image: image)
            
            completion(.success(grid))
        }
        
        request.recognitionLevel = .accurate
        request.usesLanguageCorrection = false
        request.minimumTextHeight = 0.01 // Very small to catch individual numbers
        
        let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                try handler.perform([request])
            } catch {
                completion(.failure(error))
            }
        }
    }
    
    private func extractNumberRegions(from observations: [VNRecognizedTextObservation], imageSize: CGSize) -> [(text: String, bounds: CGRect)] {
        var numberRegions: [(text: String, bounds: CGRect)] = []
        
        for observation in observations {
            guard let candidate = observation.topCandidates(1).first else { continue }
            let text = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
            
            // Check if this looks like a lottery number
            if self.isLotteryNumber(text) {
                // Convert normalized coordinates to image coordinates
                let imageBounds = VNImageRectForNormalizedRect(
                    observation.boundingBox,
                    Int(imageSize.width),
                    Int(imageSize.height)
                )
                
                numberRegions.append((text: text, bounds: imageBounds))
            }
        }
        
        return numberRegions
    }
    
    private func isLotteryNumber(_ text: String) -> Bool {
        // Check if text contains only digits and is a reasonable lottery number
        guard text.range(of: #"^[0-9]+$"#, options: .regularExpression) != nil else { return false }
        
        if let number = Int(text) {
            return (1...99).contains(number) // Most lottery numbers are 1-99
        }
        
        return false
    }
    
    private func detectGridStructure(numberRegions: [(text: String, bounds: CGRect)], image: UIImage) -> LotteryGrid {
        // Sort regions by position (top to bottom, left to right)
        let sortedRegions = numberRegions.sorted { region1, region2 in
            let dy = abs(region1.bounds.midY - region2.bounds.midY)
            return dy < 20 ? region1.bounds.minX < region2.bounds.minX : region1.bounds.midY < region2.bounds.midY
        }
        
        // Group into rows based on Y-coordinate proximity
        var rows: [[(text: String, bounds: CGRect)]] = []
        var currentRow: [(text: String, bounds: CGRect)] = []
        var lastY: CGFloat = -1
        
        for region in sortedRegions {
            if lastY == -1 || abs(region.bounds.midY - lastY) < 30 {
                currentRow.append(region)
            } else {
                if !currentRow.isEmpty {
                    rows.append(currentRow)
                }
                currentRow = [region]
            }
            lastY = region.bounds.midY
        }
        if !currentRow.isEmpty {
            rows.append(currentRow)
        }
        
        
        // Create number positions
        var numberPositions: [NumberPosition] = []
        
        for (rowIndex, row) in rows.enumerated() {
            // Sort row by X coordinate
            let sortedRow = row.sorted { $0.bounds.minX < $1.bounds.minX }
            
            for (colIndex, region) in sortedRow.enumerated() {
                // Crop individual number image
                if let numberImage = self.cropNumberImage(from: image, bounds: region.bounds) {
                    let isSpecial = colIndex >= 5 // Assume special numbers are in column 5+
                    let position = NumberPosition(
                        row: rowIndex,
                        column: colIndex,
                        image: numberImage,
                        originalBounds: region.bounds,
                        isSpecial: isSpecial
                    )
                    numberPositions.append(position)
                    
                }
            }
        }
        
        return LotteryGrid(rows: rows.count, columns: 6, numberPositions: numberPositions)
    }
    
    private func cropNumberImage(from image: UIImage, bounds: CGRect) -> UIImage? {
        // Add padding around the number (slightly larger to ensure full glyph capture)
        let padding: CGFloat = 18
        let expandedBounds = CGRect(
            x: max(0, bounds.minX - padding),
            y: max(0, bounds.minY - padding),
            width: min(image.size.width - bounds.minX + padding, bounds.width + 2 * padding),
            height: min(image.size.height - bounds.minY + padding, bounds.height + 2 * padding)
        )
        
        guard let cgImage = image.cgImage?.cropping(to: expandedBounds) else { return nil }
        return UIImage(cgImage: cgImage)
    }
    
    private func scanIndividualNumbers(grid: LotteryGrid, gameType: String, completion: @escaping (Result<[TicketRow], Error>) -> Void) {
        let dispatchGroup = DispatchGroup()
        var results: [NumberPosition: Int] = [:]
        var errors: [Error] = []
        
        // Determine constraints based on detected game
        let constraints = self.getGameConstraints(for: gameType)
        
        // Scan each number individually
        for position in grid.numberPositions {
            dispatchGroup.enter()
            
            scanSingleNumber(position: position, image: position.image, maxMain: constraints.maxMain, maxSpecial: constraints.maxSpecial) { result in
                defer { dispatchGroup.leave() }
                
                switch result {
                case .success(let number):
                    results[position] = number
                case .failure(let error):
                    errors.append(error)
                }
            }
        }
        
        // Wait for all scans to complete
        dispatchGroup.notify(queue: .main) {
            if !errors.isEmpty {
            }
            
            // Reconstruct lottery data
            let ticketRows = self.reconstructTicketRows(from: grid, results: results)
            completion(.success(ticketRows))
        }
    }
    
    private func scanSingleNumber(position: NumberPosition, image: UIImage, maxMain: Int, maxSpecial: Int, completion: @escaping (Result<Int, Error>) -> Void) {
        // Build preprocessing variants to try sequentially
        let variants = generateDigitVariants(from: image)
        
        func runOCR(on cgImage: CGImage, completion: @escaping (Result<Int, Error>) -> Void) {
            let request = VNRecognizeTextRequest { request, error in
            // Helper to normalize OCR text into a clean integer
            func normalizeToInt(_ raw: String) -> Int? {
                // Strip non-digits, fix common OCR confusions
                let cleaned = raw
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .replacingOccurrences(of: "O", with: "0")
                    .replacingOccurrences(of: "Q", with: "0")
                    .replacingOccurrences(of: "D", with: "0")
                    .replacingOccurrences(of: "I", with: "1")
                    .replacingOccurrences(of: "l", with: "1")
                    .replacingOccurrences(of: "S", with: "5")
                    .replacingOccurrences(of: "B", with: "8")
                    .replacingOccurrences(of: "G", with: "6")
                    .replacingOccurrences(of: "Z", with: "2")
                let digits = cleaned.filter({ $0.isNumber })
                guard !digits.isEmpty else { return nil }
                // Enforce per-column length: main numbers are 1-2 digits typically
                var trimmedDigits = digits
                if !position.isSpecial && trimmedDigits.count > 2 {
                    trimmedDigits = String(trimmedDigits.suffix(2))
                }
                if let value = Int(trimmedDigits) {
                    if position.isSpecial {
                        return (1...maxSpecial).contains(value) ? value : nil
                    } else {
                        return (1...maxMain).contains(value) ? value : nil
                    }
                }
                return nil
            }
            
            // Primary pass
            if error == nil,
               let observations = request.results as? [VNRecognizedTextObservation],
               let top = observations.first,
               let candidate = top.topCandidates(1).first,
               let num = normalizeToInt(candidate.string) {
                completion(.success(num))
                return
            }
            
            // Fallback: attempt with more candidates
            if let observations = request.results as? [VNRecognizedTextObservation] {
                for obs in observations {
                    for cand in obs.topCandidates(5) {
                        if let num = normalizeToInt(cand.string) {
                            completion(.success(num))
                            return
                        }
                    }
                }
            }
            
            completion(.failure(error ?? NSError(domain: "SingleNumberOCR", code: -3, userInfo: [NSLocalizedDescriptionKey: "Unable to parse number"])) )
        }
            
            request.recognitionLevel = .accurate
            request.usesLanguageCorrection = false
            request.minimumTextHeight = 0.0035 // capture smaller glyphs
            request.customWords = (0...99).map { String($0) }
            
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try handler.perform([request])
                } catch {
                    completion(.failure(error))
                }
            }
        }
        
        // Try variants sequentially until one succeeds
        func tryNext(index: Int) {
            if index >= variants.count {
                completion(.failure(NSError(domain: "SingleNumberOCR", code: -4, userInfo: [NSLocalizedDescriptionKey: "All variants failed"])) )
                return
            }
            guard let cg = variants[index].cgImage else { tryNext(index: index + 1); return }
            runOCR(on: cg) { result in
                switch result {
                case .success(let n): completion(.success(n))
                case .failure: tryNext(index: index + 1)
                }
            }
        }
        tryNext(index: 0)
    }

    // Upscale and binarize small digit crops for better OCR
    private func upscaleAndBinarizeForDigit(_ image: UIImage) -> UIImage {
        let scale: CGFloat = 3.0
        let newSize = CGSize(width: image.size.width * scale, height: image.size.height * scale)
        let renderer = UIGraphicsImageRenderer(size: newSize)
        let upscaled = renderer.image { ctx in
            UIColor.white.setFill()
            ctx.fill(CGRect(origin: .zero, size: newSize))
            image.draw(in: CGRect(origin: .zero, size: newSize))
        }
        // Light preprocessing
        return preprocessImageForOCR(upscaled)
    }

    // Produce multiple preprocessing variants to improve OCR robustness
    private func generateDigitVariants(from image: UIImage) -> [UIImage] {
        var variants: [UIImage] = []
        let base = upscaleAndBinarizeForDigit(image)
        variants.append(base)
        
        // Stronger contrast/sharpen variant
        if let cg = base.cgImage {
            var ci = CIImage(cgImage: cg)
            if let controls = CIFilter(name: "CIColorControls") {
                controls.setValue(ci, forKey: kCIInputImageKey)
                controls.setValue(2.0, forKey: kCIInputContrastKey)
                controls.setValue(0.15, forKey: kCIInputBrightnessKey)
                controls.setValue(0.0, forKey: kCIInputSaturationKey)
                if let out = controls.outputImage { ci = out }
            }
            if let sharp = CIFilter(name: "CISharpenLuminance") {
                sharp.setValue(ci, forKey: kCIInputImageKey)
                sharp.setValue(1.0, forKey: kCIInputSharpnessKey)
                if let out = sharp.outputImage { ci = out }
            }
            let ctx = CIContext()
            if let outCG = ctx.createCGImage(ci, from: ci.extent) {
                variants.append(UIImage(cgImage: outCG))
            }
        }
        
        // Inverted variant (tickets sometimes have dark text/light noise)
        if let cg = base.cgImage {
            let ci = CIImage(cgImage: cg).applyingFilter("CIColorInvert")
            let ctx = CIContext()
            if let outCG = ctx.createCGImage(ci, from: ci.extent) {
                variants.append(UIImage(cgImage: outCG))
            }
        }
        
        return variants
    }

    private func getGameConstraints(for game: String) -> (maxMain: Int, maxSpecial: Int) {
        // Comprehensive game constraints based on magayo.com documentation
        switch game {
        // USA Games
        case "us_mega_millions": return (maxMain: 70, maxSpecial: 25)
        case "us_powerball": return (maxMain: 69, maxSpecial: 26)
        case "us_lotto_america": return (maxMain: 52, maxSpecial: 10)
        case "us_cash4life": return (maxMain: 60, maxSpecial: 4)
        
        // European Games
        case "euromillions": return (maxMain: 50, maxSpecial: 12)
        case "uk_lotto": return (maxMain: 59, maxSpecial: 0)
        case "irish_lotto": return (maxMain: 47, maxSpecial: 0)
        case "spanish_lottery": return (maxMain: 49, maxSpecial: 0)
        case "italian_superenalotto": return (maxMain: 90, maxSpecial: 0)
        case "french_lotto": return (maxMain: 49, maxSpecial: 10)
        case "german_lotto": return (maxMain: 49, maxSpecial: 0)
        
        // Australian Games
        case "au_oz_lotto": return (maxMain: 45, maxSpecial: 0)
        case "au_powerball": return (maxMain: 35, maxSpecial: 20)
        case "au_saturday_lotto": return (maxMain: 45, maxSpecial: 0)
        
        // Canadian Games
        case "ca_lotto_max": return (maxMain: 50, maxSpecial: 0)
        case "ca_lotto_649": return (maxMain: 49, maxSpecial: 0)
        
        // Other International Games
        case "brazil_mega_sena": return (maxMain: 60, maxSpecial: 0)
        case "mexico_melate": return (maxMain: 56, maxSpecial: 0)
        case "south_africa_lotto": return (maxMain: 52, maxSpecial: 0)
        case "japan_lotto": return (maxMain: 43, maxSpecial: 0)
        
        default: return (maxMain: 70, maxSpecial: 25) // Default to Mega Millions
        }
    }
    
    private func detectGameTypeFromRows(_ rows: [TicketRow]) -> String {
        let allNumbers = rows.flatMap { $0.numbers }.filter { $0 > 0 }
        let allSpecials = rows.compactMap { $0.special }.filter { $0 > 0 }
        
        let maxNumber = allNumbers.max() ?? 0
        let maxSpecial = allSpecials.max() ?? 0
        
        // Detect game type based on number ranges and special ball constraints
        // USA Games
        if maxNumber <= 70 && (maxSpecial == 0 || maxSpecial <= 25) {
            return "us_mega_millions"
        }
        if maxNumber <= 69 && (maxSpecial == 0 || maxSpecial <= 26) {
            return "us_powerball"
        }
        if maxNumber <= 52 && (maxSpecial == 0 || maxSpecial <= 10) {
            return "us_lotto_america"
        }
        if maxNumber <= 60 && (maxSpecial == 0 || maxSpecial <= 4) {
            return "us_cash4life"
        }
        
        // European Games
        if maxNumber <= 50 && (maxSpecial == 0 || maxSpecial <= 12) {
            return "euromillions"
        }
        if maxNumber <= 59 && maxSpecial == 0 {
            return "uk_lotto"
        }
        if maxNumber <= 47 && maxSpecial == 0 {
            return "irish_lotto"
        }
        if maxNumber <= 90 && maxSpecial == 0 {
            return "italian_superenalotto"
        }
        if maxNumber <= 49 && (maxSpecial == 0 || maxSpecial <= 10) {
            return "french_lotto"
        }
        
        // Australian Games
        if maxNumber <= 45 && maxSpecial == 0 {
            return "au_oz_lotto"
        }
        if maxNumber <= 35 && (maxSpecial == 0 || maxSpecial <= 20) {
            return "au_powerball"
        }
        
        // Canadian Games
        if maxNumber <= 50 && maxSpecial == 0 {
            return "ca_lotto_max"
        }
        if maxNumber <= 49 && maxSpecial == 0 {
            return "ca_lotto_649"
        }
        
        // Other International Games
        if maxNumber <= 60 && maxSpecial == 0 {
            return "brazil_mega_sena"
        }
        if maxNumber <= 56 && maxSpecial == 0 {
            return "mexico_melate"
        }
        if maxNumber <= 52 && maxSpecial == 0 {
            return "south_africa_lotto"
        }
        if maxNumber <= 43 && maxSpecial == 0 {
            return "japan_lotto"
        }
        
        // Default to Mega Millions for most lottery tickets
        return "us_mega_millions"
    }
    
    private func reconstructTicketRows(from grid: LotteryGrid, results: [NumberPosition: Int]) -> [TicketRow] {
        var ticketRows: [TicketRow] = []
        
        // Group positions by row
        let positionsByRow = Dictionary(grouping: grid.numberPositions) { $0.row }
        
        for rowIndex in 0..<grid.rows {
            let rowPositions = positionsByRow[rowIndex] ?? []
            
            // Separate regular numbers and special numbers
            let regularPositions = rowPositions.filter { !$0.isSpecial }.sorted { $0.column < $1.column }
            let specialPositions = rowPositions.filter { $0.isSpecial }.sorted { $0.column < $1.column }
            
            // Extract numbers
            let regularNumbers = regularPositions.compactMap { results[$0] }
            let specialNumber = specialPositions.first.flatMap { results[$0] }
            
            // Pad regular numbers to 5 if needed
            let paddedNumbers = regularNumbers + Array(repeating: 0, count: max(0, 5 - regularNumbers.count))
            
            let ticketRow = TicketRow(numbers: paddedNumbers, special: specialNumber)
            ticketRows.append(ticketRow)
            
        }
        
        return ticketRows
    }
    
    // MARK: - Image Preprocessing
    
    private func preprocessImageForOCR(_ image: UIImage) -> UIImage {
        guard let cgImage = image.cgImage else { return image }
        
        let context = CIContext()
        var ciImage = CIImage(cgImage: cgImage)
        
        // 1. Enhance contrast and brightness
        if let contrastFilter = CIFilter(name: "CIColorControls") {
            contrastFilter.setValue(ciImage, forKey: kCIInputImageKey)
            contrastFilter.setValue(1.5, forKey: kCIInputContrastKey) // Increase contrast
            contrastFilter.setValue(0.1, forKey: kCIInputBrightnessKey) // Slight brightness increase
            contrastFilter.setValue(1.0, forKey: kCIInputSaturationKey)
            if let output = contrastFilter.outputImage {
                ciImage = output
            }
        }
        
        // 2. Apply sharpening to make text clearer
        if let sharpenFilter = CIFilter(name: "CISharpenLuminance") {
            sharpenFilter.setValue(ciImage, forKey: kCIInputImageKey)
            sharpenFilter.setValue(0.8, forKey: kCIInputSharpnessKey) // Moderate sharpening
            if let output = sharpenFilter.outputImage {
                ciImage = output
            }
        }
        
        // 3. Convert to grayscale to reduce noise
        if let grayFilter = CIFilter(name: "CIColorMonochrome") {
            grayFilter.setValue(ciImage, forKey: kCIInputImageKey)
            grayFilter.setValue(CIColor.white, forKey: kCIInputColorKey)
            grayFilter.setValue(1.0, forKey: kCIInputIntensityKey)
            if let output = grayFilter.outputImage {
                ciImage = output
            }
        }
        
        // 4. Apply noise reduction using CIGaussianBlur for gentle smoothing
        if let blurFilter = CIFilter(name: "CIGaussianBlur") {
            blurFilter.setValue(ciImage, forKey: kCIInputImageKey)
            blurFilter.setValue(0.5, forKey: kCIInputRadiusKey) // Very light blur for noise reduction
            if let output = blurFilter.outputImage {
                ciImage = output
            }
        }
        
        // Convert back to UIImage
        guard let outputCGImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            return image
        }
        
        return UIImage(cgImage: outputCGImage)
    }
    
    func parseTicket(from image: UIImage, completion: @escaping (Result<[TicketRow], Error>) -> Void) {
        // Preprocess the image to improve OCR accuracy
        let processedImage = preprocessImageForOCR(image)
        
        guard let cg = processedImage.cgImage else {
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
            
            // Enhanced filtering for lottery ticket numbers
            let digitish = lines
                .map { (text: $0.text.trimmingCharacters(in: .whitespacesAndNewlines), box: $0.box) }
                .filter { line in
                    // More flexible pattern to catch lottery numbers with various separators
                    let text = line.text
                    return text.range(of: #"^[0-9\s,\|]+$"#, options: .regularExpression) != nil &&
                           text.count >= 3 && // At least 3 characters (2 digits + separator)
                           text.range(of: #"[0-9]"#, options: .regularExpression) != nil
                }
            
            for line in digitish {
            }
            
            // group into rows by Y proximity with tighter tolerance
            let rows = Self.groupByRows(digitish, yTol: 0.015)
            
            // parse each row with enhanced logic
            let parsed: [TicketRow] = rows.compactMap { row -> TicketRow? in
                let merged = row.map(\.text).joined(separator: " ")
                    .replacingOccurrences(of: "  ", with: " ")
                    .replacingOccurrences(of: ",", with: " ")
                    .replacingOccurrences(of: "|", with: " ")
                
                // Extract numbers more intelligently
                let tokens = merged.split(separator: " ").compactMap { token -> Int? in
                    let cleaned = String(token).trimmingCharacters(in: .whitespacesAndNewlines)
                    // Handle common OCR mistakes
                    let corrected = cleaned
                        .replacingOccurrences(of: "O", with: "0")
                        .replacingOccurrences(of: "I", with: "1")
                        .replacingOccurrences(of: "l", with: "1")
                        .replacingOccurrences(of: "S", with: "5")
                        .replacingOccurrences(of: "B", with: "8")
                    return Int(corrected)
                }
                
                
                // More flexible row validation - allow rows with just numbers
                guard tokens.count >= 1 else { 
                    return nil 
                }
                
                // Detect if this looks like a lottery row (5 main numbers + 1 special)
                if tokens.count >= 5 {
                    // Likely a full lottery row
                    let specials = tokens.count > 5 ? tokens.last : nil
                    let normals = Array(tokens.prefix(5))
                    
                let saneNormals = normals.map { num in
                        (1...70).contains(num) ? num : 0
                }
                let saneSpecial = specials.flatMap { num in
                        (1...99).contains(num) ? num : 0
                }
                
                return TicketRow(numbers: saneNormals, special: saneSpecial)
                } else {
                    // Partial row - pad with zeros
                    let paddedNumbers = Array(tokens.prefix(5)) + Array(repeating: 0, count: max(0, 5 - tokens.count))
                    let saneNormals = paddedNumbers.map { num in
                        (1...70).contains(num) ? num : 0
                    }
                    
                    return TicketRow(numbers: saneNormals, special: nil)
                }
            }
            
            completion(.success(parsed))
        }
        
        req.recognitionLevel = .accurate
        req.usesLanguageCorrection = false
        req.minimumTextHeight = 0.015 // Reduced for smaller text
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
    var selectedGame: LotteryGame?
    
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
            return 
        }
        let videoInput: AVCaptureDeviceInput
        
        do {
            videoInput = try AVCaptureDeviceInput(device: videoCaptureDevice)
        } catch {
            return
        }
        
        if captureSession.canAddInput(videoInput) {
            captureSession.addInput(videoInput)
        } else {
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
            return
        }
        
        // Add photo output for capture functionality
        photoOutput = AVCapturePhotoOutput()
        if captureSession.canAddOutput(photoOutput) {
            captureSession.addOutput(photoOutput)
        } else {
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
        instructionLabel.text = "Position lottery ticket within the green frame"
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
        
        
        // Crop the image
        guard let cgImage = image.cgImage?.cropping(to: cropRect) else {
            return image
        }
        
        let croppedImage = UIImage(cgImage: cgImage)
        
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
                    return
            }
            
            // Find the largest rectangle (likely the lottery ticket)
            let largestRectangle = observations.max { $0.boundingBox.width * $0.boundingBox.height < $1.boundingBox.width * $1.boundingBox.height }
            
            if let rectangle = largestRectangle {
                
                // Convert normalized coordinates to image coordinates
                let cropRect = CGRect(
                    x: rectangle.boundingBox.origin.x * imageSize.width,
                    y: (1 - rectangle.boundingBox.origin.y - rectangle.boundingBox.height) * imageSize.height, // Flip Y coordinate
                    width: rectangle.boundingBox.width * imageSize.width,
                    height: rectangle.boundingBox.height * imageSize.height
                )
                
                
                // Crop the image to the detected rectangle
                DispatchQueue.main.async {
                    self.cropImageToRect(image, rect: cropRect, scanningFrame: scanningFrame, previewSize: previewSize)
                }
                } else {
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
                        return
                    }
            
            // Look for "LOTTERY" text to identify the ticket area
            var ticketBounds: CGRect?
            
            for observation in observations {
                if let candidate = observation.topCandidates(1).first,
                   candidate.string.uppercased().contains("LOTTERY") {
                    ticketBounds = observation.boundingBox
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
                
                
                DispatchQueue.main.async {
                    self.cropImageToRect(image, rect: expandedRect, scanningFrame: scanningFrame, previewSize: previewSize)
                }
            } else {
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
        
        // Show progress dialog during OpenAI request
        let progressAlert = UIAlertController(title: "Scanning Lottery Ticket", message: "Analyzing ticket with AI...", preferredStyle: .alert)
        present(progressAlert, animated: true)
        
        // OpenAI-only scanning path (local OCR disabled)
        ocr.scanWithOpenAI(image: image) { result in
            DispatchQueue.main.async {
                progressAlert.dismiss(animated: true) {
                switch result {
                case .success(let rows):
                    for (index, row) in rows.enumerated() {
                            let regularStr = row.numbers.map { $0 == -1 ? "ISSUE" : String($0) }.joined(separator: ", ")
                            let powerballStr = (row.special ?? -1) == -1 ? "ISSUE" : String(row.special!)
                    }
                        
                        // Count issues (-1 values)
                        let issueCount = self.countIssues(in: rows)
                        if issueCount > 0 {
                            self.showIssuesToast(count: issueCount)
                    }
                    
                    // Convert to lottery data format
                    let lotteryData = self.convertTicketRowsToLotteryData(rows)
                    
                    if !lotteryData.isEmpty {
                        AudioServicesPlaySystemSound(SystemSoundID(kSystemSoundID_Vibrate))
                        self.captureSession.stopRunning()
                        self.delegate?.didScanQRCode(lotteryData, image: self.capturedImage)
                    } else {
                        self.showNoDataFoundAlert()
                    }
                    
                case .failure(let error):
                    
                    // Check if it's a network connectivity error
                    if let nsError = error as NSError?, nsError.code == -1000 {
                        self.showNetworkErrorAlert()
                    } else {
                        self.showNoDataFoundAlert()
                    }
                    }
                }
            }
        }
    }
    
    private func convertTicketRowsToLotteryData(_ rows: [TicketRow]) -> String {
        var allRows: [String] = []
        
        for (index, row) in rows.enumerated() {
        }
        
        // Filter out completely empty rows (all 0s)
        let validRows = rows.filter { row in
            let hasValidNumbers = row.numbers.contains { $0 > 0 }
            let hasValidSpecial = (row.special ?? 0) > 0
            return hasValidNumbers || hasValidSpecial
        }
        
        
        // Process all valid rows - no hardcoded limit
        for (i, row) in validRows.enumerated() {
                // Ensure we have exactly 5 regular numbers and 1 powerball
                let regularNumbers = Array(row.numbers.prefix(5))
            let paddedRegulars = regularNumbers + Array(repeating: -1, count: max(0, 5 - regularNumbers.count))
            let powerball = row.special ?? -1
                
                let regularNumbersStr = paddedRegulars.map { String($0) }.joined(separator: " ")
                let powerballStr = String(powerball)
                let rowData = "\(regularNumbersStr) \(powerballStr)"
                allRows.append(rowData)
        }
        
        let result = "Lottery: \(allRows.joined(separator: "|")) Ticket:OCR"
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
    
    private func showNetworkErrorAlert() {
        let alert = UIAlertController(title: "No Internet Connection", 
                                    message: "Please check your internet connection and try again. Make sure you're connected to Wi-Fi or cellular data.", 
                                    preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
    
    private func countIssues(in rows: [TicketRow]) -> Int {
        var issueCount = 0
        
        for row in rows {
            // Count -1 values in regular numbers
            issueCount += row.numbers.filter { $0 == -1 }.count
            
            // Count -1 values in special numbers
            if let special = row.special, special == -1 {
                issueCount += 1
            }
        }
        
        return issueCount
    }
    
    private func showIssuesToast(count: Int) {
        let message = "\(count) issues found"
        DispatchQueue.main.async {
            // Remove existing toast if any
            let toastTag = 987654
            self.view.viewWithTag(toastTag)?.removeFromSuperview()
            
            // Container view with red background
            let toastView = UIView()
            toastView.tag = toastTag
            toastView.backgroundColor = UIColor.systemRed
            toastView.alpha = 0.0
            toastView.layer.cornerRadius = 10
            toastView.layer.masksToBounds = true
            toastView.translatesAutoresizingMaskIntoConstraints = false
            
            // Label
            let label = UILabel()
            label.text = message
            label.textColor = .white
            label.font = UIFont.boldSystemFont(ofSize: 15)
            label.numberOfLines = 0
            label.textAlignment = .center
            label.translatesAutoresizingMaskIntoConstraints = false
            toastView.addSubview(label)
            
            self.view.addSubview(toastView)
            
            // Layout: show near the top of the Lottery scan result area
            NSLayoutConstraint.activate([
                toastView.leadingAnchor.constraint(equalTo: self.view.leadingAnchor, constant: 20),
                toastView.trailingAnchor.constraint(equalTo: self.view.trailingAnchor, constant: -20),
                toastView.topAnchor.constraint(equalTo: self.view.safeAreaLayoutGuide.topAnchor, constant: 12),
                
                label.leadingAnchor.constraint(equalTo: toastView.leadingAnchor, constant: 16),
                label.trailingAnchor.constraint(equalTo: toastView.trailingAnchor, constant: -16),
                label.topAnchor.constraint(equalTo: toastView.topAnchor, constant: 12),
                label.bottomAnchor.constraint(equalTo: toastView.bottomAnchor, constant: -12)
            ])
            
            // Animate in/out
            UIView.animate(withDuration: 0.25, animations: {
                toastView.alpha = 0.95
            }) { _ in
                UIView.animate(withDuration: 0.25, delay: 2.5, options: [.curveEaseInOut], animations: {
                    toastView.alpha = 0.0
                }, completion: { _ in
                    toastView.removeFromSuperview()
                })
            }
        }
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
        let correctedImage = fixImageOrientation(image)
        
        // Store the corrected image
        self.capturedImage = correctedImage
        
        if enableCropping {
            // Crop image to scanning area and show preview
            let croppedImage = cropImageToScanningArea(correctedImage)
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
// MARK: - Lottery Data Structures
struct LotteryGame {
    let name: String
    let code: String
}

class ViewController: UIViewController {
    
    // MARK: - Lottery Data
    private var games: [LotteryGame] = []
    private var selectedGame: LotteryGame?
    private var selectedDrawDate: Date?
    
    // MARK: - Network Status
    private var networkStatusLabel: UILabel = {
        let label = UILabel()
        label.font = UIFont.systemFont(ofSize: 12, weight: .medium)
        label.textAlignment = .center
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    // MARK: - UI Elements
    private let scrollView: UIScrollView = {
        let scrollView = UIScrollView()
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        scrollView.backgroundColor = .clear
        scrollView.showsVerticalScrollIndicator = true
        scrollView.alwaysBounceVertical = true
        return scrollView
    }()
    
    private let contentView: UIView = {
        let view = UIView()
        view.translatesAutoresizingMaskIntoConstraints = false
        view.backgroundColor = .clear
        return view
    }()
    
    private let scanButton: UIButton = {
        let button = UIButton(type: .system)
        button.setTitle("Scan QR", for: .normal)
        button.titleLabel?.font = UIFont.systemFont(ofSize: 18, weight: .semibold)
        button.backgroundColor = UIColor.systemBlue
        button.setTitleColor(.white, for: .normal)
        button.layer.cornerRadius = 16
        button.layer.shadowColor = UIColor.systemBlue.cgColor
        button.layer.shadowOffset = CGSize(width: 0, height: 4)
        button.layer.shadowRadius = 8
        button.layer.shadowOpacity = 0.3
        button.translatesAutoresizingMaskIntoConstraints = false
        button.isUserInteractionEnabled = true
        button.isEnabled = true
        
        // Add gradient background
        let gradientLayer = CAGradientLayer()
        gradientLayer.colors = [UIColor.systemBlue.cgColor, UIColor.systemIndigo.cgColor]
        gradientLayer.startPoint = CGPoint(x: 0, y: 0)
        gradientLayer.endPoint = CGPoint(x: 1, y: 1)
        gradientLayer.cornerRadius = 16
        button.layer.insertSublayer(gradientLayer, at: 0)
        
        return button
    }()
    
    private let headerLabel: UILabel = {
        let label = UILabel()
        label.text = "Lottery Scanner"
        label.textAlignment = .center
        label.font = UIFont.systemFont(ofSize: 28, weight: .bold)
        label.textColor = .label
        label.numberOfLines = 1
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    private let instructionLabel: UILabel = {
        let label = UILabel()
        label.text = "Select a lottery game and required draw date to scan tickets"
        label.textAlignment = .center
        label.font = UIFont.systemFont(ofSize: 16, weight: .medium)
        label.textColor = .secondaryLabel
        label.numberOfLines = 0
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    // MARK: - Selection UI Elements
    private let gameLabel: UILabel = {
        let label = UILabel()
        label.text = "Select Game"
        label.font = UIFont.systemFont(ofSize: 18, weight: .semibold)
        label.textColor = .label
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    private let gamePicker: UIPickerView = {
        let picker = UIPickerView()
        picker.translatesAutoresizingMaskIntoConstraints = false
        picker.backgroundColor = UIColor.systemBackground
        picker.layer.cornerRadius = 12
        picker.layer.borderWidth = 1
        picker.layer.borderColor = UIColor.separator.cgColor
        picker.layer.shadowColor = UIColor.black.cgColor
        picker.layer.shadowOffset = CGSize(width: 0, height: 2)
        picker.layer.shadowRadius = 4
        picker.layer.shadowOpacity = 0.1
        return picker
    }()
    
    private let dateLabel: UILabel = {
        let label = UILabel()
        label.text = "Select Draw Date (Required)"
        label.font = UIFont.systemFont(ofSize: 18, weight: .semibold)
        label.textColor = .label
        label.translatesAutoresizingMaskIntoConstraints = false
        return label
    }()
    
    private let datePicker: UIDatePicker = {
        let picker = UIDatePicker()
        picker.translatesAutoresizingMaskIntoConstraints = false
        picker.datePickerMode = .date
        picker.preferredDatePickerStyle = .compact
        picker.backgroundColor = UIColor.systemBackground
        picker.layer.cornerRadius = 12
        picker.layer.borderWidth = 1
        picker.layer.borderColor = UIColor.separator.cgColor
        picker.layer.shadowColor = UIColor.black.cgColor
        picker.layer.shadowOffset = CGSize(width: 0, height: 2)
        picker.layer.shadowRadius = 4
        picker.layer.shadowOpacity = 0.1
        // Set default to today
        picker.date = Date()
        // Set maximum date to today (can't select future dates)
        picker.maximumDate = Date()
        return picker
    }()
    
    
    private let selectionStackView: UIStackView = {
        let stackView = UIStackView()
        stackView.axis = .vertical
        stackView.spacing = 24
        stackView.translatesAutoresizingMaskIntoConstraints = false
        return stackView
    }()
    
    private let selectionContainerView: UIView = {
        let view = UIView()
        view.backgroundColor = UIColor.systemBackground
        view.layer.cornerRadius = 20
        view.layer.shadowColor = UIColor.black.cgColor
        view.layer.shadowOffset = CGSize(width: 0, height: 8)
        view.layer.shadowRadius = 16
        view.layer.shadowOpacity = 0.1
        view.translatesAutoresizingMaskIntoConstraints = false
        return view
    }()
    
    private let backgroundGradientView: UIView = {
        let view = UIView()
        view.translatesAutoresizingMaskIntoConstraints = false
        return view
    }()
    
    // MARK: - Lifecycle
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupConstraints()
        generateInitialTicket()
        setupNetworkMonitoring()
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
        // Update gradient layer frame
        if let gradientLayer = backgroundGradientView.layer.sublayers?.first as? CAGradientLayer {
            gradientLayer.frame = backgroundGradientView.bounds
        }
        
        // Update scan button gradient
        if let gradientLayer = scanButton.layer.sublayers?.first as? CAGradientLayer {
            gradientLayer.frame = scanButton.bounds
        }
    }
    
    // MARK: - UI Setup
    private func setupUI() {
        setupBackgroundGradient()
        title = "Lottery QR Scanner"
        
        // Setup selection UI
        setupSelectionUI()
        
        // Add views to hierarchy
        view.addSubview(backgroundGradientView)
        view.addSubview(scrollView)
        scrollView.addSubview(contentView)
        
        contentView.addSubview(headerLabel)
        contentView.addSubview(selectionContainerView)
        selectionContainerView.addSubview(selectionStackView)
        contentView.addSubview(scanButton)
        contentView.addSubview(instructionLabel)
        contentView.addSubview(networkStatusLabel)
        
        scanButton.addTarget(self, action: #selector(scanButtonTapped), for: .touchUpInside)
        
        // Initially hide scan button
        scanButton.isHidden = true
        
        // Load lottery data
        loadLotteryData()
    }
    
    private func setupBackgroundGradient() {
        let gradientLayer = CAGradientLayer()
        gradientLayer.colors = [
            UIColor.systemBackground.cgColor,
            UIColor.systemGray6.cgColor
        ]
        gradientLayer.startPoint = CGPoint(x: 0, y: 0)
        gradientLayer.endPoint = CGPoint(x: 1, y: 1)
        backgroundGradientView.layer.insertSublayer(gradientLayer, at: 0)
    }
    
    private func setupSelectionUI() {
        // Add elements to stack view
        selectionStackView.addArrangedSubview(gameLabel)
        selectionStackView.addArrangedSubview(gamePicker)
        selectionStackView.addArrangedSubview(dateLabel)
        selectionStackView.addArrangedSubview(datePicker)
        
        // Setup picker delegates
        gamePicker.delegate = self
        gamePicker.dataSource = self
        
        // Setup date picker target
        datePicker.addTarget(self, action: #selector(datePickerChanged), for: .valueChanged)
        
        // Set initial date
        selectedDrawDate = Date()
    }
    
    private func setupConstraints() {
        NSLayoutConstraint.activate([
            // Background Gradient
            backgroundGradientView.topAnchor.constraint(equalTo: view.topAnchor),
            backgroundGradientView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            backgroundGradientView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            backgroundGradientView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            
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
            
            // Header Label
            headerLabel.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 20),
            headerLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            headerLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            
            // Selection Container
            selectionContainerView.topAnchor.constraint(equalTo: headerLabel.bottomAnchor, constant: 30),
            selectionContainerView.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            selectionContainerView.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            
            // Selection Stack View
            selectionStackView.topAnchor.constraint(equalTo: selectionContainerView.topAnchor, constant: 24),
            selectionStackView.leadingAnchor.constraint(equalTo: selectionContainerView.leadingAnchor, constant: 20),
            selectionStackView.trailingAnchor.constraint(equalTo: selectionContainerView.trailingAnchor, constant: -20),
            selectionStackView.bottomAnchor.constraint(equalTo: selectionContainerView.bottomAnchor, constant: -24),
            
            // Scan Button
            scanButton.centerXAnchor.constraint(equalTo: contentView.centerXAnchor),
            scanButton.topAnchor.constraint(equalTo: selectionContainerView.bottomAnchor, constant: 40),
            scanButton.widthAnchor.constraint(equalToConstant: 240),
            scanButton.heightAnchor.constraint(equalToConstant: 56),
            
            // Network Status Label
            networkStatusLabel.centerXAnchor.constraint(equalTo: contentView.centerXAnchor),
            networkStatusLabel.topAnchor.constraint(equalTo: scanButton.bottomAnchor, constant: 10),
            networkStatusLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            networkStatusLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            
            // Instruction Label
            instructionLabel.centerXAnchor.constraint(equalTo: contentView.centerXAnchor),
            instructionLabel.topAnchor.constraint(equalTo: networkStatusLabel.bottomAnchor, constant: 10),
            instructionLabel.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            instructionLabel.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            instructionLabel.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -20)
        ])
    }
    
    private func generateInitialTicket() {
        // Don't generate ticket automatically on load
        instructionLabel.text = "Select a lottery game and required draw date to scan tickets"
        instructionLabel.textColor = .secondaryLabel
    }
    
    private func setupNetworkMonitoring() {
        // Update network status initially
        updateNetworkStatus()
        
        // Monitor network changes
        NetworkConnectivityManager.shared.checkInternetConnectivity { [weak self] isConnected in
            DispatchQueue.main.async {
                self?.updateNetworkStatus()
            }
        }
    }
    
    private func updateNetworkStatus() {
        let isConnected = NetworkConnectivityManager.shared.isInternetAvailable()
        if isConnected {
            networkStatusLabel.text = "🌐 Connected"
            networkStatusLabel.textColor = .systemGreen
        } else {
            networkStatusLabel.text = "📶 No Internet"
            networkStatusLabel.textColor = .systemRed
        }
    }
    
    // MARK: - Lottery Data Loading
    private func loadLotteryData() {
        // Load popular lottery games based on magayo API documentation
        games = [
            // Multi-state US games
            LotteryGame(name: "Mega Millions", code: "us_mega_millions"),
            LotteryGame(name: "Powerball", code: "us_powerball"),
            LotteryGame(name: "Cash4Life", code: "us_cash4life"),
            LotteryGame(name: "Lotto America", code: "us_lotto_america"),
            LotteryGame(name: "Lucky for Life", code: "us_lucky_life"),
            LotteryGame(name: "Powerball Double Play", code: "us_powerball_double"),
            
            // State-specific games
            LotteryGame(name: "California Fantasy 5", code: "us_ca_fantasy"),
            LotteryGame(name: "California SuperLotto Plus", code: "us_ca_lotto"),
            LotteryGame(name: "Florida Fantasy 5", code: "us_fl_fantasy"),
            LotteryGame(name: "Florida Lotto", code: "us_fl_lotto"),
            LotteryGame(name: "New York Lotto", code: "us_ny_lotto"),
            LotteryGame(name: "Texas Lotto", code: "us_tx_lotto"),
            LotteryGame(name: "Arizona Fantasy 5", code: "us_az_fantasy"),
            
            // International games
            LotteryGame(name: "UK Lotto", code: "uk_lotto"),
            LotteryGame(name: "EuroMillions", code: "uk_euromillions"),
            LotteryGame(name: "Thunderball", code: "uk_thunderball"),
            LotteryGame(name: "Canada Lotto 6/49", code: "ca_on_lotto649"),
            LotteryGame(name: "Canada Lotto Max", code: "ca_on_lottomax")
        ]
        
        // Reload picker view
        gamePicker.reloadAllComponents()
    }
    
    private func updateScanButtonVisibility() {
        let hasValidSelection = selectedGame != nil
        
        
        if hasValidSelection {
            let gameName = selectedGame?.name ?? ""
            scanButton.setTitle(" Scan \(gameName)", for: .normal)
            
            // Animate button appearance
            UIView.animate(withDuration: 0.3, delay: 0, usingSpringWithDamping: 0.8, initialSpringVelocity: 0.5, options: [.curveEaseInOut], animations: {
                self.scanButton.isHidden = false
                self.scanButton.alpha = 1.0
                self.scanButton.transform = CGAffineTransform(scaleX: 1.05, y: 1.05)
            }) { _ in
                UIView.animate(withDuration: 0.2) {
                    self.scanButton.transform = .identity
                }
            }
        } else {
            // Animate button disappearance
            UIView.animate(withDuration: 0.2, animations: {
                self.scanButton.alpha = 0.0
                self.scanButton.transform = CGAffineTransform(scaleX: 0.95, y: 0.95)
            }) { _ in
                self.scanButton.isHidden = true
                self.scanButton.transform = .identity
            }
        }
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
        scannerVC.selectedGame = selectedGame
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
    
    // Selected game from the main screen
    var selectedGame: LotteryGame?
    
    // Selected draw date from the main screen (nil means use latest draw)
    var selectedDrawDate: Date?
    
    private let scrollView: UIScrollView = {
        let scrollView = UIScrollView()
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        scrollView.backgroundColor = .systemBackground
        scrollView.keyboardDismissMode = .onDrag
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
        label.text = "Select Date" // Will be updated with actual selected date
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
    
    // Editing state
    private var currentEditingButton: UIButton?
    private var currentEditingTextField: UITextField?
    
    // Keyboard handling
    private var keyboardHeight: CGFloat = 0
    private var originalScrollViewInsets: UIEdgeInsets = .zero
    
    // Persistent toast
    private var issuesToastView: UIView?
    private let toastTag = 987654
    
    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        setupConstraints()
        displayResult()
        
        // Set the selected draw date
        if let drawDate = selectedDrawDate {
            let formatter = DateFormatter()
            formatter.dateFormat = "dd MMM, yyyy"
            dateValueLabel.text = formatter.string(from: drawDate)
        } else {
            dateValueLabel.text = "No Date Selected"
        }
        
        closeButton.addTarget(self, action: #selector(closeButtonTapped), for: .touchUpInside)
        actionButton.addTarget(self, action: #selector(actionButtonTapped), for: .touchUpInside)
        
        // Add keyboard observers
        NotificationCenter.default.addObserver(self, selector: #selector(keyboardWillShow), name: UIResponder.keyboardWillShowNotification, object: nil)
        NotificationCenter.default.addObserver(self, selector: #selector(keyboardWillHide), name: UIResponder.keyboardWillHideNotification, object: nil)
        
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
        } else {
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
            
        // Show persistent toast if there are issues
        let issueCount = countCurrentIssues()
        if issueCount > 0 {
            showPersistentIssuesToast(count: issueCount)
        }
        
        // Update action button state
        updateActionButtonState()
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
        
        
        // Parse the scanned code to extract lottery numbers
        if scannedCode.contains("Lottery:") {
            let components = scannedCode.components(separatedBy: "Lottery: ")
            if components.count > 1 {
                let lotteryData = components[1].components(separatedBy: " Ticket:")[0]
                
                // Check if it's multi-row data (separated by |)
                if lotteryData.contains("|") {
                    let rows = lotteryData.components(separatedBy: "|")
                    for (rowIndex, row) in rows.enumerated() {
                        let numbers = row.trimmingCharacters(in: .whitespacesAndNewlines).components(separatedBy: " ")
                        
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
                        
                    }
                } else {
                    // Single row format - try both comma and space separated
                    let numbers: [String]
                    if lotteryData.contains(",") {
                        // Comma-separated format: "10,15,25,30,43,3"
                        numbers = lotteryData.components(separatedBy: ",")
                    } else {
                        // Space-separated format: "10 15 25 30 43 3"
                        numbers = lotteryData.components(separatedBy: " ")
                    }
                    
                    
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
        
        // Only show rows that were actually scanned - no padding to 5 rows
        // The UI will dynamically adjust to show only the scanned rows
        
        for (index, row) in lotteryNumbers.enumerated() {
        }
    }
    
    private func setupNumbersGrid() {
        
        // Safety check
        guard numbersStackView.superview != nil else {
            return
        }
        
        // Clear existing views
        numbersStackView.arrangedSubviews.forEach { $0.removeFromSuperview() }
        
        let rowLabels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"] // Extended labels for more rows
        
        // Only create rows for the actual scanned data
        for (index, row) in lotteryNumbers.enumerated() {
            let rowLabel = index < rowLabels.count ? rowLabels[index] : "Row \(index + 1)"
            
            // Safety check for powerball array bounds
            let powerball = index < powerballNumbers.count ? powerballNumbers[index] : 0
            
            do {
            let rowView = createNumberRowView(
                    label: rowLabel,
                numbers: row,
                    powerball: powerball
            )
            numbersStackView.addArrangedSubview(rowView)
            } catch {
        }
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
        
        // Add number buttons (limit to 5 to prevent overflow)
        let maxNumbers = min(numbers.count, 5)
        for i in 0..<maxNumbers {
            let number = numbers[i]
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
        
        // Set up constraints with priority to prevent conflicts
        let heightConstraint = rowView.heightAnchor.constraint(equalToConstant: 50)
        heightConstraint.priority = UILayoutPriority(999)
        
        let labelLeadingConstraint = labelView.leadingAnchor.constraint(equalTo: rowView.leadingAnchor)
        labelLeadingConstraint.priority = UILayoutPriority(999)
        
        let labelCenterYConstraint = labelView.centerYAnchor.constraint(equalTo: rowView.centerYAnchor)
        labelCenterYConstraint.priority = UILayoutPriority(999)
        
        let labelWidthConstraint = labelView.widthAnchor.constraint(equalToConstant: 20)
        labelWidthConstraint.priority = UILayoutPriority(999)
        
        let stackLeadingConstraint = numbersStackView.leadingAnchor.constraint(equalTo: labelView.trailingAnchor, constant: 10)
        stackLeadingConstraint.priority = UILayoutPriority(999)
        
        let stackTrailingConstraint = numbersStackView.trailingAnchor.constraint(equalTo: rowView.trailingAnchor)
        stackTrailingConstraint.priority = UILayoutPriority(999)
        
        let stackCenterYConstraint = numbersStackView.centerYAnchor.constraint(equalTo: rowView.centerYAnchor)
        stackCenterYConstraint.priority = UILayoutPriority(999)
        
        NSLayoutConstraint.activate([
            heightConstraint,
            labelLeadingConstraint,
            labelCenterYConstraint,
            labelWidthConstraint,
            stackLeadingConstraint,
            stackTrailingConstraint,
            stackCenterYConstraint
        ])
        
        return rowView
    }
    
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
        
        button.titleLabel?.font = UIFont.boldSystemFont(ofSize: 16)
        button.layer.cornerRadius = 20
        button.translatesAutoresizingMaskIntoConstraints = false
        
        // Add tap gesture for editing
        button.addTarget(self, action: #selector(numberButtonTapped(_:)), for: .touchUpInside)
        
        let widthConstraint = button.widthAnchor.constraint(equalToConstant: 40)
        let heightConstraint = button.heightAnchor.constraint(equalToConstant: 40)
        widthConstraint.priority = UILayoutPriority(999)
        heightConstraint.priority = UILayoutPriority(999)
        
        NSLayoutConstraint.activate([
            widthConstraint,
            heightConstraint
        ])
        
        return button
    }
    
    private func createPowerballButton(number: Int) -> UIButton {
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
        
        button.titleLabel?.font = UIFont.boldSystemFont(ofSize: 16)
        button.layer.cornerRadius = 20
        button.translatesAutoresizingMaskIntoConstraints = false
        
        // Add tap gesture for editing
        button.addTarget(self, action: #selector(powerballButtonTapped(_:)), for: .touchUpInside)
        button.accessibilityIdentifier = "powerball"
        
        let widthConstraint = button.widthAnchor.constraint(equalToConstant: 40)
        let heightConstraint = button.heightAnchor.constraint(equalToConstant: 40)
        widthConstraint.priority = UILayoutPriority(999)
        heightConstraint.priority = UILayoutPriority(999)
        
        NSLayoutConstraint.activate([
            widthConstraint,
            heightConstraint
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
            // Check if button is disabled due to issues
            if !actionButton.isEnabled {
                let issueCount = countCurrentIssues()
                let alert = UIAlertController(title: "Complete All Fields", 
                                           message: "Please fill in all \(issueCount) empty fields before checking for winners.", 
                                           preferredStyle: .alert)
                alert.addAction(UIAlertAction(title: "OK", style: .default))
                present(alert, animated: true)
                return
            }
            
            // Check for winners directly
            checkForWinners()
        } else {
            // Copy to clipboard
            UIPasteboard.general.string = scannedCode
            showCopyConfirmation()
        }
    }
    
    // MARK: - Data Models
    
    struct LotteryResult: Codable {
        let error: Int?
        let draw: String?
        let results: String?
    }
    
    struct LotteryAPIError: Error {
        let message: String
        let code: Int?
    }
    
    // MARK: - Configuration
    
    // magayo.com API key for lottery results
    private let apiKey = APIKeys.magayoAPIKey
    
    // MARK: - Network Service
    
    private func fetchLotteryResults(completion: @escaping (Result<LotteryResult, LotteryAPIError>) -> Void) {
        // Check internet connectivity first
        guard NetworkConnectivityManager.shared.isInternetAvailable() else {
            completion(.failure(LotteryAPIError(message: "No internet connection. Please check your network and try again.", code: -1000)))
            return
        }
        
        // Use the selected game from the picker
        guard let selectedGame = selectedGame else {
            completion(.failure(LotteryAPIError(message: "No game selected", code: nil)))
            return
        }
        
        let gameCode = selectedGame.code
        
        // Build URL with required parameters according to magayo API documentation
        // https://www.magayo.com/lottery-docs/api/get-draw-results/
        var components = URLComponents(string: "https://www.magayo.com/api/results.php")!
        
        var queryItems = [
            URLQueryItem(name: "api_key", value: apiKey),
            URLQueryItem(name: "game", value: gameCode),
            URLQueryItem(name: "format", value: "json")
        ]
        
        // Add draw date (mandatory parameter)
        guard let drawDate = selectedDrawDate else {
            completion(.failure(LotteryAPIError(message: "Draw date is required", code: nil)))
            return
        }
        let dateString = formatDateForAPI(drawDate)
        queryItems.append(URLQueryItem(name: "draw", value: dateString))
        
        components.queryItems = queryItems
        
        guard let url = components.url else {
            completion(.failure(LotteryAPIError(message: "Invalid URL", code: nil)))
            return
        }
        
        // Print the final API URL for debugging
        
        var request = URLRequest(url: url)
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                completion(.failure(LotteryAPIError(message: "Network error: \(error.localizedDescription)", code: nil)))
                return
            }
            
            guard let data = data else {
                completion(.failure(LotteryAPIError(message: "No data received", code: nil)))
                return
            }
            
            // Print raw response for debugging
            if let responseString = String(data: data, encoding: .utf8) {
            }
            
            do {
                let lotteryResult = try JSONDecoder().decode(LotteryResult.self, from: data)
                
                
                // Check if there's an API error according to magayo documentation
                if let errorCode = lotteryResult.error, errorCode != 0 {
                    let errorMessage = self.getErrorMessage(for: errorCode)
                    completion(.failure(LotteryAPIError(message: errorMessage, code: errorCode)))
                    return
                }
                
                completion(.success(lotteryResult))
            } catch {
                completion(.failure(LotteryAPIError(message: "Failed to parse response: \(error.localizedDescription)", code: nil)))
            }
        }.resume()
    }
    
    // MARK: - Date Formatting
    
    private func formatDateForAPI(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.string(from: date)
    }
    
    // MARK: - Error Handling
    
    /// Get error message for magayo API error codes according to their documentation
    /// https://www.magayo.com/lottery-docs/api/get-draw-results/
    private func getErrorMessage(for errorCode: Int) -> String {
        switch errorCode {
        case 0:
            return "No error"
        case 100:
            return "API key not provided"
        case 101:
            return "Invalid API key"
        case 102:
            return "API key not found"
        case 200:
            return "Game not provided"
        case 201:
            return "Invalid game"
        case 202:
            return "Game not found"
        case 300:
            return "Account suspended - You have violated terms of use"
        case 303:
            return "API limit reached - Consider upgrading your API plan"
        case 400:
            return "Invalid draw date - Date format must be YYYY-MM-DD"
        case 401:
            return "No draw results - There is no draw on the specified date"
        default:
            return "Unknown error (Code: \(errorCode))"
        }
    }
    
    // MARK: - Game Detection
    
    private func detectLotteryGameType() -> String {
        // Analyze the scanned numbers to determine the lottery game type
        let userNumbers = lotteryNumbers.flatMap { $0 }.filter { $0 > 0 }
        let hasPowerball = powerballNumbers.contains { $0 > 0 }
        let maxNumber = userNumbers.max() ?? 0
        
        // Check if we have Mega Millions (numbers up to 70, Mega Ball up to 25)
        if maxNumber <= 70 && powerballNumbers.allSatisfy({ $0 == 0 || $0 <= 25 }) {
            return "us_mega_millions"
        }
        
        // Check if we have Powerball (numbers up to 69, Powerball up to 26)
        if maxNumber <= 69 && powerballNumbers.allSatisfy({ $0 == 0 || $0 <= 26 }) {
            return "us_powerball"
        }
        
        // Check if we have Lotto America (numbers up to 52, Star Ball up to 10)
        if maxNumber <= 52 && powerballNumbers.allSatisfy({ $0 == 0 || $0 <= 10 }) {
            return "us_lotto_america"
        }
        
        // Check if we have Cash4Life (numbers up to 60, Cash Ball up to 4)
        if maxNumber <= 60 && powerballNumbers.allSatisfy({ $0 == 0 || $0 <= 4 }) {
            return "us_cash4life"
        }
        
        // Default to Mega Millions for most lottery tickets
        return "us_mega_millions"
    }
    
    
    private func checkForWinners() {
        // Call the lottery results API directly without loading dialog
        fetchLotteryResults { [weak self] result in
            DispatchQueue.main.async {
                switch result {
                case .success(let lotteryResult):
                    self?.showWinningResults(with: lotteryResult)
                case .failure(let error):
                    self?.showErrorAlert(error: error)
                }
            }
        }
    }
    
    private func showWinningResults(with result: LotteryResult) {
        var message = ""
        
        // Display draw date
        if let draw = result.draw, !draw.isEmpty && draw != "-" {
            message += "Draw Date: \(draw)\n\n"
        }
        
        // Display winning numbers
        if let results = result.results, !results.isEmpty && results != "-" {
            let winningNumbers = results.components(separatedBy: ",")
            message += "Winning Numbers: \(winningNumbers.joined(separator: ", "))\n\n"
            
            // Check for matches with user's numbers
            let matches = checkForMatches(winningNumbers: winningNumbers)
            if matches.count > 0 {
                message += " You have \(matches.count) matching number(s): \(matches.joined(separator: ", "))\n\n"
            } else {
                message += "No matching numbers found.\n\n"
            }
        }
        
        // Display user's scanned numbers
        if !lotteryNumbers.isEmpty {
            message += "Your Numbers:\n"
            let rowLabels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
            for (index, row) in lotteryNumbers.enumerated() {
                if !row.allSatisfy({ $0 == 0 }) {
                    let rowLabel = index < rowLabels.count ? rowLabels[index] : "Row \(index + 1)"
                    let numbersString = row.map { $0 == 0 ? "⚪" : "\($0)" }.joined(separator: " ")
                    let powerballString = powerballNumbers[index] > 0 ? " PB: \(powerballNumbers[index])" : ""
                    message += "\(rowLabel): \(numbersString)\(powerballString)\n"
                }
            }
        }
        
        if message.isEmpty {
            message = "No lottery data available. Please try again later."
        }
        
        let alert = UIAlertController(title: "Lottery Results", message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
    
    private func checkForMatches(winningNumbers: [String]) -> [String] {
        var matches: [String] = []
        
        // Convert winning numbers to integers
        let winningInts = winningNumbers.compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
        
        // Check each row of user's numbers
        for row in lotteryNumbers {
            for number in row {
                if number > 0 && winningInts.contains(number) {
                    matches.append("\(number)")
                }
            }
        }
        
        return Array(Set(matches)) // Remove duplicates
    }
    
    private func showErrorAlert(error: LotteryAPIError) {
        let title: String
        let message: String
        
        // Check if it's a network connectivity error
        if error.code == -1000 {
            title = "No Internet Connection"
            message = "Please check your internet connection and try again. Make sure you're connected to Wi-Fi or cellular data."
        } else {
            title = "Error"
            message = error.message
        }
        
        let alert = UIAlertController(title: title, message: message, preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
    
    private func showNetworkErrorAlert() {
        let alert = UIAlertController(title: "No Internet Connection", 
                                    message: "Please check your internet connection and try again. Make sure you're connected to Wi-Fi or cellular data.", 
                                    preferredStyle: .alert)
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
        startEditingButton(sender, isPowerball: false)
    }
    
    @objc private func powerballButtonTapped(_ sender: UIButton) {
        startEditingButton(sender, isPowerball: true)
    }
    
    private func startEditingButton(_ button: UIButton, isPowerball: Bool) {
        // End any current editing
        endEditing()
        
        // Set current editing button
        currentEditingButton = button
        
        // Create text field overlay
        let textField = UITextField()
        textField.text = button.title(for: .normal) ?? ""
        textField.textAlignment = .center
        textField.font = UIFont.boldSystemFont(ofSize: 16)
            textField.keyboardType = .numberPad
        textField.backgroundColor = .clear
        textField.textColor = .label
        textField.borderStyle = .none
        textField.delegate = self
        textField.translatesAutoresizingMaskIntoConstraints = false
        
        // Add text field to button
        button.addSubview(textField)
        currentEditingTextField = textField
        
        // Layout text field
        NSLayoutConstraint.activate([
            textField.centerXAnchor.constraint(equalTo: button.centerXAnchor),
            textField.centerYAnchor.constraint(equalTo: button.centerYAnchor),
            textField.widthAnchor.constraint(equalTo: button.widthAnchor, constant: -8),
            textField.heightAnchor.constraint(equalTo: button.heightAnchor, constant: -8)
        ])
        
        // Visual feedback for editing mode
        button.layer.borderColor = UIColor.systemBlue.cgColor
        button.layer.borderWidth = 2
        
        // Show keyboard
        textField.becomeFirstResponder()
        
        // Add tap gesture to dismiss editing when tapping elsewhere
        let tapGesture = UITapGestureRecognizer(target: self, action: #selector(dismissEditing))
        view.addGestureRecognizer(tapGesture)
    }
    
    @objc private func dismissEditing() {
        endEditing()
    }
    
    private func endEditing() {
        // Remove tap gesture
        view.gestureRecognizers?.forEach { gesture in
            if gesture is UITapGestureRecognizer {
                view.removeGestureRecognizer(gesture)
            }
        }
        
        // End text field editing
        currentEditingTextField?.resignFirstResponder()
        currentEditingTextField?.removeFromSuperview()
        currentEditingTextField = nil
        
        // Reset button appearance
        if let button = currentEditingButton {
            updateButtonAppearance(button)
        }
        
        currentEditingButton = nil
    }
    
    private func updateButtonAppearance(_ button: UIButton) {
        let title = button.title(for: .normal) ?? ""
        let number = Int(title) ?? 0
        
        if number == -1 || number == 0 {
            button.layer.borderColor = UIColor.systemRed.cgColor
            button.layer.borderWidth = 2
        } else {
            button.layer.borderColor = UIColor.systemGray4.cgColor
            button.layer.borderWidth = 1
        }
    }
    
    private func showCopyConfirmation() {
        let alert = UIAlertController(title: "Copied!", message: "QR code data has been copied to clipboard.", preferredStyle: .alert)
        alert.addAction(UIAlertAction(title: "OK", style: .default))
        present(alert, animated: true)
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
    }
    
    // MARK: - Keyboard Handling
    @objc private func keyboardWillShow(_ notification: Notification) {
        guard let keyboardFrame = notification.userInfo?[UIResponder.keyboardFrameEndUserInfoKey] as? CGRect,
              let animationDuration = notification.userInfo?[UIResponder.keyboardAnimationDurationUserInfoKey] as? Double else { return }
        
        keyboardHeight = keyboardFrame.height
        
        // Store original insets if not already stored
        if originalScrollViewInsets == .zero {
            originalScrollViewInsets = scrollView.contentInset
        }
        
        // Adjust scroll view insets
        let newInsets = UIEdgeInsets(top: originalScrollViewInsets.top, 
                                   left: originalScrollViewInsets.left, 
                                   bottom: keyboardHeight, 
                                   right: originalScrollViewInsets.right)
        
        UIView.animate(withDuration: animationDuration) {
            self.scrollView.contentInset = newInsets
            self.scrollView.scrollIndicatorInsets = newInsets
        }
        
        // Scroll to the editing text field
        if let textField = currentEditingTextField {
            scrollToTextField(textField)
        }
    }
    
    @objc private func keyboardWillHide(_ notification: Notification) {
        guard let animationDuration = notification.userInfo?[UIResponder.keyboardAnimationDurationUserInfoKey] as? Double else { return }
        
        keyboardHeight = 0
        
        UIView.animate(withDuration: animationDuration) {
            self.scrollView.contentInset = self.originalScrollViewInsets
            self.scrollView.scrollIndicatorInsets = self.originalScrollViewInsets
        }
    }
    
    private func scrollToTextField(_ textField: UITextField) {
        let textFieldFrame = textField.convert(textField.bounds, to: scrollView)
        let visibleFrame = CGRect(x: 0, y: scrollView.contentOffset.y, 
                                width: scrollView.bounds.width, 
                                height: scrollView.bounds.height - keyboardHeight)
        
        if !visibleFrame.contains(textFieldFrame) {
            let targetY = textFieldFrame.midY - visibleFrame.height / 2
            let maxY = max(0, scrollView.contentSize.height - visibleFrame.height)
            let clampedY = min(maxY, max(0, targetY))
            
            scrollView.setContentOffset(CGPoint(x: 0, y: clampedY), animated: true)
        }
    }
    
    // MARK: - Persistent Toast Management
    func showPersistentIssuesToast(count: Int) {
        let message = "\(count) issues found"
        
        // Remove existing toast if any
        issuesToastView?.removeFromSuperview()
        
        // Container view with red background
        let toastView = UIView()
        toastView.tag = toastTag
        toastView.backgroundColor = UIColor.systemRed
        toastView.alpha = 0.0
        toastView.layer.cornerRadius = 10
        toastView.layer.masksToBounds = true
        toastView.translatesAutoresizingMaskIntoConstraints = false
        
        // Label
        let label = UILabel()
        label.text = message
        label.textColor = .white
        label.font = UIFont.boldSystemFont(ofSize: 15)
        label.numberOfLines = 0
        label.textAlignment = .center
        label.translatesAutoresizingMaskIntoConstraints = false
        toastView.addSubview(label)
        
        view.addSubview(toastView)
        issuesToastView = toastView
        
        // Layout: show near the top of the Lottery scan result area
        NSLayoutConstraint.activate([
            toastView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            toastView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            toastView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            
            label.leadingAnchor.constraint(equalTo: toastView.leadingAnchor, constant: 16),
            label.trailingAnchor.constraint(equalTo: toastView.trailingAnchor, constant: -16),
            label.topAnchor.constraint(equalTo: toastView.topAnchor, constant: 12),
            label.bottomAnchor.constraint(equalTo: toastView.bottomAnchor, constant: -12)
        ])
        
        // Animate in
        UIView.animate(withDuration: 0.25) {
            toastView.alpha = 0.95
        }
    }
    
    func updateIssuesToast(count: Int) {
        if count > 0 {
            showPersistentIssuesToast(count: count)
        } else {
            hideIssuesToast()
        }
    }
    
    func hideIssuesToast() {
        UIView.animate(withDuration: 0.25, animations: {
            self.issuesToastView?.alpha = 0.0
        }) { _ in
            self.issuesToastView?.removeFromSuperview()
            self.issuesToastView = nil
        }
    }
    
    private func countCurrentIssues() -> Int {
        var issueCount = 0
        
        // Count issues in lottery numbers (both -1 and 0 are considered empty)
        for row in lotteryNumbers {
            for number in row {
                if number == -1 || number == 0 {
                    issueCount += 1
                }
            }
        }
        
        // Count issues in powerball numbers (both -1 and 0 are considered empty)
        for number in powerballNumbers {
            if number == -1 || number == 0 {
                issueCount += 1
            }
        }
        
        return issueCount
    }
    
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
}

// MARK: - UITextFieldDelegate
extension QRScanResultViewController: UITextFieldDelegate {
    func textField(_ textField: UITextField, shouldChangeCharactersIn range: NSRange, replacementString string: String) -> Bool {
        // Allow only numeric input
        let allowedCharacters = CharacterSet.decimalDigits
        let characterSet = CharacterSet(charactersIn: string)
        return allowedCharacters.isSuperset(of: characterSet)
    }
    
    func textFieldDidEndEditing(_ textField: UITextField) {
        guard let button = currentEditingButton else { return }
        
        // Get the entered value
        let text = textField.text ?? ""
        let number = Int(text) ?? 0
        
        // Determine if this is a powerball button
        let isPowerball = button.accessibilityIdentifier == "powerball"
        let maxValue = isPowerball ? 26 : 69
        
        // Validate and update button
        if number >= 1 && number <= maxValue {
            button.setTitle(String(number), for: .normal)
            button.backgroundColor = .white
            button.setTitleColor(.label, for: .normal)
            // Update the underlying data with valid number
            updateLotteryDataFromButton(button, newValue: number)
        } else {
            // Invalid number or empty field, set to empty (0)
            button.setTitle("", for: .normal)
            button.backgroundColor = .systemGray6
            button.setTitleColor(.clear, for: .normal)
            // Update the underlying data with 0 (empty)
            updateLotteryDataFromButton(button, newValue: 0)
        }
        
        // Update toast based on current issues
        let currentIssues = countCurrentIssues()
        updateIssuesToast(count: currentIssues)
        
        // Update action button state
        updateActionButtonState()
        
        // End editing
        endEditing()
    }
    
    private func updateLotteryDataFromButton(_ button: UIButton, newValue: Int) {
        // Find which row and column this button represents
        guard let stackView = button.superview as? UIStackView,
              let rowView = stackView.superview as? UIView else { return }
        
        // Find row index
        var rowIndex = -1
        for (index, arrangedSubview) in numbersStackView.arrangedSubviews.enumerated() {
            if arrangedSubview == rowView {
                rowIndex = index
                break
            }
        }
        
        guard rowIndex >= 0 && rowIndex < lotteryNumbers.count else { return }
        
        // Check if this is a powerball button (last button in row)
        if let lastButton = stackView.arrangedSubviews.last as? UIButton,
           lastButton == button {
            // This is a powerball button
            powerballNumbers[rowIndex] = newValue
        } else {
            // This is a regular number button
            // Find column index
            var colIndex = -1
            for (index, arrangedSubview) in stackView.arrangedSubviews.enumerated() {
                if arrangedSubview == button {
                    colIndex = index
                    break
                }
            }
            
            guard colIndex >= 0 && colIndex < lotteryNumbers[rowIndex].count else { return }
            lotteryNumbers[rowIndex][colIndex] = newValue
        }
    }
}

// MARK: - QRScannerDelegate
extension ViewController: QRScannerDelegate {
    func didScanQRCode(_ code: String, image: UIImage?) {
        instructionLabel.text = "QR Code scanned successfully!"
        instructionLabel.textColor = .systemGreen
        
        // Log image resizing information if image is provided
        if let image = image {
            logImageResizingInfo(image)
        }
        
        // Dismiss the scanner first
        dismiss(animated: true) {
            // Present scan result screen after scanner is dismissed
            let resultVC = QRScanResultViewController()
            resultVC.scannedCode = code
            resultVC.isLotteryTicket = code.contains("Lottery:")
            resultVC.scannedImage = image
            resultVC.selectedGame = self.selectedGame // Pass the selected game
            resultVC.selectedDrawDate = self.selectedDrawDate // Pass the selected draw date
            resultVC.modalPresentationStyle = .fullScreen
            
            self.present(resultVC, animated: true)
        }
    }
    
    /// Logs information about image resizing for debugging
    private func logImageResizingInfo(_ image: UIImage) {
        let originalSize = image.size
        let originalFileSize = image.fileSizeKB()
        
        if let resizedImage = image.resizedForUpload() {
            let resizedSize = resizedImage.size
            let resizedFileSize = resizedImage.fileSizeKB()
            
        } else {
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
    
    @objc private func datePickerChanged() {
        selectedDrawDate = datePicker.date
    }
    
    
    private func formatDateForAPI(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter.string(from: date)
    }
}

// MARK: - UIPickerViewDataSource & UIPickerViewDelegate
extension ViewController: UIPickerViewDataSource, UIPickerViewDelegate {
    
    func numberOfComponents(in pickerView: UIPickerView) -> Int {
        return 1
    }
    
    func pickerView(_ pickerView: UIPickerView, numberOfRowsInComponent component: Int) -> Int {
        switch pickerView {
        case gamePicker:
            return games.count
        default:
            return 0
        }
    }
    
    func pickerView(_ pickerView: UIPickerView, titleForRow row: Int, forComponent component: Int) -> String? {
        switch pickerView {
        case gamePicker:
            return games[row].name
        default:
            return nil
        }
    }
    
    func pickerView(_ pickerView: UIPickerView, didSelectRow row: Int, inComponent component: Int) {
        switch pickerView {
        case gamePicker:
            selectedGame = games[row]
            
        default:
            break
        }
        
        // Update scan button visibility
        updateScanButtonVisibility()
    }
}



