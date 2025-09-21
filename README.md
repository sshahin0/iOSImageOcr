# QRLottery - iOS Lottery Ticket QR Generator

A beautiful iOS UIKit Swift application that generates lottery ticket-style QR codes with random lottery numbers.

## Features

- **Lottery Ticket Design**: Beautiful UI that mimics a real lottery ticket with red sidebar
- **Random Lottery Numbers**: Generates 5 rows of lottery numbers (5 regular + 1 powerball per row)
- **QR Code Integration**: QR code contains all lottery data for verification
- **Ticket Number**: Unique ticket serial number for each generated ticket
- **Modern UI**: Clean, responsive interface with lottery ticket styling
- **Real-time Generation**: Each tap creates a new lottery ticket with fresh numbers

## Requirements

- iOS 17.0+
- Xcode 15.0+
- Swift 5.0+

## Project Structure

```
QRLottery/
├── QRLottery/
│   ├── AppDelegate.swift          # App lifecycle management
│   ├── SceneDelegate.swift        # Scene management
│   ├── ViewController.swift       # Main view controller with QR generation
│   ├── Info.plist                # App configuration
│   ├── Assets.xcassets/          # App icons and colors
│   └── Base.lproj/
│       ├── Main.storyboard       # Main storyboard (minimal)
│       └── LaunchScreen.storyboard # Launch screen
└── QRLottery.xcodeproj/          # Xcode project file
```

## How to Use

1. Open `QRLottery.xcodeproj` in Xcode
2. Select your target device or simulator
3. Build and run the project (⌘+R)
4. Tap the "Generate Lottery QR" button to create a new lottery ticket
5. The lottery ticket will appear with random numbers and QR code

## Lottery Ticket Features

The app generates authentic-looking lottery tickets with:
- **🎫 Professional Design**: Red sidebar with vertical "LOTTERY" text, just like real lottery tickets
- **🎲 Grid Number Layout**: 5 rows of lottery numbers in individual boxes with proper styling
- **🔴 Powerball Highlighting**: Powerball numbers displayed in red boxes with white text
- **📊 Dashed Separator Lines**: Authentic lottery ticket styling with dashed lines
- **🎟️ Unique Ticket Number**: 20-digit serial number for each ticket
- **📱 QR Code Integration**: Contains all lottery information for verification
- **💰 Price Display**: Shows $10 price like a real lottery ticket
- **✨ Shadow Effects**: Professional depth and styling

## Customization

You can easily customize the QR code content by modifying the `message` variable in the `generateQRCode()` method in `ViewController.swift`.

## Dependencies

- UIKit (built-in)
- CoreImage (built-in)

No external dependencies required!
