import Foundation

/// Push-to-talk UI / orchestration states shared by ContentView and EmergencyButtonView.
enum AppState: Equatable {
    case idle
    case listening
    case processing
    case speaking
    case error(String)
}
