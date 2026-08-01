import SwiftUI

/// Large, high-contrast, circular push-to-talk button with per-state visual treatment.
/// Tap toggles listen on/off (idle/error → start, listening → stop).
struct EmergencyButtonView: View {
    let state: AppState
    let onPressDown: () -> Void
    let onPressUp: () -> Void

    var body: some View {
        Button(action: handleTap) {
            Circle()
                .fill(fillColor)
                .frame(width: 180, height: 180)
                .overlay(
                    Text(label)
                        .font(.title2.bold())
                        .foregroundStyle(.white)
                        .multilineTextAlignment(.center)
                        .padding()
                )
                .shadow(color: fillColor.opacity(0.45), radius: 12, y: 6)
        }
        .buttonStyle(.plain)
        .disabled(!isInteractive)
        .opacity(isInteractive ? 1.0 : 0.85)
        .accessibilityLabel(label)
        .accessibilityHint(accessibilityHint)
    }

    private var isInteractive: Bool {
        switch state {
        case .idle, .listening, .error:
            return true
        case .processing, .speaking:
            return false
        }
    }

    private func handleTap() {
        switch state {
        case .idle, .error:
            onPressDown()
        case .listening:
            onPressUp()
        case .processing, .speaking:
            break
        }
    }

    private var fillColor: Color {
        switch state {
        case .idle: return .blue
        case .listening: return .red
        case .processing: return .orange
        case .speaking: return .green
        case .error: return .gray
        }
    }

    private var label: String {
        switch state {
        case .idle: return "Tap to Ask"
        case .listening: return "Tap to Send"
        case .processing: return "Processing…"
        case .speaking: return "Speaking…"
        case .error: return "Error — Tap to Retry"
        }
    }

    private var accessibilityHint: String {
        switch state {
        case .idle, .error:
            return "Starts listening and captures a camera snapshot"
        case .listening:
            return "Stops listening and runs the offline protocol pipeline"
        case .processing, .speaking:
            return "Wait for the current response to finish"
        }
    }
}

#Preview("Idle") {
    EmergencyButtonView(state: .idle, onPressDown: {}, onPressUp: {})
}

#Preview("Listening") {
    EmergencyButtonView(state: .listening, onPressDown: {}, onPressUp: {})
}
