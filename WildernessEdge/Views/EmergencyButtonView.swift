import SwiftUI

/// Large, high-contrast, circular voice control.
///
/// Interaction is unchanged from the version device-tested at Checkpoint 4: a **tap
/// toggles** listening on and off (idle/error → `onPressDown`, listening → `onPressUp`).
/// Only the presentation is new. Callback names are kept even though they no longer
/// describe a press-and-hold, because `ContentView` and the sprint plan both refer to them.
struct EmergencyButtonView: View {
    let state: AppState
    let onPressDown: () -> Void
    let onPressUp: () -> Void

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var pulse: CGFloat = 1

    private var style: StateStyle { StateStyle(state) }

    var body: some View {
        VStack(spacing: 14) {
            Button(action: handleTap) {
                ZStack {
                    // Halo, listening only. Communicates "live mic" at a glance from a
                    // distance, which the icon alone does not.
                    Circle()
                        .fill(style.tint.opacity(0.16))
                        .frame(width: Theme.Metric.primaryButtonSize + 48,
                               height: Theme.Metric.primaryButtonSize + 48)
                        .scaleEffect(pulse)
                        .opacity(state == .listening ? 1 : 0)
                        .allowsHitTesting(false)

                    Circle()
                        .fill(style.tint.gradient)
                        .frame(width: Theme.Metric.primaryButtonSize,
                               height: Theme.Metric.primaryButtonSize)
                        .overlay(Circle().strokeBorder(.white.opacity(0.22), lineWidth: 1.5))
                        .shadow(color: style.tint.opacity(0.4), radius: 18, y: 8)

                    if state == .processing {
                        ProgressView()
                            .progressViewStyle(.circular)
                            .tint(.white)
                            .scaleEffect(1.7)
                    } else {
                        Image(systemName: style.symbol)
                            .font(.system(size: 52, weight: .semibold))
                            .foregroundStyle(.white)
                    }
                }
            }
            .buttonStyle(PressableCircleStyle())
            .disabled(!isInteractive)
            .opacity(isInteractive ? 1 : 0.9)
            .animation(.easeInOut(duration: 0.2), value: state)

            VStack(spacing: 3) {
                Text(style.label)
                    .font(.headline)
                    .foregroundStyle(Theme.textPrimary)
                Text(actionHint)
                    .font(.caption)
                    .foregroundStyle(Theme.textSecondary)
                    .multilineTextAlignment(.center)
            }
            .animation(nil, value: state)
        }
        .onChange(of: state) { _, newValue in
            guard !reduceMotion else {
                pulse = 1
                return
            }
            if newValue == .listening {
                withAnimation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true)) {
                    pulse = 1.1
                }
            } else {
                withAnimation(.easeOut(duration: 0.2)) { pulse = 1 }
            }
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Voice question")
        .accessibilityValue(style.label)
        .accessibilityHint(accessibilityHint)
        .accessibilityAddTraits(.isButton)
    }

    private var isInteractive: Bool {
        switch state {
        case .idle, .listening, .error: return true
        case .processing, .speaking: return false
        }
    }

    private func handleTap() {
        switch state {
        case .idle, .error: onPressDown()
        case .listening: onPressUp()
        case .processing, .speaking: break
        }
    }

    private var actionHint: String {
        switch state {
        case .idle: return "Tap and ask your question"
        case .listening: return "Tap again to send"
        case .processing: return "Searching the offline manuals"
        case .speaking: return "Reading the checklist aloud"
        case .error: return "Tap to try again"
        }
    }

    private var accessibilityHint: String {
        switch state {
        case .idle, .error:
            return "Starts listening and captures a camera snapshot"
        case .listening:
            return "Stops listening and searches the offline protocol library"
        case .processing, .speaking:
            return "Wait for the current response to finish"
        }
    }
}

/// Press feedback without the default Button fade, which washes out the state colour.
private struct PressableCircleStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed ? 0.94 : 1)
            .animation(.spring(response: 0.26, dampingFraction: 0.7), value: configuration.isPressed)
    }
}

#Preview("States") {
    ScrollView {
        VStack(spacing: 36) {
            EmergencyButtonView(state: .idle, onPressDown: {}, onPressUp: {})
            EmergencyButtonView(state: .listening, onPressDown: {}, onPressUp: {})
            EmergencyButtonView(state: .processing, onPressDown: {}, onPressUp: {})
            EmergencyButtonView(state: .error("x"), onPressDown: {}, onPressUp: {})
        }
        .padding(40)
    }
    .background(Theme.background)
}
