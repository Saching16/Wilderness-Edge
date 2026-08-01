import SwiftUI
import UIKit

/// Design tokens for the field UI.
///
/// The constraints here are not decorative. This is read one-handed, outdoors, possibly at
/// night, possibly in gloves, by someone whose attention is on a casualty rather than a
/// screen. So: a dark-leaning palette that does not wreck night vision or burn battery on
/// OLED, contrast well above the 4.5:1 minimum, touch targets past Apple's 44pt floor, and
/// state carried by colour *and* shape *and* text — never colour alone, because a chunk of
/// the population cannot separate the red and green states.
enum Theme {

    // MARK: - Palette

    /// Adaptive colour from an explicit dark/light pair, so both appearances are deliberate
    /// rather than whatever the system happens to invert to.
    private static func adaptive(dark: UInt32, light: UInt32) -> Color {
        Color(uiColor: UIColor { traits in
            UIColor(hex: traits.userInterfaceStyle == .dark ? dark : light)
        })
    }

    static let background = adaptive(dark: 0x0A0C10, light: 0xF4F6F9)
    static let surface = adaptive(dark: 0x151922, light: 0xFFFFFF)
    static let surfaceRaised = adaptive(dark: 0x1D2230, light: 0xFFFFFF)
    static let hairline = adaptive(dark: 0x2A3142, light: 0xD8DEE8)

    static let textPrimary = adaptive(dark: 0xF2F5FA, light: 0x0E1420)
    static let textSecondary = adaptive(dark: 0x9AA5B8, light: 0x5A6478)
    static let citation = adaptive(dark: 0x7FB4FF, light: 0x1B5FC4)

    // State colours. Each is paired with an SF Symbol and a label in `StateStyle` so the
    // colour is reinforcement, not the only signal.
    static let idle = adaptive(dark: 0x3B82F6, light: 0x2563EB)
    static let listening = adaptive(dark: 0xEF4444, light: 0xDC2626)
    static let processing = adaptive(dark: 0xF59E0B, light: 0xD97706)
    static let speaking = adaptive(dark: 0x10B981, light: 0x059669)
    static let danger = adaptive(dark: 0xF87171, light: 0xDC2626)

    // MARK: - Metrics

    enum Metric {
        /// Comfortably past the 44pt minimum: this gets tapped with cold or gloved hands.
        static let minimumTouchTarget: CGFloat = 56
        static let primaryButtonSize: CGFloat = 168
        static let cornerRadius: CGFloat = 16
        static let cornerRadiusSmall: CGFloat = 10
        static let gutter: CGFloat = 20
        static let stackSpacing: CGFloat = 16
    }
}

// MARK: - State presentation

/// Everything the UI needs to render a state, in one place, so `ContentView`,
/// `EmergencyButtonView` and the status pill can never disagree about what "processing"
/// looks like.
struct StateStyle {
    let tint: Color
    let symbol: String
    let label: String
    /// Spoken by VoiceOver and shown under the button. Present tense, plain language.
    let hint: String

    init(_ state: AppState) {
        switch state {
        case .idle:
            tint = Theme.idle
            symbol = "mic.fill"
            label = "Ready"
            hint = "Hold to speak, or type your question"
        case .listening:
            tint = Theme.listening
            symbol = "waveform"
            label = "Listening"
            hint = "Release to send"
        case .processing:
            tint = Theme.processing
            symbol = "gearshape.2.fill"
            label = "Searching manuals"
            hint = "Working offline"
        case .speaking:
            tint = Theme.speaking
            symbol = "speaker.wave.3.fill"
            label = "Reading aloud"
            hint = "Tap to stop"
        case .error:
            tint = Theme.danger
            symbol = "exclamationmark.triangle.fill"
            label = "Problem"
            hint = "Tap to try again"
        }
    }
}

// MARK: - Helpers

extension UIColor {
    /// 0xRRGGBB.
    convenience init(hex: UInt32) {
        self.init(
            red: CGFloat((hex >> 16) & 0xFF) / 255,
            green: CGFloat((hex >> 8) & 0xFF) / 255,
            blue: CGFloat(hex & 0xFF) / 255,
            alpha: 1
        )
    }
}

extension View {
    /// Standard raised panel: surface fill, hairline border, consistent radius.
    func fieldPanel(raised: Bool = false) -> some View {
        self
            .background(raised ? Theme.surfaceRaised : Theme.surface)
            .clipShape(RoundedRectangle(cornerRadius: Theme.Metric.cornerRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: Theme.Metric.cornerRadius, style: .continuous)
                    .strokeBorder(Theme.hairline, lineWidth: 1)
            )
    }
}
