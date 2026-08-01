import SwiftUI

/// Large, high-contrast, circular push-to-talk button with per-state visual treatment.
///
/// See plans/daniel.md Task C2 for the full implementation spec.
struct EmergencyButtonView: View {
    let state: AppState
    let onPressDown: () -> Void
    let onPressUp: () -> Void

    var body: some View {
        // TODO(daniel): implement per plans/daniel.md Task C2.
        Circle()
            .fill(Color.blue)
            .frame(width: 180, height: 180)
    }
}
