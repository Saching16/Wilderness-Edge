import SwiftUI

/// High-contrast overlay card displaying the active source citation and spoken checklist text.
///
/// See plans/daniel.md Task C3 for the full implementation spec.
struct SubtitleCardView: View {
    let citation: String?
    let checklistText: String
    let isError: Bool

    var body: some View {
        // TODO(daniel): implement per plans/daniel.md Task C3.
        Text(checklistText)
    }
}
