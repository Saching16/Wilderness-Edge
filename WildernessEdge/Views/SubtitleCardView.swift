import SwiftUI

/// The answer surface: source citation above, checklist below.
///
/// The citation is deliberately given its own banded header rather than being styled as
/// secondary text. It is the difference between a protocol and an opinion, and it is the
/// one thing a responder must be able to check before acting — so it never scrolls away
/// under the steps it belongs to.
///
/// Signature is unchanged from the merged version so `ContentView` needs no call-site edit.
struct SubtitleCardView: View {
    let citation: String?
    let checklistText: String
    let isError: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let citation, !citation.isEmpty {
                citationHeader(citation)
                Divider().overlay(Theme.hairline)
            }

            ScrollView {
                Text(checklistText)
                    .font(.body)
                    .lineSpacing(5)
                    .foregroundStyle(isError ? Theme.danger : Theme.textPrimary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                    .padding(Theme.Metric.gutter)
            }
            .scrollBounceBehavior(.basedOnSize)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(isError ? Theme.danger.opacity(0.10) : Theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: Theme.Metric.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Metric.cornerRadius, style: .continuous)
                .strokeBorder(isError ? Theme.danger.opacity(0.8) : Theme.hairline, lineWidth: 1)
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel(isError ? "Error" : "Retrieved protocol")
    }

    private func citationHeader(_ citation: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: "book.closed.fill")
                .font(.caption)
                .foregroundStyle(Theme.citation)
            Text(citation)
                .font(.footnote.weight(.semibold))
                .foregroundStyle(Theme.citation)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, Theme.Metric.gutter)
        .padding(.vertical, 12)
        .background(Theme.citation.opacity(0.10))
        .accessibilityLabel("Source: \(citation)")
    }
}

#Preview("Checklist") {
    SubtitleCardView(
        citation: "[Source: NASEMSO National Model EMS Clinical Guidelines v3.0, Extremity Trauma, p. 128]",
        checklistText: """
        1. Expose and inspect the injured extremity.
        2. Check distal pulse, motor, and sensory function.
        3. Immobilise in the position found unless distal circulation is absent.
        """,
        isError: false
    )
    .padding()
    .frame(maxHeight: 320)
    .background(Theme.background)
}

#Preview("Error") {
    SubtitleCardView(citation: nil, checklistText: "I didn't catch that. Try again.", isError: true)
        .padding()
        .background(Theme.background)
}
