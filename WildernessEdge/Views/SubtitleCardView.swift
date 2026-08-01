import SwiftUI

/// High-contrast overlay card displaying the active source citation and spoken checklist text.
struct SubtitleCardView: View {
    let citation: String?
    let checklistText: String
    let isError: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let citation, !citation.isEmpty {
                Text(citation)
                    .font(.footnote.bold())
                    .foregroundStyle(isError ? .red : .secondary)
            }
            Text(checklistText)
                .font(.body)
                .foregroundStyle(isError ? .red : .primary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(isError ? Color.red.opacity(0.12) : Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(isError ? Color.red : Color.clear, lineWidth: 2)
        )
    }
}

#Preview("Checklist") {
    SubtitleCardView(
        citation: "[Source: NASEMSO National Model EMS Clinical Guidelines v3.0, Extremity Trauma, p. 128]",
        checklistText: "1. Expose and inspect the injured extremity.\n2. Check distal pulse, motor, and sensory function.",
        isError: false
    )
    .padding()
}

#Preview("Error") {
    SubtitleCardView(
        citation: nil,
        checklistText: "I didn't catch that. Try again.",
        isError: true
    )
    .padding()
}
