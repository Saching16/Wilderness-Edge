import PhotosUI
import SwiftUI
import UIKit

/// Which way the responder is asking right now.
///
/// Voice is the default because the hands-busy case is the design centre. Text exists
/// because voice fails in wind, in noise, near a running engine, and when the casualty
/// should not overhear the question.
enum QueryInputMode: String, CaseIterable, Identifiable {
    case voice
    case text

    var id: String { rawValue }
    var label: String { self == .voice ? "Voice" : "Type" }
    var symbol: String { self == .voice ? "mic.fill" : "keyboard.fill" }
}

/// Segmented switch between voice and typed entry.
struct InputModeSwitch: View {
    @Binding var mode: QueryInputMode
    let isEnabled: Bool

    var body: some View {
        HStack(spacing: 4) {
            ForEach(QueryInputMode.allCases) { candidate in
                Button {
                    withAnimation(.easeInOut(duration: 0.18)) { mode = candidate }
                } label: {
                    Label(candidate.label, systemImage: candidate.symbol)
                        .font(.subheadline.weight(.semibold))
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 10)
                        .foregroundStyle(mode == candidate ? Theme.textPrimary : Theme.textSecondary)
                        .background(
                            RoundedRectangle(cornerRadius: Theme.Metric.cornerRadiusSmall, style: .continuous)
                                .fill(mode == candidate ? Theme.surfaceRaised : .clear)
                        )
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(mode == candidate ? [.isButton, .isSelected] : .isButton)
            }
        }
        .padding(4)
        .background(Theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: Theme.Metric.cornerRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: Theme.Metric.cornerRadius, style: .continuous)
                .strokeBorder(Theme.hairline, lineWidth: 1)
        )
        .disabled(!isEnabled)
        .opacity(isEnabled ? 1 : 0.5)
    }
}

/// Typed question entry with an inline send action.
struct TextQueryField: View {
    @Binding var text: String
    let isEnabled: Bool
    let onSubmit: () -> Void

    @FocusState private var isFocused: Bool

    private var canSend: Bool {
        isEnabled && !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        HStack(spacing: 10) {
            TextField("Describe the injury or ask a question", text: $text, axis: .vertical)
                .lineLimit(1...4)
                .font(.body)
                .foregroundStyle(Theme.textPrimary)
                .focused($isFocused)
                .submitLabel(.send)
                .onSubmit(send)
                .padding(.horizontal, 14)
                .padding(.vertical, 12)
                .accessibilityLabel("Question")

            Button(action: send) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 34))
                    .foregroundStyle(canSend ? Theme.idle : Theme.textSecondary.opacity(0.4))
            }
            .buttonStyle(.plain)
            .disabled(!canSend)
            .padding(.trailing, 8)
            .accessibilityLabel("Send question")
        }
        .fieldPanel()
        .disabled(!isEnabled)
        .opacity(isEnabled ? 1 : 0.5)
    }

    private func send() {
        guard canSend else { return }
        isFocused = false
        onSubmit()
    }
}

/// Camera-snapshot / photo attachment row.
///
/// The image is shown back to the responder at a legible size before it is sent, because a
/// misframed or motion-blurred frame is worse than no frame: it gives the model something
/// confident to be wrong about.
struct SnapshotStrip: View {
    let image: UIImage?
    let isEnabled: Bool
    let onCapture: () -> Void
    let onClear: () -> Void
    @Binding var pickedItem: PhotosPickerItem?

    var body: some View {
        HStack(spacing: 12) {
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 76, height: 76)
                    .clipShape(RoundedRectangle(cornerRadius: Theme.Metric.cornerRadiusSmall, style: .continuous))
                    .overlay(
                        RoundedRectangle(cornerRadius: Theme.Metric.cornerRadiusSmall, style: .continuous)
                            .strokeBorder(Theme.hairline, lineWidth: 1)
                    )
                    .accessibilityLabel("Attached photo")

                VStack(alignment: .leading, spacing: 3) {
                    Text("Photo attached")
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(Theme.textPrimary)
                    Text("Sent with your next question")
                        .font(.caption)
                        .foregroundStyle(Theme.textSecondary)
                }

                Spacer(minLength: 0)

                Button(action: onClear) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title2)
                        .foregroundStyle(Theme.textSecondary)
                }
                .buttonStyle(.plain)
                .frame(width: 44, height: 44)
                .accessibilityLabel("Remove photo")
            } else {
                attachButton(
                    title: "Camera",
                    symbol: "camera.fill",
                    action: onCapture
                )

                PhotosPicker(selection: $pickedItem, matching: .images, photoLibrary: .shared()) {
                    attachLabel(title: "Photos", symbol: "photo.on.rectangle")
                }
                .buttonStyle(.plain)
                .disabled(!isEnabled)
                .accessibilityLabel("Choose a photo")
            }
        }
        .padding(12)
        .fieldPanel()
        .opacity(isEnabled ? 1 : 0.5)
        .animation(.easeInOut(duration: 0.18), value: image == nil)
    }

    private func attachButton(title: String, symbol: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            attachLabel(title: title, symbol: symbol)
        }
        .buttonStyle(.plain)
        .disabled(!isEnabled)
        .accessibilityLabel(title == "Camera" ? "Take a photo" : title)
    }

    private func attachLabel(title: String, symbol: String) -> some View {
        Label(title, systemImage: symbol)
            .font(.subheadline.weight(.medium))
            .foregroundStyle(Theme.textPrimary)
            .frame(maxWidth: .infinity)
            .frame(height: Theme.Metric.minimumTouchTarget - 8)
            .background(Theme.surfaceRaised)
            .clipShape(RoundedRectangle(cornerRadius: Theme.Metric.cornerRadiusSmall, style: .continuous))
    }
}

#Preview("Input dock") {
    VStack(spacing: 14) {
        InputModeSwitch(mode: .constant(.text), isEnabled: true)
        SnapshotStrip(
            image: nil,
            isEnabled: true,
            onCapture: {},
            onClear: {},
            pickedItem: .constant(nil)
        )
        TextQueryField(text: .constant("bleeding from the thigh"), isEnabled: true, onSubmit: {})
    }
    .padding()
    .frame(maxHeight: .infinity)
    .background(Theme.background)
}
