import SwiftUI
import UIKit

/// Reference imagery for a retrieved flora/fauna hazard card.
///
/// This view exists to keep identification in human hands. Gemma sees the camera frame and
/// describes what is visible, the description drives text retrieval, and the responder then
/// compares the real organism against these licensed reference photographs and decides.
/// The app never claims to know what the organism is — `SafetyFilter`'s `species_id` and
/// `harmless_reassurance` patterns intercept the model if it tries.
///
/// Images are decoded lazily and only for the top retrieved chunk, so a corpus with dozens
/// of hazard cards costs nothing until one is actually shown.
struct SpeciesCardView: View {
    let images: [VectorRAGManager.ReferenceImage]

    @State private var selection: Int = 0

    var body: some View {
        if images.isEmpty {
            EmptyView()
        } else {
            VStack(alignment: .leading, spacing: 8) {
                Label("Confirm this yourself", systemImage: "eye.trianglebadge.exclamationmark")
                    .font(.footnote.bold())
                    .foregroundStyle(.orange)

                Text("Compare the plant or animal against these reference photographs. This app does not identify species.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                TabView(selection: $selection) {
                    ForEach(Array(images.enumerated()), id: \.offset) { index, image in
                        referenceImage(image)
                            .tag(index)
                    }
                }
                .tabViewStyle(.page(indexDisplayMode: images.count > 1 ? .automatic : .never))
                .frame(height: 220)

                // Attribution is a license condition for the CC BY / CC BY-SA images in the
                // pack, not decoration — it must stay visible wherever the image is shown.
                if images.indices.contains(selection) {
                    let current = images[selection]
                    Text("\(current.attribution) — \(current.license)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(2)
                        .accessibilityLabel("Image credit: \(current.attribution), licensed \(current.license)")
                }
            }
            .padding()
            .background(Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
    }

    @ViewBuilder
    private func referenceImage(_ image: VectorRAGManager.ReferenceImage) -> some View {
        if let uiImage = UIImage(data: image.data) {
            Image(uiImage: uiImage)
                .resizable()
                .scaledToFit()
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .accessibilityLabel("Reference photograph \(image.ordinal + 1) of \(images.count)")
        } else {
            // A corrupt blob must not take the checklist down with it.
            RoundedRectangle(cornerRadius: 8)
                .fill(Color(.tertiarySystemBackground))
                .overlay(
                    Label("Reference image unavailable", systemImage: "photo")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                )
        }
    }
}

#Preview {
    // Solid-colour placeholder stands in for a bundled JPEG blob.
    let placeholder: Data = {
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: 320, height: 200))
        let image = renderer.image { context in
            UIColor.systemGreen.setFill()
            context.fill(CGRect(x: 0, y: 0, width: 320, height: 200))
        }
        return image.jpegData(compressionQuality: 0.8) ?? Data()
    }()

    return SpeciesCardView(images: [
        .init(
            speciesSlug: "poison-ivy",
            ordinal: 0,
            data: placeholder,
            license: "CC BY-SA 4.0",
            attribution: "Example Photographer",
            sourceURL: "https://commons.wikimedia.org/"
        ),
        .init(
            speciesSlug: "poison-ivy",
            ordinal: 1,
            data: placeholder,
            license: "Public domain",
            attribution: "US Fish and Wildlife Service",
            sourceURL: "https://commons.wikimedia.org/"
        ),
    ])
    .padding()
}
