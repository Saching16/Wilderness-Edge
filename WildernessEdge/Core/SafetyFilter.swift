import Foundation

/// Client-side non-diagnostic / non-prescriptive sanitizer.
/// All LLM (and demo) text must pass through this layer before display or TTS.
enum SafetyFilter {
    struct Result: Equatable {
        let text: String
        let wasModified: Bool
        let matchedPatterns: [String]
    }

    /// Standard framing used when diagnostic, drug-prescriptive, species-identifying, or
    /// edibility-advising language is detected.
    static let replacementCitation =
        "Displaying retrieved protocol checklist. This assistant does not diagnose conditions, prescribe treatments, identify species, or advise on what is safe to eat. Compare against the reference images yourself, and follow only the cited field-manual steps within your training and scope."

    /// Patterns that indicate independent diagnosis or drug prescription language.
    private static let bannedPatterns: [(name: String, regex: NSRegularExpression)] = {
        let patternStrings: [(String, String)] = [
            ("diagnosis_is", #"\b(the\s+)?diagnosis\s+is\b"#),
            ("you_have_condition", #"\byou\s+have\s+(a\s+)?(fracture|sprain|concussion|infection|dislocation|hypothermia|frostbite|heat\s*stroke|pneumonia|stroke|heart\s+attack)\b"#),
            ("patient_has_grade", #"\b(patient|victim)\s+has\s+(a\s+)?(grade[-\s]?\d+|severe|mild|moderate)\b"#),
            ("this_is_a_grade", #"\bthis\s+is\s+(a\s+)?(grade[-\s]?\d+\s+)?(fracture|sprain|concussion|dislocation)\b"#),
            ("prescribe_take_drug", #"\b(take|administer|give|prescribe)\s+\d+(\.\d+)?\s*(mg|mcg|ml|g)\b"#),
            ("named_drug_dose", #"\b(ibuprofen|acetaminophen|aspirin|morphine|epinephrine|naloxone|antibiotics?)\s+\d+(\.\d+)?\s*(mg|mcg|ml)\b"#),
            ("you_should_take", #"\byou\s+should\s+(take|be\s+given)\s+\w+"#),
            ("i_diagnose", #"\bi\s+(diagnose|diagnosed)\b"#),
            ("my_diagnosis", #"\bmy\s+diagnosis\b"#),

            // --- Flora & fauna guardrails -------------------------------------------------
            // A confident species identification from a single camera frame is the same class
            // of risk as a diagnosis, and a small multimodal model is not reliable at it. The
            // hazard cards are written to describe field marks and let the *responder* decide,
            // so the model asserting an identity is always a failure. Conditional framing
            // ("if this is a copperhead, the protocol says...") is deliberately still allowed,
            // which is what the `if`/`whether` lookbehinds protect.
            ("species_id", #"(?<!if )(?<!whether )\b(this|that|it)(?:\s+(?:is|was|must\s+be)|'s)\s+(an?\s+)?(?:[a-z]+[-\s]){0,2}(rattlesnake|copperhead|cottonmouth|water\s+moccasin|coral\s+snake|black\s+widow|brown\s+recluse|poison\s+(ivy|oak|sumac)|hogweed|hemlock|death\s+cap|nettle|grizzly|black\s+bear|brown\s+bear|mountain\s+lion|cougar|puma|moose|tick|spider|snake|mushroom|berry)\b"#),
            // The catastrophic output: telling a responder the animal that bit their patient
            // was harmless. Phrased to require an assertion about the subject, so a hazard
            // card's own look-alike text ("Harmless water snakes share this habitat") passes.
            ("harmless_reassurance", #"\b(is|are|it'?s|that'?s)\s+(an?\s+)?(probably\s+|likely\s+|definitely\s+|most\s+likely\s+)?(harmless|non-?venomous|non-?poisonous|not\s+(venomous|poisonous|dangerous))\b"#),
            // Foraging and edibility advice must never be spoken. This is also the mitigation
            // for FM 3-05.70 Appendix B (the discredited "Universal Edibility Test") if that
            // manual is ingested wholesale — see OffLineTools/SOURCES.md.
            ("edibility_advice", #"\b(safe|ok|okay|fine)\s+to\s+eat\b|\b(is|are)\s+edible\b|\byou\s+can\s+eat\b|\buniversal\s+edibility\s+test\b"#),
        ]

        return patternStrings.compactMap { name, pattern in
            guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
                return nil
            }
            return (name, regex)
        }
    }()

    static func sanitize(_ input: String) -> Result {
        let range = NSRange(input.startIndex..<input.endIndex, in: input)
        var matched: [String] = []

        for (name, regex) in bannedPatterns {
            if regex.firstMatch(in: input, options: [], range: range) != nil {
                matched.append(name)
            }
        }

        if matched.isEmpty {
            return Result(text: input, wasModified: false, matchedPatterns: [])
        }

        return Result(text: replacementCitation, wasModified: true, matchedPatterns: matched)
    }
}
