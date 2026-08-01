import Foundation

/// Client-side non-diagnostic / non-prescriptive sanitizer.
/// All LLM (and demo) text must pass through this layer before display or TTS.
enum SafetyFilter {
    struct Result: Equatable {
        let text: String
        let wasModified: Bool
        let matchedPatterns: [String]
    }

    /// Standard framing used when diagnostic or drug-prescriptive language is detected.
    static let replacementCitation =
        "Displaying retrieved protocol checklist. This assistant does not diagnose conditions or prescribe treatments. Follow only the cited field-manual steps within your training and scope."

    /// Patterns that indicate independent diagnosis or drug prescription language.
    private static let bannedPatterns: [(name: String, regex: NSRegularExpression)] = {
        let patternStrings: [(String, String)] = [
            ("diagnosis_is", #"\b(the\s+)?diagnosis\s+is\b"#),
            ("you_have_condition", #"\byou\s+have\s+(a\s+)?(fracture|sprain|concussion|infection|dislocation|hypothermia|frostbite|heat\s*stroke|pneumonia|stroke|heart\s+attack)\b"#),
            ("patient_has_grade", #"\b(patient|victim)\s+has\s+(a\s+)?(grade[-\s]?\d+|severe|mild|moderate)\b"#),
            ("this_is_a_grade", #"\bthis\s+is\s+(a\s+)?(grade[-\s]?\d+\s+)?(fracture|sprain|concussion|dislocation)\b"#),
            ("prescribe_take_drug", #"\b(take|administer|give|prescribe)\s+\d+\s*(mg|mcg|ml|g)\b"#),
            ("named_drug_dose", #"\b(ibuprofen|acetaminophen|aspirin|morphine|epinephrine|naloxone|antibiotics?)\s+\d+\s*(mg|mcg|ml)\b"#),
            ("you_should_take", #"\byou\s+should\s+(take|be\s+given)\s+\w+"#),
            ("i_diagnose", #"\bi\s+(diagnose|diagnosed)\b"#),
            ("my_diagnosis", #"\bmy\s+diagnosis\b"#),
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
