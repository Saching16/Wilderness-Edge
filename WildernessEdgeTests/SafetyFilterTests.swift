import XCTest

// SafetyFilter and SpeechManager are compiled directly into this test bundle
// (see project.yml) so these tests run without a host-app install.

final class SafetyFilterTests: XCTestCase {
    func testDiagnosticPhrasesAreIntercepted() {
        let bait: [String] = [
            "The diagnosis is a fracture",
            "You have a fracture of the tibia",
            "The patient has a grade-2 sprain",
            "This is a grade 2 fracture",
            "I diagnose hypothermia",
            "My diagnosis: concussion",
        ]

        for phrase in bait {
            let result = SafetyFilter.sanitize(phrase)
            XCTAssertTrue(result.wasModified, "Expected interception for: \(phrase)")
            XCTAssertEqual(result.text, SafetyFilter.replacementCitation)
            XCTAssertFalse(result.matchedPatterns.isEmpty, "Expected matched patterns for: \(phrase)")
        }
    }

    func testPrescriptiveDrugLanguageIsIntercepted() {
        let bait: [String] = [
            "Take 400mg ibuprofen",
            "Administer 0.3 mg epinephrine",
            "Give 325 mg aspirin now",
            "Prescribe 500mg acetaminophen",
            "ibuprofen 400 mg every 6 hours",
            "You should take antibiotics immediately",
        ]

        for phrase in bait {
            let result = SafetyFilter.sanitize(phrase)
            XCTAssertTrue(result.wasModified, "Expected interception for: \(phrase)")
            XCTAssertEqual(result.text, SafetyFilter.replacementCitation)
        }
    }

    // MARK: - Dosage guardrails

    /// Regression tests for a gap found by running the real pipeline against the real
    /// corpus: `named_drug_dose` required a listed drug name IMMEDIATELY followed by a
    /// number, so all of these reached the output unblocked and would have been spoken.
    func testDosagePhrasingVariantsAreIntercepted() {
        let bait: [String] = [
            "Morphine sulfate at 0.1 mg/kg IM, IV or IO (maximum initial dose is 10 mg).",
            "Ketamine 0.3 mg/kg slow IV push",
            "Naloxone 0.4 to 2 mg IV/IM",
            "Acetaminophen 15 mg/kg PO or IV (maximum dose 1 g)",
            "Hydroxocobalamin 5 g IV over 15 minutes",
            "Give tranexamic acid, 1 g, slowly",
            "Amiodarone 150 mg IV over 10 minutes",
        ]

        for phrase in bait {
            let result = SafetyFilter.sanitize(phrase)
            XCTAssertTrue(result.wasModified, "Dosage reached the output unblocked: \(phrase)")
            XCTAssertEqual(result.text, SafetyFilter.replacementCitation)
        }
    }

    /// The dosage patterns must stay dose-shaped, not "any number next to a unit". Blanking
    /// a shock checklist because it estimates blood loss in cc would be its own failure, so
    /// these assessment strings — all taken from the shipped corpus — must survive intact.
    func testAssessmentNumbersAreNotMistakenForDoses() {
        let legitimate: [String] = [
            "Radial pulse >100 (with 4,000cc - 1,000cc loss).",
            "Apply a tourniquet 2-3 inches above the wound.",
            "Elevate the head up to 30 degrees when possible.",
            "Increase PEEP stepwise by 2-3 cmH2O as needed.",
            "Core temperature below 89 F indicates severe hypothermia.",
            "Use an 18 g needle for decompression.",
            "Administer oxygen at 15 L/min via non-rebreather.",
        ]

        for phrase in legitimate {
            let result = SafetyFilter.sanitize(phrase)
            XCTAssertFalse(
                result.wasModified,
                "Assessment text was wrongly blanked as a dose (\(result.matchedPatterns)): \(phrase)"
            )
            XCTAssertEqual(result.text, phrase)
        }
    }

    // MARK: - Flora & fauna guardrails

    func testSpeciesIdentificationAssertionsAreIntercepted() {
        let bait: [String] = [
            "This is a copperhead",
            "That is a black widow spider",
            "It is a western diamondback rattlesnake",
            "This is poison ivy",
            "This is a death cap mushroom",
            "It is a grizzly bear, so play dead",
        ]

        for phrase in bait {
            let result = SafetyFilter.sanitize(phrase)
            XCTAssertTrue(result.wasModified, "Expected interception for: \(phrase)")
            XCTAssertEqual(result.text, SafetyFilter.replacementCitation)
            XCTAssertTrue(result.matchedPatterns.contains("species_id"), "Expected species_id for: \(phrase)")
        }
    }

    /// The catastrophic failure mode: reassuring a responder that the animal was harmless.
    func testHarmlessReassuranceIsIntercepted() {
        let bait: [String] = [
            "That's a non-venomous water snake",
            "The snake is harmless",
            "It's probably non-venomous, so no evacuation is needed",
            "That spider is not dangerous",
        ]

        for phrase in bait {
            let result = SafetyFilter.sanitize(phrase)
            XCTAssertTrue(result.wasModified, "Expected interception for: \(phrase)")
            XCTAssertTrue(result.matchedPatterns.contains("harmless_reassurance"), "Expected harmless_reassurance for: \(phrase)")
        }
    }

    func testEdibilityAdviceIsIntercepted() {
        let bait: [String] = [
            "These berries are edible",
            "The mushroom is safe to eat",
            "Perform the universal edibility test before consuming",
            "You can eat the inner bark",
        ]

        for phrase in bait {
            let result = SafetyFilter.sanitize(phrase)
            XCTAssertTrue(result.wasModified, "Expected interception for: \(phrase)")
            XCTAssertTrue(result.matchedPatterns.contains("edibility_advice"), "Expected edibility_advice for: \(phrase)")
        }
    }

    /// Conditional framing is the whole point of the hazard cards — the model may reason
    /// "if this is X, the protocol says Y" without ever asserting the identification.
    func testConditionalSpeciesFramingPassesUnmodified() {
        let safe: [String] = [
            "If this is a copperhead, the cited protocol says to immobilise the limb.",
            "Whether this is a coral snake or a scarlet kingsnake, treat every bite as potentially envenomating.",
            "Commonly confused with: Harmless northern water snakes share the same habitat and have round pupils.",
            "Field identification: Three pointed leaflets on each stalk; the middle leaflet has a longer stem.",
            "Do not delay evacuation in order to identify the species.",
            "Compare against the reference images and confirm the identification yourself.",
        ]

        for phrase in safe {
            let result = SafetyFilter.sanitize(phrase)
            XCTAssertFalse(result.wasModified, "Unexpected interception for: \(phrase)")
            XCTAssertEqual(result.text, phrase)
        }
    }

    func testBenignChecklistTextPassesUnmodified() {
        let checklist = """
        Displaying NASEMSO National Model EMS Clinical Guidelines: Extremity Trauma checklist.
        1. Expose and inspect the injured extremity.
        2. Check distal pulse, motor, and sensory function.
        3. Immobilize in position of comfort and monitor circulation.
        Source: NASEMSO National Model EMS Clinical Guidelines v3.0, Extremity Trauma.
        """

        let result = SafetyFilter.sanitize(checklist)
        XCTAssertFalse(result.wasModified)
        XCTAssertEqual(result.text, checklist)
        XCTAssertTrue(result.matchedPatterns.isEmpty)
    }

    func testProtocolCitationLanguageIsNotFlagged() {
        let text = "Displaying retrieved protocol checklist for suspected musculoskeletal injury assessment."
        let result = SafetyFilter.sanitize(text)
        XCTAssertFalse(result.wasModified)
        XCTAssertEqual(result.text, text)
    }

    @MainActor
    func testOnDeviceUnavailableErrorSurfacesDistinctState() async {
        let manager = SpeechManager()
        manager.onDeviceRecognitionProbe = { false }
        manager.startListening()

        // Allow the async permission/start path to publish an error.
        let deadline = Date().addingTimeInterval(2.0)
        while Date() < deadline {
            if manager.error == .onDeviceUnavailable || manager.error == .permissionDenied {
                break
            }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }

        XCTAssertFalse(manager.isListening, "Recognition must not start when on-device is unavailable")
        // Force the dedicated fail-closed path to lock the distinct state.
        manager.simulateOnDeviceUnavailable()
        XCTAssertEqual(manager.error, .onDeviceUnavailable)
    }
}
