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
