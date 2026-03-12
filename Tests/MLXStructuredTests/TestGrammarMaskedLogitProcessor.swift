import Testing
@testable import MLXStructured
import MLX

@Test func testProcessPadsShortMaskToLogitWidth() {
    let processor = GrammarMaskedLogitProcessor(
        grammarMatcher: StubGrammarMatcher(mask: MLXArray([0.0, -Float.infinity, 0.0]))
    )

    let processed = processor.process(logits: MLXArray.zeros([1, 5]))
    let allowed = processed[0].exp().asArray(Int.self)

    #expect(allowed == [1, 0, 1, 0, 0])
}

@Test func testProcessTruncatesLongMaskToLogitWidth() {
    let processor = GrammarMaskedLogitProcessor(
        grammarMatcher: StubGrammarMatcher(mask: MLXArray([0.0, -Float.infinity, 0.0, -Float.infinity]))
    )

    let processed = processor.process(logits: MLXArray.zeros([1, 3]))
    let allowed = processed[0].exp().asArray(Int.self)

    #expect(allowed == [1, 0, 1])
}

private struct StubGrammarMatcher: GrammarMatcher {
    let mask: MLXArray

    func nextTokenMask() -> MLXArray { mask }
    func advance(token: MLXArray) {}
    func reset() {}
    func isTerminated() -> Bool { false }
}
