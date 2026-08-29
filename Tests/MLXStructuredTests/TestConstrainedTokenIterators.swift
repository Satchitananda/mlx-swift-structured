import Testing
@testable import MLXStructured
import MLX
import MLXLMCommon
import MLXLLM

struct ConstrainedTokenIteratorTests {
    @Test func `lazy iterator emits nothing after its first mask failure`() throws {
        let matcher = DesyncOnMaskMatcher()
        var iterator = try GrammarConstrainedTokenIterator(
            input: LMInput(tokens: MLXArray([1])),
            model: smallModel(),
            grammarMatcher: matcher,
            parameters: GenerateParameters(maxTokens: 2, temperature: 0)
        )

        #expect(iterator.next() == nil)
        #expect(matcher.maskRequests == 1)
        #expect(matcher.isDesynced)
    }

    @Test func `lazy iterator does not emit a token rejected during advance`() throws {
        let matcher = DesyncOnAcceptMatcher()
        var iterator = try GrammarConstrainedTokenIterator(
            input: LMInput(tokens: MLXArray([1])),
            model: smallModel(),
            grammarMatcher: matcher,
            parameters: GenerateParameters(maxTokens: 2, temperature: 0)
        )

        #expect(iterator.next() == nil)
        #expect(matcher.acceptedTokens == 1)
        #expect(matcher.isDesynced)
    }

    @Test func `jump-forward iterator emits nothing after its first mask failure`() throws {
        let matcher = DesyncOnMaskMatcher()
        var iterator = try GrammarConstrainedJumpForwardTokenIterator(
            input: LMInput(tokens: MLXArray([1])),
            model: smallModel(),
            tokenizer: EmptyTokenizer(),
            grammarMatcher: matcher,
            parameters: GenerateParameters(maxTokens: 2, temperature: 0)
        )

        #expect(iterator.next() == nil)
        #expect(matcher.maskRequests == 1)
        #expect(matcher.isDesynced)
    }

    private func smallModel() -> LlamaModel {
        LlamaModel(
            .init(
                hiddenSize: 8,
                hiddenLayers: 1,
                intermediateSize: 16,
                attentionHeads: 1,
                rmsNormEps: 1e-5,
                vocabularySize: 3,
                kvHeads: 1
            )
        )
    }
}

private final class DesyncOnAcceptMatcher: GrammarMatcher {
    private(set) var isDesynced = false
    private(set) var acceptedTokens = 0

    func isTerminated() -> Bool { false }
    func nextTokenMask() -> MLXArray { MLXArray.zeros([3]) }
    func findJumpForwardString() -> String { "" }

    func accept(token: MLXArray) {
        acceptedTokens += 1
        isDesynced = true
    }

    func accept(string: String) {}

    func reset() {
        isDesynced = false
        acceptedTokens = 0
    }
}

private final class DesyncOnMaskMatcher: GrammarMatcher {
    private(set) var isDesynced = false
    private(set) var maskRequests = 0

    func isTerminated() -> Bool { false }

    func nextTokenMask() -> MLXArray {
        maskRequests += 1
        isDesynced = true
        return MLXArray.zeros([3])
    }

    func findJumpForwardString() -> String { "" }
    func accept(token: MLXArray) {}
    func accept(string: String) {}

    func reset() {
        isDesynced = false
        maskRequests = 0
    }
}

private struct EmptyTokenizer: Tokenizer {
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }

    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}
