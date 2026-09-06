import Testing
@testable import MLXStructured
import MLX
import MLXLMCommon
import MLXLLM
import MLXNN
import os

struct ConstrainedTokenIteratorTests {
    @Test(arguments: [false, true])
    func cacheConstructionFailurePropagates(useJumpForward: Bool) throws {
        let model = RuntimeContractModel(failCacheCreation: true)

        #expect(throws: RuntimeContractModel.CacheFailure.unavailable) {
            _ = try runtimeIterator(useJumpForward: useJumpForward, model: model)
        }
        #expect(model.cacheCreationCount == 1)
        #expect(model.prepareCount == 0)
    }

    @Test(arguments: [false, true])
    func suppliedCacheBypassesCreation(useJumpForward: Bool) throws {
        let model = RuntimeContractModel(failCacheCreation: true)
        let suppliedCache = KVCacheSimple()

        let iterator = try runtimeIterator(
            useJumpForward: useJumpForward,
            model: model,
            cache: [suppliedCache]
        )

        #expect(model.cacheCreationCount == 0)
        #expect(model.prepareCount == 1)
        let receivedCache = try #require(model.receivedCache.first as? KVCacheSimple)
        #expect(receivedCache === suppliedCache)
        #expect(iterator.state?[RuntimeContractModel.positionKey] == 4)
    }

    @Test(arguments: [false, true])
    func explicitPrefillAndReturnedStateReachTheCaller(useJumpForward: Bool) throws {
        let progress = OSAllocatedUnfairLock(initialState: [Int]())
        var parameters = GenerateParameters(maxTokens: 3, temperature: 0)
        parameters.prefill = PrefillParameters(stepSize: 3, chunking: .remainder) { processed, total in
            progress.withLock { $0.append(contentsOf: [processed, total]) }
        }
        let model = RuntimeContractModel()

        var iterator = try runtimeIterator(
            useJumpForward: useJumpForward,
            model: model,
            parameters: parameters
        )

        #expect(model.cacheCreationCount == 1)
        #expect(model.receivedPrefill?.stepSize == 3)
        #expect(model.receivedPrefill?.chunking == .remainder)
        #expect(progress.withLock { $0 } == [2, 4, 4, 4])
        #expect(iterator.state?[RuntimeContractModel.positionKey] == 4)

        // Both iterators advance the model by their second emitted token;
        // jump-forward first drains the token sampled during prepare.
        #expect(iterator.next() == 1)
        #expect(iterator.next() == 1)
        #expect(model.receivedPositions.first == 4)
        let finalPosition = try #require(model.receivedPositions.last)
        #expect(iterator.state?[RuntimeContractModel.positionKey] == finalPosition + 1)
    }

    private func runtimeIterator(
        useJumpForward: Bool,
        model: RuntimeContractModel,
        cache: [KVCache]? = nil,
        parameters: GenerateParameters = GenerateParameters(maxTokens: 3, temperature: 0)
    ) throws -> any TokenIteratorProtocol {
        let input = LMInput(tokens: MLXArray([1, 2, 1, 2]))
        let matcher = RuntimeContractMatcher()
        if useJumpForward {
            return try GrammarConstrainedJumpForwardTokenIterator(
                input: input,
                model: model,
                cache: cache,
                tokenizer: EmptyTokenizer(),
                grammarMatcher: matcher,
                parameters: parameters
            )
        }
        return try GrammarConstrainedTokenIterator(
            input: input,
            model: model,
            cache: cache,
            grammarMatcher: matcher,
            parameters: parameters
        )
    }

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

/// No weights, downloads or inference: only the iterator's runtime contract.
private final class RuntimeContractModel: Module, LanguageModel {
    enum CacheFailure: Error, Equatable {
        case unavailable
    }

    static let positionKey = LMOutput.Key<Int>("structured.runtime.position")

    private let failCacheCreation: Bool
    private(set) var cacheCreationCount = 0
    private(set) var prepareCount = 0
    private(set) var receivedCache: [KVCache] = []
    private(set) var receivedPrefill: PrefillParameters?
    private(set) var receivedPositions: [Int] = []

    init(failCacheCreation: Bool = false) {
        self.failCacheCreation = failCacheCreation
        super.init()
    }

    func newCache(parameters: GenerateParameters?) throws -> [KVCache] {
        cacheCreationCount += 1
        if failCacheCreation { throw CacheFailure.unavailable }
        return [KVCacheSimple()]
    }

    func prepare(
        _ input: LMInput,
        cache: [KVCache],
        state: LMOutput.State?,
        prefill: PrefillParameters
    ) throws -> PrepareResult {
        prepareCount += 1
        receivedCache = cache
        receivedPrefill = prefill
        prefill.progress?(2, input.text.tokens.size)
        prefill.progress?(input.text.tokens.size, input.text.tokens.size)
        return .logits(output(position: input.text.tokens.size))
    }

    func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?
    ) -> LMOutput {
        let position = state?[Self.positionKey] ?? -1
        receivedPositions.append(position)
        return output(position: position + 1)
    }

    private func output(position: Int) -> LMOutput {
        var state = LMOutput.State()
        state[Self.positionKey] = position
        return LMOutput(logits: MLXArray([Float(0), 1, 0]).reshaped([1, 1, 3]), state: state)
    }
}

private final class RuntimeContractMatcher: GrammarMatcher {
    var isDesynced: Bool { false }
    func isTerminated() -> Bool { false }
    func nextTokenMask() -> MLXArray { MLXArray.zeros([3]) }
    func findJumpForwardString() -> String { "" }
    func accept(token: MLXArray) {}
    func accept(string: String) {}
    func reset() {}
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
