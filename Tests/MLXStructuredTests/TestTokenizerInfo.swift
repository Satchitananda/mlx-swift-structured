import Testing
@testable import MLXStructured

struct TokenizerInfoTests {
    @Test func `runtime and tokenizer stop tokens share one grammar set`() {
        let vocab = ["<pad>", "<eos>", "x", "<|tool_response>", "<turn|>"]

        let ids = TokenizerInfo.stopTokenIDs(
            configurationIDs: [1, 3, 4],
            extraTokenStrings: ["<turn|>"],
            tokenizerEOSTokenID: 1,
            vocab: vocab
        )

        #expect(ids == [1, 3, 4])
    }

    @Test func `invalid runtime stop IDs cannot escape the vocabulary`() {
        let ids = TokenizerInfo.stopTokenIDs(
            configurationIDs: [-1, 1, 99],
            extraTokenStrings: ["missing"],
            tokenizerEOSTokenID: nil,
            vocab: ["<pad>", "<eos>"]
        )

        #expect(ids == [1])
    }
}
