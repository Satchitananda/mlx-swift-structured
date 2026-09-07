import Foundation
import Testing
import MLX
import MLXLMCommon
@testable import MLXStructured

struct SpecialTokenMaskingTests {
    @Test func jsonMasksReservedTokensButPreservesLiteralTextAndStops() async throws {
        let directory = try makeTokenizerFixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let configuration = ModelConfiguration(directory: directory, eosTokenIds: [0])
        let grammar = Grammar.schema(#"{"type":"string"}"#,
            format: JSONSchemaFormatOptions(strict: true, whitespace: .none))

        // Establish the observed defect in the unfiltered vocabulary: a model's
        // reserved image token is legal JSON string content to XGrammar.
        let info = try await TokenizerInfo.from(configuration: configuration)
        #expect(Set(info.specialTokenIds) == [0, 6, 7])
        let original = try XGrammar(compiledGrammar: GrammarCompiler(tokenizerInfo: info).compile(grammar: grammar))
        original.accept(token: MLXArray(1))
        #expect(original.nextTokenMask().exp().asArray(Int.self)[6] == 1)

        let processor = try await GrammarMaskedLogitProcessor.from(configuration: configuration, grammar: grammar)
        let matcher = processor.grammarMatcher
        matcher.accept(token: MLXArray(1))
        let allowed = matcher.nextTokenMask().exp().asArray(Int.self)
        #expect(allowed[0] == 0) // EOS waits for a complete JSON value.
        #expect(allowed[6] == 0) // Reserved image token.
        #expect(allowed[7] == 0) // Reserved reasoning token.
        #expect(allowed[8] == 1) // Ordinary added tokens remain available.
        // Quoting the marker as ordinary text remains legal; no substring scrub.
        for token in [2, 3, 4, 5, 1] {
            #expect(matcher.nextTokenMask().exp().asArray(Int.self)[token] == 1)
            matcher.accept(token: MLXArray(token))
        }
        #expect(!matcher.isDesynced)
        #expect(matcher.nextTokenMask().exp().asArray(Int.self)[0] == 1)
    }

    @Test func compilerCacheKeepsExplicitControlGrammarsSeparateFromJSON() async throws {
        let directory = try makeTokenizerFixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let configuration = ModelConfiguration(directory: directory, eosTokenIds: [0])
        let json = Grammar.schema(#"{"type":"string"}"#,
            format: JSONSchemaFormatOptions(strict: true, whitespace: .none))
        for grammar in [json, .ebnf(#"root ::= "<image|>""#), json] {
            let processor = try await GrammarMaskedLogitProcessor.from(configuration: configuration, grammar: grammar)
            let matcher = processor.grammarMatcher
            if case .schema = grammar {
                matcher.accept(token: MLXArray(1))
                #expect(matcher.nextTokenMask().exp().asArray(Int.self)[6] == 0)
            } else {
                #expect(matcher.nextTokenMask().exp().asArray(Int.self)[6] == 1)
                matcher.accept(token: MLXArray(6))
                #expect(!matcher.isDesynced)
            }
        }
    }

    private func makeTokenizerFixture() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("grammar-special-tokens-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let files: [String: String] = [
            "config.json": #"{"model_type":"fixture","vocab_size":9,"eos_token_id":0}"#,
            "tokenizer_config.json": #"{"eos_token":"<eos>","tokenizer_class":"PreTrainedTokenizerFast"}"#,
            "tokenizer.json": #"{"model":{"type":"BPE","vocab":{"<eos>":0,"\"":1,"<":2,"image":3,"|":4,">":5}},"added_tokens":[{"id":0,"content":"<eos>","special":true},{"id":6,"content":"<image|>","special":true},{"id":7,"content":"<think>","special":true},{"id":8,"content":"<ordinary>","special":false}],"decoder":{"type":"Fuse"}}"#,
        ]
        for (name, text) in files {
            try text.write(to: directory.appendingPathComponent(name), atomically: true, encoding: .utf8)
        }
        return directory
    }
}
