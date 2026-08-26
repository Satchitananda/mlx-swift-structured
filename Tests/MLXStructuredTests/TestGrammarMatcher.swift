//
//  TestGrammarMatcher.swift
//  MLXStructured
//
//  Created by Ivan Petrukha on 16.09.2025.
//

import Testing
@testable import MLXStructured
import MLX

struct GrammarMatcherTests {

    @Test func `EBNF grammar matcher finds and accepts jump forward string`() async throws {
        let vocab = ["Y", "E", "S", "!", "?"]
        let grammar = Grammar.ebnf(#"root ::= "YES" ("!" | "?")"#)
        let grammarMatcher = try XGrammar(vocab: vocab, stopTokenIds: [], grammar: grammar)

        #expect(grammarMatcher.findJumpForwardString() == "YES")
        grammarMatcher.accept(string: "YES")

        let mask = grammarMatcher.nextTokenMask()
        let allowed = mask.exp().asArray(Int.self)
        #expect(allowed == [0, 0, 0, 1, 1])
    }

    @Test func `EBNF grammar matcher respects accepted jump forward string`() async throws {
        let vocab = ["Y", "E", "S", "!", "?"]
        let grammar = Grammar.ebnf(#"root ::= "YES" ("!" | "?")"#)
        let grammarMatcher = try XGrammar(vocab: vocab, stopTokenIds: [], grammar: grammar)

        #expect(grammarMatcher.findJumpForwardString() == "YES")
        grammarMatcher.accept(string: "YES")

        let mask = grammarMatcher.nextTokenMask()
        let allowed = mask.exp().asArray(Int.self)
        #expect(allowed == [0, 0, 0, 1, 1])

        grammarMatcher.accept(token: MLXArray(3))
        #expect(grammarMatcher.findJumpForwardString() == "")
    }

    @Test func `EBNF grammar matcher accepts YES sequence`() async throws {
        let vocab = ["Y", "E", "S", "N", "O"]
        let grammar = Grammar.ebnf(#"root ::= ("YES" | "NO")"#)
        let grammarMatcher = try XGrammar(vocab: vocab, stopTokenIds: [], grammar: grammar)

        let advances: [Int] = "YES".map(String.init).compactMap({ vocab.firstIndex(of: $0) })
        let expectations: [[Int]] = [
            [1, 0, 0, 1, 0],  // "Y" or "N"
            [0, 1, 0, 0, 0],  // "E"
            [0, 0, 1, 0, 0],  // "S"
        ]

        for (expectation, advance) in zip(expectations, advances) {
            let mask = grammarMatcher.nextTokenMask()
            let allowed = mask.exp().asArray(Int.self)
            #expect(allowed == expectation)
            grammarMatcher.accept(token: MLXArray(advance))
        }
    }

    @Test func `EBNF grammar matcher allows EOS after valid sequence`() async throws {
        let vocab = ["<eos>", "Y", "E", "S", "N", "O"]
        let grammar = Grammar.ebnf(#"root ::= ("YES" | "NO")"#)
        let grammarMatcher = try XGrammar(vocab: vocab, stopTokenIds: [0], grammar: grammar)

        let advances: [Int] = "YES".map(String.init).compactMap({ vocab.firstIndex(of: $0) }) + [0]
        let expectations: [[Int]] = [
            [0, 1, 0, 0, 1, 0],  // "Y" or "N"
            [0, 0, 1, 0, 0, 0],  // "E"
            [0, 0, 0, 1, 0, 0],  // "S"
            [1, 0, 0, 0, 0, 0],  // "<eos>"
        ]

        for (expectation, advance) in zip(expectations, advances) {
            let mask = grammarMatcher.nextTokenMask()
            let allowed = mask.exp().asArray(Int.self)
            #expect(allowed == expectation)
            grammarMatcher.accept(token: MLXArray(advance))
        }
    }

    @Test func `Regex email grammar matcher enforces token constraints`() async throws {
        let vocab = ["<eos>", "a", "b", "c", "@", "."]
        let grammar = Grammar.regex(#"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"#)  // Simple email regex
        let grammarMatcher = try XGrammar(vocab: vocab, stopTokenIds: [0], grammar: grammar)

        let advances: [Int] = "abc@ab.cc".map(String.init).compactMap({ vocab.firstIndex(of: $0) }) + [0]
        let expectations: [[Int]] = [
            [0, 1, 1, 1, 0, 1],  // Not "@" nor "<eos>
            [0, 1, 1, 1, 1, 1],  // Not "<eos>"
            [0, 1, 1, 1, 1, 1],  // Not "<eos>"
            [0, 1, 1, 1, 1, 1],  // Not "<eos>"
            [0, 1, 1, 1, 0, 1],  // Not "@" nor "<eos>"
            [0, 1, 1, 1, 0, 1],  // Not "@" nor "<eos>"
            [0, 1, 1, 1, 0, 1],  // Not "@" nor "<eos>"
            [0, 1, 1, 1, 0, 1],  // Not "@" nor "<eos>"
            [0, 1, 1, 1, 0, 1],  // Not "@" nor "<eos>"
            [1, 1, 1, 1, 0, 1],  // Not "@"
        ]

        for (expectation, advance) in zip(expectations, advances) {
            let mask = grammarMatcher.nextTokenMask()
            let allowed = mask.exp().asArray(Int.self)
            #expect(allowed == expectation)
            grammarMatcher.accept(token: MLXArray(advance))
        }
    }

    @Test func `Regex phone grammar matcher enforces token constraints`() async throws {
        let vocab = ["<eos>", "+", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", " "]
        let grammar = Grammar.regex(#"^\+[0-9\s]{7,15}$"#)  // Simple phone regex
        let grammarMatcher = try XGrammar(vocab: vocab, stopTokenIds: [0], grammar: grammar)

        let advances: [Int] = "+1 234 5678".map(String.init).compactMap({ vocab.firstIndex(of: $0) }) + [0]
        let expectations: [[Int]] = [
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  // "+"
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  // Any decimal but not "+" or "<eos>"
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  // Any decimal but not "+" or "<eos>"
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  // Any decimal but not "+" or "<eos>"
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  // Any decimal but not "+" or "<eos>"
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  // Any decimal but not "+" or "<eos>"
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  // Any decimal but not "+" or "<eos>"
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  // Any decimal but not "+" or "<eos>"
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  // Any decimal but not "+"
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  // Any decimal but not "+"
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  // Any decimal but not "+"
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  // Any decimal but not "+"
        ]

        for (expectation, advance) in zip(expectations, advances) {
            let mask = grammarMatcher.nextTokenMask()
            let allowed = mask.exp().asArray(Int.self)
            #expect(allowed == expectation)
            grammarMatcher.accept(token: MLXArray(advance))
        }
    }

    @Test func `JSON schema grammar matcher accepts required object payload`() async throws {
        let vocab = ["<eos>", "{", "}", ":", " ", "\"", "a", "b"]
        let grammar = try Grammar.schema(.object(properties: ["a": .string()], required: ["a"]))
        let grammarMatcher = try XGrammar(vocab: vocab, stopTokenIds: [0], grammar: grammar)

        let advances: [Int] = #"{"a": "b"}"#.map(String.init).compactMap({ vocab.firstIndex(of: $0) }) + [0]
        let expectations: [[Int]] = [
            [0, 1, 0, 0, 0, 0, 0, 0],  // "{"
            [0, 0, 0, 0, 1, 1, 0, 0],  // optional space or """
            [0, 0, 0, 0, 0, 0, 1, 0],  // "a"
            [0, 0, 0, 0, 0, 1, 0, 0],  // """
            [0, 0, 0, 1, 1, 0, 0, 0],  // optional space or ":"
            [0, 0, 0, 0, 1, 1, 0, 0],  // optional space or """
            [0, 0, 0, 0, 1, 1, 0, 0],  // more space or """
            [0, 1, 1, 1, 1, 1, 1, 1],  // Any char except "<eos>"
            [0, 1, 1, 1, 1, 1, 1, 1],  // Any char except "<eos>"
            [0, 0, 1, 0, 1, 0, 0, 0],  // optional space or "}"
            [1, 0, 0, 0, 0, 0, 0, 0],  // "<eos>"
        ]

        for (expectation, advance) in zip(expectations, advances) {
            let mask = grammarMatcher.nextTokenMask()
            let allowed = mask.exp().asArray(Int.self)
            #expect(allowed == expectation)
            grammarMatcher.accept(token: MLXArray(advance))
        }
    }
}

/// Production regressions (SightRoll Director, 2026-08-26): the JSON-schema
/// grammar for `whitespace: .none` demanded `": "` / `", "` (xgrammar's
/// default separators carry spaces), so a model prompted with compact JSON
/// disagreed with the matcher at the first separator — and the silent
/// accept()->reset() path then let generation run fully unconstrained while
/// looking healthy.
struct GrammarDesyncRegressionTests {

    private let jsonVocab = ["{", "}", "\"", ":", ",", "a", "b", "x", " "]

    @Test func `whitespace none compiles a truly compact grammar`() async throws {
        let schema = #"{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}"#
        let matcher = try XGrammar(
            vocab: jsonVocab,
            grammar: .schema(schema, format: JSONSchemaFormatOptions(strict: true, whitespace: .none))
        )
        for ch in #"{"a":"x"}"# {
            matcher.accept(string: String(ch))
        }
        #expect(!matcher.isDesynced, "compact JSON must walk a whitespace-none grammar without a space after ':'")
        // Root object closed — anything further is illegal, proving the walk
        // really consumed the grammar rather than skating on a lax one.
        matcher.accept(string: "x")
        #expect(matcher.isDesynced)
    }

    @Test func `rejected accept marks desync instead of silently resetting`() async throws {
        let matcher = try XGrammar(
            vocab: ["Y", "E", "S", "N", "O", "!"],
            grammar: .ebnf(#"root ::= "YES" "!""#)
        )
        matcher.accept(string: "N")
        #expect(matcher.isDesynced, "an illegal string must mark the matcher desynced")
        // The old reset() made the matcher accept "YES" again from the start,
        // hiding the divergence. Desync is sticky until an explicit reset.
        matcher.accept(string: "YES")
        #expect(matcher.isDesynced)
        matcher.reset()
        #expect(!matcher.isDesynced)
        matcher.accept(string: "YES")
        matcher.accept(string: "!")
        #expect(!matcher.isDesynced, "a legal walk after reset must stay in sync")
    }

    @Test func `all allowed mask is not a desync`() async throws {
        let matcher = try XGrammar(
            vocab: ["<eos>", "a", "b"],
            stopTokenIds: [0],
            grammar: .regex(".*")
        )

        let allowed = matcher.nextTokenMask().exp().asArray(Int.self)

        #expect(allowed == [1, 1, 1])
        #expect(!matcher.isDesynced, "xgrammar returns false when no mask is needed; that is not an error")
    }

    @Test func `lazy processor does not request a mask after accepting stop token`() async throws {
        let matcher = TerminatingMatcher()
        let processor = GrammarMaskedLogitProcessor(grammarMatcher: matcher)
        processor.didSample(token: MLXArray(0))

        _ = processor.process(logits: MLXArray.zeros([3]))

        #expect(matcher.acceptedTokens == 1)
        #expect(matcher.maskRequests == 0)
        #expect(!matcher.isDesynced)
    }

}

private final class TerminatingMatcher: GrammarMatcher {
    private(set) var acceptedTokens = 0
    private(set) var maskRequests = 0
    private(set) var isDesynced = false
    private var terminated = false

    func isTerminated() -> Bool { terminated }

    func nextTokenMask() -> MLXArray {
        maskRequests += 1
        isDesynced = true
        return MLXArray.zeros([3])
    }

    func findJumpForwardString() -> String { "" }

    func accept(token: MLXArray) {
        acceptedTokens += 1
        terminated = true
    }

    func accept(string: String) {}

    func reset() {
        acceptedTokens = 0
        maskRequests = 0
        isDesynced = false
        terminated = false
    }
}
