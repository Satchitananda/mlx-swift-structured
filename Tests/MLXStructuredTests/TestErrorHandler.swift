//
//  TestErrorHandler.swift
//  MLXStructured
//
//  Created by Ivan Petrukha on 18.09.2025.
//

import Dispatch
import Foundation
import Testing
@testable import MLXStructured

struct ErrorHandlerTests {

    @Test func `Concurrent callers retain their own native error`() {
        let firstRecorded = DispatchSemaphore(value: 0)
        let secondRecorded = DispatchSemaphore(value: 0)
        let firstRead = DispatchSemaphore(value: 0)
        let completed = DispatchSemaphore(value: 0)
        let results = ErrorResults()

        Thread {
            CErrorHandler.record("first error")
            firstRecorded.signal()
            secondRecorded.wait()
            results.setFirst(CErrorHandler.lastErrorMessage)
            firstRead.signal()
            completed.signal()
        }.start()

        Thread {
            firstRecorded.wait()
            CErrorHandler.record("second error")
            secondRecorded.signal()
            firstRead.wait()
            results.setSecond(CErrorHandler.lastErrorMessage)
            completed.signal()
        }.start()

        completed.wait()
        completed.wait()

        #expect(results.values == ("first error", "second error"))
    }

    @Test func `Empty EBNF grammar throws`() async throws {
        #expect(
            performing: {
                let grammar = Grammar.ebnf("")
                let _ = try XGrammar(vocab: ["a", "b", "c"], grammar: grammar)
            },
            throws: { error in
                switch error {
                case XGrammarError.emptyGrammar:
                    return true
                default:
                    return false
                }
            }
        )
    }

    @Test func `Invalid EBNF grammar includes parser message`() async throws {
        #expect(
            performing: {
                let grammar = Grammar.ebnf("*")
                let _ = try XGrammar(vocab: ["a", "b", "c"], grammar: grammar)
            },
            throws: { error in
                switch error {
                case XGrammarError.invalidGrammar(let message):
                    return message.contains("The root rule with name \"root\" is not found")
                default:
                    return false
                }
            }
        )
    }

    @Test func `Empty regex grammar throws`() async throws {
        #expect(
            performing: {
                let grammar = Grammar.regex("")
                let _ = try XGrammar(vocab: ["a", "b", "c"], grammar: grammar)
            },
            throws: { error in
                switch error {
                case XGrammarError.emptyGrammar:
                    return true
                default:
                    return false
                }
            }
        )
    }

    @Test func `Invalid regex grammar includes parser message`() async throws {
        #expect(
            performing: {
                let grammar = Grammar.regex("*")
                let _ = try XGrammar(vocab: ["a", "b", "c"], grammar: grammar)
            },
            throws: { error in
                switch error {
                case XGrammarError.invalidGrammar(let message):
                    return message.contains("Expect element, but got *")
                default:
                    return false
                }
            }
        )
    }

    @Test func `Empty JSON schema grammar throws`() async throws {
        #expect(
            performing: {
                let grammar = Grammar.schema("")
                let _ = try XGrammar(vocab: ["a", "b", "c"], grammar: grammar)
            },
            throws: { error in
                switch error {
                case XGrammarError.emptyGrammar:
                    return true
                default:
                    return false
                }
            }
        )
    }

    @Test func `Invalid JSON schema grammar includes compiler message`() async throws {
        #expect(
            performing: {
                let grammar = Grammar.schema(#"{"type": "foo"}"#)
                let _ = try XGrammar(vocab: ["a", "b", "c"], grammar: grammar)
            },
            throws: { error in
                switch error {
                case XGrammarError.invalidGrammar(let message):
                    return message.contains("Unsupported type \"foo\"")
                default:
                    return false
                }
            }
        )
    }
}

private final class ErrorResults: @unchecked Sendable {
    private let lock = NSLock()
    private var first: String?
    private var second: String?

    var values: (String?, String?) {
        lock.withLock { (first, second) }
    }

    func setFirst(_ value: String) {
        lock.withLock { first = value }
    }

    func setSecond(_ value: String) {
        lock.withLock { second = value }
    }
}

extension XGrammar {
    convenience init(
        vocab: [String],
        vocabType: Int32 = 0,
        stopTokenIds: [Int32] = [],
        grammar: Grammar
    ) throws {
        let tokenizerInfo = TokenizerInfo(vocab: vocab, vocabType: vocabType, stopTokenIds: stopTokenIds)
        let compiler = try GrammarCompiler(tokenizerInfo: tokenizerInfo)
        let compiledGrammar = try compiler.compile(grammar: grammar)
        try self.init(compiledGrammar: compiledGrammar)
    }
}
