//
//  GrammarMatcherFactory.swift
//  MLXStructured
//
//  Created by Ivan Petrukha on 20.09.2025.
//

import MLXLMCommon
import Hub

extension GrammarMaskedLogitProcessor {

    private struct CompilerKey: Hashable, Sendable {
        let configuration: ModelConfiguration
        let jsonTextOnly: Bool
    }

    private static let cache = Cache<CompilerKey, GrammarCompiler>()

    public static func from(
        hub: HubApi = .shared,
        configuration: ModelConfiguration,
        grammar: Grammar
    ) async throws -> GrammarMaskedLogitProcessor {
        // EBNF/regex/structural grammars may deliberately match control tokens.
        // JSON schema compilation gets its own vocabulary and compiler cache.
        let jsonTextOnly: Bool
        if case .schema = grammar { jsonTextOnly = true } else { jsonTextOnly = false }
        let key = CompilerKey(configuration: configuration, jsonTextOnly: jsonTextOnly)
        let compiler: GrammarCompiler
        if let cached = await cache.value(for: key) {
            compiler = cached
        } else {
            let tokenizerInfo = try await TokenizerInfo.from(hub: hub, configuration: configuration)
            compiler = try GrammarCompiler(tokenizerInfo: jsonTextOnly
                ? tokenizerInfo.excludingNonStopSpecialTokens() : tokenizerInfo)
            await cache.set(compiler, for: key)
        }

        let compiledGrammar = try compiler.compile(grammar: grammar)
        let grammarMatcher = try XGrammar(compiledGrammar: compiledGrammar)
        let processor = GrammarMaskedLogitProcessor(grammarMatcher: grammarMatcher)
        return processor
    }
}

extension ModelConfiguration: @retroactive Hashable {
    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
        hasher.combine(tokenizerSource)
        hasher.combine(defaultPrompt)
        hasher.combine(extraEOSTokens)
        hasher.combine(eosTokenIds)
        hasher.combine(toolCallFormat)
    }
}

extension ModelConfiguration.Identifier: @retroactive Hashable {
    public func hash(into hasher: inout Hasher) {
        switch self {
        case .id(let id, let revision):
            hasher.combine(0)
            hasher.combine(id)
            hasher.combine(revision)
        case .directory(let directory):
            hasher.combine(1)
            hasher.combine(directory.path)
        }
    }
}

extension TokenizerSource: @retroactive Hashable {
    public func hash(into hasher: inout Hasher) {
        switch self {
        case .id(let id, let revision):
            hasher.combine(0)
            hasher.combine(id)
            hasher.combine(revision)
        case .directory(let directory):
            hasher.combine(1)
            hasher.combine(directory.path)
        }
    }
}
