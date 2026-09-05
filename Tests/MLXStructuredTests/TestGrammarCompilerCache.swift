import Testing
@testable import MLXStructured

struct GrammarCompilerCacheTests {
    private let tokenizer = TokenizerInfo(
        vocab: ["<eos>"] + Array("clip0123456789").map(String.init), vocabType: 0, stopTokenIds: [0]
    )

    @Test func defaultCompilerHasAFiniteCacheBudget() throws {
        let compiler = try GrammarCompiler(tokenizerInfo: tokenizer)
        #expect(compiler.cacheLimitBytes > 0)
        #expect(compiler.cacheLimitBytes <= 128 * 1_024 * 1_024)
    }

    @Test func dynamicGrammarChurnEvictsWithoutInvalidatingActiveMatchers() throws {
        let budget: Int64 = 65_536
        let compiler = try GrammarCompiler(tokenizerInfo: tokenizer, maxMemoryBytes: budget)
        let retained = try XGrammar(compiledGrammar: compiler.compile(grammar: .regex("clip0")))
        for index in 1...256 {
            let compiled = try compiler.compile(grammar: .regex("clip\(index)"))
            let matcher = try XGrammar(compiledGrammar: compiled)
            #expect(matcher.findJumpForwardString() == "clip\(index)")
            #expect(compiler.cacheSizeBytes >= 0)
            #expect(compiler.cacheSizeBytes <= budget)
        }
        #expect(retained.findJumpForwardString() == "clip0")
        retained.accept(string: "clip0")
        #expect(!retained.isDesynced)
        let recompiled = try XGrammar(compiledGrammar: compiler.compile(grammar: .regex("clip0")))
        #expect(recompiled.findJumpForwardString() == "clip0")
    }
}
