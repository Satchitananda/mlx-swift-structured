import Testing
@testable import MLXStructured

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
