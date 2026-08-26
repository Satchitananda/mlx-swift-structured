//
//  GrammarMaskedLogitProcessor.swift
//  MLXStructured
//
//  Created by Ivan Petrukha on 14.09.2025.
//

import MLXLMCommon
import MLX

public class GrammarMaskedLogitProcessor: LogitProcessor, @unchecked Sendable {

    let grammarMatcher: GrammarMatcher
    var pendingToken: MLXArray?

    public init(grammarMatcher: GrammarMatcher) {
        self.grammarMatcher = grammarMatcher
    }

    public func prompt(_ prompt: MLXArray) {
        pendingToken = nil
        grammarMatcher.reset()
    }

    public func process(logits: MLXArray) -> MLXArray {
        asyncEval(logits)

        if let token = pendingToken {
            pendingToken = nil
            grammarMatcher.accept(token: token)
            // The lazy iterator evaluates one token ahead so it can return the
            // previous token. After accepting EOS, XGrammar is terminated and
            // FillNextTokenBitmask is no longer a legal operation. The sampled
            // logits here are never emitted because generateTask stops on that
            // previous EOS token.
            if grammarMatcher.isTerminated() || grammarMatcher.isDesynced {
                return logits
            }
        }

        let mask = grammarMatcher.nextTokenMask()
        let maskWidth = mask.dim(-1)
        let logitsWidth = logits.dim(-1)

        if maskWidth == logitsWidth {
            return logits + mask
        }

        if maskWidth < logitsWidth {
            let padding = full([logitsWidth - maskWidth], values: -Float.infinity)
            let mask = concatenated([mask, padding])
            return logits + mask
        }

        return logits + mask[0..<logitsWidth]
    }

    public func didSample(token: MLXArray) {
        if !grammarMatcher.isTerminated() {
            pendingToken = token
        }
    }
}

public class EagerGrammarMaskedLogitProcessor: GrammarMaskedLogitProcessor, @unchecked Sendable {

    public override func didSample(token: MLXArray) {
        grammarMatcher.accept(token: token)
    }
}
