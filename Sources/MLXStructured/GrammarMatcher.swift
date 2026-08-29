//
//  GrammarMatcher.swift
//  MLXStructured
//
//  Created by Ivan Petrukha on 16.09.2025.
//

import MLX

public protocol GrammarMatcher {
    func isTerminated() -> Bool
    /// True once the matcher and the generated text have diverged (a token or
    /// string the grammar rejects, or a bitmask fill failure). A desynced
    /// matcher cannot constrain anything anymore — callers must STOP
    /// generation rather than continue under masks that no longer mean
    /// anything. The former behavior (silently reset to the grammar start and
    /// keep going) let production generation run fully unconstrained while
    /// looking healthy.
    var isDesynced: Bool { get }
    func nextTokenMask() -> MLXArray  // 0 or -infinity
    func findJumpForwardString() -> String
    func accept(token: MLXArray)
    func accept(string: String)
    func reset()
}

public extension GrammarMatcher {
    var isDesynced: Bool { false }
}
