//
//  Grammar+Schema.swift
//  MLXStructured
//
//  Created by Ivan Petrukha on 04.10.2025.
//

import Foundation
import JSONSchema

public extension Grammar {

    static func schema(_ schema: JSONSchema = .object(), indent: Int) throws -> Grammar {
        try Grammar.schema(schema, options: JSONSchemaFormatOptions(indent: indent))
    }

    static func schema(_ schema: JSONSchema = .object(), options: JSONSchemaFormatOptions = .init()) throws -> Grammar {
        let data = try JSONEncoder.sorted.encode(schema)
        let string = String(decoding: data, as: UTF8.self).sanitizedSchema
        return .schema(string, options: options)
    }
}

public struct JSONSchemaFormatOptions: Sendable, Equatable {

    public struct Separators: Sendable, Equatable {

        public let comma: String
        public let colon: String

        public init(comma: String, colon: String) {
            self.comma = comma
            self.colon = colon
        }
    }

    public let indent: Int?
    public let anyWhitespace: Bool
    public let separators: Separators?
    public let strictMode: Bool
    public let maxWhitespaceCount: Int?

    public init(
        indent: Int? = nil,
        anyWhitespace: Bool = false,
        separators: Separators? = nil,
        strictMode: Bool = true,
        maxWhitespaceCount: Int? = nil
    ) {
        self.indent = indent
        self.anyWhitespace = anyWhitespace
        self.separators = separators
        self.strictMode = strictMode
        self.maxWhitespaceCount = maxWhitespaceCount
    }
}
