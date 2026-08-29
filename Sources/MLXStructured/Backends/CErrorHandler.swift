//
//  CErrorHandler.swift
//  mlx-swift-structured
//
//  Created by Ivan Petrukha on 03.04.2026.
//

import Foundation
import CMLXStructured

enum CErrorHandler {
    /// Native compilation reports errors synchronously on the calling thread.
    /// Keep the message thread-local so concurrent grammar compilations cannot
    /// overwrite one another between clear, C call, and Swift error creation.
    private static let threadDictionaryKey = "mlx-structured.c-error-message"

    private static let installHandler: Void = {
        set_error_handler(errorHandlerClosure)
    }()

    private static let errorHandlerClosure: @convention(c) (UnsafePointer<CChar>?) -> Void = {
        record($0.map { String(cString: $0) })
    }

    static func initialize() {
        _ = installHandler
    }

    static func clearLastError() {
        record(nil)
    }

    static var lastErrorMessage: String {
        Thread.current.threadDictionary[threadDictionaryKey] as? String ?? "Unknown Error"
    }

    static func record(_ message: String?) {
        let dictionary = Thread.current.threadDictionary
        if let message {
            dictionary[threadDictionaryKey] = message
        } else {
            dictionary.removeObject(forKey: threadDictionaryKey)
        }
    }
}

@inline(__always)
func withCErrorHandling<T>(_ body: () throws -> T) rethrows -> T {
    CErrorHandler.initialize()
    CErrorHandler.clearLastError()
    return try body()
}
