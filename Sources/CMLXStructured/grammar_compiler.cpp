#include "mlx_structured/grammar_compiler.h"
#include "mlx_structured/error_handler.h"
#include <xgrammar/matcher.h>

using namespace xgrammar;

extern "C" void *compile_ebnf_grammar(
    void *tokenizer_info,
    const char *ebnf_utf8,
    size_t ebnf_len
) {
    try {
        const std::string ebnf(ebnf_utf8, ebnf_len);
        auto &tokenizer_info_ptr = *static_cast<TokenizerInfo *>(tokenizer_info);
        auto *compiled_grammar_ptr = new CompiledGrammar(
            GrammarCompiler(tokenizer_info_ptr).CompileGrammar(Grammar::FromEBNF(ebnf))
        );
        return compiled_grammar_ptr;
    } catch (const std::exception &e) {
        catch_error(e.what());
        return nullptr;
    }
}

extern "C" void *compile_regex_grammar(
    void *tokenizer_info,
    const char *regex_utf8,
    size_t regex_len
) {
    try {
        const std::string regex(regex_utf8, regex_len);
        auto &tokenizer_info_ptr = *static_cast<TokenizerInfo *>(tokenizer_info);
        auto *compiled_grammar_ptr =
            new CompiledGrammar(GrammarCompiler(tokenizer_info_ptr).CompileRegex(regex));
        return compiled_grammar_ptr;
    } catch (const std::exception &e) {
        catch_error(e.what());
        return nullptr;
    }
}

extern "C" void *compile_json_schema_grammar(
    void *tokenizer_info,
    const char *schema_utf8,
    size_t schema_len,
    const json_schema_compile_options_t *options
) {
    try {
        const json_schema_compile_options_t default_options{
            .indent = -1,
            .any_whitespace = 1,
            .strict_mode = 1,
            .max_whitespace_cnt = -1,
            .has_separators = 0,
            .separators = {nullptr, 0, nullptr, 0},
        };

        const std::string schema(schema_utf8, schema_len);
        const auto &compile_options = options ? *options : default_options;
        const bool any_whitespace = static_cast<bool>(compile_options.any_whitespace);
        const bool strict_mode = static_cast<bool>(compile_options.strict_mode);

        std::optional<int> indent = std::nullopt;
        if (compile_options.indent >= 0) {
            indent = compile_options.indent;
        }

        std::optional<int> max_whitespace_cnt = std::nullopt;
        if (compile_options.max_whitespace_cnt >= 0) {
            max_whitespace_cnt = compile_options.max_whitespace_cnt;
        }

        std::optional<std::pair<std::string, std::string>> separators = std::nullopt;
        if (compile_options.has_separators && compile_options.separators.comma_separator_utf8 &&
            compile_options.separators.colon_separator_utf8) {
            separators = std::make_pair(
                std::string(
                    compile_options.separators.comma_separator_utf8,
                    compile_options.separators.comma_separator_len
                ),
                std::string(
                    compile_options.separators.colon_separator_utf8,
                    compile_options.separators.colon_separator_len
                )
            );
        }

        auto &tokenizer_info_ptr = *static_cast<TokenizerInfo *>(tokenizer_info);
        const auto compiled_grammar = GrammarCompiler(tokenizer_info_ptr)
                                          .CompileJSONSchema(
                                              schema,
                                              any_whitespace,
                                              indent,
                                              separators,
                                              strict_mode,
                                              max_whitespace_cnt
                                          );

        auto *compiled_grammar_ptr = new CompiledGrammar(compiled_grammar);
        return compiled_grammar_ptr;
    } catch (const std::exception &e) {
        catch_error(e.what());
        return nullptr;
    }
}

extern "C" void *compile_structural_tag(
    void *tokenizer_info,
    const char *structural_tag_utf8,
    size_t structural_tag_len
) {
    try {
        const std::string structural_tag(structural_tag_utf8, structural_tag_len);
        auto &tokenizer_info_ptr = *static_cast<TokenizerInfo *>(tokenizer_info);
        auto *compiled_grammar_ptr = new CompiledGrammar(
            GrammarCompiler(tokenizer_info_ptr).CompileStructuralTag(structural_tag)
        );
        return compiled_grammar_ptr;
    } catch (const std::exception &e) {
        catch_error(e.what());
        return nullptr;
    }
}

extern "C" void compiled_grammar_free(void *compiled_grammar) {
    if (compiled_grammar) {
        delete static_cast<CompiledGrammar *>(compiled_grammar);
    }
}
