#pragma once

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    const char *comma_separator_utf8;
    size_t comma_separator_len;
    const char *colon_separator_utf8;
    size_t colon_separator_len;
} json_schema_separators_t;

typedef struct {
    int indent;
    int any_whitespace;
    int strict_mode;
    int max_whitespace_cnt;
    int has_separators;
    json_schema_separators_t separators;
} json_schema_compile_options_t;

void *compile_ebnf_grammar(void *tokenizer_info, const char *ebnf_utf8, size_t ebnf_len);

void *compile_regex_grammar(void *tokenizer_info, const char *regex_utf8, size_t regex_len);

void *compile_json_schema_grammar(
    void *tokenizer_info,
    const char *schema_utf8,
    size_t schema_len,
    const json_schema_compile_options_t *options
);

void *compile_structural_tag(
    void *tokenizer_info,
    const char *structural_tag_utf8,
    size_t structural_tag_len
);

void compiled_grammar_free(void *compiled_grammar);

#ifdef __cplusplus
}
#endif
