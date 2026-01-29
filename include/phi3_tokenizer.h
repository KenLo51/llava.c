#ifndef PHI3_TOKENIZER_H
#define PHI3_TOKENIZER_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "gguf_reader.h"

typedef struct {
    char *str;
    int id;
} TokenIndex;


#define TOKEN_TYPE_NORMAL 1
#define TOKEN_TYPE_UNKNOWN 2
#define TOKEN_TYPE_CONTROL 3
#define TOKEN_TYPE_USER_DEFINED 4
#define TOKEN_TYPE_UNUSED 5
#define TOKEN_TYPE_BYTE 6

typedef struct {
    char** vocab;
    int* vocab_lengths;
    float* vocab_scores;
    int* vocab_types;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;

    int bos_token; // ID of Beginning-Of-Sequence token
    int eos_token; // ID of End-Of-Sequence token
    int pad_token; // ID of Padding token
    int unk_token; // ID of Unknown token

    int* special_tokens;
    int special_tokens_size;
    int* rstrip_tokens;
    int rstrip_tokens_size;
} Tokenizer;


// Initialize tokenizer
void init_tokenizer_empty(Tokenizer* tokenizer);
void init_tokenizer_vocabsize(Tokenizer* tokenizer, int vocab_size);
void init_tokenizer_vocabarr(Tokenizer* tokenizer, char** vocab, int* types, float* scores, int* vocab_len, int vocab_size);

Tokenizer* build_tokenizer_from_gguf(gguf_context* ctx);
// Clear tokenizer
void delete_tokenizer(Tokenizer* t);

int compare_tokens(const void *a, const void *b);
char* decode(Tokenizer* tokenizer, int* tokens, int tokens_size, int* num_token_decoded);
void safe_printf(char *piece);
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size);
int str_lookup_len(char *str, int len, Tokenizer* t);
int str_lookup_prefix(char *str, int max_len, TokenIndex *sorted_vocab, int vocab_size, int* matched_len);
void encode(Tokenizer* t, char *text, int *tokens, int *n_tokens);

#endif // PHI3_TOKENIZER_H