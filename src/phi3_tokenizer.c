#include "phi3_tokenizer.h"


// Initialize tokenizer
void init_tokenizer_empty(Tokenizer* tokenizer){
    // Allocate memory for the tokenizer
    if(tokenizer == NULL){
        fprintf(stderr, "Failed to allocate memory for tokenizer\n");
        exit(EXIT_FAILURE);
    }
}
void init_tokenizer_vocabsize(Tokenizer* tokenizer, int vocab_size){
    // Allocate memory for the tokenizer
    if(tokenizer == NULL){
        fprintf(stderr, "Failed to allocate memory for tokenizer\n");
        exit(EXIT_FAILURE);
    }

    // Init vocab array
    tokenizer->vocab_size = vocab_size;
    tokenizer->vocab = (char**)calloc(vocab_size, sizeof(char*));
    if(tokenizer->vocab == NULL){
        fprintf(stderr, "Failed to allocate memory for tokenizer vocab\n");
        exit(EXIT_FAILURE);
    }
    tokenizer->vocab_lengths = (int*)calloc(vocab_size, sizeof(int));
    if(tokenizer->vocab_lengths == NULL){
        fprintf(stderr, "Failed to allocate memory for tokenizer vocab_lengths\n");
        exit(EXIT_FAILURE);
    }
    // Init vocab types array
    tokenizer->vocab_types = (int*)malloc(vocab_size * sizeof(int));
    if(tokenizer->vocab_types == NULL){
        fprintf(stderr, "Failed to allocate memory for tokenizer vocab_types\n");
        exit(EXIT_FAILURE);
    }
    // Init vocab scores array
    tokenizer->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    if(tokenizer->vocab_scores == NULL){
        fprintf(stderr, "Failed to allocate memory for tokenizer vocab_scores\n");
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < vocab_size; i++){
        tokenizer->vocab_scores[i] = 0.0f;
    }
    tokenizer->sorted_vocab = NULL;
    tokenizer->max_token_length = 0;
}
void init_tokenizer_vocabarr(Tokenizer* tokenizer, char** vocab, int* types, float* scores, int* vocab_len, int vocab_size){
    init_tokenizer_vocabsize(tokenizer, vocab_size);
    // Copy vocab strings
    for(int i = 0; i < vocab_size; i++){
        size_t len = vocab_len[i];        
        tokenizer->vocab[i] = (char*)malloc(len+1);
        if(tokenizer->vocab[i] == NULL){
            fprintf(stderr, "Failed to allocate memory for tokenizer vocab[%d]\n", i);
            exit(EXIT_FAILURE);
        }
        memcpy(tokenizer->vocab[i], vocab[i], len);
        tokenizer->vocab[i][len] = '\0';
        tokenizer->vocab_lengths[i] = len;
        if (len > tokenizer->max_token_length) {
            tokenizer->max_token_length = len;
        }
    }
    // Copy vocab scores
    for(int i = 0; i < vocab_size; i++){
        tokenizer->vocab_scores[i] = scores[i];
    }
    // Copy vocab types
    for(int i = 0; i < vocab_size; i++){
        tokenizer->vocab_types[i] = types[i];
    }
    // Initialize sorted vocab
    tokenizer->sorted_vocab = (TokenIndex*)malloc(tokenizer->vocab_size * sizeof(TokenIndex));
    if(tokenizer->sorted_vocab == NULL){
        fprintf(stderr, "Failed to allocate memory for sorted_vocab\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < tokenizer->vocab_size; i++) {
        tokenizer->sorted_vocab[i].str = tokenizer->vocab[i];
        tokenizer->sorted_vocab[i].id = i;
    }
    qsort(tokenizer->sorted_vocab, tokenizer->vocab_size, sizeof(TokenIndex), compare_tokens);
    // Init special and rstrip tokens
    tokenizer->special_tokens_size = 0;
    for(int i = 0; i < vocab_size; i++)
        if(tokenizer->vocab_types[i]==TOKEN_TYPE_CONTROL || tokenizer->vocab_types[i]==TOKEN_TYPE_USER_DEFINED) // control or user defined
            tokenizer->special_tokens_size++;
    tokenizer->special_tokens = (int*)malloc(tokenizer->special_tokens_size * sizeof(int));
    int special_idx = 0;
    for(int i = 0; i < vocab_size; i++){
        if(tokenizer->vocab_types[i]==TOKEN_TYPE_CONTROL || tokenizer->vocab_types[i]==TOKEN_TYPE_USER_DEFINED){ // control or user defined
#ifdef DEBUG
            printf("Adding special token ID %d: '%s'\n", i, tokenizer->vocab[i]);
#endif
            tokenizer->special_tokens[special_idx++] = i;
        }
    }

    tokenizer->rstrip_tokens_size = 1 + vocab_size - 32000+1;
    tokenizer->rstrip_tokens = (int*)malloc(tokenizer->rstrip_tokens_size * sizeof(int));
    int rstrip_idx = 0;
    tokenizer->rstrip_tokens[0] = 3; //</s>
    for(int i = 32000; i < vocab_size; i++)
        tokenizer->rstrip_tokens[rstrip_idx++] = i;

}

Tokenizer* build_tokenizer_from_gguf(gguf_context* ctx){
    if (!ctx) {
        fprintf(stderr, "NULL gguf_context passed to build_tokenizer_from_gguf\n");
        exit(EXIT_FAILURE);
    }
    

    // Build tokenizer from GGUF metadata
    gguf_metadata_kv* vocab_size_kv = gguf_get_metadata(ctx, "tokenizer.ggml.tokens");
    int vocab_size = 0;
    if (vocab_size_kv && vocab_size_kv->type == GGUF_TYPE_ARRAY) {
        gguf_array* tokens_array = (gguf_array*)vocab_size_kv->value.arr;
        vocab_size = tokens_array->len;

#ifdef DEBUG
        printf("Vocabulary size from GGUF: %d\n", vocab_size);
#endif
    }
    // Get token types
    int* token_types = malloc(vocab_size * sizeof(int32_t));
    if(token_types == NULL){
        fprintf(stderr, "Failed to allocate memory for token types\n");
        exit(EXIT_FAILURE);
    }
    gguf_metadata_kv* token_types_kv = gguf_get_metadata(ctx, "tokenizer.ggml.token_type");
    if (token_types_kv && token_types_kv->type == GGUF_TYPE_ARRAY) {
        gguf_array* types_array = (gguf_array*)token_types_kv->value.arr;
        gguf_value* type_data = types_array->data;
        
        for (int i = 0; i < vocab_size && i < types_array->len; i++) {
            int32_t token_type = type_data[i].int32;
            token_types[i] = token_type;
        }
    } else {
        // Default all to 1 if not present. (1=normal, 2=unknown, 3=control, 4=user defined, 5=unused, 6=byte).
        for (int i = 0; i < vocab_size; i++) {
            token_types[i] = 1;
        }
    }

    // Get token scores
    float* token_scores = malloc(vocab_size * sizeof(float));
    if(token_scores == NULL){
        fprintf(stderr, "Failed to allocate memory for token scores\n");
        exit(EXIT_FAILURE);
    }
    gguf_metadata_kv* token_scores_kv = gguf_get_metadata(ctx, "tokenizer.ggml.scores");
    if (token_scores_kv && token_scores_kv->type == GGUF_TYPE_ARRAY) {
        gguf_array* scores_array = (gguf_array*)token_scores_kv->value.arr;
        gguf_value* score_data = (gguf_value*)scores_array->data;
        
        for (int i = 0; i < vocab_size && i < scores_array->len; i++) {
            float score = score_data[i].float32;
            token_scores[i] = score;
        }
    } else {
        // Default all to 0.0f if not present.
        for (int i = 0; i < vocab_size; i++) {
            token_scores[i] = 0.0f;
        }
    }

    // Copy vocab from GGUF to temporary arrays
    gguf_metadata_kv* tokens_kv = gguf_get_metadata(ctx, "tokenizer.ggml.tokens");
    char **vocab = (char**)malloc(vocab_size * sizeof(char*));
    int *vocab_len = (int*)malloc(vocab_size * sizeof(int));
    if(vocab == NULL || vocab_len == NULL){
        fprintf(stderr, "Failed to allocate memory for vocab or vocab_len\n");
        exit(EXIT_FAILURE);
    }

    if (tokens_kv && tokens_kv->type == GGUF_TYPE_ARRAY) {
        gguf_array* tokens_array = (gguf_array*)tokens_kv->value.arr;
        gguf_string* token_strings = (gguf_string*)tokens_array->data;
        
        for (int i = 0; i < vocab_size; i++) {
            const char* token = token_strings[i].data;
            size_t len = token_strings[i].len;
            vocab[i] = (char*)malloc(len);
            if(vocab[i] == NULL){
                fprintf(stderr, "Failed to allocate memory for vocab[%d]\n", i);
                exit(EXIT_FAILURE);
            }
            memcpy(vocab[i], token, len);
            vocab_len[i] = len;
        }
    }

    // Initialize tokenizer with vocab
    Tokenizer* tokenizer = (Tokenizer*)calloc(1, sizeof(Tokenizer));
    init_tokenizer_vocabarr(tokenizer, vocab, token_types, token_scores, vocab_len, vocab_size);

    // Free temporary vocab arrays
    free(vocab);
    free(vocab_len);

    // Load token scores (if available)
    gguf_metadata_kv* scores_kv = gguf_get_metadata(ctx, "tokenizer.ggml.scores");
    if (scores_kv && scores_kv->type == GGUF_TYPE_ARRAY) {
        gguf_array* scores_array = (gguf_array*)scores_kv->value.arr;
        float* score_data = (float*)scores_array->data;
        
        for (int i = 0; i < vocab_size && i < scores_array->len; i++) {
            tokenizer->vocab_scores[i] = score_data[i];
        }
    }

    // Get special token IDs
    gguf_metadata_kv* kv;
    kv = gguf_get_metadata(ctx, "tokenizer.ggml.bos_token_id");
    if (kv && kv->type == GGUF_TYPE_UINT32) {
        tokenizer->bos_token = kv->value.uint32;
    }
    kv = gguf_get_metadata(ctx, "tokenizer.ggml.eos_token_id");
    if (kv && kv->type == GGUF_TYPE_UINT32) {
        tokenizer->eos_token = kv->value.uint32;
    }
    kv = gguf_get_metadata(ctx, "tokenizer.ggml.pad_token_id");
    if (kv && kv->type == GGUF_TYPE_UINT32) {
        tokenizer->pad_token = kv->value.uint32;
    }
    kv = gguf_get_metadata(ctx, "tokenizer.ggml.unk_token_id");
    if (kv && kv->type == GGUF_TYPE_UINT32) {
        tokenizer->unk_token = kv->value.uint32;
    }


    return tokenizer;
}

// Clear tokenizer
void delete_tokenizer(Tokenizer* t){
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->vocab_lengths);
    free(t->vocab_scores);
    free(t->vocab_types);
    free(t->sorted_vocab);
    free(t->special_tokens);
    free(t->rstrip_tokens);
}

// 
int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}


char* decode(Tokenizer* tokenizer, int* tokens, int tokens_size, int* num_token_decoded){
    int decoded_str_size = 16;
    int decoded_str_ptr = 0;
    char* decoded_str = (char*)malloc(decoded_str_size);
    if(num_token_decoded) *num_token_decoded = 0;

    for(int i = 0; i < tokens_size; i++){
        int token_id = tokens[i];
        
        int token_type = tokenizer->vocab_types[token_id];
        int token_str_len = tokenizer->vocab_lengths[token_id];
        char* token_str = tokenizer->vocab[token_id];
        //printf("Decoding token ID %d: type %d, str '%s'\n", token_id, token_type, token_str);
        

        if(token_type == TOKEN_TYPE_NORMAL || token_type == TOKEN_TYPE_UNKNOWN){
            // normal or unknown token, decode it
            // Ensure enough space in decoded_str
            while(decoded_str_ptr + token_str_len >= decoded_str_size){
                decoded_str_size *= 2;
                char* new_decoded_str = (char*)realloc(decoded_str, decoded_str_size);
                if(new_decoded_str == NULL){
                    fprintf(stderr, "Failed to reallocate memory for decoded string\n");
                    exit(EXIT_FAILURE);
                }
                decoded_str = new_decoded_str;
            }
            // Append token string to decoded_str. and replace '▁' with space
            for(int j = 0; j < token_str_len; j++){
                bool need_space_replacement = false;
                if(j+2 < token_str_len){
                    if((unsigned char)token_str[j] == 0xE2 && (unsigned char)token_str[j+1] == 0x96 && (unsigned char)token_str[j+2] == 0x81)
                        need_space_replacement = true;
                }

                if(need_space_replacement){
                    decoded_str[decoded_str_ptr++] = ' ';
                    j += 2;
                }
                else{
                    decoded_str[decoded_str_ptr++] = token_str[j];
                }
            }
            if(num_token_decoded) (*num_token_decoded)++;
            continue;
        }
        if(token_type == TOKEN_TYPE_CONTROL || token_type == TOKEN_TYPE_USER_DEFINED){
            // skip special tokens
            continue;
        }
        if(token_type == TOKEN_TYPE_BYTE){
            // byte token, format is <0xXX>
            uint8_t byte_val;

            byte_val = (uint8_t)strtol(token_str+3, NULL, 16);
            int num_bytes_needed = 0;
            if((byte_val&0x80) == 0x00){
                num_bytes_needed = 1;
            } 
            else if((byte_val&0xE0) == 0xC0){
                num_bytes_needed = 2;
            }
            else if((byte_val&0xF0) == 0xE0) {
                num_bytes_needed = 3;
            }
            else if((byte_val&0xF8) == 0xF0) {
                num_bytes_needed = 4;
            }
            else{
                fprintf(stderr, "Invalid byte token value: 0x%02X\n", byte_val);
                continue;
            }

            if(i+num_bytes_needed-1 >= tokens_size) break; // Not enough tokens to decode byte
            // Verify that the next tokens are also byte tokens
            for(int b = 0; b < num_bytes_needed; b++){
                token_id = tokens[i+b];
                token_type = tokenizer->vocab_types[token_id];
                if(token_type != TOKEN_TYPE_BYTE){
                    fprintf(stderr, "Expected byte token, got token ID %d of type %d\n", token_id, token_type);
                    break;
                }
            }

            for(int b = 0; b < num_bytes_needed; b++){
                token_id = tokens[i+b];
                token_str = tokenizer->vocab[token_id];
                byte_val = (uint8_t)strtol(token_str+3, NULL, 16);
                if(decoded_str_ptr + 1 >= decoded_str_size){
                    decoded_str_size *= 2;
                    char* new_decoded_str = (char*)realloc(decoded_str, decoded_str_size);
                    if(new_decoded_str == NULL){
                        fprintf(stderr, "Failed to reallocate memory for decoded string\n");
                        exit(EXIT_FAILURE);
                    }
                    decoded_str = new_decoded_str;
                }
                decoded_str[decoded_str_ptr++] = (char)byte_val;
                if(num_token_decoded) (*num_token_decoded)++;
            }
            continue;
        }
    }

    // Null-terminate the decoded string
    if (decoded_str_ptr >= decoded_str_size) {
        decoded_str_size += 1;
        char* new_decoded_str = (char*)realloc(decoded_str, decoded_str_size);
        if (new_decoded_str == NULL) {
            fprintf(stderr, "Failed to reallocate memory for decoded string\n");
            free(decoded_str);
            exit(EXIT_FAILURE);
        }
        decoded_str = new_decoded_str;
    }
    decoded_str[decoded_str_ptr] = '\0';

    return decoded_str;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

// Length-aware string lookup - compares exact binary content with length
int str_lookup_len(char *str, int len, Tokenizer* t) {
    // Check if string is already null-terminated at the right position
    int can_use_bsearch = 1;
    for (int i = 0; i < len; i++) {
        if (str[i] == '\0') {
            can_use_bsearch = 0;
            break;
        }
    }
    
    if (can_use_bsearch && str[len] == '\0') {
        // Can use binary search safely
        int result = str_lookup(str, t->sorted_vocab, t->vocab_size);
        return result;
    }
    
    // Linear search with length-based comparison
    for (int i = 0; i < t->vocab_size; i++) {
        if (t->vocab_lengths[i] == len && memcmp(t->vocab[i], str, len) == 0) {
            return i;
        }
    }
    return -1;
}

// Find longest prefix match in vocab
int str_lookup_prefix(char *str, int max_len, TokenIndex *sorted_vocab, int vocab_size, int* matched_len) {
    int best_id = -1;
    int best_len = 0;
    
    // Try to find the longest matching token by checking all possible lengths
    for (int len = max_len; len >= 1; len--) {
        // Create temporary string of this length
        char temp[512];
        if (len >= sizeof(temp)) continue;
        memcpy(temp, str, len);
        temp[len] = '\0';
        
        int id = str_lookup(temp, sorted_vocab, vocab_size);
        if (id != -1) {
            // Found a match
            if (len > best_len) {
                best_len = len;
                best_id = id;
            }
        }
    }
    
    *matched_len = best_len;
    return best_id;
}

void encode(Tokenizer* t, char *text, int *tokens, int *n_tokens) {
    if (text == NULL || text[0] == '\0') {
        *n_tokens = 0;
        return;
    }
    
    int text_len = strlen(text);
    
    // Temporary structure to hold pieces (either token IDs or string segments)
    typedef struct {
        int is_token;  // 1 if this is a token ID, 0 if it's a string segment
        int token_id;  // Valid if is_token == 1
        char* str;     // Valid if is_token == 0
        int str_len;   // Length of string segment
    } Piece;
    
    Piece* pieces = (Piece*)malloc(text_len * sizeof(Piece));
    int pieces_count = 0;
    
    // Step 1: Tokenize special tokens first
    int i = 0;
    while (i < text_len) {
        int matched = 0;
        
        // Try to match special tokens
        for (int s = 0; s < t->special_tokens_size; s++) {
            int special_id = t->special_tokens[s];
            int special_len = t->vocab_lengths[special_id];
            
            if (i + special_len <= text_len && 
                memcmp(text + i, t->vocab[special_id], special_len) == 0) {
                
                // Add special token ID
                pieces[pieces_count].is_token = 1;
                pieces[pieces_count].token_id = special_id;
                pieces[pieces_count].str = NULL;
                pieces[pieces_count].str_len = 0;
                pieces_count++;
                
                // Check if this is an rstrip token
                int is_rstrip = 0;
                for (int r = 0; r < t->rstrip_tokens_size; r++) {
                    if (special_id == t->rstrip_tokens[r]) {
                        is_rstrip = 1;
                        break;
                    }
                }
                
                i += special_len;
                
                // Remove trailing spaces if rstrip token
                if (is_rstrip) {
                    while (i < text_len && 
                           (text[i] == ' ' || text[i] == '\t' || 
                            text[i] == '\n' || text[i] == '\r')) {
                        i++;
                    }
                }
                
                matched = 1;
                break;
            }
        }
        
        if (!matched) {
            // Add character to current string piece or create new piece
            if (pieces_count == 0 || pieces[pieces_count - 1].is_token) {
                // Create new string piece
                pieces[pieces_count].is_token = 0;
                pieces[pieces_count].token_id = -1;
                pieces[pieces_count].str = (char*)malloc(text_len);
                pieces[pieces_count].str[0] = text[i];
                pieces[pieces_count].str_len = 1;
                pieces_count++;
            } else {
                // Append to existing string piece
                pieces[pieces_count - 1].str[pieces[pieces_count - 1].str_len] = text[i];
                pieces[pieces_count - 1].str_len++;
            }
            i++;
        }
    }
    
    // Step 2: Replace ' ' by '▁' (UTF-8: 0xE2 0x96 0x81) in string pieces
    for (int p = 0; p < pieces_count; p++) {
        if (!pieces[p].is_token) {
            char* old_str = pieces[p].str;
            int old_len = pieces[p].str_len;
            
            // Count spaces to allocate new buffer (each space becomes 3 bytes)
            int space_count = 0;
            for (int j = 0; j < old_len; j++) {
                if (old_str[j] == ' ') space_count++;
            }
            
            // Allocate new buffer: add '▁' prefix (3 bytes) + space replacements (2 extra bytes each)
            char* new_str = (char*)malloc(old_len + 3 + space_count * 2);
            int new_len = 0;
            
            // Add '▁' prefix
            new_str[new_len++] = 0xE2;
            new_str[new_len++] = 0x96;
            new_str[new_len++] = 0x81;
            
            // Replace spaces with '▁'
            for (int j = 0; j < old_len; j++) {
                if (old_str[j] == ' ') {
                    new_str[new_len++] = 0xE2;
                    new_str[new_len++] = 0x96;
                    new_str[new_len++] = 0x81;
                } else {
                    new_str[new_len++] = old_str[j];
                }
            }
            
            free(old_str);
            pieces[p].str = new_str;
            pieces[p].str_len = new_len;
        }
    }
    
    // Step 3-4: Process each string piece
    int token_count = 0;
    
    for (int p = 0; p < pieces_count; p++) {
        if (pieces[p].is_token) {
            tokens[token_count++] = pieces[p].token_id;
        } else {
            // Process string piece
            char* piece = pieces[p].str;
            int piece_len = pieces[p].str_len;
            
            // Encode string into initial byte tokens
            int* piece_ids = (int*)malloc(piece_len * 6 * sizeof(int));  // Worst case: each char becomes 6 tokens
            float* piece_scores = (float*)malloc(piece_len * 6 * sizeof(float));
            int piece_ids_count = 0;
            
            int pos = 0;
            while (pos < piece_len) {
                // Try to match token from vocabulary
                int matched_len = 0;
                int matched_id = -1;
                
                // Try to find longest match in vocabulary
                for (int len = (piece_len - pos < t->max_token_length ? piece_len - pos : t->max_token_length); 
                     len >= 1; len--) {
                    int id = str_lookup_len(piece + pos, len, t);
                    if (id != -1) {
                        matched_id = id;
                        matched_len = len;
                        break;
                    }
                }
                
                if (matched_id != -1) {
                    piece_ids[piece_ids_count] = matched_id;
                    piece_scores[piece_ids_count] = t->vocab_scores[matched_id];
                    piece_ids_count++;
                    pos += matched_len;
                } else {
                    // Fallback to byte-level tokenization
                    unsigned char byte = (unsigned char)piece[pos];
                    char byte_token[8];
                    snprintf(byte_token, sizeof(byte_token), "<0x%02X>", byte);
                    
                    int byte_id = str_lookup(byte_token, t->sorted_vocab, t->vocab_size);
                    if (byte_id != -1) {
                        piece_ids[piece_ids_count] = byte_id;
                        piece_scores[piece_ids_count] = t->vocab_scores[byte_id];
                        piece_ids_count++;
                    }
                    pos++;
                }
            }
            
            // Step 4: Merge tokens greedily with highest score
            while (piece_ids_count > 1) {
                float best_score = -1e10f;
                int best_idx = -1;
                int best_merged_id = -1;
                
                // Find best pair to merge
                for (int j = 0; j < piece_ids_count - 1; j++) {
                    int id1 = piece_ids[j];
                    int id2 = piece_ids[j + 1];
                    
                    // Combine tokens
                    int combined_len = t->vocab_lengths[id1] + t->vocab_lengths[id2];
                    char* combined = (char*)malloc(combined_len + 1);
                    memcpy(combined, t->vocab[id1], t->vocab_lengths[id1]);
                    memcpy(combined + t->vocab_lengths[id1], t->vocab[id2], t->vocab_lengths[id2]);
                    combined[combined_len] = '\0';
                    
                    int merged_id = str_lookup_len(combined, combined_len, t);
                    free(combined);
                    
                    if (merged_id != -1) {
                        float score = t->vocab_scores[merged_id];
                        if (score > best_score) {
                            best_score = score;
                            best_idx = j;
                            best_merged_id = merged_id;
                        }
                    }
                }
                
                if (best_idx == -1) {
                    break;  // No more merges possible
                }
                
                // Perform merge at best_idx
                piece_ids[best_idx] = best_merged_id;
                piece_scores[best_idx] = best_score;
                
                // Shift remaining tokens
                for (int j = best_idx + 1; j < piece_ids_count - 1; j++) {
                    piece_ids[j] = piece_ids[j + 1];
                    piece_scores[j] = piece_scores[j + 1];
                }
                piece_ids_count--;
            }
            
            // Add final tokens
            for (int j = 0; j < piece_ids_count; j++) {
                tokens[token_count++] = piece_ids[j];
            }
            
            free(piece_ids);
            free(piece_scores);
        }
    }
    
    // Clean up pieces
    for (int p = 0; p < pieces_count; p++) {
        if (!pieces[p].is_token && pieces[p].str != NULL) {
            free(pieces[p].str);
        }
    }
    free(pieces);
    
    *n_tokens = token_count;
}