#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>

#include "nn_ops.h"
#include "gguf_reader.h"
#include "phi3_tokenizer.h"
#include "sampler.h"

#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Phi3_Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights for matmuls. note dim == n_heads * head_size
    float* wqkv; // (layer, dim, 3 * n_heads * head_size)
    float* wo; // (layer, dim, dim)
    // weights for ffn
    float* w_up; // (layer, 2 * hidden_dim, dim)
    float* w_down; // (layer, dim, hidden_dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} Phi3_Weights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (2 * hidden_dim)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} Phi3_RunState;

typedef struct {
    Phi3_Config config; // the hyperparameters of the architecture (the blueprint)
    Phi3_Weights weights; // the weights of the model
    Phi3_RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    ssize_t file_size; // size of the checkpoint file in bytes
} Phi3_Transformer;


// Instantiation and cleanup functions
void load_phi3_config_from_gguf(Phi3_Config* phi3_config, gguf_context* ctx);
void load_phi3_weights_from_gguf(Phi3_Config* config, Phi3_Weights* phi3_weights, gguf_context* ctx);
Phi3_Transformer* init_phi3_from_gguf(gguf_context* ctx);

void delete_phi3_transformer(Phi3_Transformer* phi3);

// Inference functions
void phi3_rotary_embedding_inplace(float* x, int dim, int seq_len, int pos);
void phi3_rotary_embedding(float* out, float* in, int dim, int seq_len, int pos);

void phi3_attention_forward(Phi3_Transformer* phi3,
                            int layer_index, int pos);
void phi3_feed_forward(Phi3_Transformer* phi3, int layer_index);

// Forward pass with a new token
float* phi3_forward(Phi3_Transformer* phi3, int token, int pos);


char* phi3_generate(Phi3_Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int max_tokens_gen);
void phi3_generate_stream(Phi3_Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int max_tokens_gen, void (*callback)(const char*, size_t));