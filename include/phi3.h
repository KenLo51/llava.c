/**
 * @file phi3.h
 * @brief PHI-3 Transformer Model Implementation
 * 
 * This file contains the data structures and function declarations for the PHI-3
 * language model. It provides functionality for loading model configurations and
 * weights from GGUF files, running inference with KV caching, and generating text.
 * 
 * The implementation includes:
 * - Model configuration and weight structures
 * - Runtime state management with KV cache
 * - Forward pass with rotary position embeddings (RoPE)
 * - Multi-head attention with grouped query attention
 * - Feed-forward networks with SiLU activation
 * - Text generation with sampling strategies
 */

#ifndef PHI3_H
#define PHI3_H
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

/**
 * @struct Phi3_Config
 * @brief Configuration parameters for the PHI-3 transformer model
 */
typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Phi3_Config;

/**
 * @struct Phi3_Weights
 * @brief Model weights for the PHI-3 transformer
 * 
 * Contains all learnable parameters including embeddings, attention weights,
 * feed-forward network weights, and normalization parameters.
 */
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
    float* w_up; // (layer, dim, 2 * hidden_dim)
    float* w_down; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} Phi3_Weights;

/**
 * @struct Phi3_RunState
 * @brief Runtime activation buffers and KV cache for inference
 * 
 * Contains temporary buffers for storing intermediate activations during
 * the forward pass, as well as key-value caches for efficient autoregressive generation.
 */
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

/**
 * @struct Phi3_Model
 * @brief Main PHI-3 transformer model structure
 * 
 * Combines configuration, weights, and runtime state into a single model instance.
 */
typedef struct {
    Phi3_Config config; // the hyperparameters of the architecture (the blueprint)
    Phi3_Weights weights; // the weights of the model
    Phi3_RunState state; // buffers for the "wave" of activations in the forward pass
} Phi3_Model;


// ============================================================================
// Instantiation and Cleanup Functions
// ============================================================================

/**
 * @brief Load PHI-3 model configuration from GGUF metadata
 * @param phi3_config Pointer to config structure to populate
 * @param ctx GGUF context containing model metadata
 */
void phi3_load_config_from_gguf(Phi3_Config* phi3_config, gguf_context* ctx);

/**
 * @brief Load PHI-3 model weights from GGUF tensors
 * @param config Pointer to model configuration
 * @param phi3_weights Pointer to weights structure to populate
 * @param ctx GGUF context containing model tensors
 */
void phi3_load_weights_from_gguf(Phi3_Config* config, Phi3_Weights* phi3_weights, gguf_context* ctx);

/**
 * @brief Allocate memory for runtime state buffers
 * @param s Pointer to runtime state structure
 * @param p Pointer to model configuration
 */
void phi3_malloc_run_state(Phi3_RunState* s, Phi3_Config* p);

/**
 * @brief Initialize a complete PHI-3 transformer from GGUF file
 * @param ctx GGUF context containing model data
 * @return Pointer to initialized transformer (must be freed with phi3_delete)
 */
Phi3_Model* phi3_init_from_gguf(gguf_context* ctx);

/**
 * @brief Free all memory associated with a PHI-3 transformer
 * @param phi3 Pointer to transformer to delete
 */
void phi3_delete(Phi3_Model* phi3);

// ============================================================================
// Inference Functions
// ============================================================================

/**
 * @brief Apply rotary position embeddings (RoPE) in-place
 * @param x Input/output vector to apply RoPE to
 * @param dim Total dimension of the vector
 * @param head_dim Dimension per attention head
 * @param pos Position index in the sequence
 */
void phi3_rotary_embedding_inplace(float* x, int dim, int head_dim, int pos);

/**
 * @brief Apply rotary position embeddings (RoPE) with separate output
 * @param out Output vector for result
 * @param in Input vector
 * @param dim Total dimension of the vector
 * @param head_dim Dimension per attention head
 * @param pos Position index in the sequence
 */
void phi3_rotary_embedding(float* out, float* in, int dim, int head_dim, int pos);

/**
 * @brief Perform multi-head self-attention forward pass with KV cache
 * @param phi3 Pointer to transformer model
 * @param layer_index Index of the current layer
 * @param pos Current position in the sequence
 */
void phi3_attention_forward(Phi3_Model* phi3,
                            int layer_index, int pos);

/**
 * @brief Perform feed-forward network forward pass
 * @param phi3 Pointer to transformer model
 * @param layer_index Index of the current layer
 */
void phi3_FFN_forward(Phi3_Model* phi3, int layer_index);

/**
 * @brief Run forward pass through the entire transformer for a single token
 * @param phi3 Pointer to transformer model
 * @param token Input token ID
 * @param pos Current position in the sequence
 * @return Pointer to logits array for next token prediction
 */
float* phi3_forward(Phi3_Model* phi3, int token, int pos);

/**
 * @brief Generate text from a prompt (non-streaming)
 * @param transformer Pointer to transformer model
 * @param tokenizer Pointer to tokenizer
 * @param sampler Pointer to sampling strategy
 * @param prompt Input text prompt (NULL for empty prompt)
 * @param max_tokens_gen Maximum number of tokens to generate
 * @return Generated text as a string (must be freed by caller)
 */
char* phi3_generate(Phi3_Model *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int max_tokens_gen);

/**
 * @brief Generate text from a prompt with streaming callback
 * @param transformer Pointer to transformer model
 * @param tokenizer Pointer to tokenizer
 * @param sampler Pointer to sampling strategy
 * @param prompt Input text prompt (NULL for empty prompt)
 * @param max_tokens_gen Maximum number of tokens to generate
 * @param callback Function to call with each generated text chunk
 */
void phi3_generate_stream(Phi3_Model *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int max_tokens_gen, void (*callback)(const char*, size_t));

#endif // PHI3_H