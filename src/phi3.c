/**
 * @file phi3.c
 * @brief PHI-3 Transformer Model Implementation
 * 
 * This file implements the PHI-3 language model, a transformer-based architecture
 * designed for efficient inference. The implementation includes:
 * 
 * Key Features:
 * - GGUF file format support for model loading
 * - Grouped Query Attention (GQA) for efficient KV caching
 * - Rotary Position Embeddings (RoPE) for position encoding
 * - SiLU activation in feed-forward networks
 * - RMSNorm for layer normalization
 * - Autoregressive text generation with sampling
 * - Streaming generation support
 * 
 * The model supports multi-threaded inference using OpenMP for parallel operations.
 */

#include "phi3.h"

// ============================================================================
// Instantiation and Cleanup Functions
// ============================================================================

/**
 * @brief Load PHI-3 model configuration from GGUF metadata
 * 
 * Extracts hyperparameters from GGUF metadata including dimensions, layer counts,
 * attention head configuration, vocabulary size, and context length.
 * 
 * @param phi3_config Pointer to config structure to populate
 * @param ctx GGUF context containing model metadata
 */
void load_phi3_config_from_gguf(Phi3_Config* phi3_config, gguf_context* ctx){
    if(!phi3_config || !ctx){
        fprintf(stderr, "NULL pointer passed to load_phi3_config_from_gguf\n");
        exit(EXIT_FAILURE);
    }

    gguf_metadata_kv* kv = NULL;

    // dim, transformer dimension
    kv = gguf_get_metadata(ctx, "phi3.embedding_length");
    if(!kv){
        fprintf(stderr, "Failed to get phi3.embedding_length from GGUF metadata\n");
        exit(EXIT_FAILURE);
    }
    phi3_config->dim = kv->value.uint32;
#ifdef DEBUG
    printf("Loaded phi3.dim = %d\n", phi3_config->dim);
#endif

    // hidden_dim, for ffn layers
    kv = gguf_get_metadata(ctx, "phi3.feed_forward_length");
    if(!kv){
        fprintf(stderr, "Failed to get phi3.feed_forward_length from GGUF metadata\n");
        exit(EXIT_FAILURE);
    }
    phi3_config->hidden_dim = kv->value.uint32;
#ifdef DEBUG
    printf("Loaded phi3.hidden_dim = %d\n", phi3_config->hidden_dim);
#endif
    // n_layers, number of layers
    kv = gguf_get_metadata(ctx, "phi3.block_count");
    if(!kv){
        fprintf(stderr, "Failed to get phi3.block_count from GGUF metadata\n");
        exit(EXIT_FAILURE);
    }
    phi3_config->n_layers = kv->value.uint32;
#ifdef DEBUG
    printf("Loaded phi3.n_layers = %d\n", phi3_config->n_layers);
#endif
    // n_heads, number of query heads
    kv = gguf_get_metadata(ctx, "phi3.attention.head_count");
    if(!kv){
        fprintf(stderr, "Failed to get phi3.attention.head_count from GGUF metadata\n");
        exit(EXIT_FAILURE);
    }
    phi3_config->n_heads = kv->value.uint32;
#ifdef DEBUG
    printf("Loaded phi3.n_heads = %d\n", phi3_config->n_heads);
#endif
    // n_kv_heads, number of key/value heads (can be < query heads because of multiquery)
    kv = gguf_get_metadata(ctx, "phi3.attention.head_count_kv");
    if(!kv){
        fprintf(stderr, "Failed to get phi3.attention.head_count_kv from GGUF metadata\n");
        exit(EXIT_FAILURE);
    }
    phi3_config->n_kv_heads = kv->value.uint32;
#ifdef DEBUG
    printf("Loaded phi3.n_kv_heads = %d\n", phi3_config->n_kv_heads);
#endif
    // vocab_size, vocabulary size, usually 256 (byte-level)
    kv = gguf_get_metadata(ctx, "tokenizer.ggml.tokens");
    if(!kv){
        fprintf(stderr, "Failed to get phi3.vocab_size from GGUF metadata\n");
        exit(EXIT_FAILURE);
    }
    phi3_config->vocab_size = ((gguf_array*)(kv->value.arr))->len;
#ifdef DEBUG
    printf("Loaded phi3.vocab_size = %d\n", phi3_config->vocab_size);
#endif
    // seq_len, max sequence length
    kv = gguf_get_metadata(ctx, "phi3.context_length");
    if(!kv){
        fprintf(stderr, "Failed to get phi3.context_length from GGUF metadata\n");
        exit(EXIT_FAILURE);
    }
    phi3_config->seq_len = kv->value.uint32;
#ifdef DEBUG
    printf("Loaded phi3.seq_len = %d\n", phi3_config->seq_len);
#endif
}


/**
 * @brief Load PHI-3 model weights from GGUF tensors
 * 
 * Allocates memory for all weight matrices and loads them from GGUF tensors.
 * Handles token embeddings, attention weights (fused QKV), output projections,
 * feed-forward weights, and normalization parameters.
 * 
 * @param config Pointer to model configuration (for dimension information)
 * @param phi3_weights Pointer to weights structure to populate
 * @param ctx GGUF context containing model tensors
 */
void load_phi3_weights_from_gguf(Phi3_Config* config, Phi3_Weights* phi3_weights, gguf_context* ctx){
    if(!phi3_weights || !ctx){
        fprintf(stderr, "NULL pointer passed to load_phi3_weights_from_gguf\n");
        exit(EXIT_FAILURE);
    }

    // Allocate memory for all weights
    phi3_weights->token_embedding_table = (float*)malloc(sizeof(float) * config->vocab_size * config->dim);
    phi3_weights->rms_att_weight = (float*)malloc(sizeof(float) * config->n_layers * config->dim);
    phi3_weights->rms_ffn_weight = (float*)malloc(sizeof(float) * config->n_layers * config->dim);
    phi3_weights->wqkv = (float*)malloc(sizeof(float) * config->n_layers * config->dim * 3 * config->dim);
    phi3_weights->wo = (float*)malloc(sizeof(float) * config->n_layers * config->dim * config->dim);
    phi3_weights->w_up = (float*)malloc(sizeof(float) * config->n_layers * 2 * config->hidden_dim * config->dim);
    phi3_weights->w_down = (float*)malloc(sizeof(float) * config->n_layers * config->dim * config->hidden_dim);
    phi3_weights->rms_final_weight = (float*)malloc(sizeof(float) * config->dim);
    phi3_weights->wcls = (float*)malloc(sizeof(float) * config->vocab_size * config->dim);
    
    // check mallocs
    if(!phi3_weights->token_embedding_table || !phi3_weights->rms_att_weight
        || !phi3_weights->rms_ffn_weight || !phi3_weights->wqkv
        || !phi3_weights->wo || !phi3_weights->w_up
        || !phi3_weights->w_down || !phi3_weights->rms_final_weight
        || !phi3_weights->wcls){
        fprintf(stderr, "Failed to allocate memory for Phi3 weights\n");
        exit(EXIT_FAILURE);
    }

    // Iterate over tensors in GGUF and load weights
    for(unsigned int i=0; i<ctx->tensor_count; i++){
        gguf_tensor* tensor = &ctx->tensors[i];

#ifdef DEBUG
        printf("Processing tensor: %s\n", tensor->name);
#endif
        
        // input and output embeddings
        if(strcmp(tensor->name, "token_embd.weight") == 0){
            copy_tensor_data_to_float_array(tensor, phi3_weights->token_embedding_table);
        }
        else if(strcmp(tensor->name, "output_norm.weight") == 0){
            copy_tensor_data_to_float_array(tensor, phi3_weights->rms_final_weight);
        }
        else if(strcmp(tensor->name, "output.weight") == 0){
            copy_tensor_data_to_float_array(tensor, phi3_weights->wcls);
        }
        // transformer blocks
        else if(strncmp(tensor->name, "blk.", 4) == 0){
            int layer_index = -1;
            // Extract layer index
            if (sscanf(tensor->name, "blk.%d.", &layer_index) != 1 || layer_index < 0 || layer_index >= config->n_layers) {
                fprintf(stderr, "Failed to extract valid layer index from tensor name: %s\n", tensor->name);
                continue;
            }
            // ffn_norm
            if(strstr(tensor->name, "ffn_norm.weight")){
                copy_tensor_data_to_float_array(tensor, 
                    &phi3_weights->rms_ffn_weight[layer_index * config->dim]);
            }
            // attn_qkv
            if(strstr(tensor->name, "attn_qkv.weight")){
                copy_tensor_data_to_float_array(tensor, 
                    &phi3_weights->wqkv[layer_index * config->dim * 3 * config->dim]);
            }
            // attn_output
            if(strstr(tensor->name, "attn_output.weight")){
                copy_tensor_data_to_float_array(tensor, 
                    &phi3_weights->wo[layer_index * config->dim * config->dim]);
            }
            // attn_norm
            if(strstr(tensor->name, "attn_norm.weight")){
                copy_tensor_data_to_float_array(tensor, 
                    &phi3_weights->rms_att_weight[layer_index * config->dim]);
            }
            // ffn_up
            if(strstr(tensor->name, "ffn_up.weight")){
                copy_tensor_data_to_float_array(tensor, 
                    &phi3_weights->w_up[layer_index * 2 * config->hidden_dim * config->dim]);
            }
            // ffn_down
            if(strstr(tensor->name, "ffn_down.weight")){
                copy_tensor_data_to_float_array(tensor, 
                    &phi3_weights->w_down[layer_index * config->dim * config->hidden_dim]);
            }
        }

        else {
            fprintf(stderr, "Unrecognized tensor name in GGUF: %s\n", tensor->name);
            // exit(EXIT_FAILURE);
        }
    }
}

/**
 * @brief Initialize a complete PHI-3 transformer from GGUF file
 * 
 * Creates and initializes all components of the transformer including
 * configuration, weights, and runtime state.
 * 
 * @param ctx GGUF context containing model data
 * @return Pointer to initialized transformer (must be freed with delete_phi3_transformer)
 */
Phi3_Transformer* init_phi3_from_gguf(gguf_context* ctx){
    Phi3_Transformer* phi3 = (Phi3_Transformer*)calloc(1, sizeof(Phi3_Transformer));
    if(!phi3){
        fprintf(stderr, "Failed to allocate memory for Phi3_Transformer\n");
        exit(EXIT_FAILURE);
    }
    load_phi3_config_from_gguf(&phi3->config, ctx);
    load_phi3_weights_from_gguf(&phi3->config, &phi3->weights, ctx);

    return phi3;
}

/**
 * @brief Free all memory associated with a PHI-3 transformer
 * 
 * Deallocates weights, runtime state buffers, and the transformer structure itself.
 * 
 * @param phi3 Pointer to transformer to delete
 */
void delete_phi3_transformer(Phi3_Transformer* phi3){
    if(phi3){
        free(phi3->weights.token_embedding_table);
        free(phi3->weights.rms_att_weight);
        free(phi3->weights.rms_ffn_weight);
        free(phi3->weights.wqkv);
        free(phi3->weights.wo);
        free(phi3->weights.w_up);
        free(phi3->weights.w_down);
        free(phi3->weights.rms_final_weight);
        free(phi3->weights.wcls);

        free(phi3->state.x);
        free(phi3->state.xb);
        free(phi3->state.xb2);
        free(phi3->state.hb);
        free(phi3->state.hb2);
        free(phi3->state.q);
        free(phi3->state.k);
        free(phi3->state.v);
        free(phi3->state.att);
        free(phi3->state.logits);
        free(phi3->state.key_cache);
        free(phi3->state.value_cache);

        free(phi3);
    }
}

/**
 * @brief Allocate memory for runtime state buffers
 * 
 * Allocates all temporary buffers needed for forward pass computation,
 * including activation buffers and KV cache.
 * 
 * @param s Pointer to runtime state structure
 * @param p Pointer to model configuration
 */
void malloc_run_state(Phi3_RunState* s, Phi3_Config* p) {
    // we calloc instead of malloc to keep valgrind happy
    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(p->dim, sizeof(float));
    s->xb2 = calloc(p->dim, sizeof(float));
    s->hb = calloc(2 * p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->q = calloc(p->dim, sizeof(float));
    s->k = calloc(p->dim, sizeof(float));
    s->v = calloc(p->dim, sizeof(float));
    s->key_cache = calloc(p->n_layers * p->seq_len * p->dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * p->seq_len * p->dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q || !s->k || !s->v
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

// ============================================================================
// Inference Functions
// ============================================================================

/**
 * @brief Apply rotary position embeddings (RoPE) in-place
 * 
 * Implements RoPE by rotating pairs of values in each attention head based on
 * position index. This encodes relative position information into the representations.
 * Uses OpenMP for parallel processing across multiple heads.
 * 
 * @param x Input/output vector to apply RoPE to
 * @param dim Total dimension of the vector
 * @param head_dim Dimension per attention head
 * @param pos Position index in the sequence
 */
void phi3_rotary_embedding_inplace(float* x, int dim, int head_dim, int pos) {
    int half_dim = head_dim / 2;
    int num_heads = dim / head_dim;

    // 1. Iterate over each head
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < num_heads; h++) {
        
        // 
        float* current_head_ptr = x + h * head_dim;

        // 2. Apply RoPE to each pair within the head
        for (int i = 0; i < half_dim; i++) {
            // calculate the rotary embedding factors
            float freq_exponent = (2.0f * i) / (float)head_dim;
            float freq = 1.0f / powf(10000.0f, freq_exponent);
            
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);


            float v0 = current_head_ptr[i];            // x1
            float v1 = current_head_ptr[i + half_dim]; // x2
            
            // perform the rotation
            // x1_new = x1 * cos - x2 * sin
            // x2_new = x2 * cos + x1 * sin
            current_head_ptr[i]            = v0 * fcr - v1 * fci;
            current_head_ptr[i + half_dim] = v1 * fcr + v0 * fci;
        }
    }
}

/**
 * @brief Apply rotary position embeddings (RoPE) with separate output
 * 
 * Same as phi3_rotary_embedding_inplace but writes to a separate output buffer
 * instead of modifying the input in-place.
 * 
 * @param out Output vector for result
 * @param in Input vector
 * @param dim Total dimension of the vector
 * @param head_dim Dimension per attention head
 * @param pos Position index in the sequence
 */
void phi3_rotary_embedding(float* out, float* in, int dim, int head_dim, int pos) {
    // RoPE: Rotary Position Embedding (separate output version)
    int half_dim = head_dim / 2;
    int num_heads = dim / head_dim;

    // 1. Iterate over each head
    for (int h = 0; h < num_heads; h++) {
        
        // 
        float* in_head_ptr = in + h * head_dim;
        float* out_head_ptr = out + h * head_dim;

        // 2. Apply RoPE to each pair within the head
        for (int i = 0; i < half_dim; i++) {
            // calculate the rotary embedding factors
            float freq_exponent = (2.0f * i) / (float)head_dim;
            float freq = 1.0f / powf(10000.0f, freq_exponent);
            
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);


            float v0 = in_head_ptr[i];            // x1
            float v1 = in_head_ptr[i + half_dim]; // x2
            
            // perform the rotation
            // x1_new = x1 * cos - x2 * sin
            // x2_new = x2 * cos + x1 * sin
            out_head_ptr[i]            = v0 * fcr - v1 * fci;
            out_head_ptr[i + half_dim] = v1 * fcr + v0 * fci;
        }
    }
}

/**
 * @brief Perform feed-forward network forward pass
 * 
 * Implements the FFN with gated SiLU activation:
 * 1. Projects input to 2*hidden_dim using gate_up weights
 * 2. Splits into gate and up components
 * 3. Applies SiLU(gate) * up
 * 4. Projects back to model dimension using down weights
 * 
 * Uses OpenMP for parallel activation computation.
 * 
 * @param phi3 Pointer to transformer model
 * @param layer_index Index of the current layer
 */
void phi3_feed_forward(Phi3_Transformer* phi3, int layer_index){
    Phi3_RunState* state = &phi3->state;
    Phi3_Weights* weights = &phi3->weights;
    float* weight_up = &weights->w_up[layer_index * 2 * phi3->config.hidden_dim * phi3->config.dim];
    float* weight_down = &weights->w_down[layer_index * phi3->config.dim * phi3->config.hidden_dim];
    int dim = phi3->config.dim;
    int hidden_dim = phi3->config.hidden_dim;

    // up_states = self.gate_up_proj(hidden_states)
    matmul(state->hb, state->xb, weight_up, dim, 2*hidden_dim);

    // gate, up_states = up_states.chunk(2, dim=-1)
    // up_states = up_states * self.activation_fn(gate) (SiLUActivation)
    // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < hidden_dim; i++) {
        state->hb[i] = state->hb[i] / (1.0f + expf(-state->hb[i]));
        state->hb2[i] = state->hb[i] * state->hb[i+hidden_dim];
    }

    // out = self.down_proj(up_states)
    matmul(state->xb, state->hb2, weight_down, hidden_dim, dim);
}

/**
 * @brief Project input vector into Query, Key, and Value vectors
 * 
 * Uses a fused weight matrix to compute Q, K, and V projections in a single
 * operation. The weight matrix has shape (3*dim, dim) with Q, K, V weights
 * concatenated along the first dimension.
 * 
 * Uses OpenMP for parallel computation of each projection.
 * 
 * @param q_out Output array for Query vector (size: dim)
 * @param k_out Output array for Key vector (size: dim)
 * @param v_out Output array for Value vector (size: dim)
 * @param in Input vector (size: dim)
 * @param weight_qkv Fused weight matrix for Q, K, V (size: 3*dim × dim)
 * @param dim Dimension of input and output vectors
 */
void phi3_proj_qkv(float* q_out, float* k_out, float* v_out,
                   float* in, float* weight_qkv,
                   int dim) {


    // ----------------------------------------------------------------------
    // 1. Calculate Query (Q)
    // The first 'dim' rows of the fused matrix correspond to Q.
    // Target Q size: [dim]
    // ----------------------------------------------------------------------
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < dim; i++) {
        float val = 0.0f;
        // Point to the start of row 'i' in the weight matrix
        float* w_row = weight_qkv + i * dim;
        
        // Dot product: weight_row . input_vector
        for (int j = 0; j < dim; j++) {
            val += w_row[j] * in[j];
        }
        
        q_out[i] = val;
    }


    // Offset the weight pointer to skip past the Q weights.
    // We have processed 'dim' rows, each of length 'dim'.
    float* weight_k_start = weight_qkv + (dim * dim);

    // ----------------------------------------------------------------------
    // 2. Calculate Key (K)
    // ----------------------------------------------------------------------
    #pragma omp parallel for private(i)
    for (i = 0; i < dim; i++) {
        float val = 0.0f;
        float* w_row = weight_k_start + i * dim;
        
        for (int j = 0; j < dim; j++) {
            val += w_row[j] * in[j];
        }
        k_out[i] = val;
    }

    // Offset the weight pointer to skip past the K weights.
    // We have processed 'dim' rows.
    float* weight_v_start = weight_k_start + (dim * dim);

    // ----------------------------------------------------------------------
    // 3. Calculate Value (V)
    // ----------------------------------------------------------------------
    #pragma omp parallel for private(i)
    for (i = 0; i < dim; i++) {
        float val = 0.0f;
        float* w_row = weight_v_start + i * dim;
        
        for (int j = 0; j < dim; j++) {
            val += w_row[j] * in[j];
        }
        v_out[i] = val;
    }
}

/**
 * @brief Perform multi-head self-attention forward pass with KV cache
 * 
 * Implements scaled dot-product attention with:
 * - Fused QKV projection
 * - Rotary position embeddings (RoPE)
 * - KV caching for efficient autoregressive generation
 * - Grouped Query Attention (GQA) support
 * - Parallel computation across attention heads using OpenMP
 * 
 * @param phi3 Pointer to transformer model
 * @param layer_index Index of the current layer
 * @param pos Current position in the sequence
 */
void phi3_attention_forward(Phi3_Transformer* phi3,
                            int layer_index, int pos) {


    Phi3_Config* config = &phi3->config;
    Phi3_RunState* state = &phi3->state;
    Phi3_Weights* weights = &phi3->weights;
    
    int dim = config->dim;
    int n_heads = config->n_heads;
    int n_kv_heads = config->n_kv_heads;
    int seq_len = config->seq_len;
    
    // Get layer-specific pointers
    int loff = layer_index * seq_len * dim; // kv cache layer offset
    float* hidden_state = state->xb;
    float* key_cache = state->key_cache + loff;
    float* value_cache = state->value_cache + loff;

    float* weight_qkv = weights->wqkv + layer_index * dim * (3 * dim);
    float* weight_o = weights->wo + layer_index * dim * dim;
    
    int head_size = dim / n_heads;
    int kv_mul = n_heads / n_kv_heads; // integer multiplier of the kv sharing in multiquery
    
    // Use state buffers instead of allocating temporary ones
    float* q = state->q;
    float* k = state->k;
    float* v = state->v;
    float* att = state->att;
    float* xb = state->xb2; // Use xb2 as temporary buffer for attention output

    // 1. project q, k, v using the fused QKV projection
    phi3_proj_qkv(q, k, v, hidden_state, weight_qkv, dim);
    
    // Apply RoPE (Rotary Position Embedding) to q and k
    phi3_rotary_embedding_inplace(q, dim, dim / n_heads, pos);
    phi3_rotary_embedding_inplace(k, dim, dim / n_heads, pos);

    // Store k and v in the cache at position pos
    float* key_cache_pos = key_cache + pos * dim;
    float* value_cache_pos = value_cache + pos * dim;
    memcpy(key_cache_pos, k, dim * sizeof(float));
    memcpy(value_cache_pos, v, dim * sizeof(float));
    
    // 2. attention scores (scaled dot-product)
    // multihead attention. iterate over all heads
    int h;
    #pragma omp parallel for private(h)
    for (h = 0; h < n_heads; h++) {
        // get the query vector for this head
        float* q_head = q + h * head_size;
        // attention scores for this head
        float* att_head = att + h * seq_len;
        
        // iterate over all timesteps, including the current one
        for (int t = 0; t <= pos; t++) {
            // get the key vector for this head and at this timestep
            float* k_head = key_cache + t * dim + (h / kv_mul) * head_size;
            // calculate the attention score as the dot product of q and k
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) {
                score += q_head[i] * k_head[i];
            }
            score /= sqrtf(head_size);
            // save the score to the attention buffer
            att_head[t] = score;
        }

        // 3. softmax the scores to get attention weights, from 0..pos inclusively
        softmax_inplace(att_head, pos + 1);
        
        // 4. weighted sum of the values, store back into xb
        float* xb_head = xb + h * head_size;
        memset(xb_head, 0, head_size * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            // get the value vector for this head and at this timestep
            float* v_head = value_cache + t * dim + (h / kv_mul) * head_size;
            // get the attention weight for this timestep
            float a = att_head[t];
            // accumulate the weighted value into xb
            for (int i = 0; i < head_size; i++) {
                xb_head[i] += a * v_head[i];
            }
        }
    }
    
    // 5. final matmul to get the output of the attention
    matmul(hidden_state, xb, weight_o, dim, dim);
}

/**
 * @brief Forward pass through a single decoder layer
 * 
 * Implements a standard transformer decoder block:
 * 1. RMSNorm + Self-Attention + Residual
 * 2. RMSNorm + Feed-Forward + Residual
 * 
 * Uses KV cache for efficient autoregressive generation.
 * 
 * @param phi3 Pointer to transformer model
 * @param layer_index Index of the current layer
 * @param pos Current position in the sequence
 */
void phi3_decoder_layer_forward(Phi3_Transformer* phi3,
                                int layer_index,
                                int pos) {


    
    Phi3_Config* config = &phi3->config;
    Phi3_Weights* weight = &phi3->weights;
    Phi3_RunState* state = &phi3->state;
    
    int dim = config->dim;
    int hidden_dim = config->hidden_dim;

    // 1. attention layernorm (RMSNorm)
    rmsnorm(state->xb, state->x, weight->rms_att_weight + layer_index * dim, dim);

    // 2. self-attention with KV cache
    phi3_attention_forward(phi3, layer_index, pos);
    

    // 3. residual connection
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < dim; i++) {
        state->x[i] += state->xb[i];
    }

    // 4. ffn layernorm (RMSNorm)
    rmsnorm(state->xb, state->x, weight->rms_ffn_weight + layer_index * dim, dim);

    // 5. feed-forward network
    phi3_feed_forward(phi3, layer_index);

    // 6. residual connection
    #pragma omp parallel for private(i)
    for (i = 0; i < dim; i++) {
        state->x[i] += state->xb[i];
    }
}

/**
 * @brief Run forward pass through the entire transformer for a single token
 * 
 * Complete forward pass:
 * 1. Token embedding lookup
 * 2. Forward through all decoder layers
 * 3. Final RMSNorm
 * 4. Output projection to vocabulary logits
 * 
 * @param phi3 Pointer to transformer model
 * @param token Input token ID
 * @param pos Current position in the sequence
 * @return Pointer to logits array for next token prediction (size: vocab_size)
 */
float* phi3_forward(Phi3_Transformer* phi3, int token, int pos){


    Phi3_Config* config = &phi3->config;
    Phi3_Weights* weight = &phi3->weights;
    Phi3_RunState* state = &phi3->state;
    float* x = state->x;
    int dim = config->dim;
    
    // 1. Embedding lookup
    float* content_row = weight->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(float));
    
    // 2. Forward all layers
    for (int l = 0; l < config->n_layers; l++) {
        phi3_decoder_layer_forward(phi3, l, pos);
    }
    
    // 3. Final RMSNorm
    rmsnorm_inplace(x, weight->rms_final_weight, dim);

    // 4. Output projection to logits
    matmul(state->logits, x, weight->wcls, dim, config->vocab_size);
    
    return state->logits;
}

/**
 * @brief Generate text from a prompt (non-streaming)
 * 
 * Performs autoregressive text generation:
 * 1. Encodes the prompt into tokens
 * 2. Processes prompt tokens through the model
 * 3. Samples and generates new tokens until EOS or max_tokens_gen
 * 4. Decodes generated tokens into text
 * 
 * @param transformer Pointer to transformer model
 * @param tokenizer Pointer to tokenizer
 * @param sampler Pointer to sampling strategy
 * @param prompt Input text prompt (NULL for empty prompt)
 * @param max_tokens_gen Maximum number of tokens to generate
 * @return Generated text as a string (must be freed by caller)
 */
char* phi3_generate(Phi3_Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int max_tokens_gen) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(transformer->config.seq_len * sizeof(int));
    encode(tokenizer, prompt, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int* generated_tokens = (int*)malloc(max_tokens_gen * sizeof(int));
    int generated_tokens_count = 0;
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < max_tokens_gen) {

        // forward the transformer to get logits for the next token
        float* logits = phi3_forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        //  terminating condition: EOS token delimits sequences. (2:<\s>, 32000:<|endoftext|>, 32007:<|end|>)
        if ((pos >= num_prompt_tokens) &&
            (next == 2 || next == 32000 || next == 32007)) { break; }

        // print the token as string, decode it with the Tokenizer object
        // printf(" Predicted token %d: ", token);
        if(pos >= num_prompt_tokens){
            generated_tokens[generated_tokens_count++] = next;
            // char* piece = decode(tokenizer, &next, 1);
            // safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        }
        token = next;

        if (next == tokenizer->eos_token) {
            break;
        }
    }

    free(prompt_tokens);

    char* decoded_str = decode(tokenizer, generated_tokens, generated_tokens_count, NULL);
    free(generated_tokens);

    return decoded_str;
}

/**
 * @brief Generate text from a prompt with streaming callback
 * 
 * Similar to phi3_generate but calls a callback function with each decoded
 * text chunk as it's generated, allowing for real-time streaming output.
 * 
 * @param transformer Pointer to transformer model
 * @param tokenizer Pointer to tokenizer
 * @param sampler Pointer to sampling strategy
 * @param prompt Input text prompt (NULL for empty prompt)
 * @param max_tokens_gen Maximum number of tokens to generate
 * @param callback Function to call with each generated text chunk (text, length)
 */
void phi3_generate_stream(Phi3_Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int max_tokens_gen, void (*callback)(const char*, size_t)) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(transformer->config.seq_len * sizeof(int));
    encode(tokenizer, prompt, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int* generated_tokens = (int*)malloc(5 * sizeof(int));
    int generated_tokens_count = 0;
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < max_tokens_gen) {

        // forward the transformer to get logits for the next token
        float* logits = phi3_forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        //  terminating condition: EOS token delimits sequences. (2:<\s>, 32000:<|endoftext|>, 32007:<|end|>)
        if ((pos >= num_prompt_tokens) &&
            (next == 2 || next == 32000 || next == 32007)) { break; }

        // print the token as string, decode it with the Tokenizer object
        // printf(" Predicted token %d: ", token);
        if(pos >= num_prompt_tokens){
            generated_tokens[generated_tokens_count++] = next;
            int num_decode_tokens = 0;
            char* piece = decode(tokenizer, &next, generated_tokens_count, &num_decode_tokens);
            if(num_decode_tokens > 0){
                callback(piece, strlen(piece));
                generated_tokens_count = 0; // reset count after callback
            }
            free(piece);
        }
        token = next;

        if (next == tokenizer->eos_token) {
            break;
        }
    }

    free(prompt_tokens);
    free(generated_tokens);
}
