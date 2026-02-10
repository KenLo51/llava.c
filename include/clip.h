#ifndef CLIP_H
#define CLIP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stb_image.h"
#include "stb_image_resize2.h"

#include "gguf_reader.h"
#include "nn_ops.h"

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int image_size; // input image size (assumed square)
    float image_mean[3]; // (3,) mean for each RGB channel
    float image_std[3]; // (3,) std for each RGB channel
    float layer_norm_epsilon; // epsilon for layer normalization
    int patch_size; // size of each image patch (assumed square)
    int proj_dim; // final projection dimension for the output embedding (unused in LLaVA)
} CLIP_Vision_Config;

typedef struct {
    // embeddings
    float* e_cls; // (dim,) embedding for the [CLS] token
    float* e_patch; // (patch_size*patch_size*3, dim) weights for patch embedding
    float* e_pos; // (num_patches+1, dim) positional embeddings for patches + cls token
    // pre layernorm
    float* w_pre_ln; // (dim,) weights for pre-layernorm
    float* b_pre_ln; // (dim,) bias for pre-layernorm
    // weights & bias for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* bq; // (layer, n_heads * head_size)
    float* wk; // (layer, dim, n_heads * head_size)
    float* bk; // (layer, n_heads * head_size)
    float* wv; // (layer, dim, n_heads * head_size)
    float* bv; // (layer, n_heads * head_size)
    float* wo; // (layer, dim, dim)
    float* bo; // (layer, dim)
    // weights & bias for ffn
    float* w_fc1; // (layer, dim, hidden_dim)
    float* b_fc1; // (layer, dim, hidden_dim)
    float* w_fc2; // (layer, hidden_dim, dim)
    float* b_fc2; // (layer, hidden_dim, dim)
    // layernorm (in encoder blocks)
    float* w_ln1; // (layer, dim)
    float* b_ln1; // (layer, dim)
    float* w_ln2; // (layer, dim)
    float* b_ln2; // (layer, dim)
    // final(post) layernorm 
    float* w_post_ln; // (dim,) weights for post-layernorm
    float* b_post_ln; // (dim,) bias for post-layernorm
    // 
    float* w_proj; // (dim, proj_dim) projection matrix for output embedding (unused in LLaVA)
} CLIP_Vision_Weights;

/**
 * @struct CLIP_Vision_RunState
 * @brief Runtime activation buffers and KV cache for inference
 * 
 * Contains temporary buffers for storing intermediate activations during
 * the forward pass, as well as key-value caches for efficient autoregressive generation.
 */
typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (num_patches+1, dim)
    float *xb; // same, but inside a residual branch (num_patches+1, dim)
    float *xb2; // an additional buffer just for convenience (num_patches+1, dim)
    float *hb; // buffer for hidden dimension in the ffn (num_patches+1, hidden_dim)
    float *hb2; // buffer for hidden dimension in the ffn (num_patches+1, hidden_dim)
    float *q; // query (num_patches+1, dim)
    float *k; // key(num_patches+1, dim)
    float *v; // value (num_patches+1, dim)
    float *att; // buffer for scores/attention values (n_heads, num_patches+1, num_patches+1)
    float *f; // output features after projection (num_patches+1, proj_dim) (unused in LLaVA)
} CLIP_Vision_RunState;

/**
 * @struct CLIP_Vision_Model
 * @brief Main CLIP Vision transformer model structure
 * 
 * Combines configuration, weights, and runtime state into a single model instance.
 */
typedef struct {
    CLIP_Vision_Config config; // the hyperparameters of the architecture (the blueprint)
    CLIP_Vision_Weights weights; // the weights of the model
    CLIP_Vision_RunState state; // buffers for the "wave" of activations in the forward pass
} CLIP_Vision_Model;



// ============================================================================
// Instantiation and Cleanup Functions
// ============================================================================

/**
 * @brief Load CLIP Vision model configuration from GGUF metadata
 * 
 * Extracts hyperparameters from GGUF metadata including transformer dimensions,
 * layer counts, attention head configuration, image size, normalization parameters,
 * patch size, and projection dimensions.
 * 
 * @param config Pointer to config structure to populate
 * @param ctx GGUF context containing model metadata
 */
void clip_vision_load_config_from_gguf(CLIP_Vision_Config* config, gguf_context* ctx);

/**
 * @brief Load CLIP Vision model weights from GGUF tensors
 * 
 * Allocates memory for all weight matrices and loads them from GGUF tensors.
 * Handles embeddings (class token, patch, position), attention weights (Q, K, V),
 * output projections, feed-forward weights, and normalization parameters.
 * 
 * @param config Pointer to model configuration (for dimension information)
 * @param weights Pointer to weights structure to populate
 * @param ctx GGUF context containing model tensors
 */
void clip_vision_load_weights_from_gguf(CLIP_Vision_Config* config, CLIP_Vision_Weights* weights, gguf_context* ctx);

/**
 * @brief Initialize runtime state buffers for CLIP Vision model
 * 
 * Allocates all temporary buffers needed for forward pass computation,
 * including activation buffers for embeddings, attention, and feed-forward layers.
 * Buffer sizes are calculated based on the number of image patches.
 * 
 * @param state Pointer to runtime state structure to initialize
 * @param config Pointer to model configuration
 */
void clip_vision_init_run_state(CLIP_Vision_RunState* state, CLIP_Vision_Config* config);

/**
 * @brief Create and initialize a complete CLIP Vision model from GGUF file
 * 
 * Creates and initializes all components of the CLIP Vision model including
 * configuration, weights, and runtime state buffers.
 * 
 * @param ctx GGUF context containing model data
 * @return Pointer to initialized model (must be freed with clip_vision_delete_model)
 */
CLIP_Vision_Model* clip_vision_create_from_gguf(gguf_context* ctx);

/**
 * @brief Free all memory associated with a CLIP Vision model
 * 
 * Deallocates weights, runtime state buffers, and the model structure itself.
 * 
 * @param model Pointer to model to delete
 */
void clip_vision_delete_model(CLIP_Vision_Model* model);

// ============================================================================
// Inference Functions
// ============================================================================

/**
 * @brief Extract image features from an image file
 * 
 * Loads an image from the specified file path, preprocesses it (resize, pad, normalize),
 * and runs it through the CLIP Vision model to extract features. The image is:
 * 1. Loaded from file
 * 2. Resized maintaining aspect ratio
 * 3. Padded to square dimensions
 * 4. Normalized using model-specific mean and std
 * 5. Processed through the vision transformer
 * 6. Projected to the output embedding space
 * 
 * @param model Pointer to the CLIP Vision model
 * @param image_path Path to the input image file
 * @param output_embedding Output buffer for features (size: (num_patches+1) * proj_dim)
 */
void clip_get_image_features_from_file(CLIP_Vision_Model* model, const char* image_path, float* output_embedding);

/**
 * @brief Extract image features from a pixel array
 * 
 * Processes a pre-loaded image (as pixel array) through the CLIP Vision model.
 * The input image should be in range [0, 1] and will be normalized using the
 * model's mean and std values. The function:
 * 1. Normalizes the pixel values
 * 2. Runs forward pass through the vision transformer
 * 3. Projects features to output embedding space
 * 
 * @param model Pointer to the CLIP Vision model
 * @param image Input pixel array (size: image_size * image_size * 3, normalized [0:1])
 * @param output_embedding Output buffer for features (size: (num_patches+1) * proj_dim)
 */
void clip_get_image_features(CLIP_Vision_Model* model, float* image, float* output_embedding);

/**
 * @brief Run forward pass through the CLIP Vision transformer
 * 
 * Performs the complete forward pass through the vision transformer:
 * 1. Embedding layer: Converts image patches to embeddings with position encodings
 * 2. Pre-LayerNorm: Normalizes embeddings before encoder
 * 3. Transformer encoder: Processes through all encoder layers
 * 4. Post-LayerNorm: Normalizes the output (applied to CLS token)
 * 
 * The output features are stored in model->state.x and represent the final
 * hidden states for all patches (including the CLS token at index 0).
 * 
 * @param model Pointer to the CLIP Vision model
 * @param patches Input image pixel values (size: image_size * image_size * 3)
 * @param output_embedding Output buffer for final embeddings (size: (num_patches+1) * dim)
 */
void clip_vision_forward(CLIP_Vision_Model* model, float* patches, float* output_embedding);


#endif // CLIP_H