#include "clip.h"


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
void clip_vision_load_config_from_gguf(CLIP_Vision_Config* config, gguf_context* ctx){
    if(!config || !ctx){
        fprintf(stderr, "NULL pointer passed to clip_vision_load_config_from_gguf\n");
        exit(EXIT_FAILURE);
    }

    gguf_metadata_kv* kv = NULL;

    // dim, transformer dimension
    kv = gguf_get_metadata(ctx, "clip.vision.embedding_length");
    if(!kv){
        fprintf(stderr, "Failed to get clip.vision.embedding_length from GGUF metadata\n");
        exit(EXIT_FAILURE);
    }
    config->dim = kv->value.uint32;
    printf("Loaded clip.vision.dim = %d\n", config->dim);

    // hidden_dim, for ffn layers
    kv = gguf_get_metadata(ctx, "clip.vision.feed_forward_length");
    if(!kv){
        fprintf(stderr, "Failed to get clip.vision.feed_forward_length from GGUF metadata\n");
        exit(EXIT_FAILURE);
    }
    config->hidden_dim = kv->value.uint32;
    printf("Loaded clip.vision.hidden_dim = %d\n", config->hidden_dim);

    // n_layers, number of layers
    kv = gguf_get_metadata(ctx, "clip.vision.block_count");
    if(!kv){
        fprintf(stderr, "Failed to get clip.vision.block_count from GGUF metadata\n");
        exit(EXIT_FAILURE);
    }
    config->n_layers = kv->value.uint32;
    printf("Loaded clip.vision.n_layers = %d\n", config->n_layers);

    // n_heads, number of attention heads
    kv = gguf_get_metadata(ctx, "clip.vision.attention.head_count");
    if(!kv){
        fprintf(stderr, "Failed to get clip.vision.attention.head_count from GGUF metadata\n");
        exit(EXIT_FAILURE);
    }
    config->n_heads = kv->value.uint32;
    printf("Loaded clip.vision.n_heads = %d\n", config->n_heads);

    // image_size, input image size
    kv = gguf_get_metadata(ctx, "clip.vision.image_size");
    if(!kv){
        fprintf(stderr, "Failed to get clip.vision.image_size from GGUF metadata\n");
        exit(EXIT_FAILURE);
    }
    config->image_size = kv->value.uint32;
    printf("Loaded clip.vision.image_size = %d\n", config->image_size);

    // image_mean, mean for each RGB channel
    kv = gguf_get_metadata(ctx, "clip.vision.image_mean");
    if(!kv || kv->type != GGUF_TYPE_ARRAY){
        fprintf(stderr, "Failed to get clip.vision.image_mean from GGUF metadata\n");
        exit(EXIT_FAILURE);
    }
    gguf_array* mean_array = (gguf_array*)(kv->value.arr);
    if(mean_array->len != 3){
        fprintf(stderr, "Expected 3 values for image_mean, got %zu\n", mean_array->len);
        exit(EXIT_FAILURE);
    }
    if(!config->image_mean){
        fprintf(stderr, "Failed to allocate memory for image_mean\n");
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < 3; i++){
        gguf_value mean_data = mean_array->data[i];
        config->image_mean[i] = mean_data.float32;
    }
    printf("Loaded clip.vision.image_mean = [%f, %f, %f]\n", 
           config->image_mean[0], config->image_mean[1], config->image_mean[2]);

    // image_std, std for each RGB channel
    kv = gguf_get_metadata(ctx, "clip.vision.image_std");
    if(!kv || kv->type != GGUF_TYPE_ARRAY){
        fprintf(stderr, "Failed to get clip.vision.image_std from GGUF metadata\n");
        exit(EXIT_FAILURE);
    }
    gguf_array* std_array = (gguf_array*)(kv->value.arr);
    if(std_array->len != 3){
        fprintf(stderr, "Expected 3 values for image_std, got %zu\n", std_array->len);
        exit(EXIT_FAILURE);
    }
    if(!config->image_std){
        fprintf(stderr, "Failed to allocate memory for image_std\n");
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < 3; i++){
        gguf_value std_data = std_array->data[i];
        config->image_std[i] = std_data.float32;
    }
    printf("Loaded clip.vision.image_std = [%f, %f, %f]\n", 
           config->image_std[0], config->image_std[1], config->image_std[2]);

    // layer_norm_epsilon
    kv = gguf_get_metadata(ctx, "clip.vision.attention.layer_norm_epsilon");
    if(!kv){
        fprintf(stderr, "Failed to get clip.vision.attention.layer_norm_epsilon from GGUF metadata\n");
        exit(EXIT_FAILURE);
    }
    config->layer_norm_epsilon = kv->value.float32;
    printf("Loaded clip.vision.layer_norm_epsilon = %f\n", config->layer_norm_epsilon);

    // patch_size, size of each image patch
    kv = gguf_get_metadata(ctx, "clip.vision.patch_size");
    if(!kv){
        fprintf(stderr, "Failed to get clip.vision.patch_size from GGUF metadata\n");
        exit(EXIT_FAILURE);
    }
    config->patch_size = kv->value.uint32;
    printf("Loaded clip.vision.patch_size = %d\n", config->patch_size);

    // proj_dim, final projection dimension
    kv = gguf_get_metadata(ctx, "clip.vision.projection_dim");
    if(!kv){
        fprintf(stderr, "Failed to get clip.vision.projection_dim from GGUF metadata\n");
        exit(EXIT_FAILURE);
    }
    config->proj_dim = kv->value.uint32;
    printf("Loaded clip.vision.proj_dim = %d\n", config->proj_dim);
}

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
void clip_vision_load_weights_from_gguf(CLIP_Vision_Config* config, CLIP_Vision_Weights* weights, gguf_context* ctx){
    if(!weights || !ctx){
        fprintf(stderr, "NULL pointer passed to clip_vision_load_weights_from_gguf\n");
        exit(EXIT_FAILURE);
    }

    int num_patch = 1 + (config->image_size/config->patch_size) * (config->image_size/config->patch_size);

    // Allocate memory for all weights
    // embeddings
    weights->e_cls = (float*)malloc(sizeof(float) * config->dim); // (dim,) embedding for the [CLS] token
    weights->e_patch = (float*)malloc(sizeof(float) * config->patch_size * config->patch_size * 3 * config->dim); // (patch_size*patch_size*3, dim) weights for patch embedding
    weights->e_pos = (float*)malloc(sizeof(float) * num_patch * config->dim); // (num_patches+1, dim) positional embeddings for patches + cls token
    weights->w_pre_ln = (float*)malloc(sizeof(float) * config->dim); // (dim,) weights for pre-layernorm
    weights->b_pre_ln = (float*)malloc(sizeof(float) * config->dim); // (dim,) bias for pre-layernorm
    weights->wq = (float*)malloc(sizeof(float) * config->n_layers * config->dim * config->dim); // (layer, dim, n_heads * head_size)
    weights->bq = (float*)malloc(sizeof(float) * config->n_layers * config->dim); // (layer, n_heads * head_size)
    weights->wk = (float*)malloc(sizeof(float) * config->n_layers * config->dim * config->dim); // (layer, dim, n_heads * head_size)
    weights->bk = (float*)malloc(sizeof(float) * config->n_layers * config->dim); // (layer, n_heads * head_size)
    weights->wv = (float*)malloc(sizeof(float) * config->n_layers * config->dim * config->dim); // (layer, dim, n_heads * head_size)
    weights->bv = (float*)malloc(sizeof(float) * config->n_layers * config->dim); // (layer, n_heads * head_size)
    weights->wo = (float*)malloc(sizeof(float) * config->n_layers * config->dim * config->dim); // (layer, dim, dim)
    weights->bo = (float*)malloc(sizeof(float) * config->n_layers * config->dim); // (layer, dim)
    weights->w_fc1 = (float*)malloc(sizeof(float) * config->n_layers * config->dim * config->hidden_dim); // (layer, dim, hidden_dim)
    weights->b_fc1 = (float*)malloc(sizeof(float) * config->n_layers * config->dim * config->hidden_dim); // (layer, dim, hidden_dim)
    weights->w_fc2 = (float*)malloc(sizeof(float) * config->n_layers * config->dim * config->hidden_dim); // (layer, hidden_dim, dim)
    weights->b_fc2 = (float*)malloc(sizeof(float) * config->n_layers * config->dim * config->hidden_dim); // (layer, hidden_dim, dim)
    weights->w_ln1 = (float*)malloc(sizeof(float) * config->n_layers * config->dim); // (layer, dim)
    weights->b_ln1 = (float*)malloc(sizeof(float) * config->n_layers * config->dim); // (layer, dim)
    weights->w_ln2 = (float*)malloc(sizeof(float) * config->n_layers * config->dim); // (layer, dim)
    weights->b_ln2 = (float*)malloc(sizeof(float) * config->n_layers * config->dim); // (layer, dim)
    weights->w_post_ln = (float*)malloc(sizeof(float) * config->dim); // (dim,) weights for post-layernorm
    weights->b_post_ln = (float*)malloc(sizeof(float) * config->dim); // (dim,) bias for post-layernorm
    weights->w_proj = (float*)malloc(sizeof(float) * config->dim * config->proj_dim); // (dim, proj_dim) projection matrix for output embedding (unused in LLaVA)

    // check mallocs
    if(!weights->e_cls || !weights->e_patch || !weights->e_pos || 
       !weights->w_pre_ln || !weights->b_pre_ln || 
       !weights->wq || !weights->bq || !weights->wk || !weights->bk || 
       !weights->wv || !weights->bv || !weights->wo || !weights->bo || 
       !weights->w_fc1 || !weights->b_fc1 || !weights->w_fc2 || !weights->b_fc2 || 
       !weights->w_ln1 || !weights->b_ln1 || !weights->w_ln2 || !weights->b_ln2 || 
       !weights->w_post_ln || !weights->b_post_ln || !weights->w_proj){
        fprintf(stderr, "Failed to allocate memory for CLIP Vision weights\n");
        exit(EXIT_FAILURE);
    }

    // Iterate over tensors in GGUF and load weights
    for(unsigned int i=0; i<ctx->tensor_count; i++){
        gguf_tensor* tensor = &ctx->tensors[i];

#ifdef DEBUG
        printf("Processing tensor: %s\n", tensor->name);
#endif
        
        // input and output embeddings
        if(strcmp(tensor->name, "v.class_embd") == 0){
            copy_tensor_data_to_float_array(tensor, weights->e_cls);
        }
        else if(strcmp(tensor->name, "v.patch_embd.weight") == 0){
            copy_tensor_data_to_float_array(tensor, weights->e_patch);
        }
        else if(strcmp(tensor->name, "v.position_embd.weight") == 0){
            copy_tensor_data_to_float_array(tensor, weights->e_pos);
        }
        else if(strcmp(tensor->name, "v.pre_ln.weight") == 0){
            copy_tensor_data_to_float_array(tensor, weights->w_pre_ln);
        }
        else if(strcmp(tensor->name, "v.pre_ln.bias") == 0){
            copy_tensor_data_to_float_array(tensor, weights->b_pre_ln);
        }
        else if(strcmp(tensor->name, "v.post_ln.weight") == 0){
            copy_tensor_data_to_float_array(tensor, weights->w_post_ln);
        }
        else if(strcmp(tensor->name, "v.post_ln.bias") == 0){
            copy_tensor_data_to_float_array(tensor, weights->b_post_ln);
        }
        else if(strcmp(tensor->name, "vproj.weight") == 0){
            copy_tensor_data_to_float_array(tensor, weights->w_proj);
        }
        // transformer blocks
        else if(strncmp(tensor->name, "v.blk.", 4) == 0){
            int layer_index = -1;
            // Extract layer index
            if (sscanf(tensor->name, "v.blk.%d.", &layer_index) != 1 || layer_index < 0 || layer_index >= config->n_layers) {
                fprintf(stderr, "Failed to extract valid layer index from tensor name: %s\n", tensor->name);
                continue;
            }
            // attn qkvo
            if(strstr(tensor->name, "attn_q.weight")){
                copy_tensor_data_to_float_array(tensor, &weights->wq[layer_index * config->dim * config->dim]);
            }
            else if(strstr(tensor->name, "attn_q.bias")){
                copy_tensor_data_to_float_array(tensor, &weights->bq[layer_index * config->dim]);
            }
            else if(strstr(tensor->name, "attn_k.weight")){
                copy_tensor_data_to_float_array(tensor, &weights->wk[layer_index * config->dim * config->dim]);
            }
            else if(strstr(tensor->name, "attn_k.bias")){
                copy_tensor_data_to_float_array(tensor, &weights->bk[layer_index * config->dim]);
            }
            else if(strstr(tensor->name, "attn_v.weight")){
                copy_tensor_data_to_float_array(tensor, &weights->wv[layer_index * config->dim * config->dim]);
            }
            else if(strstr(tensor->name, "attn_v.bias")){
                copy_tensor_data_to_float_array(tensor, &weights->bv[layer_index * config->dim]);
            }
            else if(strstr(tensor->name, "attn_out.weight")){
                copy_tensor_data_to_float_array(tensor, &weights->wo[layer_index * config->dim * config->dim]);
            }
            else if(strstr(tensor->name, "attn_out.bias")){
                copy_tensor_data_to_float_array(tensor, &weights->bo[layer_index * config->dim]);
            }
            // feedforward
            else if(strstr(tensor->name, "ffn_down.weight")){
                copy_tensor_data_to_float_array(tensor, &weights->w_fc1[layer_index * config->dim * config->hidden_dim]);
            }
            else if(strstr(tensor->name, "ffn_down.bias")){
                copy_tensor_data_to_float_array(tensor, &weights->b_fc1[layer_index * config->hidden_dim]);
            }
            else if(strstr(tensor->name, "ffn_up.weight")){
                copy_tensor_data_to_float_array(tensor, &weights->w_fc2[layer_index * config->dim * config->hidden_dim]);
            }
            else if(strstr(tensor->name, "ffn_up.bias")){
                copy_tensor_data_to_float_array(tensor, &weights->b_fc2[layer_index * config->dim]);
            }
            // layernorms
            else if(strstr(tensor->name, "ln1.weight")){
                copy_tensor_data_to_float_array(tensor, &weights->w_ln1[layer_index * config->dim]);
            }
            else if(strstr(tensor->name, "ln1.bias")){
                copy_tensor_data_to_float_array(tensor, &weights->b_ln1[layer_index * config->dim]);
            }
            else if(strstr(tensor->name, "ln2.weight")){
                copy_tensor_data_to_float_array(tensor, &weights->w_ln2[layer_index * config->dim]);
            }
            else if(strstr(tensor->name, "ln2.bias")){
                copy_tensor_data_to_float_array(tensor, &weights->b_ln2[layer_index * config->dim]);
            }
        }

        else {
            fprintf(stderr, "Unrecognized tensor name in GGUF: %s\n", tensor->name);
            // exit(EXIT_FAILURE);
        }
    }
}

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
void clip_vision_init_run_state(CLIP_Vision_RunState* state, CLIP_Vision_Config* config){
    if(!state || !config){
        fprintf(stderr, "NULL pointer passed to clip_vision_init_run_state\n");
        exit(EXIT_FAILURE);
    }

    // Calculate number of patches (including CLS token)
    int num_patches = (config->image_size / config->patch_size) * (config->image_size / config->patch_size);

    // we calloc instead of malloc to keep valgrind happy
    state->x = calloc((1 + num_patches) * config->dim, sizeof(float)); // (num_patches+1, dim)
    state->xb = calloc((1 + num_patches) * config->dim, sizeof(float)); // (num_patches+1, dim)
    state->xb2 = calloc((1 + num_patches) * config->dim, sizeof(float)); // (num_patches+1, dim)
    state->hb = calloc((1 + num_patches) * config->hidden_dim, sizeof(float)); // (num_patches+1, hidden_dim)
    state->hb2 = calloc((1 + num_patches) * config->hidden_dim, sizeof(float)); // (num_patches+1, hidden_dim)
    state->q = calloc((1 + num_patches) * config->dim, sizeof(float)); // (num_patches+1, dim)
    state->k = calloc((1 + num_patches) * config->dim, sizeof(float)); // (num_patches+1, dim)
    state->v = calloc((1 + num_patches) * config->dim, sizeof(float)); // (num_patches+1, dim)
    state->att = calloc(config->n_heads * (1 + num_patches) * (1 + num_patches), sizeof(float)); // (n_heads, num_patches+1, num_patches+1)
    state->f = calloc((1 + num_patches) * config->proj_dim, sizeof(float)); // (num_patches+1, proj_dim) (unused in LLaVA)

    if(!state->x || !state->xb || !state->xb2 || !state->hb || !state->hb2 || 
       !state->q || !state->k || !state->v || !state->att || !state->f){
        fprintf(stderr, "Failed to allocate memory for CLIP_Vision_RunState\n");
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Create and initialize a complete CLIP Vision model from GGUF file
 * 
 * Creates and initializes all components of the CLIP Vision model including
 * configuration, weights, and runtime state buffers.
 * 
 * @param ctx GGUF context containing model data
 * @return Pointer to initialized model (must be freed with clip_vision_delete_model)
 */
CLIP_Vision_Model* clip_vision_create_from_gguf(gguf_context* ctx){
    CLIP_Vision_Model* model = (CLIP_Vision_Model*)calloc(1, sizeof(CLIP_Vision_Model));
    if(!model){
        fprintf(stderr, "Failed to allocate memory for CLIP_Vision_Model\n");
        exit(EXIT_FAILURE);
    }
    clip_vision_load_config_from_gguf(&model->config, ctx);
    clip_vision_load_weights_from_gguf(&model->config, &model->weights, ctx);
    clip_vision_init_run_state(&model->state, &model->config);
    return model;
}

/**
 * @brief Free all memory associated with a CLIP Vision model
 * 
 * Deallocates weights, runtime state buffers, and the model structure itself.
 * 
 * @param model Pointer to model to delete
 */
void clip_vision_delete_model(CLIP_Vision_Model* model){
    if(!model) return;
    
    // Free weights
    if(model->weights.e_cls) free(model->weights.e_cls);
    if(model->weights.e_patch) free(model->weights.e_patch);
    if(model->weights.e_pos) free(model->weights.e_pos);
    if(model->weights.w_pre_ln) free(model->weights.w_pre_ln);
    if(model->weights.b_pre_ln) free(model->weights.b_pre_ln);
    if(model->weights.wq) free(model->weights.wq);
    if(model->weights.bq) free(model->weights.bq);
    if(model->weights.wk) free(model->weights.wk);
    if(model->weights.bk) free(model->weights.bk);
    if(model->weights.wv) free(model->weights.wv);
    if(model->weights.bv) free(model->weights.bv);
    if(model->weights.wo) free(model->weights.wo);
    if(model->weights.bo) free(model->weights.bo);
    if(model->weights.w_fc1) free(model->weights.w_fc1);
    if(model->weights.b_fc1) free(model->weights.b_fc1);
    if(model->weights.w_fc2) free(model->weights.w_fc2);
    if(model->weights.b_fc2) free(model->weights.b_fc2);
    if(model->weights.w_ln1) free(model->weights.w_ln1);
    if(model->weights.b_ln1) free(model->weights.b_ln1);
    if(model->weights.w_ln2) free(model->weights.w_ln2);
    if(model->weights.b_ln2) free(model->weights.b_ln2);
    if(model->weights.w_post_ln) free(model->weights.w_post_ln);
    if(model->weights.b_post_ln) free(model->weights.b_post_ln);
    
    // Free run state
    if(model->state.x) free(model->state.x);
    if(model->state.xb) free(model->state.xb);
    if(model->state.xb2) free(model->state.xb2);
    if(model->state.hb) free(model->state.hb);
    if(model->state.hb2) free(model->state.hb2);
    if(model->state.q) free(model->state.q);
    if(model->state.k) free(model->state.k);
    if(model->state.v) free(model->state.v);
    if(model->state.att) free(model->state.att);
    
    // Free the model structure itself
    free(model);
}

// ============================================================================
// Inference Functions
// ============================================================================

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
void clip_get_image_features(CLIP_Vision_Model* model, float* image, float* output_embedding){
    int i;

    // 1. Image preprocessing (normalization)
    float* mean = model->config.image_mean;
    float* std = model->config.image_std;
    int image_size = model->config.image_size;

    for(i=0; i<image_size*image_size*3; i++){
        int c = i / (image_size * image_size); // channel index
        int x = (i % (image_size * image_size)) / image_size; // x coordinate
        int y = (i % (image_size * image_size)) % image_size; // y coordinate
        float pixel = image[i];
        pixel = (pixel - mean[c]) / std[c];
        image[i] = pixel;
    }
    // 2. Forward
    clip_vision_forward(model, image, output_embedding);

    // 3. Project
    int num_patches = (image_size / model->config.patch_size) * (image_size / model->config.patch_size);
    for(i=0; i < num_patches + 1; i++){
        float* out = model->state.f + i * model->config.proj_dim; // (proj_dim,) output embedding for this patch
        float* in = model->state.x + i * model->config.dim; // (dim,) input embedding for this patch
        float* w = model->weights.w_proj; // (dim, proj_dim) projection matrix
        matmul(out, in, w, model->config.dim, model->config.proj_dim);
    }
}
/**
 * @brief Extract image features from an image file
 * 
 * Loads an image from the specified file path, preprocesses it (resize, pad, normalize),
 * and runs it through the CLIP Vision model to extract features. The image is:
 * 1. Loaded from file using stb_image
 * 2. Resized maintaining aspect ratio using stb_image_resize
 * 3. Padded to square dimensions (centered)
 * 4. Converted from HWC uint8 to CHW float32 layout
 * 5. Normalized to [0, 1] range
 * 6. Processed through the vision transformer
 * 
 * @param model Pointer to the CLIP Vision model
 * @param image_path Path to the input image file
 * @param output_embedding Output buffer for features (size: (num_patches+1) * proj_dim)
 */
void clip_get_image_features_from_file(CLIP_Vision_Model* model, const char* image_path, float* output_embedding){
    // Load image from file
    int src_img_x, src_img_y, src_img_n;
    unsigned char *data = stbi_load(image_path, &src_img_x, &src_img_y, &src_img_n, 3);
    if (!data) {
        fprintf(stderr, "Failed to load image: %s\n", image_path);
        exit(EXIT_FAILURE);
    }
    #ifdef DEBUG
    printf("data:");
    for(int i = 0; i < 20; i++) printf(" %3u", (unsigned int)data[i]);
    printf("\n");
    #endif

    int target_size = model->config.image_size;
    
    // Calculate resize dimensions maintaining aspect ratio
    float aspect_ratio = (float)src_img_x / (float)src_img_y;
    int resized_width, resized_height;
    
    if (src_img_x > src_img_y) {
        resized_width = target_size;
        resized_height = (int)(target_size / aspect_ratio);
    } else {
        resized_height = target_size;
        resized_width = (int)(target_size * aspect_ratio);
    }
    
    // Allocate buffer for resized image
    unsigned char* resized_data = (unsigned char*)malloc(resized_width * resized_height * 3);
    if (!resized_data) {
        fprintf(stderr, "Failed to allocate memory for resized image\n");
        stbi_image_free(data);
        exit(EXIT_FAILURE);
    }
    
    // Resize image using stb_image_resize
    if (!stbir_resize_uint8_srgb(data, src_img_x, src_img_y, 0,
                                resized_data, resized_width, resized_height, 0, 3)) {
        fprintf(stderr, "Failed to resize image\n");
        free(resized_data);
        stbi_image_free(data);
        exit(EXIT_FAILURE);
    }
    #ifdef DEBUG
    printf("resized_data:");
    for(int i = 0; i < 20; i++) printf(" %3u", (unsigned int)resized_data[i]);
    printf("\n");
    #endif

    // Create padded square image and convert to float with normalization
    float* pixel_values = (float*)calloc(target_size * target_size * 3, sizeof(float));
    if (!pixel_values) {
        fprintf(stderr, "Failed to allocate memory for pixel values\n");
        free(resized_data);
        stbi_image_free(data);
        exit(EXIT_FAILURE);
    }
    
    // Calculate padding offsets to center the image
    int offset_x = (target_size - resized_width) / 2;
    int offset_y = (target_size - resized_height) / 2;
    
    // Copy resized image to padded buffer and normalize
    // Convert from HWC (uint8) to CHW (float32) layout
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < resized_height; h++) {
            for (int w = 0; w < resized_width; w++) {
                int src_idx = (h * resized_width + w) * 3 + c;
                int dst_idx = c * target_size * target_size + (h + offset_y) * target_size + (w + offset_x);
                
                // Convert to [0, 1] and normalize
                float pixel = resized_data[src_idx] / 255.0f;
                pixel_values[dst_idx] = pixel;
            }
        }
    }
    free(resized_data);

    #ifdef DEBUG
    printf("Pixel value:");
    for(int i = 0; i < 20; i++) printf(" %7.2f", pixel_values[i]);
    printf("\n");
    #endif

    // Process the image through the model
    clip_get_image_features(model, pixel_values, output_embedding);
    
    // Cleanup
    free(pixel_values);
    stbi_image_free(data);
}

/**
 * @brief Compute patch embeddings and apply position encodings
 * 
 * Performs the embedding layer operations:
 * 1. Extracts image patches from the input pixel array
 * 2. Applies patch embedding (linear projection) to each patch
 * 3. Prepends the learnable CLS token embedding
 * 4. Adds positional embeddings to all tokens (CLS + patches)
 * 
 * The output is stored in model->state.x with shape (num_patches+1, dim),
 * where the first token (index 0) is the CLS token.
 * 
 * @param model Pointer to the CLIP Vision model
 * @param pixel_values Input image pixel values (size: image_size * image_size * 3)
 */
void clip_vision_embedding_forward(CLIP_Vision_Model* model, float* pixel_values){
    int patch_size = model->config.patch_size;
    int image_size = model->config.image_size;
    int dim = model->config.dim;
    int num_patches_per_dim = image_size / patch_size; // patches per dimension
    int num_patches = num_patches_per_dim * num_patches_per_dim;
    
    // 1. Patch embedding - equivalent to Conv2d with kernel_size=patch_size, stride=patch_size
    // Extract patches and apply linear transformation
    float* patch_embedding_weights = model->weights.e_patch; // (dim, 3, patch_size, patch_size)
    #ifdef DEBUG
    printf("Patch embedding weights:");
    for(int i = 0; i < 5; i++) printf(" %f", patch_embedding_weights[i]);
    printf(" ... ");
    for(int i = patch_size*patch_size*3*dim-5; i < patch_size*patch_size*3*dim; i++) printf(" %f", patch_embedding_weights[i]);
    printf("\n");
    #endif

    int patch_idx;
    #pragma omp parallel for private(patch_idx)
    for(patch_idx = 0; patch_idx < num_patches; patch_idx++){
        int patch_y = patch_idx / num_patches_per_dim;
        int patch_x = patch_idx % num_patches_per_dim;
        float* output_embedding = &model->state.x[(1 + patch_idx) * dim]; // (dim,) output embedding for this patch
        
        // Initialize output embedding to zero
        for(int d = 0; d < dim; d++){
            output_embedding[d] = 0.0f;
        }
        
        // Apply Conv2d: for each dimension, accumulate weighted pixel values
        // Weight layout: (dim, 3, patch_size, patch_size)
        for(int d = 0; d < dim; d++){
            for(int c = 0; c < 3; c++){
                for(int ph = 0; ph < patch_size; ph++){
                    for(int pw = 0; pw < patch_size; pw++){
                        int img_h = patch_y * patch_size + ph;
                        int img_w = patch_x * patch_size + pw;
                        
                        float pixel_val = pixel_values[(c * image_size * image_size) + (img_h * image_size) + img_w];
                        // Weight index for layout (dim, 3, patch_size, patch_size)
                        int weight_idx = d * (3 * patch_size * patch_size) + c * (patch_size * patch_size) + ph * patch_size + pw;
                        
                        output_embedding[d] += pixel_val * patch_embedding_weights[weight_idx];
                    }
                }
            }
        }
    }
    
    // 2. Concatenate [CLS] token at position 0
    memcpy(&model->state.x[0], model->weights.e_cls, sizeof(float) * dim);
    
    // 3. Add positional embedding (without interpolation for now)
    // Add position embeddings to all tokens (CLS + patches)
    float* pos_embeddings = model->weights.e_pos; // (num_patches+1, dim)
    for(int pos = 0; pos < num_patches + 1; pos++){
        for(int d = 0; d < dim; d++){
            model->state.x[pos * dim + d] += pos_embeddings[pos * dim + d];
        }
    }
}

/**
 * @brief Run forward pass through a single CLIP Vision encoder layer
 * 
 * Implements a standard transformer encoder layer with:
 * 1. Multi-head self-attention with residual connection:
 *    - LayerNorm1 (pre-normalization)
 *    - Self-attention (Q, K, V projections and attention computation)
 *    - Output projection
 *    - Residual addition
 * 2. Feed-forward network with residual connection:
 *    - LayerNorm2 (pre-normalization)
 *    - FC1: Linear projection to hidden_dim with GELU activation
 *    - FC2: Linear projection back to dim
 *    - Residual addition
 * 
 * The input is read from model->state.x and the output is written back to
 * model->state.x. Intermediate buffers (xb, xb2, q, k, v, att, hb, hb2) are
 * used for computations.
 * 
 * @param model Pointer to the CLIP Vision model
 * @param layer_idx Index of the encoder layer to execute (0 to n_layers-1)
 */
void clip_vision_encoder_layer_forward(CLIP_Vision_Model* model, int layer_idx){
    int num_patches =(model->config.image_size / model->config.patch_size) * (model->config.image_size / model->config.patch_size);
    
    #ifdef DEBUG
    if(true){
        int log_idx = 1;
        printf("Layer %d input x[%d]:", layer_idx, log_idx);
        for(int i = 0; i<5; i++) printf(" %f", model->state.x[log_idx * model->config.dim + i]);
        printf(" ... ");
        for(int i = model->config.dim-5; i<model->config.dim; i++) printf(" %f", model->state.x[log_idx * model->config.dim + i]);
        printf("\n");
    }
    #endif

    // 1. layernorm
    for(int i=0; i < num_patches + 1; i++){
        float* in = &model->state.x[i * model->config.dim];
        float* out = &model->state.xb[i * model->config.dim];
        float* w_ln = &model->weights.w_ln1[layer_idx * model->config.dim];
        float* b_ln = &model->weights.b_ln1[layer_idx * model->config.dim];
        layernorm(out, in, w_ln, b_ln, model->config.dim, model->config.layer_norm_epsilon);
    }
    #ifdef DEBUG
    if(true){
        int log_idx = 1;
        printf("After LN1[%d]:", log_idx);
        for(int i = 0; i<5; i++) printf(" %f", model->state.xb[log_idx * model->config.dim + i]);
        printf(" ... ");
        for(int i = model->config.dim-5; i<model->config.dim; i++) printf(" %f", model->state.xb[log_idx * model->config.dim + i]);
        printf("\n");
    }
    #endif

    // 2. Self-attention
    //     2.1 Project input to q, k, v
    for(int i=0; i < num_patches + 1; i++){
        float* in = &model->state.xb[i * model->config.dim];
        float* q = &model->state.q[i * model->config.dim];
        float* k = &model->state.k[i * model->config.dim];
        float* v = &model->state.v[i * model->config.dim];
        float* wq = &model->weights.wq[layer_idx * model->config.dim * model->config.dim];
        float* bq = &model->weights.bq[layer_idx * model->config.dim];
        float* wk = &model->weights.wk[layer_idx * model->config.dim * model->config.dim];
        float* bk = &model->weights.bk[layer_idx * model->config.dim];
        float* wv = &model->weights.wv[layer_idx * model->config.dim * model->config.dim];
        float* bv = &model->weights.bv[layer_idx * model->config.dim];
        // Compute q, k, v
        linear(q, in, wq, bq, model->config.dim, model->config.dim);
        linear(k, in, wk, bk, model->config.dim, model->config.dim);
        linear(v, in, wv, bv, model->config.dim, model->config.dim);
    }    
    #ifdef DEBUG // OK
    if(true){
        int log_idx = 1;
        printf("Q[%d]:", log_idx);
        for(int i = 0; i<5; i++) printf(" %f", model->state.q[log_idx * model->config.dim + i]);
        printf(" ... ");
        for(int i = model->config.dim-5; i<model->config.dim; i++) printf(" %f", model->state.q[log_idx * model->config.dim + i]);
        printf("\nK[%d]:", log_idx);
        for(int i = 0; i<5; i++) printf(" %f", model->state.k[log_idx * model->config.dim + i]);
        printf(" ... ");
        for(int i = model->config.dim-5; i<model->config.dim; i++) printf(" %f", model->state.k[log_idx * model->config.dim + i]);
        printf("\nV[%d]:", log_idx);
        for(int i = 0; i<5; i++) printf(" %f", model->state.v[log_idx * model->config.dim + i]);
        printf(" ... ");
        for(int i = model->config.dim-5; i<model->config.dim; i++) printf(" %f", model->state.v[log_idx * model->config.dim + i]);
        printf("\n");
    }
    #endif

    //     2.2 Compute attention scores and apply softmax
    int head_dim = model->config.dim / model->config.n_heads;
    float scale = 1.0f / sqrtf((float)head_dim);
    
    for(int head = 0; head < model->config.n_heads; head++){
        float* att_scores = &model->state.att[head * (num_patches + 1) * (num_patches + 1)]; // (num_patches+1, num_patches+1)
        
        // Compute scaled dot-product attention scores for this head
        for(int i = 0; i < num_patches + 1; i++){
            for(int j = 0; j < num_patches + 1; j++){
                float score = 0.0f;
                float* q_vec = &model->state.q[i * model->config.dim + head * head_dim]; // (head_dim,) query vector for token i and this head
                float* k_vec = &model->state.k[j * model->config.dim + head * head_dim]; // (head_dim,) key vector for token j and this head
                for(int d = 0; d < head_dim; d++){
                    float q_val = q_vec[d];
                    float k_val = k_vec[d];
                    score += q_val * k_val;
                }
                score = score * scale; // scale the score
                att_scores[i * (num_patches + 1) + j] = score;
            }
            // Apply softmax to each row of attention scores
            softmax_inplace(&att_scores[i * (num_patches + 1)], num_patches + 1);
        }
    }
    #ifdef DEBUG
    if(true){
        int head_idx = 0;
        printf("Attention Scores (head %d):",head_idx);
        float* att = &model->state.att[head_idx * (num_patches + 1) * (num_patches + 1)];
        for(int i = 0; i<5; i++) printf(" %f", att[i]);
        printf("...");
        for(int i =(num_patches + 1) * (num_patches + 1) - 5; i < (num_patches + 1) * (num_patches + 1); i++) printf(" %f", att[i]);
        printf("\n");
    }
    #endif
    
    //     2.3 Compute attention output by weighted sum of v
    // Zero out xb2 to accumulate attention outputs
    memset(model->state.xb2, 0, (num_patches + 1) * model->config.dim * sizeof(float));
    
    for(int head = 0; head < model->config.n_heads; head++){
        float* att_scores = &model->state.att[head * (num_patches + 1) * (num_patches + 1)];
        
        for(int i = 0; i < num_patches + 1; i++){
            float* output = &model->state.xb2[i * model->config.dim + head * head_dim];
            
            for(int j = 0; j < num_patches + 1; j++){
                float att_weight = att_scores[i * (num_patches + 1) + j];
                float* v_vec = &model->state.v[j * model->config.dim + head * head_dim]; // (head_dim,) value vector for token j and this head
                
                for(int d = 0; d < head_dim; d++){
                    output[d] += att_weight * v_vec[d];
                }
            }
        }
    }
    #ifdef DEBUG
    if(true){
        int log_idx = 1;
        printf("Attention Output[%d]:", log_idx);
        for(int i = 0; i<5; i++) printf(" %f", model->state.xb2[log_idx * model->config.dim + i]);
        printf(" ... ");
        for(int i = model->config.dim-5; i<model->config.dim; i++) printf(" %f", model->state.xb2[log_idx * model->config.dim + i]);
        printf("\n");
    }
    #endif
    
    //     2.4 Final linear projection for attention output
    for(int i = 0; i < num_patches + 1; i++){
        float* in = &model->state.xb2[i * model->config.dim];
        float* out = &model->state.xb[i * model->config.dim];
        float* wo = &model->weights.wo[layer_idx * model->config.dim * model->config.dim];
        float* bo = &model->weights.bo[layer_idx * model->config.dim];
        linear(out, in, wo, bo, model->config.dim, model->config.dim);
    }
    #ifdef DEBUG
    if(true){
        int log_idx = 1;
        printf("After Projection[%d]:", log_idx);
        for(int i = 0; i<5; i++) printf(" %f", model->state.xb[log_idx * model->config.dim + i]);
        printf(" ... ");
        for(int i = model->config.dim-5; i<model->config.dim; i++) printf(" %f", model->state.xb[log_idx * model->config.dim + i]);
        printf("\n");
    }
    #endif

    // 3. Residual connection
    for(int i=0; i < num_patches + 1; i++){
        for(int d=0; d < model->config.dim; d++){
            float* in1 = &model->state.x[i * model->config.dim + d];
            float* in2 = &model->state.xb[i * model->config.dim + d];
            float* out = &model->state.x[i * model->config.dim + d];
            out[0] = in1[0] + in2[0];
        }
    }
    #ifdef DEBUG
    if(true){
        int log_idx = 1;
        printf("After Residual1[%d]:", log_idx);
        for(int i = 0; i<5; i++) printf(" %f", model->state.x[log_idx * model->config.dim + i]);
        printf(" ... ");
        for(int i = model->config.dim-5; i<model->config.dim; i++) printf(" %f", model->state.x[log_idx * model->config.dim + i]);
        printf("\n");
    }
    #endif
    // 4. layernorm
    for(int i=0; i < num_patches + 1; i++){
        float* in = &model->state.x[i * model->config.dim];
        float* out = &model->state.xb[i * model->config.dim];
        float* w_ln = &model->weights.w_ln2[layer_idx * model->config.dim];
        float* b_ln = &model->weights.b_ln2[layer_idx * model->config.dim];
        layernorm(out, in, w_ln, b_ln, model->config.dim, model->config.layer_norm_epsilon);
    }
    #ifdef DEBUG
    if(true){
        int log_idx = 1;
        printf("After LN2[%d]:", log_idx);
        for(int i = 0; i<5; i++) printf(" %f", model->state.xb[log_idx * model->config.dim + i]);
        printf(" ... ");
        for(int i = model->config.dim-5; i<model->config.dim; i++) printf(" %f", model->state.xb[log_idx * model->config.dim + i]);
        printf("\n");
    }
    #endif

    // 5. Feedforward
    for(int i=0; i < num_patches + 1; i++){
        float* x = &model->state.xb[i * model->config.dim];
        float* hb = &model->state.hb[i * model->config.hidden_dim];

        float* w_fc1 = &model->weights.w_fc1[layer_idx * model->config.dim * model->config.hidden_dim];
        float* b_fc1 = &model->weights.b_fc1[layer_idx * model->config.hidden_dim];
        float* w_fc2 = &model->weights.w_fc2[layer_idx * model->config.hidden_dim * model->config.dim];
        float* b_fc2 = &model->weights.b_fc2[layer_idx * model->config.dim];
        // 5.1 First linear layer
        linear(hb, x, w_fc1, b_fc1, model->config.dim, model->config.hidden_dim);
        #ifdef DEBUG
        int log_idx = 1;
        if((true) && i == log_idx){
            printf("After FC1[%d]:", log_idx);
            for(int i = 0; i<5; i++) printf(" %f", model->state.hb[log_idx * model->config.hidden_dim + i]);
            printf(" ... ");
            for(int i = model->config.hidden_dim-5; i<model->config.hidden_dim; i++) printf(" %f", model->state.hb[log_idx * model->config.hidden_dim + i]);
            printf("\n");
        }
        #endif

        // 5.2 Activation (GELU)
        quick_gelu_inplace(hb, model->config.hidden_dim);
        #ifdef DEBUG
        if((true) && i == log_idx){
            printf("After GELU[%d]:", log_idx);
            for(int i = 0; i<5; i++) printf(" %f", model->state.hb[log_idx * model->config.hidden_dim + i]);
            printf(" ... ");
            for(int i = model->config.hidden_dim-5; i<model->config.hidden_dim; i++) printf(" %f", model->state.hb[log_idx * model->config.hidden_dim + i]);
            printf("\n");
        }
        #endif

        // 5.3 Second linear layer
        linear(x, hb, w_fc2, b_fc2, model->config.hidden_dim, model->config.dim);
    }
    #ifdef DEBUG
    if(true){
        int log_idx = 1;
        printf("After FFN[%d]:", log_idx);
        for(int i = 0; i<5; i++) printf(" %f", model->state.xb[log_idx * model->config.dim + i]);
        printf(" ... ");
        for(int i = model->config.dim-5; i<model->config.dim; i++) printf(" %f", model->state.xb[log_idx * model->config.dim + i]);
        printf("\n");
    }
    #endif

    // 6. Residual connection
    for(int i=0; i < num_patches + 1; i++){
        for(int d=0; d < model->config.dim; d++){
            float* in1 = &model->state.x[i * model->config.dim + d];
            float* in2 = &model->state.xb[i * model->config.dim + d];
            float* out = &model->state.x[i * model->config.dim + d];
            out[0] = in1[0] + in2[0];
        }
    }
    #ifdef DEBUG
    if(true){
        int log_idx = 1;
        printf("After Residual2[%d]:", log_idx);
        for(int i = 0; i<5; i++) printf(" %f", model->state.x[log_idx * model->config.dim + i]);
        printf(" ... ");
        for(int i = model->config.dim-5; i<model->config.dim; i++) printf(" %f", model->state.x[log_idx * model->config.dim + i]);
        printf("\n");
    }
    #endif

}

/**
 * @brief Run complete forward pass through the CLIP Vision transformer
 * 
 * Executes the full vision transformer pipeline:
 * 1. Embedding layer: Converts image patches to embeddings with position encodings
 * 2. Pre-LayerNorm: Normalizes embeddings before encoder
 * 3. Transformer encoder: Sequentially processes through all n_layers encoder layers
 * 4. Post-LayerNorm: Normalizes the CLS token output
 * 
 * The final output embeddings are stored in model->state.x with shape (num_patches+1, dim),
 * where index 0 contains the CLS token and indices 1+ contain patch tokens.
 * The CLS token after post-normalization represents the global image features.
 * 
 * @param model Pointer to the CLIP Vision model
 * @param patches Input image pixel values (size: image_size * image_size * 3)
 * @param output_embedding Output buffer for final embeddings (currently unused, output in model->state.x)
 */
void clip_vision_forward(CLIP_Vision_Model* model, float* patches, float* output_embedding){
    int num_patches =(model->config.image_size / model->config.patch_size) * (model->config.image_size / model->config.patch_size);
    // 1. Embedding
    clip_vision_embedding_forward(model, patches);
    #ifdef DEBUG // OK
    int log_idx = 1;
    printf("After Embedding[%d]:", log_idx);
    for(int i = 0; i<5; i++) printf(" %f", model->state.x[log_idx * model->config.dim + i]);
    printf(" ... ");
    for(int i = model->config.dim-5; i<model->config.dim; i++) printf(" %f", model->state.x[log_idx * model->config.dim + i]);
    printf("\n");
    #endif

    // 2. Pre-layernorm
    for(int i=0; i < num_patches + 1; i++){
        float* in = &model->state.x[i * model->config.dim];
        float* out = &model->state.x[i * model->config.dim];
        float* w_ln = model->weights.w_pre_ln;
        float* b_ln = model->weights.b_pre_ln;
        layernorm(out, in, w_ln, b_ln, model->config.dim, model->config.layer_norm_epsilon);
    }
    #ifdef DEBUG // OK
    printf("After Pre-LN[%d]:", log_idx);
    for(int i = 0; i<5; i++) printf(" %f", model->state.x[log_idx * model->config.dim + i]);
    printf(" ... ");
    for(int i = model->config.dim-5; i<model->config.dim; i++) printf(" %f", model->state.x[log_idx * model->config.dim + i]);
    printf("\n");
    #endif

    // 3. Encoder
    for(int i=0; i<model->config.n_layers; i++){
        #ifdef DEBUG
        printf("Entering Encoder Layer %d\n", i);
        #endif
        clip_vision_encoder_layer_forward(model, i);
    }
    #ifdef DEBUG
    printf("After Encoder[%d]:", log_idx);
    for(int i = 0; i<5; i++) printf(" %f", model->state.x[log_idx * model->config.dim + i]);
    printf(" ... ");
    for(int i = model->config.dim-5; i<model->config.dim; i++) printf(" %f", model->state.x[log_idx * model->config.dim + i]);
    printf("\n");
    #endif

    // 4. Post-layernorm
    for(int i=0; i < num_patches + 1; i++){
        float* in = &model->state.x[i * model->config.dim];
        float* out = &model->state.x[i * model->config.dim];
        float* w_ln = model->weights.w_post_ln;
        float* b_ln = model->weights.b_post_ln;
        layernorm(out, in, w_ln, b_ln, model->config.dim, model->config.layer_norm_epsilon);
    }
    #ifdef DEBUG
    printf("After Post-LN[%d]:", log_idx);
    for(int i = 0; i<5; i++) printf(" %f", model->state.x[log_idx * model->config.dim + i]);
    printf(" ... ");
    for(int i = model->config.dim-5; i<model->config.dim; i++) printf(" %f", model->state.x[log_idx * model->config.dim + i]);
    printf("\n");
    #endif
}
