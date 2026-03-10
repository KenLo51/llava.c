#include "llava_phi3.h"


void llava_phi3_load_projector_weights_from_gguf(LLaVA_Phi3* llava, gguf_context* vision_ctx){
    int vision_dim = llava->vidion_tower->config.dim;
    int llm_dim = llava->model->config.dim;

    // Malloc memory for projector weights
    llava->weight.w1 = (float*)malloc(sizeof(float) * vision_dim * llm_dim);
    llava->weight.b1 = (float*)malloc(sizeof(float) * llm_dim);
    llava->weight.w2 = (float*)malloc(sizeof(float) * llm_dim * llm_dim);
    llava->weight.b2 = (float*)malloc(sizeof(float) * llm_dim);
    if (!llava->weight.w1 || !llava->weight.b1 || !llava->weight.w2 || !llava->weight.b2) {
        fprintf(stderr, "Failed to allocate memory for projector weights\n");
        exit(EXIT_FAILURE);
    }

    // Iterate over tensors in GGUF and load weights
    for(unsigned int i=0; i<vision_ctx->tensor_count; i++){
        gguf_tensor* tensor = &vision_ctx->tensors[i];

        #ifdef DEBUG
        printf("Processing tensor: %s\n", tensor->name);
        #endif

        // tensor names are "mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias"
        if(strcmp(tensor->name, "mm.0.weight") == 0){
            copy_tensor_data_to_float_array(tensor, llava->weight.w1);
        }
        else if(strcmp(tensor->name, "mm.0.bias") == 0){
            copy_tensor_data_to_float_array(tensor, llava->weight.b1);
        }
        else if(strcmp(tensor->name, "mm.2.weight") == 0){
            copy_tensor_data_to_float_array(tensor, llava->weight.w2);
        }
        else if(strcmp(tensor->name, "mm.2.bias") == 0){
            copy_tensor_data_to_float_array(tensor, llava->weight.b2);
        }
    }
}

void llava_phi3_init_run_state(LLaVA_Phi3* llava){
    int llm_dim = llava->model->config.dim;
    int n_patches = (llava->vidion_tower->config.image_size / llava->vidion_tower->config.patch_size) * 
                    (llava->vidion_tower->config.image_size / llava->vidion_tower->config.patch_size);

    llava->state.x = (float*)malloc(sizeof(float) * n_patches * llm_dim);
    if (!llava->state.x) {
        fprintf(stderr, "Failed to allocate memory for projector run state\n");
        exit(EXIT_FAILURE);
    }
}

LLaVA_Phi3* llava_phi3_create_from_gguf(gguf_context* llm_ctx, gguf_context* vision_ctx){
    LLaVA_Phi3* llava = (LLaVA_Phi3*)malloc(sizeof(LLaVA_Phi3));
    if (!llava) {
        fprintf(stderr, "Failed to allocate memory for LLaVA_Phi3\n");
        return NULL;
    }

    llava->config.proj_layers = 2; // currently only support 2-layer MLP projector
    
    // Load PHI-3 model from LLM GGUF context
    llava->model = phi3_init_from_gguf(llm_ctx);
    
    // Load CLIP vision tower from vision GGUF context
    llava->vidion_tower = clip_vision_create_from_gguf(vision_ctx);
    
    // Load projector weights from LLM GGUF context
    llava_phi3_load_projector_weights_from_gguf(llava, vision_ctx);

    // Initialize runtime state buffers for projector
    llava_phi3_init_run_state(llava);

    return llava;
}

void llava_phi3_delete(LLaVA_Phi3* llava){
    if (!llava) return;

    // Free projector weights
    free(llava->weight.w1);
    llava->weight.w1 = NULL;
    free(llava->weight.b1);
    llava->weight.b1 = NULL;
    free(llava->weight.w2);
    llava->weight.w2 = NULL;
    free(llava->weight.b2);
    llava->weight.b2 = NULL;

    // Free projector run state
    free(llava->state.x);
    llava->state.x = NULL;

    // Free vision tower and PHI-3 model states
    clip_vision_delete(llava->vidion_tower);
    phi3_delete(llava->model);

    // Finally, free the LLaVA_Phi3 struct itself
    free(llava);
}


void llava_phi3_projector_forward(LLaVA_Phi3* llava, float* image_embedding, float* output){
    // image_embedding: (n_patches, vision_dim)
    // output: (n_patches, llm_dim)

    int vision_dim = llava->vidion_tower->config.dim;
    int llm_dim = llava->model->config.dim;
    int n_patches = (llava->vidion_tower->config.image_size / llava->vidion_tower->config.patch_size) * 
                    (llava->vidion_tower->config.image_size / llava->vidion_tower->config.patch_size);
    if(!image_embedding){
        // default to using the hidden state of the vision tower as the image embedding if not provided
        image_embedding = llava->vidion_tower->state.x + vision_dim; // skip cls token 
    }

    // First layer
    for(int i=0; i<n_patches; i++){
        float* in= image_embedding + i * vision_dim;
        float* out = llava->state.x + i * llm_dim;
        float* w = llava->weight.w1;
        float* b = llava->weight.b1;
        linear(out, in, w, b, vision_dim, llm_dim);
    }

    // Apply activation function (GELU)
    gelu_inplace(llava->state.x, n_patches * llm_dim);

    // Second layer
    if(output == NULL) output = llava->state.x; // inplace if output buffer not provided
    for(int i=0; i<n_patches; i++){
        float* in= llava->state.x + i * llm_dim;
        float* out = output + i * llm_dim;
        float* w = llava->weight.w2;
        float* b = llava->weight.b2;
        linear(out, in, w, b, llm_dim, llm_dim);
    }

}

// Shared tokenization helper: splits prompt on <image>, inserts image_token_id (-200),
// populates prompt_tokens and num_prompt_tokens. Returns the image token position (-1 if none).
static int _llava_tokenize_prompt(LLaVA_Phi3 *model, Tokenizer *tokenizer, char *prompt,
                                  bool has_image, int *prompt_tokens, int *num_prompt_tokens) {
    const char* image_placeholder = "<image>";
    int image_token_id = -200;
    int placeholder_len = strlen(image_placeholder);
    int max_prompt_tokens = model->model->config.seq_len;
    *num_prompt_tokens = 0;

    char* image_pos = strstr(prompt, image_placeholder);

    if (image_pos == NULL) {
        encode(tokenizer, prompt, prompt_tokens, num_prompt_tokens);
    } else {
        int pre_len = image_pos - prompt;
        if (pre_len > 0) {
            char* pre_text = (char*)malloc((pre_len + 1) * sizeof(char));
            strncpy(pre_text, prompt, pre_len);
            pre_text[pre_len] = '\0';
            int pre_count = 0;
            int* pre_tokens = (int*)malloc(max_prompt_tokens * sizeof(int));
            encode(tokenizer, pre_text, pre_tokens, &pre_count);
            memcpy(prompt_tokens, pre_tokens, pre_count * sizeof(int));
            *num_prompt_tokens = pre_count;
            free(pre_text);
            free(pre_tokens);
        }

        if (has_image)
            prompt_tokens[(*num_prompt_tokens)++] = image_token_id;

        char* post_text = image_pos + placeholder_len;
        if (strlen(post_text) > 0) {
            int post_count = 0;
            int* post_tokens = (int*)malloc(max_prompt_tokens * sizeof(int));
            encode(tokenizer, post_text, post_tokens, &post_count);
            memcpy(prompt_tokens + *num_prompt_tokens, post_tokens, post_count * sizeof(int));
            *num_prompt_tokens += post_count;
            free(post_tokens);
        }
    }

    int image_token_id_val = -200;
    for (int i = 0; i < *num_prompt_tokens; i++)
        if (prompt_tokens[i] == image_token_id_val) return i;
    return -1;
}

// Shared generation loop: runs the LLM decode loop and delivers tokens via callback.
static void _llava_generation_loop(LLaVA_Phi3 *model, Tokenizer *tokenizer, Sampler *sampler,
                                   int *prompt_tokens, int num_prompt_tokens,
                                   int image_token_pos, float *image_embeddings,
                                   int max_tokens_gen, void (*callback)(const char*, size_t),
                                   char **out_buf, size_t *out_len, size_t *out_cap) {
    int next;
    int* generated_tokens = (int*)malloc(5 * sizeof(int));
    int generated_tokens_count = 0;
    int token = prompt_tokens[0];
    int pos = 0, pos_img = 0, pos_txt = 0;
    int num_image_tokens = model->vidion_tower->config.image_size / model->vidion_tower->config.patch_size *
                           model->vidion_tower->config.image_size / model->vidion_tower->config.patch_size;
    float* logits = model->model->state.logits;

    while (pos < max_tokens_gen) {
        bool is_image_step = (pos >= image_token_pos) && (pos_img < num_image_tokens) && (image_token_pos != -1);

        if (is_image_step) {
            float* image_embed = image_embeddings + pos_img * model->model->config.dim;
            // logits = phi3_forward_embed(model->model, image_embed, pos);
        } else {
            logits = phi3_forward(model->model, token, pos);
        }

        if (pos_txt >= num_prompt_tokens - 1)
            next = sample(sampler, logits);
        else
            next = prompt_tokens[pos_txt + 1];

        if (pos_img + 1 == num_image_tokens) pos_txt++;
        if (is_image_step) pos_img++;
        else pos_txt++;
        pos++;

        if (next == 2 || next == 32000 || next == 32007) break;

        if (pos_txt >= num_prompt_tokens) {
            generated_tokens[generated_tokens_count++] = next;
            int num_decode_tokens = 0;
            char* piece = decode(tokenizer, &next, generated_tokens_count, &num_decode_tokens);
            if (num_decode_tokens > 0) {
                if (callback) {
                    callback(piece, strlen(piece));
                } else if (out_buf) {
                    size_t piece_len = strlen(piece);
                    if (*out_len + piece_len + 1 > *out_cap) {
                        *out_cap = (*out_len + piece_len + 1) * 2;
                        *out_buf = (char*)realloc(*out_buf, *out_cap);
                    }
                    memcpy(*out_buf + *out_len, piece, piece_len);
                    *out_len += piece_len;
                    (*out_buf)[*out_len] = '\0';
                }
                generated_tokens_count = 0;
            }
            free(piece);
        }
        token = next;
    }
    free(generated_tokens);
}

char* llava_phi3_generate(LLaVA_Phi3 *model, Tokenizer *tokenizer, Sampler *sampler, char *prompt, float* image, int max_tokens_gen){
    // 1. forward pass through vision tower to get image embedding
    float *image_embeddings = NULL;
    if (image) {
        clip_vision_forward_early_exit(model->vidion_tower, image, NULL, model->vidion_tower->config.n_layers - 2);
        clip_vision_forward(model->vidion_tower, image, NULL);
        llava_phi3_projector_forward(model, NULL, NULL);
        image_embeddings = model->state.x;
    }

    // 2. Tokenize prompt
    char *empty_prompt = "";
    if (prompt == NULL) prompt = empty_prompt;
    int max_prompt_tokens = model->model->config.seq_len;
    int* prompt_tokens = (int*)malloc(max_prompt_tokens * sizeof(int));
    int num_prompt_tokens = 0;
    int image_token_pos = _llava_tokenize_prompt(model, tokenizer, prompt, image != NULL,
                                                 prompt_tokens, &num_prompt_tokens);

    // 3. Decode and generation loop — accumulate into returned string
    size_t out_cap = 1024, out_len = 0;
    char* output = (char*)malloc(out_cap);
    output[0] = '\0';
    _llava_generation_loop(model, tokenizer, sampler, prompt_tokens, num_prompt_tokens,
                           image_token_pos, image_embeddings, max_tokens_gen,
                           NULL, &output, &out_len, &out_cap);

    free(prompt_tokens);
    return output;
}

void llava_phi3_generate_stream(LLaVA_Phi3 *model, Tokenizer *tokenizer, Sampler *sampler, char *prompt, float* image, int max_tokens_gen, void (*callback)(const char*, size_t)){
    // 1. forward pass through vision tower to get image embedding
    float *image_embeddings = NULL;
    if (image) {
        clip_vision_forward_early_exit(model->vidion_tower, image, NULL, model->vidion_tower->config.n_layers - 2);
        clip_vision_forward(model->vidion_tower, image, NULL);
        llava_phi3_projector_forward(model, NULL, NULL);
        image_embeddings = model->state.x;
    }

    // 2. Tokenize prompt
    char *empty_prompt = "";
    if (prompt == NULL) prompt = empty_prompt;
    int max_prompt_tokens = model->model->config.seq_len;
    int* prompt_tokens = (int*)malloc(max_prompt_tokens * sizeof(int));
    int num_prompt_tokens = 0;
    int image_token_pos = _llava_tokenize_prompt(model, tokenizer, prompt, image != NULL,
                                                 prompt_tokens, &num_prompt_tokens);

    // 3. Decode and generation loop — stream tokens via callback
    _llava_generation_loop(model, tokenizer, sampler, prompt_tokens, num_prompt_tokens,
                           image_token_pos, image_embeddings, max_tokens_gen,
                           callback, NULL, NULL, NULL);

    free(prompt_tokens);
}

char* llava_phi3_generate_imgpath(LLaVA_Phi3 *model, Tokenizer *tokenizer, Sampler *sampler, char *prompt, const char* image_path, int max_tokens_gen){
    // 1. image preprocessing + forward pass through vision tower
    float *image_embeddings = NULL;
    float *pixel_values = NULL;
    if (image_path) {
        printf("Encoding image: %s\n", image_path);
        int image_size = model->vidion_tower->config.image_size;
        pixel_values = (float*)calloc(image_size * image_size * 3, sizeof(float));
        // clip_preprocess_image_path(model->vidion_tower, image_path, pixel_values);

        // 2. forward pass through vision tower to get image embedding
        clip_vision_forward_early_exit(model->vidion_tower, pixel_values, NULL, model->vidion_tower->config.n_layers - 2);
        clip_vision_forward(model->vidion_tower, pixel_values, NULL);
        free(pixel_values);

        #ifdef DEBUG
        int n_patches = (model->vidion_tower->config.image_size / model->vidion_tower->config.patch_size) *
                        (model->vidion_tower->config.image_size / model->vidion_tower->config.patch_size);
        int vision_dim = model->vidion_tower->config.dim;
        printf("Image features : ");
        for(int i=0; i<5; i++)
            printf("%f ", model->vidion_tower->state.x[vision_dim + i]);
        printf(" ... ");
        for(int i=vision_dim + (n_patches-1)*vision_dim; i<vision_dim + (n_patches-1)*vision_dim + 5; i++)
            printf("%f ", model->vidion_tower->state.x[i]);
        printf("\n");
        #endif

        llava_phi3_projector_forward(model, NULL, NULL);
        image_embeddings = model->state.x;
    }

    // 3. Tokenize prompt
    char *empty_prompt = "";
    if (prompt == NULL) prompt = empty_prompt;
    int max_prompt_tokens = model->model->config.seq_len;
    int* prompt_tokens = (int*)malloc(max_prompt_tokens * sizeof(int));
    int num_prompt_tokens = 0;
    int image_token_pos = _llava_tokenize_prompt(model, tokenizer, prompt, image_path != NULL,
                                                 prompt_tokens, &num_prompt_tokens);

    #ifdef DEBUG
    printf("Prompt token ids: ");
    for(int i=0; i<num_prompt_tokens; i++) printf("%d ", prompt_tokens[i]);
    printf("\n");
    #endif

    // 4. Decode and generation loop — accumulate into returned string
    size_t out_cap = 1024, out_len = 0;
    char* output = (char*)malloc(out_cap);
    output[0] = '\0';
    _llava_generation_loop(model, tokenizer, sampler, prompt_tokens, num_prompt_tokens,
                           image_token_pos, image_embeddings, max_tokens_gen,
                           NULL, &output, &out_len, &out_cap);

    free(prompt_tokens);
    return output;
}

void llava_phi3_generate_stream_imgpath(LLaVA_Phi3 *model, Tokenizer *tokenizer, Sampler *sampler, char *prompt, const char* image_path, int max_tokens_gen, void (*callback)(const char*, size_t)){
    // 1. image preprocessing
    float *image_embeddings;
    if (image_path) {
        printf("Encoding image: %s\n", image_path);
        int image_size = model->vidion_tower->config.image_size;
        float* pixel_values = (float*)calloc(image_size * image_size * 3, sizeof(float));
        // clip_preprocess_image_path(model->vidion_tower, image_path, pixel_values);

        clip_vision_forward_early_exit(model->vidion_tower, pixel_values, NULL, model->vidion_tower->config.n_layers - 2); // early exit before the last 2 layers of the vision tower to get intermediate image embedding for better performance. The final 2 layers will be done together with the projector to save time.

        // 2. forward pass through vision tower to get image embedding
        clip_vision_forward(model->vidion_tower, pixel_values, NULL);
        
        free(pixel_values);


        #ifdef DEBUG
        int n_patches = (model->vidion_tower->config.image_size / model->vidion_tower->config.patch_size) * 
                        (model->vidion_tower->config.image_size / model->vidion_tower->config.patch_size);
        int vision_dim = model->vidion_tower->config.dim;
        printf("Image features : ");
        for(int i=0; i<5; i++)
            printf("%f ", model->vidion_tower->state.x[vision_dim + i]); // print the first 5 dimensions of the first patch embedding (skip cls token)
        printf(" ... ");
        for(int i=vision_dim + (n_patches-1)*vision_dim; i<vision_dim + (n_patches-1)*vision_dim + 5; i++)
            printf("%f ", model->vidion_tower->state.x[i]); // print the first 5 dimensions of the last patch embedding
        printf("\n");

        #endif

        llava_phi3_projector_forward(model, NULL, NULL); // use default image embedding from vision tower state
        image_embeddings = model->state.x; // (n_patches, llm_dim)
    }
    // 3. Tokenize prompt
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    //     Seperate string by '<image>' into chunks
    const char* image_placeholder = "<image>";
    int image_token_id = -200; // placeholder token id for image, should be a special token in the tokenizer vocab
    int placeholder_len = strlen(image_placeholder);
    
    //     Find the position of '<image>' in the prompt
    char* image_pos = strstr(prompt, image_placeholder);
    
    //     Allocate buffer for prompt tokens (estimate max size)
    int max_prompt_tokens = model->model->config.seq_len; // max sequence length of the model
    int* prompt_tokens = (int*)malloc(max_prompt_tokens * sizeof(int));
    int num_prompt_tokens = 0;
    
    if (image_pos == NULL) {
        // No image placeholder, encode entire prompt
        encode(tokenizer, prompt, prompt_tokens, &num_prompt_tokens);
    } else {
        // Split prompt into pre-image and post-image parts
        int pre_len = image_pos - prompt;
        
        // Encode pre-image text
        if (pre_len > 0) {
            char* pre_text = (char*)malloc((pre_len + 1) * sizeof(char));
            strncpy(pre_text, prompt, pre_len);
            pre_text[pre_len] = '\0';
            
            int pre_tokens_count = 0;
            int* pre_tokens = (int*)malloc(max_prompt_tokens * sizeof(int));
            encode(tokenizer, pre_text, pre_tokens, &pre_tokens_count);
            
            // Copy pre tokens
            memcpy(prompt_tokens, pre_tokens, pre_tokens_count * sizeof(int));
            num_prompt_tokens = pre_tokens_count;
            
            free(pre_text);
            free(pre_tokens);
        }
        
        // Insert image token (-200)
        if(image_path)
            prompt_tokens[num_prompt_tokens++] = image_token_id;
        
        // Encode post-image text
        char* post_text = image_pos + placeholder_len;
        if (strlen(post_text) > 0) {
            int post_tokens_count = 0;
            int* post_tokens = (int*)malloc(max_prompt_tokens * sizeof(int));
            encode(tokenizer, post_text, post_tokens, &post_tokens_count);
            
            // Append post tokens
            memcpy(prompt_tokens + num_prompt_tokens, post_tokens, post_tokens_count * sizeof(int));
            num_prompt_tokens += post_tokens_count;
            
            free(post_tokens);
        }
    }

    //     Find position of image token in prompt
    int image_token_pos = -1;
    for(int i = 0; i < num_prompt_tokens; i++){
        if(prompt_tokens[i] == image_token_id) {
            image_token_pos = i;
            break;
        }
    }

    #ifdef DEBUG
    printf("Prompt token ids: ");
    for(int i=0; i<num_prompt_tokens; i++){
        printf("%d ", prompt_tokens[i]);
    }
    printf("\n");
    #endif

    // 4. Decode and generation loop
    int next;        // will store the next token in the sequence
    int* generated_tokens = (int*)malloc(5 * sizeof(int));
    int generated_tokens_count = 0;
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0; // position in the sequence
    int pos_img = 0, pos_txt = 0; // position counters for image and text embeddings.
    int num_image_tokens = model->vidion_tower->config.image_size / model->vidion_tower->config.patch_size * model->vidion_tower->config.image_size / model->vidion_tower->config.patch_size; // number of tokens occupied by image embedding (excluding cls token)
    float* logits = model->model->state.logits; // buffer to hold logits for next token prediction
    while (pos < max_tokens_gen) {
        bool is_image_step = (pos >= image_token_pos) && (pos_img < num_image_tokens) && (image_token_pos != -1);
        // 1. Forward llm
        if(is_image_step){
            // If we are at the image token position, feed the image embedding for the next num_image_tokens steps
            
            // Get the corresponding image embedding for the current position
            float* image_embed = image_embeddings + pos_img * model->model->config.dim; // (dim,)
            // Forward the LLM with the image embedding as the "next token"
            //logits = phi3_forward_embed(model->model, image_embed, pos);
        }
        else{
            logits = phi3_forward(model->model, token, pos);
        }

        // 2. sample next token
        if(pos_txt >= num_prompt_tokens - 1){
            // Decoding stage, sample from logits and generate tokens
            next = sample(sampler, logits);
        } else {
            // Prefilling stage, use prompt tokens
            next = prompt_tokens[pos_txt + 1];
        }

        if(pos_img+1 == num_image_tokens) pos_txt++; // skip the image token
        if(is_image_step) pos_img++;
        else pos_txt++;
        pos++;

        if(next == 2 || next == 32000 || next == 32007) break;

        // 3. decode

        // print the token as string, decode it with the Tokenizer object
        // printf(" Predicted token %d: ", token);
        if(pos_txt >= num_prompt_tokens){
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
    }

    free(prompt_tokens);
    free(generated_tokens);
}
