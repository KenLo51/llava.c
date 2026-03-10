#ifndef LLAVA_PHI3_H
#define LLAVA_PHI3_H

#include "phi3.h"
#include "clip.h"

typedef struct {
    float* w1; // (in, out) first layer with diff input/output dim. mm.0
    float* b1; // (out)
    float* w2; // (out, out) mm.2. (mm.1 is GELU activation)
    float* b2; // (out)
} LLaVA_Phi3_projector_weight;

typedef struct {
    int proj_layers; // number of linear layers in the projector (currently only support 2-layer MLP)
} LLaVA_Phi3_Config;

typedef struct {
    float* x; // (n_patches, dim) 
} LLaVA_Phi3_RunState;

typedef struct {
    LLaVA_Phi3_Config config;
    Phi3_Model* model;
    CLIP_Vision_Model* vidion_tower;
    LLaVA_Phi3_projector_weight weight;
    LLaVA_Phi3_RunState state;
} LLaVA_Phi3;


void llava_phi3_load_projector_weights_from_gguf(LLaVA_Phi3* llava, gguf_context* vision_ctx);
void llava_phi3_init_run_state(LLaVA_Phi3* llava);
LLaVA_Phi3* llava_phi3_create_from_gguf(gguf_context* llm_ctx, gguf_context* vision_ctx);
void llava_phi3_delete(LLaVA_Phi3* llava);

void llava_phi3_projector_forward(LLaVA_Phi3* llava, float* image_embedding, float* output);

char* llava_phi3_generate(LLaVA_Phi3 *model, Tokenizer *tokenizer, Sampler *sampler, char *prompt, float* image, int max_tokens_gen);
void llava_phi3_generate_stream(LLaVA_Phi3 *model, Tokenizer *tokenizer, Sampler *sampler, char *prompt, float* image, int max_tokens_gen, void (*callback)(const char*, size_t));

char* llava_phi3_generate_imgpath(LLaVA_Phi3 *model, Tokenizer *tokenizer, Sampler *sampler, char *prompt, const char* image_path, int max_tokens_gen);
void llava_phi3_generate_stream_imgpath(LLaVA_Phi3 *model, Tokenizer *tokenizer, Sampler *sampler, char *prompt, const char* image_path, int max_tokens_gen, void (*callback)(const char*, size_t));

#endif // LLAVA_PHI3_H