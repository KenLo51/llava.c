#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "clip.h"
#include "gguf_reader.h"

void error_usage() {
    fprintf(stderr, "Usage:   run_clip <gguf file> [options]\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -i <path>  : input image\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        error_usage();
    }
    const char* filename = argv[1];
    char* image_path = NULL; // 
    for (int i = 2; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if (argv[i][1] == 'i') { image_path = argv[i + 1]; }
        else { error_usage(); }
    }

    gguf_context* gguf_ctx = gguf_init_from_file(filename);
    if (!gguf_ctx) {
        fprintf(stderr, "Failed to load GGUF file\n");
        return 1;
    }
    
    // Initialize CLIP Vision from GGUF
    CLIP_Vision_Model* clip_vision = clip_vision_create_from_gguf(gguf_ctx);
    if (!clip_vision) {
        fprintf(stderr, "Failed to initialize CLIP Vision from GGUF file\n");
        return 1;
    }

    // Free GGUF context
    gguf_free(gguf_ctx);

    float* output_embedding = (float*)malloc(clip_vision->config.proj_dim * sizeof(float));
    if (!output_embedding) {
        fprintf(stderr, "Failed to allocate memory for output embedding\n");
        exit(EXIT_FAILURE);
    }
    clip_get_image_features_from_file(clip_vision, image_path, output_embedding);

    int patch_idx = 0; // CLS
    for(int i=0; i<clip_vision->config.proj_dim; i++)
        printf("%.5f,", clip_vision->state.f[patch_idx * clip_vision->config.proj_dim + i]);
    printf("\n");

    // Cleanup
    free(output_embedding);
    clip_vision_delete(clip_vision);

    return 0;
}
