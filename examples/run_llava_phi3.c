#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llava.h"
#include "gguf_reader.h"


void error_usage() {
    fprintf(stderr, "Usage:   run_llava_phi3 <model gguf file> <mmproj gguf file> [options]\n");
    fprintf(stderr, "Example: run_llava_phi3 model.gguf mmproj.gguf -i image.png -p \"What is this?\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -i <path>    : input image path\n");
    fprintf(stderr, "  -p <string>  : prompt string\n");
    exit(EXIT_FAILURE);
}

void streaming_callback(char* piece, int piece_size){
    fwrite(piece, sizeof(char), piece_size, stdout);
    fflush(stdout);
}

int main(int argc, char* argv[]) {
    // 1. Parse command line arguments ////////////////////////////////////////
    if (argc < 3) {
        error_usage();
    }
    const char* model_path = argv[1];
    const char* mmproj_path = argv[2];
    char* image_path = NULL;
    char* prompt = NULL;
    
    for (int i = 3; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if (argv[i][1] == 'i') { image_path = argv[i + 1]; }
        else if (argv[i][1] == 'p') { prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // 2. Load GGUF files and initialize LLaVA_Phi3 model /////////////////////
    gguf_context* mmproj_ctx = gguf_init_from_file(mmproj_path);
    if (!mmproj_ctx) {
        fprintf(stderr, "Failed to load GGUF file\n");
        return 1;
    }

    gguf_context* model_ctx = gguf_init_from_file(model_path);
    if (!model_ctx) {
        fprintf(stderr, "Failed to load GGUF file\n");
        return 1;
    }

    LLaVA_Phi3* llava = llava_phi3_create_from_gguf(model_ctx, mmproj_ctx);

    // Init tokenizer
    Tokenizer* tokenizer = build_tokenizer_from_gguf(model_ctx);
    if (!tokenizer) {
        fprintf(stderr, "Failed to initialize Tokenizer from GGUF file\n");
        llava_phi3_delete(llava);
        return 1;
    }

    // Free GGUF context
    gguf_free(model_ctx);
    gguf_free(mmproj_ctx);

    // Init sampler
    Sampler sampler;
    build_sampler(&sampler, tokenizer->vocab_size, 0.7f, 0.9f, 42);

    // 3. Run the transformer with the prompt //////////////////////////////////
    // Run the transformer on the prompt
    char input_prompt[1024];
    if (prompt == NULL) {
        // Wait for user input if no prompt provided
        printf("User: ");
        if (fgets(input_prompt, sizeof(input_prompt), stdin) != NULL) {
            // Remove newline character if present
            input_prompt[strcspn(input_prompt, "\n")] = 0;
            prompt = input_prompt;
        }
    }
    printf("prompt: %s\n", prompt);
    llava_phi3_generate_stream_imgpath(llava, tokenizer, &sampler, prompt, image_path, 512, streaming_callback);
    
    // Cleanup
    delete_tokenizer(tokenizer);
    free_sampler(&sampler);
    llava_phi3_delete(llava);
    
    return 0;
}
