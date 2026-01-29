#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "phi3.h"
#include "gguf_reader.h"

void error_usage() {
    fprintf(stderr, "Usage:   run <gguf file> [options]\n");
    fprintf(stderr, "Example: run model.gguf -p \"What is the capital of France? Answer shortly.\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -p <string>  : prompt string\n");
    exit(EXIT_FAILURE);
}

void streaming_callback(char* piece, int piece_size){
    fwrite(piece, sizeof(char), piece_size, stdout);
    fflush(stdout);
}

int main(int argc, char* argv[]) {

     if (argc < 2) {
        error_usage();
    }
    const char* filename = argv[1];
    char* prompt = NULL;
    for (int i = 2; i < argc; i+=2) {
        if (i + 1 >= argc) { error_usage(); }
        if (argv[i][0] != '-') { error_usage(); }
        if (strlen(argv[i]) != 2) { error_usage(); }
        if (argv[i][1] == 'p') { prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    gguf_context* gguf_ctx = gguf_init_from_file(filename);
    if (!gguf_ctx) {
        fprintf(stderr, "Failed to load GGUF file\n");
        return 1;
    }
    
    // Initialize Phi3 Transformer from GGUF
    Phi3_Transformer* phi3 = init_phi3_from_gguf(gguf_ctx);
    if (!phi3) {
        fprintf(stderr, "Failed to initialize Phi3 Transformer from GGUF file\n");
        return 1;
    }

    // Init tokenizer
    Tokenizer* tokenizer = build_tokenizer_from_gguf(gguf_ctx);
    if (!tokenizer) {
        fprintf(stderr, "Failed to initialize Tokenizer from GGUF file\n");
        delete_phi3_transformer(phi3);
        return 1;
    }

    // Free GGUF context
    gguf_free(gguf_ctx);

    // Init sampler
    Sampler sampler;
    build_sampler(&sampler, tokenizer->vocab_size, 0.7f, 0.9f, 42);

    // Run the transformer on the prompt
    malloc_run_state(&phi3->state, &phi3->config);
    printf("prompt: %s\n", prompt);

    // char* generated_text = phi3_generate(phi3, tokenizer, &sampler, (char*)prompt, 100);
    // printf("Generated text: %s\n", generated_text);
    // free(generated_text);
    phi3_generate_stream(phi3, tokenizer, &sampler, (char*)prompt, 100, streaming_callback);

    // Cleanup
    delete_tokenizer(tokenizer);
    free_sampler(&sampler);
    delete_phi3_transformer(phi3);

    return 0;
}
