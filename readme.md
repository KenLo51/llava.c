```bash
make all

# Phi3 Inference
./build/run_phi3 ./res/Phi-3-mini-4k-instruct-fp16-hf.gguf -p "What is the capital of France? Answer shortly."

# CLIP Inference
./build/run_clip ./res/clip-vit-large-patch14.gguf -i ./res/test_image.png
```

# 待做
1. Projector
2. Chat mode
3. CUDA

# 參考
1. [karpathy/llama2.c](https://github.com/karpathy/llama2.c)
2. [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)