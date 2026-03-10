

1. 模型轉gguf
```bash
python phi3-hf2gguf.py \
    --model_path "microsoft/Phi-3-mini-4k-instruct" \
    --output_path "Phi-3-mini-4k-instruct-fp16.gguf"

python clip-hf2gguf.py \
    --model_path "openai/clip-vit-large-patch14" \
    --output_path "clip-vit-large-patch14-fp16.gguf"

llava-phi3-hf2gguf.py \
    --model_path "xtuner/llava-phi-3-mini-hf" \
    --output_path "llava-phi-3-mini-hf-fp16.gguf"
```

```bash
make all

# Phi3 Inference
./build/run_phi3 ./res/Phi-3-mini-4k-instruct-fp16-hf.gguf -p "What is the capital of France? Answer shortly."

# CLIP Inference
./build/run_clip ./res/clip-vit-large-patch14.gguf -i ./res/test_image.png

# LLaVA Inference
./build/run_llava ./res/llava-phi-3-mini-hf-fp16.gguf -p "What is in the image?" -i ./res/test_image.png
```

# 待做
1. Projector
2. Chat mode
3. CUDA

# 參考
1. [karpathy/llama2.c](https://github.com/karpathy/llama2.c)
2. [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)