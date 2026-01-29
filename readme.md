# 已完成
1. Phi-3推理
    ```bash
    make release
    ./build/run ../res/Phi-3-mini-4k-instruct-fp16-hf.gguf -p "What is the capital of France? Answer shortly."
    ```
2. OpenMP加速

# 待做
1. 註解與文檔
2. CLIP-ViT-L-@224, Projector
3. Chat mode
4. 硬體加速

# 參考
1. [karpathy/llama2.c](https://github.com/karpathy/llama2.c)
2. [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)