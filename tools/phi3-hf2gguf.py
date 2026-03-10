"""
Phi-3 HuggingFace to GGUF Converter

This program converts Phi-3 models from HuggingFace format to GGUF format for use with llama.cpp.

Example Usage:
    # Convert a Phi-3 model to GGUF format
    python phi3-hf2gguf.py \
        --model_path /path/to/Phi-3-mini-4k-instruct \
        --output_path output.gguf
        
    # Example command to run the converted model by llama.cpp:
    llama.cpp/build/bin/llama-cli \
        -m 'output.gguf' \
        -p 'What is the capital of France?' -c 2048
    
Features:
    - Converts Phi-3 language model to GGUF format
    - Handles parameter name mapping between PyTorch and GGUF formats
    - Supports tokenizer conversion with proper token types
    - Preserves model metadata and configuration

Limitations:
    - Only supports Phi-3 models specifically
    - Requires sufficient system memory to load the entire model
    - Output files are in float32 format only

Output Files:
    - {output_path}: Main language model in GGUF format
    
References:
    1. [gguf format documentation](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
"""

import argparse
import os
import torch
import numpy as np
import re
from gguf import GGUFWriter
from transformers import AutoTokenizer, AutoModelForCausalLM

# Convert PyTorch parameter names to GGUF tensor names and vice versa
def pytorch_to_gguf_name(pytorch_name):
    """
    Convert a PyTorch parameter name to GGUF tensor name.
    
    This function maps PyTorch model parameter names to their corresponding GGUF tensor names
    for Phi-3 models, including embeddings, attention, MLP, and normalization components.
    
    Args:
        pytorch_name (str): The PyTorch parameter name (e.g., 'model.embed_tokens.weight')
        
    Returns:
        str or None: The corresponding GGUF tensor name if mapping exists, None otherwise
        
    Behavior:
        - Maps language model parameters (embeddings, attention, MLP, normalization)
        - Uses regex patterns to handle layer indices dynamically
    """
    # Embedding layer
    if pytorch_name == 'model.embed_tokens.weight':
        return 'token_embd.weight'
    
    # Layer mappings
    layer_patterns = [
        (r'model\.layers\.(\d+)\.self_attn\.qkv_proj\.weight', 'blk.{}.attn_qkv.weight'),
        (r'model\.layers\.(\d+)\.self_attn\.o_proj\.weight', 'blk.{}.attn_output.weight'),
        (r'model\.layers\.(\d+)\.mlp\.gate_up_proj\.weight', 'blk.{}.ffn_up.weight'),
        (r'model\.layers\.(\d+)\.mlp\.down_proj\.weight', 'blk.{}.ffn_down.weight'),
        (r'model\.layers\.(\d+)\.input_layernorm\.weight', 'blk.{}.attn_norm.weight'),
        (r'model\.layers\.(\d+)\.post_attention_layernorm\.weight', 'blk.{}.ffn_norm.weight'),
    ]
    
    for pattern, template in layer_patterns:
        match = re.match(pattern, pytorch_name)
        if match:
            return template.format(*match.groups())
    
    # Output layers
    if pytorch_name == 'model.norm.weight':
        return 'output_norm.weight'
    elif pytorch_name == 'lm_head.weight':
        return 'output.weight'
    
    # No mapping found
    return None

def gguf_to_pytorch_name(gguf_name):
    """
    Convert a GGUF tensor name to PyTorch parameter name.
    
    This function provides the reverse mapping from GGUF tensor names back to PyTorch
    parameter names, useful for validation and debugging purposes.
    
    Args:
        gguf_name (str): The GGUF tensor name (e.g., 'token_embd.weight')
        
    Returns:
        str or None: The corresponding PyTorch parameter name if mapping exists, None otherwise
        
    Behavior:
        - Reverses the mapping performed by pytorch_to_gguf_name
        - Maintains consistency with forward mapping function
    """
    # Embedding layer
    if gguf_name == 'token_embd.weight':
        return 'model.embed_tokens.weight'
    
    # Layer mappings
    layer_patterns = [
        (r'blk\.(\d+)\.attn_qkv\.weight', 'model.layers.{}.self_attn.qkv_proj.weight'),
        (r'blk\.(\d+)\.attn_output\.weight', 'model.layers.{}.self_attn.o_proj.weight'),
        (r'blk\.(\d+)\.ffn_up\.weight', 'model.layers.{}.mlp.gate_up_proj.weight'),
        (r'blk\.(\d+)\.ffn_down\.weight', 'model.layers.{}.mlp.down_proj.weight'),
        (r'blk\.(\d+)\.attn_norm\.weight', 'model.layers.{}.input_layernorm.weight'),
        (r'blk\.(\d+)\.ffn_norm\.weight', 'model.layers.{}.post_attention_layernorm.weight'),
    ]
    
    for pattern, template in layer_patterns:
        match = re.match(pattern, gguf_name)
        if match:
            return template.format(*match.groups())
    
    # Output layers
    if gguf_name == 'output_norm.weight':
        return 'model.norm.weight'
    elif gguf_name == 'output.weight':
        return 'lm_head.weight'
    
    # No mapping found
    return None

def get_llm_properties(model, tokenizer, context_len=None):
    """
    Extract language model properties and tokenizer information for GGUF format.
    
    This function extracts all necessary metadata from the Phi-3 language model
    and tokenizer to create a complete GGUF file with proper configuration.
    
    Args:
        model: Phi-3 model instance
        tokenizer: Tokenizer instance for vocabulary and special tokens
        context_len (int, optional): Context length override, uses model default if None
        
    Returns:
        dict: Dictionary containing language model properties including:
            - Model architecture and dimensions
            - Attention configuration
            - Tokenizer vocabulary and special tokens
            - Token types and scores
            - Chat template if available
            
    Behavior:
        - Extracts dimensions from actual model layers
        - Processes tokenizer vocabulary into bytes format
        - Assigns token types (normal, unknown, control, byte)
        - Handles special tokens (BOS, EOS, UNK, PAD)
        - Sets default scores to 0.0 for all tokens
    """
    properties = dict()
    
    properties["general.architecture"] = "phi3"
    properties["general.name"] = "Phi3"
    
    # Get context length from config if not provided
    if context_len is None:
        context_len = model.config.max_position_embeddings
    
    properties["general.context_length"] = context_len
    properties["general.embedding_length"] = model.model.embed_tokens.embedding_dim  # 3072
    properties["general.feed_forward_length"] = model.model.layers[0].mlp.gate_up_proj.out_features // 2  # 8192
    properties["general.block_count"] = len(model.model.layers)  # 32
    properties["phi3.attention.head_count"] = model.model.layers[0].self_attn.num_heads
    properties["phi3.attention.head_count_kv"] = model.model.layers[0].self_attn.num_key_value_heads
    properties["phi3.attention.layer_norm_rms_epsilon"] = model.model.layers[0].input_layernorm.variance_epsilon
    
    properties["general.rope.dimension_count"] = model.model.layers[0].self_attn.rotary_emb.dim
    properties["general.file_type"] = 0  # Float32
    properties["general.description"] = "Phi-3 Language Model"
    
    properties["tokenizer.ggml.model"] = "llama"
    properties["tokenizer.ggml.pre"] = None  # Needs additional processing
    
    # Get vocabulary
    vocab = tokenizer.get_vocab()
    vocab_size = len(vocab)
    
    tokens = [b''] * vocab_size
    for token_str, token_id in vocab.items():
        tokens[token_id] = token_str.encode('utf-8')
    properties["tokenizer.ggml.tokens"] = tokens
    properties["tokenizer.ggml.scores"] = [0.0] * vocab_size  # Assuming no scores are provided
    
    # Set token types: 1=normal, 2=unknown, 3=control, 4=user defined, 5=unused, 6=byte
    token_types = []
    for token_id in range(vocab_size):
        if token_id == tokenizer.unk_token_id:
            token_types.append(2)  # Unknown token
        elif token_id in [1, 2, 32000, 32001, 32002, 32003, 32004, 32005, 32006, 32007, 32008, 32009, 32010]:
            token_types.append(3)  # Control tokens
        elif token_id >= 0 and token_id < (256 + 3): # 3 is for special tokens <unk>, <s>, </s> at the index 0, 1, 2
            token_types.append(6)  # Byte token
        else:
            token_types.append(1)  # Normal token
    properties["tokenizer.ggml.token_type"] = token_types
    
    properties["tokenizer.ggml.bos_token_id"] = tokenizer.bos_token_id
    properties["tokenizer.ggml.eos_token_id"] = tokenizer.eos_token_id
    properties["tokenizer.ggml.unknown_token_id"] = tokenizer.unk_token_id
    properties["tokenizer.ggml.padding_token_id"] = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    properties["tokenizer.ggml.add_bos_token"] = getattr(tokenizer, "add_bos_token", False)
    properties["tokenizer.ggml.add_eos_token"] = getattr(tokenizer, "add_eos_token", False)
    properties["tokenizer.ggml.chat_template"] = getattr(tokenizer, "chat_template", None)
    
    return properties

def write_gguf_model_llm(gguf_file, model, tokenizer, context_len=None):
    """
    Write the language model to a GGUF file.
    
    This function creates the GGUF file containing the Phi-3 language model
    with complete tokenizer information.
    
    Args:
        gguf_file (str): Output file path for the language model GGUF file
        model: Phi-3 model instance
        tokenizer: Tokenizer instance for vocabulary processing
        context_len (int, optional): Context length override
        
    Returns:
        None
        
    Behavior:
        - Creates GGUF writer with Phi-3 architecture
        - Writes language model metadata and tokenizer information
        - Handles vocabulary size padding for embedding and output layers
        - Maps PyTorch parameter names to GGUF tensor names
        - Converts all tensors to float32 format
        - Saves complete GGUF file
        
    Side Effects:
        - Creates a new GGUF file at the specified path
        - May pad vocabulary tensors if tokenizer size differs from model
        - Prints progress information and tensor statistics
        - Prints warnings for unmapped parameters
    """
    llm_prop = get_llm_properties(model, tokenizer, context_len)
    
    # Create GGUF writer
    writer = GGUFWriter(gguf_file, llm_prop["general.architecture"])
    
    # Write properties
    writer.add_name(llm_prop["general.name"])
    writer.add_file_type(llm_prop["general.file_type"])
    writer.add_description(llm_prop["general.description"])
    
    writer.add_context_length(llm_prop["general.context_length"])
    writer.add_embedding_length(llm_prop["general.embedding_length"])
    writer.add_feed_forward_length(llm_prop["general.feed_forward_length"])
    writer.add_block_count(llm_prop["general.block_count"])
    writer.add_uint32("phi3.attention.head_count", llm_prop["phi3.attention.head_count"])
    writer.add_uint32("phi3.attention.head_count_kv", llm_prop["phi3.attention.head_count_kv"])
    writer.add_float32("phi3.attention.layer_norm_rms_epsilon", llm_prop["phi3.attention.layer_norm_rms_epsilon"])

    writer.add_rope_dimension_count(llm_prop["general.rope.dimension_count"])
    
    writer.add_tokenizer_model(llm_prop["tokenizer.ggml.model"])
    writer.add_token_list(llm_prop["tokenizer.ggml.tokens"])
    writer.add_token_scores(llm_prop["tokenizer.ggml.scores"])
    writer.add_token_types(llm_prop["tokenizer.ggml.token_type"])
    
    writer.add_bos_token_id(llm_prop["tokenizer.ggml.bos_token_id"])
    writer.add_eos_token_id(llm_prop["tokenizer.ggml.eos_token_id"])
    writer.add_unk_token_id(llm_prop["tokenizer.ggml.unknown_token_id"])
    writer.add_pad_token_id(llm_prop["tokenizer.ggml.padding_token_id"])
    writer.add_add_bos_token(llm_prop["tokenizer.ggml.add_bos_token"])
    writer.add_add_eos_token(llm_prop["tokenizer.ggml.add_eos_token"])
    if llm_prop["tokenizer.ggml.chat_template"] is not None:
        writer.add_chat_template(llm_prop["tokenizer.ggml.chat_template"])
    
    # Write parameters
    for name, param in model.state_dict().items():
        # Convert parameter to numpy array
        tensor_data = param.to(torch.float32).cpu().numpy()
        
        # Find corresponding GGUF tensor name using the mapping function
        gguf_name = pytorch_to_gguf_name(name)
        
        if gguf_name is not None:
            # Handle special case for token_embd.weight and output.weight which might need padding
            if gguf_name == "token_embd.weight":
                expected_vocab_size = len(tokenizer.get_vocab())
                actual_vocab_size, actual_embd_dim = tensor_data.shape
                if expected_vocab_size != actual_vocab_size:
                    ori_shape = tensor_data.shape
                    tensor_data = np.pad(tensor_data, ((0, int(expected_vocab_size - actual_vocab_size)), (0, 0)), mode='constant', constant_values=0.0)
                    print(f"Padded token embedding weight from {ori_shape} to {tensor_data.shape}")
                writer.add_tensor(gguf_name, tensor_data, tensor_data.shape)
                continue
            if gguf_name == "output.weight":
                expected_vocab_size = len(tokenizer.get_vocab())
                actual_vocab_size, actual_embd_dim = tensor_data.shape
                if expected_vocab_size != actual_vocab_size:
                    ori_shape = tensor_data.shape
                    tensor_data = np.pad(tensor_data, ((0, int(expected_vocab_size - actual_vocab_size)), (0, 0)), mode='constant', constant_values=0.0)
                    print(f"Padded output weight from {ori_shape} to {tensor_data.shape}")
                writer.add_tensor(gguf_name, tensor_data, tensor_data.shape)
                continue
            
            # Add the tensor to the GGUF file
            print(f"Adding tensor: {gguf_name}, Shape: {tensor_data.shape}, dtype: {tensor_data.dtype}")
            writer.add_tensor(gguf_name, tensor_data, tensor_data.shape)
        else:
            print(f"Warning: No mapping found for {name}")

    # Save the main model GGUF file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"Successfully wrote GGUF model to {gguf_file}")

def convert_hf_to_gguf(hf_model_path, output_path='output.gguf'):
    """
    Convert a Phi-3 HuggingFace model to GGUF format.
    
    This is the main conversion function that orchestrates the entire process
    of loading the HuggingFace model and converting it to GGUF format.
    
    Args:
        hf_model_path (str): Path to the HuggingFace model directory
        output_path (str): Output path for the language model GGUF file
        
    Returns:
        None
        
    Behavior:
        - Loads the Phi-3 model from HuggingFace format
        - Converts model to float32 precision
        - Creates GGUF file with language model component
        - Handles device placement (CPU)
        
    Side Effects:
        - Creates a GGUF file in the filesystem
        - Loads entire model into CPU memory
        - Prints conversion progress and statistics
    """
    print(f"Loading model from {hf_model_path}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_path, 
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    
    print(f"Model loaded. Vocab size: {len(tokenizer.get_vocab())}")
    
    # Ensure output path ends with .gguf
    if not output_path.endswith('.gguf'):
        output_path += '.gguf'
    
    # Convert and write to GGUF
    write_gguf_model_llm(output_path, model, tokenizer)
    
    print(f"Conversion complete! Output saved to {output_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Phi-3 HF model to GGUF format")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Hugging Face model directory")
    parser.add_argument("--output_path", type=str, default="output.gguf", help="Output GGUF file path")

    args = parser.parse_args()
    
    # Export the model to GGUF format
    convert_hf_to_gguf(args.model_path, args.output_path)
