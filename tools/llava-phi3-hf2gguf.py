"""
LLaVA Phi-3 HuggingFace to GGUF Converter

This program converts LLaVA Phi-3 models from HuggingFace format to GGUF format for use with llama.cpp.
It separates the vision tower (including multimodal projector) and the language model into separate GGUF files.

Example Usage:
    # Convert a LLaVA Phi-3 model to GGUF format. The output will be two files: output.gguf and output_mmproj.gguf
    python llava-phi3-convert-hf-to-gguf.py \
        --model_path /path/to/Phi-3-mini-4k-instruct \
        --output_path output.gguf
        
    # Convert a LLaVA Phi-3 model with LoRA weights. The output will be two files: output.gguf and output_mmproj.gguf
    python llava-phi3-convert-hf-to-gguf.py \
        --model_path /path/to/llava-phi3-lora \
        --model_base /path/to/Phi-3-mini-4k-instruct \
        --output_path output.gguf

    # Example command to run the converted model by llama.cpp:
    llama.cpp/build/bin/llama-mtmd-cli \
        -m 'output.gguf' \
        --mmproj 'output_mmproj.gguf' \
        --image 'unhealthy/0001.jpg' \
        -p 'is the shrimp fresh?' -c 2048
    
Features:
    - Converts both vision tower and language model components
    - Handles parameter name mapping between PyTorch and GGUF formats
    - Supports tokenizer conversion with proper token types
    - Preserves model metadata and configuration

Limitations:
    - Only supports LLaVA Phi-3 models specifically
    - Requires sufficient system memory to load the entire model
    - Output files are in float32 format only
    - Vision tower uses CLIP architecture assumptions

Output Files:
    - {output_path}: Main language model in GGUF format
    - {output_path}_mmproj.gguf: Vision tower and multimodal projector in GGUF format
    
References:
    1. [gguf format documentation](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
"""

import argparse
import os
import torch
import numpy as np
import re
from gguf import GGUFWriter 
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.model import *

# Convert PyTorch parameter names to GGUF tensor names and vice versa
def pytorch_to_gguf_name(pytorch_name):
    """
    Convert a PyTorch parameter name to GGUF tensor name.
    
    This function maps PyTorch model parameter names to their corresponding GGUF tensor names
    for LLaVA Phi-3 models, including language model, vision tower, and multimodal projector components.
    
    Args:
        pytorch_name (str): The PyTorch parameter name (e.g., 'model.embed_tokens.weight')
        
    Returns:
        str or None: The corresponding GGUF tensor name if mapping exists, None otherwise
        
    Behavior:
        - Maps language model parameters (embeddings, attention, MLP, normalization)
        - Maps vision tower parameters (CLIP vision encoder components)
        - Maps multimodal projector parameters
        - Uses regex patterns to handle layer indices dynamically
    """
    # Transformer layers
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
    
    # Vision tower mappings
    vision_tower_patterns = [
        # Vision embeddings
        (r'model\.vision_tower\.vision_tower\.vision_model\.embeddings\.class_embedding', 'v.class_embd'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.embeddings\.patch_embedding\.weight', 'v.patch_embd.weight'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.embeddings\.position_embedding\.weight', 'v.position_embd.weight'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.pre_layrnorm\.weight', 'v.pre_ln.weight'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.pre_layrnorm\.bias', 'v.pre_ln.bias'),
        
        # Vision encoder layers
        (r'model\.vision_tower\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.weight', 'v.blk.{}.attn_k.weight'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.bias', 'v.blk.{}.attn_k.bias'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.weight', 'v.blk.{}.attn_v.weight'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.bias', 'v.blk.{}.attn_v.bias'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.weight', 'v.blk.{}.attn_q.weight'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.bias', 'v.blk.{}.attn_q.bias'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.weight', 'v.blk.{}.attn_out.weight'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.bias', 'v.blk.{}.attn_out.bias'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.layer_norm1\.weight', 'v.blk.{}.ln1.weight'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.layer_norm1\.bias', 'v.blk.{}.ln1.bias'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.weight', 'v.blk.{}.ffn_down.weight'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.bias', 'v.blk.{}.ffn_down.bias'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc2\.weight', 'v.blk.{}.ffn_up.weight'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc2\.bias', 'v.blk.{}.ffn_up.bias'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.layer_norm2\.weight', 'v.blk.{}.ln2.weight'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.encoder\.layers\.(\d+)\.layer_norm2\.bias', 'v.blk.{}.ln2.bias'),
        
        # Vision post layer norm
        (r'model\.vision_tower\.vision_tower\.vision_model\.post_layernorm\.weight', 'v.post_ln.weight'),
        (r'model\.vision_tower\.vision_tower\.vision_model\.post_layernorm\.bias', 'v.post_ln.bias'),
    ]
    
    for pattern, template in vision_tower_patterns:
        match = re.match(pattern, pytorch_name)
        if match:
            return template.format(*match.groups()) if '{}' in template else template
    
    # MM projector
    mm_projector_patterns = [
        (r'model\.mm_projector\.0\.weight', 'mm.0.weight'),
        (r'model\.mm_projector\.0\.bias', 'mm.0.bias'),
        (r'model\.mm_projector\.2\.weight', 'mm.2.weight'),
        (r'model\.mm_projector\.2\.bias', 'mm.2.bias'),
    ]
    
    for pattern, template in mm_projector_patterns:
        match = re.match(pattern, pytorch_name)
        if match:
            return template
    
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
        - Handles all component types: language model, vision tower, multimodal projector
        - Maintains consistency with forward mapping function
    """
    # Transformer layers
    if gguf_name == 'token_embd.weight':
        return 'model.embed_tokens.weight'
    
    # Layer mappings
    layer_patterns = [
        (r'blk\.(\d+)\.attn_qkv\.weight', 'model.layers.{}.self_attn.qkv_proj.weight'),
        (r'blk\.(\d+)\.attn_output\.weight', 'model.layers.{}.self_attn.o_proj.weight'),
        (r'blk\.(\d+)\.ffn_gate_up\.weight', 'model.layers.{}.mlp.gate_up_proj.weight'),
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
    
    # Vision tower mappings
    vision_tower_patterns = [
        # Vision embeddings
        (r'v\.class_embd', 'model.vision_tower.vision_tower.vision_model.embeddings.class_embedding'),
        (r'v\.patch_embd\.weight', 'model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight'),
        (r'v\.position_embd\.weight', 'model.vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight'),
        (r'v\.pre_ln\.weight', 'model.vision_tower.vision_tower.vision_model.pre_layrnorm.weight'),
        (r'v\.pre_ln\.bias', 'model.vision_tower.vision_tower.vision_model.pre_layrnorm.bias'),
        
        # Vision encoder layers
        (r'v\.blk\.(\d+)\.attn_k\.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.{}.self_attn.k_proj.weight'),
        (r'v\.blk\.(\d+)\.attn_k\.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.{}.self_attn.k_proj.bias'),
        (r'v\.blk\.(\d+)\.attn_v\.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.{}.self_attn.v_proj.weight'),
        (r'v\.blk\.(\d+)\.attn_v\.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.{}.self_attn.v_proj.bias'),
        (r'v\.blk\.(\d+)\.attn_q\.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.{}.self_attn.q_proj.weight'),
        (r'v\.blk\.(\d+)\.attn_q\.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.{}.self_attn.q_proj.bias'),
        (r'v\.blk\.(\d+)\.attn_out\.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.{}.self_attn.out_proj.weight'),
        (r'v\.blk\.(\d+)\.attn_out\.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.{}.self_attn.out_proj.bias'),
        (r'v\.blk\.(\d+)\.ln1\.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.{}.layer_norm1.weight'),
        (r'v\.blk\.(\d+)\.ln1\.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.{}.layer_norm1.bias'),
        (r'v\.blk\.(\d+)\.ffn_down\.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.{}.mlp.fc1.weight'),
        (r'v\.blk\.(\d+)\.ffn_down\.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.{}.mlp.fc1.bias'),
        (r'v\.blk\.(\d+)\.ffn_up\.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.{}.mlp.fc2.weight'),
        (r'v\.blk\.(\d+)\.ffn_up\.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.{}.mlp.fc2.bias'),
        (r'v\.blk\.(\d+)\.ln2\.weight', 'model.vision_tower.vision_tower.vision_model.encoder.layers.{}.layer_norm2.weight'),
        (r'v\.blk\.(\d+)\.ln2\.bias', 'model.vision_tower.vision_tower.vision_model.encoder.layers.{}.layer_norm2.bias'),
        
        # Vision post layer norm
        (r'v\.post_ln\.weight', 'model.vision_tower.vision_tower.vision_model.post_layernorm.weight'),
        (r'v\.post_ln\.bias', 'model.vision_tower.vision_tower.vision_model.post_layernorm.bias'),
    ]
    
    for pattern, template in vision_tower_patterns:
        match = re.match(pattern, gguf_name)
        if match:
            return template.format(*match.groups()) if '{}' in template else template
    
    # MM projector
    mm_projector_patterns = [
        (r'mm\.0\.weight', 'model.mm_projector.0.weight'),
        (r'mm\.0\.bias', 'model.mm_projector.0.bias'),
        (r'mm\.2\.weight', 'model.mm_projector.2.weight'),
        (r'mm\.2\.bias', 'model.mm_projector.2.bias'),
    ]
    
    for pattern, template in mm_projector_patterns:
        match = re.match(pattern, gguf_name)
        if match:
            return template
    
    # No mapping found
    return None

def get_vision_tower_properties(model, image_processor):
    """
    Extract vision tower properties and metadata for GGUF format.
    
    This function extracts configuration and metadata from the vision tower component
    of the LLaVA model to write proper GGUF headers and properties.
    
    Args:
        model: LLaVA model instance containing the vision tower
        image_processor: Image processor with preprocessing configuration
        
    Returns:
        dict: Dictionary containing vision tower properties including:
            - Architecture information (CLIP-based)
            - Image processing parameters (size, mean, std)
            - Model dimensions and layer counts
            - Attention configuration
            
    Behavior:
        - Assumes CLIP vision encoder architecture
        - Extracts dimensions from actual model parameters
        - Sets file type to float32 (type 0)
        - Configures for LLaVA-specific usage (no text encoder, has projector)
    """
    properties = dict()
    
    properties["general.architecture"] = "clip"
    
    properties["clip.has_text_encoder"] = False
    properties["clip.has_vision_encoder"] = True
    properties["clip.has_llava_projector"] = True
    properties["clip.projector_type"] = 'mlp'
    properties["clip.use_gelu"] = False
    
    
    
    properties["general.file_type"] = 0 # Float32
    
    properties["general.name"] = "clip-vit"
    properties["general.description"] = "image encoder for LLaVA Phi-3"
    
    properties["clip.vision.image_size"] = image_processor.size['shortest_edge']
    properties["clip.vision.image_mean"] = image_processor.image_mean
    properties["clip.vision.image_std"] = image_processor.image_std
    
    properties["clip.vision.patch_size"] = model.get_vision_tower().vision_tower.vision_model.embeddings.patch_embedding.kernel_size[0]
    properties["clip.vision.embedding_length"] = model.get_vision_tower().vision_tower.vision_model.pre_layrnorm.weight.shape[0]
    properties["clip.vision.feed_forward_length"] = model.get_vision_tower().vision_tower.vision_model.encoder.layers[0].mlp.fc1.weight.shape[0]
    properties["clip.vision.projection_dim"] = 768 # Not used in LLaVA Phi-3
    properties["clip.vision.attention.head_count"] = model.get_vision_tower().vision_tower.vision_model.encoder.layers[0].self_attn.num_heads
    properties["clip.vision.attention.layer_norm_epsilon"] = model.get_vision_tower().vision_tower.vision_model.encoder.layers[0].layer_norm1.eps
    properties["clip.vision.attention.block_count"] = len(model.get_vision_tower().vision_tower.vision_model.encoder.layers)
    
    
    return properties

def get_llm_properties(model, tokenizer, image_processor, context_len=None):
    """
    Extract language model properties and tokenizer information for GGUF format.
    
    This function extracts all necessary metadata from the Phi-3 language model component
    and tokenizer to create a complete GGUF file with proper configuration.
    
    Args:
        model: LLaVA model instance containing the language model
        tokenizer: Tokenizer instance for vocabulary and special tokens
        image_processor: Image processor (for completeness, not directly used)
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
    
    properties["general.context_length"] = context_len
    properties["general.embedding_length"] = model.model.embed_tokens.embedding_dim  # 3072
    properties["general.feed_forward_length"] = model.model.layers[0].mlp.gate_up_proj.out_features // 2  # 8192
    properties["general.block_count"] = len(model.model.layers)  # 32
    properties["phi3.attention.head_count"] = model.model.layers[0].self_attn.num_heads
    properties["phi3.attention.head_count_kv"] = model.model.layers[0].self_attn.num_key_value_heads
    properties["phi3.attention.layer_norm_rms_epsilon"] = model.model.layers[0].input_layernorm.variance_epsilon
    
    properties["general.rope.dimension_count"] = model.model.layers[0].self_attn.rotary_emb.dim
    properties["general.file_type"] = 0  # Float32
    properties["general.description"] = "LLaVA Phi-3 LLM"
    
    properties["tokenizer.ggml.model"] = "llama"
    properties["tokenizer.ggml.pre"] = None  # Needs additional processing
    tokens = [b''] * model.vocab_size
    for token_str, token_id in tokenizer.get_vocab().items():
        tokens[token_id] = token_str.encode('utf-8')
    properties["tokenizer.ggml.tokens"] = tokens
    properties["tokenizer.ggml.scores"] = [0.0] * model.vocab_size  # Assuming no scores are provided
    # Set token types: 1=normal, 2=unknown, 3=control, 4=user defined, 5=unused, 6=byte
    token_types = []
    for token_id in range(model.vocab_size):
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
    properties["tokenizer.ggml.padding_token_id"] = tokenizer.pad_token_id
    properties["tokenizer.ggml.add_bos_token"] = tokenizer.add_bos_token
    properties["tokenizer.ggml.add_eos_token"] = tokenizer.add_eos_token
    properties["tokenizer.ggml.chat_template"] = getattr(tokenizer, "chat_template", None)
    
    return properties

def write_gguf_model_vision_tower(gguf_file, model, image_processor):
    """
    Write the vision tower and multimodal projector to a GGUF file.
    
    This function creates a separate GGUF file containing only the vision tower
    (CLIP encoder) and multimodal projector components, which can be used
    independently for image processing in llama.cpp.
    
    Args:
        gguf_file (str): Output file path for the vision tower GGUF file
        model: LLaVA model instance containing vision components
        image_processor: Image processor with preprocessing configuration
        
    Returns:
        None
        
    Behavior:
        - Creates GGUF writer with CLIP architecture
        - Writes vision tower metadata and properties
        - Filters and converts only vision_tower and mm_projector parameters
        - Maps PyTorch parameter names to GGUF tensor names
        - Converts tensors to float32 format
        - Saves complete GGUF file with header, metadata, and tensors
        
    Side Effects:
        - Creates a new GGUF file at the specified path
        - Prints progress information for each tensor added
        - Prints warnings for unmapped parameters
    """
    vision_prop = get_vision_tower_properties(model, image_processor)
    
    # Create GGUF writer
    writer = GGUFWriter(gguf_file, vision_prop["general.architecture"])
    
    # Write properties
    writer.add_name(vision_prop["general.name"])
    writer.add_file_type(vision_prop["general.file_type"])
    writer.add_description(vision_prop["general.description"])
    
    writer.add_bool("clip.has_text_encoder", vision_prop["clip.has_text_encoder"])
    writer.add_bool("clip.has_vision_encoder", vision_prop["clip.has_vision_encoder"])
    writer.add_bool("clip.has_llava_projector", vision_prop["clip.has_llava_projector"])
    writer.add_string("clip.projector_type", vision_prop["clip.projector_type"])
    writer.add_bool("clip.use_gelu", vision_prop["clip.use_gelu"])
    
    writer.add_uint32("clip.vision.image_size", vision_prop["clip.vision.image_size"])
    writer.add_array("clip.vision.image_mean", vision_prop["clip.vision.image_mean"])
    writer.add_array("clip.vision.image_std", vision_prop["clip.vision.image_std"])
    
    writer.add_uint32("clip.vision.patch_size", vision_prop["clip.vision.patch_size"])
    writer.add_uint32("clip.vision.embedding_length", vision_prop["clip.vision.embedding_length"])
    writer.add_uint32("clip.vision.feed_forward_length", vision_prop["clip.vision.feed_forward_length"])
    writer.add_uint32("clip.vision.projection_dim", vision_prop["clip.vision.projection_dim"])
    
    writer.add_uint32("clip.vision.attention.head_count", vision_prop["clip.vision.attention.head_count"])
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", vision_prop["clip.vision.attention.layer_norm_epsilon"])
    writer.add_uint32("clip.vision.block_count", vision_prop["clip.vision.attention.block_count"])
    
    # Write parameters
    for name, param in model.state_dict().items():
        # Only include vision tower and mm projector tensors
        if 'vision_tower' in name or 'mm_projector' in name:
            
            # Skip the layers after "mm_vision_select_layer"
            mm_vision_layears = len(model.model.vision_tower.vision_tower.vision_model.encoder.layers)
            mm_vision_select_layer = model.config.mm_vision_select_layer # -2 normally, 
            #     Extract layer index. (model.vision_tower.vision_tower.vision_model.encoder.layers.23....)
            encoder_layer_idx = mm_vision_layears + 1 # default to a value larger than any layer index
            match = re.search(r'vision_model\.encoder\.layers\.(\d+)\.', name)
            if match:
                encoder_layer_idx = int(match.group(1))
            if encoder_layer_idx > mm_vision_layears + mm_vision_select_layer:
                continue
            
            # Skip post_layernorm ("model.vision_tower.vision_tower.vision_model.post_layernorm....")
            if 'post_layernorm' in name:
                continue
            
            # Convert parameter to numpy array
            tensor_data = param.to(torch.float32).cpu().numpy()
            raw_shape = tensor_data.shape
            
            # Find corresponding GGUF tensor name using the mapping function
            gguf_name = pytorch_to_gguf_name(name)
            
            if gguf_name is not None:
                # Add the tensor to the GGUF file
                tensor_data = tensor_data.astype(np.float32)
                print(f"Adding tensor: {gguf_name}, Shape: {tensor_data.shape}, dtype: {tensor_data.dtype}")
                writer.add_tensor(gguf_name, tensor_data, raw_shape)
            else:
                print(f"Warning: No mapping found for {name}")
    
    # Save the vision tower GGUF file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

def write_gguf_model_llm(gguf_file, model, tokenizer, image_processor, context_len=None):
    """
    Write the language model to a GGUF file.
    
    This function creates the main GGUF file containing the Phi-3 language model
    with complete tokenizer information, excluding vision components.
    
    Args:
        gguf_file (str): Output file path for the language model GGUF file
        model: LLaVA model instance containing the language model
        tokenizer: Tokenizer instance for vocabulary processing
        image_processor: Image processor (for metadata completeness)
        context_len (int, optional): Context length override
        
    Returns:
        None
        
    Behavior:
        - Creates GGUF writer with Phi-3 architecture
        - Writes language model metadata and tokenizer information
        - Filters out vision_tower and mm_projector parameters
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
    llm_prop = get_llm_properties(model, tokenizer, image_processor, context_len)
    
    # Write properties
    writer = GGUFWriter(gguf_file, llm_prop["general.architecture"])
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
    writer.add_chat_template(llm_prop["tokenizer.ggml.chat_template"])
    
    # Write parameters
    for name, param in model.state_dict().items():
        # Skip vision tower and mm projector tensors for the main model
        if 'vision_tower' in name or 'mm_projector' in name:
            continue
            
        # Convert parameter to numpy array
        tensor_data = param.to(torch.float32).cpu().numpy()
        
        # Find corresponding GGUF tensor name using the mapping function
        gguf_name = pytorch_to_gguf_name(name)
        
        if gguf_name is not None:
            # Handle special case for token_embd.weight and output.weight  which might need padding
            if gguf_name == "token_embd.weight":
                expected_vocab_size = len(tokenizer.get_vocab())
                actual_vocab_size, actual_embd_dim = tensor_data.shape
                ori_shape = tensor_data.shape
                tensor_data = np.pad(tensor_data, ((0, int(expected_vocab_size - actual_vocab_size)), (0, 0)), mode='constant', constant_values=0.0)
                print(f"Padded output weight from {ori_shape} to {tensor_data.shape}")
                writer.add_tensor(gguf_name, tensor_data, tensor_data.shape)
                continue
            if gguf_name == "output.weight":
                expected_vocab_size = len(tokenizer.get_vocab())
                actual_vocab_size, actual_embd_dim = tensor_data.shape
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

def convert_hf_to_gguf(hf_model_path, hf_model_base=None, output_path='output.gguf'):
    """
    Convert a LLaVA Phi-3 HuggingFace model to GGUF format.
    
    This is the main conversion function that orchestrates the entire process
    of loading the HuggingFace model and converting it to two separate GGUF files.
    
    Args:
        hf_model_path (str): Path to the HuggingFace model directory
        hf_model_base (str, optional): Base model path for loading pretrained weights
        output_path (str): Output path for the main language model GGUF file
        
    Returns:
        None
        
    Behavior:
        - Loads the complete LLaVA model from HuggingFace format
        - Converts model to float32 precision
        - Creates two output files:
            * Main GGUF file: Language model component
            * _mmproj.gguf file: Vision tower and multimodal projector
        - Handles model name extraction and device placement
        
    Side Effects:
        - Creates two GGUF files in the filesystem
        - Loads entire model into CPU memory
        - Prints conversion progress and statistics
    """
    model_name = get_model_name_from_path(hf_model_path)
    print(f"Model name: {model_name}")
    tokenizer, model, image_processor, context_len = load_pretrained_model(hf_model_path, hf_model_base, model_name, device='cpu')
    model = model.float()  # Ensure model is in float32 format
    if not output_path.endswith('.gguf'): output_path += '.gguf' # Ensure output path ends with .gguf
    write_gguf_model_vision_tower(output_path.replace('.gguf', '_mmproj.gguf'), model, image_processor)
    write_gguf_model_llm(output_path, model, tokenizer, image_processor, context_len)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LLaVA Phi-3 HF model to GGUF format")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Hugging Face model directory")
    parser.add_argument("--model_base", type=str, default=None, help="Base model path for loading pretrained weights")
    parser.add_argument("--output_path", type=str, default="output.gguf", help="Output GGUF file path")

    args = parser.parse_args()
    
    # Force to disable cuda
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Export the model to GGUF format
    convert_hf_to_gguf(args.model_path, args.model_base, args.output_path)

