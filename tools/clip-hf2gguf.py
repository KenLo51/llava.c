"""
CLIP HuggingFace to GGUF Converter

This program converts CLIP vision models from HuggingFace format to GGUF format.
It extracts only the vision encoder and visual projection components, saving all layers
including the final projector.

Example Usage:
    # Convert a CLIP model to GGUF format (vision encoder only)
    python clip-hf2gguf.py \
        --model_path openai/clip-vit-large-patch14 \
        --output_path clip-vision.gguf
    
    # Convert a local CLIP model
    python clip-hf2gguf.py \
        --model_path /path/to/local/clip-model \
        --output_path output.gguf

Features:
    - Extracts vision encoder from CLIP models
    - Includes all encoder layers (no layer skipping)
    - Saves visual projection layer
    - Preserves model metadata and configuration
    - Compatible with GGUF format specification

Output:
    - Single GGUF file containing complete vision encoder and projector
    
References:
    1. [gguf format documentation](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
    2. [CLIP paper](https://arxiv.org/abs/2103.00020)
"""

import argparse
import os
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor
from gguf import GGUFWriter


def get_vision_properties(model, processor):
    """
    Extract vision model properties and metadata for GGUF format.
    
    Args:
        model: CLIP model instance
        processor: CLIP processor with image preprocessing configuration
        
    Returns:
        dict: Dictionary containing vision model properties
    """
    vision_config = model.config.vision_config
    properties = {}
    
    # General architecture
    properties["general.architecture"] = "clip"
    properties["general.name"] = "clip-vit"
    properties["general.file_type"] = 0  # Float32
    properties["general.description"] = "CLIP Vision Model exported from HuggingFace transformers"
    
    # CLIP component flags
    properties["clip.has_text_encoder"] = False
    properties["clip.has_vision_encoder"] = True
    properties["clip.has_llava_projector"] = False
    
    # Vision model configuration
    properties["clip.vision.image_size"] = vision_config.image_size
    properties["clip.vision.image_mean"] = processor.image_processor.image_mean
    properties["clip.vision.image_std"] = processor.image_processor.image_std
    properties["clip.vision.patch_size"] = vision_config.patch_size
    properties["clip.vision.embedding_length"] = vision_config.hidden_size
    properties["clip.vision.feed_forward_length"] = vision_config.intermediate_size
    properties["clip.vision.projection_dim"] = model.config.projection_dim
    properties["clip.vision.attention.head_count"] = vision_config.num_attention_heads
    properties["clip.vision.attention.layer_norm_epsilon"] = vision_config.layer_norm_eps
    properties["clip.vision.attention.block_count"] = vision_config.num_hidden_layers
    
    return properties


def add_tensor_to_writer(writer, name, tensor):
    """
    Add a tensor to the GGUF writer with proper formatting.
    
    Args:
        writer: GGUFWriter instance
        name: Tensor name in GGUF format
        tensor: PyTorch tensor or numpy array
    """
    if tensor is None:
        print(f"Warning: Skipping {name} (None)")
        return
    
    # Convert to numpy if it's a torch tensor
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    
    # Ensure float32
    if tensor.dtype != np.float32:
        tensor = tensor.astype(np.float32)
    
    original_shape = tensor.shape
    writer.add_tensor(name, tensor)
    print(f"Added: {name} {list(original_shape)}")


def write_vision_model_to_gguf(model, processor, output_path):
    """
    Write the complete vision model to a GGUF file.
    
    This includes:
    - Class embedding
    - Patch embedding (Conv2d)
    - Position embedding
    - Pre-layer normalization
    - All encoder blocks (attention + MLP)
    - Post-layer normalization
    - Visual projection
    
    Args:
        model: CLIP model instance
        processor: CLIP processor instance
        output_path: Output file path for GGUF file
    """
    # Get properties
    properties = get_vision_properties(model, processor)
    
    # Create GGUF writer
    writer = GGUFWriter(output_path, properties["general.architecture"])
    
    # Write metadata
    writer.add_name(properties["general.name"])
    writer.add_file_type(properties["general.file_type"])
    writer.add_description(properties["general.description"])
    
    writer.add_bool("clip.has_text_encoder", properties["clip.has_text_encoder"])
    writer.add_bool("clip.has_vision_encoder", properties["clip.has_vision_encoder"])
    writer.add_bool("clip.has_llava_projector", properties["clip.has_llava_projector"])
    
    writer.add_uint32("clip.vision.image_size", properties["clip.vision.image_size"])
    writer.add_array("clip.vision.image_mean", properties["clip.vision.image_mean"])
    writer.add_array("clip.vision.image_std", properties["clip.vision.image_std"])
    writer.add_uint32("clip.vision.patch_size", properties["clip.vision.patch_size"])
    writer.add_uint32("clip.vision.embedding_length", properties["clip.vision.embedding_length"])
    writer.add_uint32("clip.vision.feed_forward_length", properties["clip.vision.feed_forward_length"])
    writer.add_uint32("clip.vision.projection_dim", properties["clip.vision.projection_dim"])
    writer.add_uint32("clip.vision.attention.head_count", properties["clip.vision.attention.head_count"])
    writer.add_float32("clip.vision.attention.layer_norm_epsilon", properties["clip.vision.attention.layer_norm_epsilon"])
    writer.add_uint32("clip.vision.block_count", properties["clip.vision.attention.block_count"])
    
    print(f"\n=== CLIP Vision Model Configuration ===")
    print(f"Image size: {properties['clip.vision.image_size']}")
    print(f"Patch size: {properties['clip.vision.patch_size']}")
    print(f"Hidden size: {properties['clip.vision.embedding_length']}")
    print(f"Intermediate size: {properties['clip.vision.feed_forward_length']}")
    print(f"Number of layers: {properties['clip.vision.attention.block_count']}")
    print(f"Number of attention heads: {properties['clip.vision.attention.head_count']}")
    print(f"Projection dim: {properties['clip.vision.projection_dim']}")
    print(f"\n=== Exporting Tensors ===")
    
    vision_model = model.vision_model
    
    # 1. Export class embedding
    add_tensor_to_writer(writer, "v.class_embd", vision_model.embeddings.class_embedding)
    
    # 2. Export patch embedding (Conv2d weights)
    add_tensor_to_writer(writer, "v.patch_embd.weight", vision_model.embeddings.patch_embedding.weight)
    
    # 3. Export position embedding
    add_tensor_to_writer(writer, "v.position_embd.weight", vision_model.embeddings.position_embedding.weight)
    
    # 4. Export pre-layernorm
    add_tensor_to_writer(writer, "v.pre_ln.weight", vision_model.pre_layrnorm.weight)
    add_tensor_to_writer(writer, "v.pre_ln.bias", vision_model.pre_layrnorm.bias)
    
    # 5. Export all encoder layers
    num_layers = len(vision_model.encoder.layers)
    print(f"\nExporting {num_layers} encoder blocks...")
    
    for i, layer in enumerate(vision_model.encoder.layers):
        prefix = f"v.blk.{i}"
        
        # Attention layers (Q, K, V projections)
        add_tensor_to_writer(writer, f"{prefix}.attn_q.weight", layer.self_attn.q_proj.weight)
        add_tensor_to_writer(writer, f"{prefix}.attn_q.bias", layer.self_attn.q_proj.bias)
        
        add_tensor_to_writer(writer, f"{prefix}.attn_k.weight", layer.self_attn.k_proj.weight)
        add_tensor_to_writer(writer, f"{prefix}.attn_k.bias", layer.self_attn.k_proj.bias)
        
        add_tensor_to_writer(writer, f"{prefix}.attn_v.weight", layer.self_attn.v_proj.weight)
        add_tensor_to_writer(writer, f"{prefix}.attn_v.bias", layer.self_attn.v_proj.bias)
        
        # Attention output projection
        add_tensor_to_writer(writer, f"{prefix}.attn_out.weight", layer.self_attn.out_proj.weight)
        add_tensor_to_writer(writer, f"{prefix}.attn_out.bias", layer.self_attn.out_proj.bias)
        
        # Layer norm 1
        add_tensor_to_writer(writer, f"{prefix}.ln1.weight", layer.layer_norm1.weight)
        add_tensor_to_writer(writer, f"{prefix}.ln1.bias", layer.layer_norm1.bias)
        
        # MLP / Feed-forward
        add_tensor_to_writer(writer, f"{prefix}.ffn_down.weight", layer.mlp.fc1.weight)
        add_tensor_to_writer(writer, f"{prefix}.ffn_down.bias", layer.mlp.fc1.bias)
        
        add_tensor_to_writer(writer, f"{prefix}.ffn_up.weight", layer.mlp.fc2.weight)
        add_tensor_to_writer(writer, f"{prefix}.ffn_up.bias", layer.mlp.fc2.bias)
        
        # Layer norm 2
        add_tensor_to_writer(writer, f"{prefix}.ln2.weight", layer.layer_norm2.weight)
        add_tensor_to_writer(writer, f"{prefix}.ln2.bias", layer.layer_norm2.bias)
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{num_layers} blocks...")
    
    # 6. Export post-layernorm
    add_tensor_to_writer(writer, "v.post_ln.weight", vision_model.post_layernorm.weight)
    add_tensor_to_writer(writer, "v.post_ln.bias", vision_model.post_layernorm.bias)
    
    # 7. Export visual projection
    add_tensor_to_writer(writer, "vproj.weight", model.visual_projection.weight)
    
    print("\n All tensors exported!")
    
    # Write the GGUF file
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"\n GGUF file written to: {output_path}")
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")


def convert_clip_to_gguf(model_path, output_path):
    """
    Main conversion function to convert CLIP HuggingFace model to GGUF format.
    
    Args:
        model_path: Path to HuggingFace CLIP model (local path or model ID)
        output_path: Output path for GGUF file
    """
    print(f"Loading CLIP model from: {model_path}")
    
    # Load model and processor
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    
    # Convert to float32 and CPU
    model = model.float()
    model = model.to('cpu')
    
    print(f"Model loaded successfully")
    print(f"Model type: {type(model).__name__}")
    
    # Ensure output path ends with .gguf
    if not output_path.endswith('.gguf'):
        output_path += '.gguf'
    
    # Write to GGUF
    write_vision_model_to_gguf(model, processor, output_path)
    
    print(f"\n Conversion complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CLIP HuggingFace model to GGUF format (vision encoder only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert from HuggingFace model hub
  python clip-hf2gguf.py --model_path openai/clip-vit-large-patch14 --output_path clip-vision.gguf
  
  # Convert from local directory
  python clip-hf2gguf.py --model_path /path/to/clip-model --output_path output.gguf
        """
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the HuggingFace CLIP model (local path or model ID like 'openai/clip-vit-large-patch14')"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="clip-vision.gguf",
        help="Output GGUF file path (default: clip-vision.gguf)"
    )
    
    args = parser.parse_args()
    
    # Perform conversion
    convert_clip_to_gguf(args.model_path, args.output_path)
