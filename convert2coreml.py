"""
CoreML Conversion Script for Flow-1B and Pulse-1B Models

This script converts PyTorch models to CoreML format for deployment on Apple devices
(iPhone, iPad, Mac) with Apple Neural Engine (ANE) acceleration.

Usage:
    # Convert Flow-1B model
    python convert2coreml.py --model_path flow_1b_gsm8k.bin --model_type regular --output flow_1b
    
    # Convert Pulse-1B model
    python convert2coreml.py --model_path pulse_1b_gsm8k.bin --model_type spiking --output pulse_1b
    
    # Custom model architecture
    python convert2coreml.py --model_path custom.bin --model_type regular --d_model 1024 --num_layers 12
"""

import argparse
import torch
import coremltools as ct
from safetensors.torch import load_file
import snntorch as snn

from models.transformer.transformer import SpikingLLM, RegularLLM


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert PyTorch models to CoreML format for Apple devices",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to PyTorch model weights (.bin or .safetensors)"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["regular", "spiking"],
        required=True,
        help="Model architecture type: 'regular' (Transformer) or 'spiking' (SNN)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output filename (without extension, will save as .mlpackage)"
    )
    
    # Model architecture arguments
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32768,
        help="Vocabulary size"
    )
    
    parser.add_argument(
        "--d_model",
        type=int,
        default=2048,
        help="Model hidden dimension"
    )
    
    parser.add_argument(
        "--n_heads",
        type=int,
        default=16,
        help="Number of attention heads"
    )
    
    parser.add_argument(
        "--n_kv_heads",
        type=int,
        default=8,
        help="Number of key-value heads for GQA"
    )
    
    parser.add_argument(
        "--num_layers",
        type=int,
        default=18,
        help="Number of transformer layers"
    )
    
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Maximum sequence length"
    )
    
    # Conversion settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for conversion (usually 1 for mobile)"
    )
    
    parser.add_argument(
        "--seq_len",
        type=int,
        default=1024,
        help="Sequence length for tracing"
    )
    
    parser.add_argument(
        "--ios_version",
        type=str,
        default="iOS18",
        choices=["iOS15", "iOS16", "iOS17", "iOS18"],
        help="Minimum iOS deployment target"
    )
    
    parser.add_argument(
        "--compute_units",
        type=str,
        default="ALL",
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
        help="Compute units for inference (ALL = CPU + GPU + Neural Engine)"
    )
    
    # SNN-specific arguments
    parser.add_argument(
        "--num_steps",
        type=int,
        default=3,
        help="Number of spiking timesteps (only for spiking models)"
    )
    
    return parser.parse_args()


def patch_snn_for_coreml():
    """Patch snntorch.Leaky to make it CoreML compatible."""
    def coreml_friendly_init_leaky(self):
        return None
    
    def coreml_friendly_forward(self, input_, mem=None):
        if mem is None:
            mem = torch.zeros_like(input_)
        mem = self.beta * mem + input_
        spk = (mem > self.threshold).float()
        mem = mem - (spk * self.threshold)
        return spk, mem
    
    snn.Leaky.forward = coreml_friendly_forward
    snn.Leaky.init_leaky = coreml_friendly_init_leaky
    print("🔧 snntorch.Leaky patched for CoreML compatibility")


class CoreMLWrapper(torch.nn.Module):
    """Wrapper to extract only logits for CoreML conversion."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, input_ids):
        logits, _ = self.model(input_ids)
        return logits


def convert_to_coreml(args):
    """Main conversion function."""
    print("="*70)
    print("CoreML Conversion for Apple Devices")
    print("="*70)
    print(f"Model Path: {args.model_path}")
    print(f"Model Type: {args.model_type}")
    print(f"Output: {args.output}.mlpackage")
    print("="*70 + "\n")
    
    # Patch SNN if needed
    if args.model_type == "spiking":
        patch_snn_for_coreml()
    
    # Create model skeleton
    print("🏗️  Initializing model architecture...")
    if args.model_type == "spiking":
        model = SpikingLLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            num_steps=args.num_steps,
            dtype=torch.float32,
            device="cpu"
        )
    else:
        model = RegularLLM(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            num_layers=args.num_layers,
            max_seq_len=args.max_seq_len,
            dtype=torch.float32,
            device="cpu"
        )
    
    # Load weights
    print(f"📥 Loading weights from {args.model_path}...")
    if args.model_path.endswith(".safetensors"):
        state_dict = load_file(args.model_path)
    else:
        state_dict = torch.load(args.model_path, map_location="cpu")
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    print(f"✅ Model loaded with {model.get_num_params():,} parameters")
    
    # Create example input
    example_input_ids = torch.randint(
        0, args.vocab_size, 
        (args.batch_size, args.seq_len)
    ).long()
    
    print(f"📝 Example input shape: {example_input_ids.shape}")
    
    # Wrap and trace model
    print("🕵️  Tracing model with PyTorch JIT...")
    wrapped_model = CoreMLWrapper(model)
    traced_model = torch.jit.trace(wrapped_model, example_input_ids, check_trace=False)
    
    # Convert to CoreML
    print("🍎 Converting to CoreML format...")
    
    # Map deployment target
    ios_target_map = {
        "iOS15": ct.target.iOS15,
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
        "iOS18": ct.target.iOS18,
    }
    
    # Map compute units
    compute_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
    }
    
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="input_ids", shape=example_input_ids.shape, dtype=int)
        ],
        outputs=[
            ct.TensorType(name="logits")
        ],
        minimum_deployment_target=ios_target_map[args.ios_version],
        compute_units=compute_map[args.compute_units]
    )
    
    # Save CoreML model
    save_path = f"{args.output}.mlpackage"
    mlmodel.save(save_path)
    
    print("\n" + "="*70)
    print("✅ Conversion successful!")
    print("="*70)
    print(f"📦 Model saved to: {save_path}")
    print(f"📱 Drag and drop this file into your Xcode project")
    print(f"🚀 Deployment target: {args.ios_version}")
    print(f"⚡ Compute units: {args.compute_units}")
    print("="*70 + "\n")


def main():
    """Main execution function."""
    args = parse_args()
    convert_to_coreml(args)


if __name__ == "__main__":
    main()