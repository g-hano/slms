"""
Inference Script for Flow-1B and Pulse-1B Models

This script loads pretrained Flow-1B (Regular Transformer) or Pulse-1B (Spiking Neural Network)
models from HuggingFace and performs text generation.

Usage:
    # Interactive chat mode (default)
    python inference.py --model_type flow
    
    # Single prompt generation
    python inference.py --model_type pulse --prompt "What is 2+2?"
    
    # Batch generation from file
    python inference.py --model_type flow --prompts_file questions.txt --output results.jsonl
"""

import os
import argparse
import torch
import json
from transformers import AutoTokenizer
from tqdm import tqdm
import requests

from models.transformer.transformer import RegularLLM, SpikingLLM


MODEL_CONFIGS = {
    "flow": {
        "url": "https://huggingface.co/Chan-Y/Flow-1B-gsm8k/resolve/main/flow_1b_gsm8k.bin",
        "model_class": RegularLLM,
        "description": "Flow-1B: Regular Transformer model fine-tuned on GSM8K"
    },
    "pulse": {
        "url": "https://huggingface.co/Chan-Y/Pulse-1B-gsm8k/resolve/main/pulse_1b_gsm8k.bin",
        "model_class": SpikingLLM,
        "description": "Pulse-1B: Spiking Neural Network model fine-tuned on GSM8K"
    }
}

# Model architecture configuration
MODEL_ARCH = {
    "vocab_size": 32768,
    "d_model": 2048,
    "n_heads": 16,
    "n_kv_heads": 8,
    "num_layers": 18,
    "max_seq_len": 1024,
}


def download_weights(url, save_path):
    """Download model weights from HuggingFace."""
    print(f"📥 Downloading weights from {url}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"✅ Weights saved to {save_path}")


def load_model(model_type, device="cuda", weights_dir="./pretrained_weights"):
    """Load pretrained model and tokenizer."""
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model type. Choose from: {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_type]
    weights_path = os.path.join(weights_dir, f"{model_type}_1b_gsm8k.bin")
    
    # Download weights if not already cached
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}")
        download_weights(config["url"], weights_path)
    else:
        print(f"✅ Using cached weights from {weights_path}")
    
    # Load tokenizer
    print(f"📚 Loading tokenizer from Chan-Y/flow-pulse-tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Chan-Y/flow-pulse-tokenizer")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize model
    print(f"🏗️  Initializing {model_type.upper()} model...")
    model = config["model_class"](
        vocab_size=MODEL_ARCH["vocab_size"],
        d_model=MODEL_ARCH["d_model"],
        n_heads=MODEL_ARCH["n_heads"],
        n_kv_heads=MODEL_ARCH["n_kv_heads"],
        num_layers=MODEL_ARCH["num_layers"],
        max_seq_len=MODEL_ARCH["max_seq_len"],
        dtype=torch.bfloat16,
        device=device
    )
    
    # Load weights
    print(f"⚙️  Loading model weights...")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✅ {config['description']} loaded successfully!")
    print(f"📊 Total parameters: {model.get_num_params():,}")
    
    return model, tokenizer


@torch.no_grad()
def generate_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=512,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    device="cuda"
):
    """Generate text from a prompt using the model."""
    
    # Format prompt with chat template
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=MODEL_ARCH["max_seq_len"] - max_new_tokens
    ).to(device)
    
    input_ids = inputs["input_ids"]
    batch_size = input_ids.shape[0]
    
    generated_tokens = input_ids.clone()
    past_key_values = None
    
    # Generation loop
    for _ in range(max_new_tokens):
        # Forward pass
        if past_key_values is None:
            # First iteration: process entire prompt
            logits, past_key_values = model(input_ids, use_cache=True, past_key_values=None)
            next_token_logits = logits[:, -1, :]
        else:
            # Subsequent iterations: only process last token
            logits, past_key_values = model(input_ids[:, -1:], use_cache=True, past_key_values=past_key_values)
            next_token_logits = logits[:, -1, :]
        
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to generated sequence
        generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
        input_ids = next_token
        
        # Stop if EOS token is generated
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    # Decode only the generated part (skip the input prompt)
    input_length = inputs["input_ids"].shape[1]
    generated_text = tokenizer.decode(generated_tokens[0, input_length:], skip_special_tokens=True)
    
    return generated_text


def interactive_chat(model, tokenizer, device="cuda"):
    """Interactive chat mode."""
    print("\n" + "="*70)
    print("Interactive Chat Mode - Type 'exit' or 'quit' to stop")
    print("="*70 + "\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if prompt.lower() in ["exit", "quit", "q"]:
                print("\n👋 Goodbye!")
                break
            
            if not prompt:
                continue
            
            print("\nAssistant: ", end="", flush=True)
            response = generate_text(model, tokenizer, prompt, device=device)
            print(response)
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def batch_generate(model, tokenizer, prompts_file, output_file, device="cuda"):
    """Generate responses for prompts from a file."""
    print(f"\n📄 Reading prompts from {prompts_file}...")
    
    with open(prompts_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"✅ Found {len(prompts)} prompts")
    
    results = []
    
    print(f"\n🔄 Generating responses...")
    for i, prompt in enumerate(tqdm(prompts, desc="Processing"), 1):
        try:
            response = generate_text(model, tokenizer, prompt, device=device)
            results.append({
                "id": i,
                "prompt": prompt,
                "response": response
            })
        except Exception as e:
            print(f"\n⚠️  Error on prompt {i}: {e}")
            results.append({
                "id": i,
                "prompt": prompt,
                "response": None,
                "error": str(e)
            })
    
    # Save results
    print(f"\n💾 Saving results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"✅ Batch generation complete! Results saved to {output_file}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Inference with Flow-1B or Pulse-1B models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive chat with Flow model
  python inference.py --model_type flow
  
  # Single prompt with Pulse model
  python inference.py --model_type pulse --prompt "What is 5 * 7?"
  
  # Batch generation
  python inference.py --model_type flow --prompts_file questions.txt --output results.jsonl
  
  # Custom generation parameters
  python inference.py --model_type flow --temperature 0.9 --top_k 100
        """
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["flow", "pulse"],
        default="flow",
        help="Model to use: 'flow' (Regular Transformer) or 'pulse' (Spiking Neural Network)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for generation (if not provided, enters interactive mode)"
    )
    
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="File containing prompts (one per line) for batch generation"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="batch_results.jsonl",
        help="Output file for batch generation results"
    )
    
    parser.add_argument(
        "--weights_dir",
        type=str,
        default="./pretrained_weights",
        help="Directory to cache downloaded model weights"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on ('cuda' or 'cpu')"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (higher = more random)"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    print("="*70)
    print(f"🚀 {MODEL_CONFIGS[args.model_type]['description']}")
    print("="*70)
    
    # Load model and tokenizer
    model, tokenizer = load_model(
        args.model_type,
        device=args.device,
        weights_dir=args.weights_dir
    )
    
    # Store generation kwargs
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "device": args.device
    }
    
    # Determine mode
    if args.prompts_file:
        # Batch generation mode
        batch_generate(model, tokenizer, args.prompts_file, args.output, args.device)
    
    elif args.prompt:
        # Single prompt mode
        print(f"\n📝 Prompt: {args.prompt}\n")
        print("Assistant: ", end="", flush=True)
        response = generate_text(model, tokenizer, args.prompt, **gen_kwargs)
        print(response)
        print()
    
    else:
        # Interactive chat mode
        interactive_chat(model, tokenizer, args.device)


if __name__ == "__main__":
    main()
