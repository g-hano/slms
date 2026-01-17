"""
Knowledge Distillation Data Generation Script

This script generates synthetic training data from a teacher model (e.g., Qwen3-4B) 
for knowledge distillation into smaller student models (e.g., Flow-1B, Pulse-1B).

Usage:
    python distill/generate_distillation_data.py \
        --teacher_model "Qwen2.5-Math-7B-Instruct" \
        --dataset "gsm8k" \
        --output_file "./distill_data/gsm8k_teacher_outputs.jsonl" \
        --batch_size 4 \
        --max_new_tokens 1024
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate knowledge distillation data from a teacher model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--teacher_model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="HuggingFace model ID for the teacher model"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        help="Dataset name from HuggingFace (e.g., 'gsm8k', 'math_qa')"
    )
    
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="test",
        help="Dataset split to use (e.g., 'train', 'test', 'validation')"
    )
    
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="main",
        help="Dataset configuration name (e.g., 'main' for GSM8K)"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        default="./distill_data/teacher_outputs.jsonl",
        help="Path to save the generated JSONL file"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference (reduce if OOM occurs)"
    )
    
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate per response"
    )
    
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a helpful assistant.",
        help="System prompt for the chat template"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on ('auto', 'cuda', 'cpu')"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (None for all)"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    print("="*70)
    print("Knowledge Distillation Data Generation")
    print("="*70)
    print(f"Teacher Model: {args.teacher_model}")
    print(f"Dataset: {args.dataset} ({args.dataset_config}/{args.dataset_split})")
    print(f"Output File: {args.output_file}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Tokens: {args.max_new_tokens}")
    print("="*70)
    
    # Load tokenizer
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    
    # Ensure the tokenizer has a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("⚠️  Warning: Tokenizer has no pad_token, using eos_token as fallback")
    
    # Load model
    print("\n[2/3] Loading teacher model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        torch_dtype="auto",
        device_map=args.device if args.device != "auto" else "auto"
    )
    
    device = model.device
    print(f"✓ Model loaded on device: {device}")
    
    # Load dataset
    print(f"\n[3/3] Loading dataset: {args.dataset}...")
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"✓ Using {len(dataset)} samples (limited by --max_samples)")
    else:
        print(f"✓ Loaded {len(dataset)} samples")
    
    # Start inference
    print(f"\n{'='*70}")
    print("Starting Inference...")
    print(f"{'='*70}\n")
    
    samples_processed = 0
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        # Iterate over the dataset in batches
        for i in tqdm(range(0, len(dataset), args.batch_size), desc="Generating batches"):
                        # Slice the batch
            batch = dataset[i : i + args.batch_size]

            # Prepare prompts for the batch
            texts = []
            questions = []
            ground_truths = []
            
            # batch is a dictionary of lists: {'question': [...], 'answer': [...]}
            # We iterate by index (0 to batch_size-1) instead of iterating over the batch object.
            num_items_in_batch = len(batch['question'])
            
            for k in range(num_items_in_batch):
                questions.append(batch['question'][k])
                ground_truths.append(batch['answer'][k])
                
                messages = [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": batch['question'][k]}
                ]
                
                # Apply chat template to get the string prompt
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                texts.append(text)
            
            # Tokenize the batch with padding enabled
            model_inputs = tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True
            ).to(device)
            
            # Generate
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            # Process the generated outputs
            batch_responses = []
            for j in range(len(model_inputs.input_ids)):
                # Calculate the length of the input (excluding padding)
                input_length = model_inputs.attention_mask[j].sum().item()
                
                # Slice only the generated part (skip the input prompt)
                generated_tokens = generated_ids[j][input_length:]
                
                # Decode
                response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                batch_responses.append(response)
            
            # Write results to JSONL
            for k in range(len(batch_responses)):
                result = {
                    "question": questions[k],
                    "ground_truth": ground_truths[k],
                    "teacher_response": batch_responses[k],
                    "teacher_model": args.teacher_model
                }
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                samples_processed += 1
    
    print(f"\n{'='*70}")
    print("Inference Complete!")
    print(f"{'='*70}")
    print(f"✓ Processed {samples_processed} samples")
    print(f"✓ Results saved to: {args.output_file}")
    print(f"✓ File size: {os.path.getsize(args.output_file) / (1024**2):.2f} MB")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
