"""
Knowledge Distillation Training Script

This script fine-tunes a pretrained student model (Flow-1B or Pulse-1B) on teacher-generated
reasoning traces for knowledge distillation.

Usage:
    accelerate launch distill/distil.py \\
        --cache_file "./distill_data/gsm8k_teacher_outputs.jsonl" \\
        --student_checkpoint "./checkpoints/phase3_final/pytorch_model.bin" \\
        --tokenizer_id "YOUR_USERNAME/flow-pulse-tokenizer" \\
        --output_dir "./checkpoints_distilled" \\
        --model_type "regular" \\
        --batch_size 2 \\
        --epochs 3
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from accelerate import Accelerator
from tqdm import tqdm

from models.transformer.transformer import RegularLLM, SpikingLLM


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune student model with knowledge distillation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        "--cache_file",
        type=str,
        default="./teacher_outputs.jsonl",
        help="Path to teacher outputs JSONL file"
    )
    
    # Model arguments
    parser.add_argument(
        "--student_checkpoint",
        type=str,
        default="./model_weights/best_phase2_1024/pytorch_model.bin",
        help="Path to pretrained student model weights"
    )
    
    parser.add_argument(
        "--tokenizer_id",
        type=str,
        default="Chan-Y/flow-pulse-tokenizer",
        help="HuggingFace tokenizer ID"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./flow_trained_model",
        help="Directory to save trained models"
    )
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="spiking",
        choices=["regular", "spiking"],
        help="Model architecture type: 'regular' or 'spiking'"
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
        "--max_length",
        type=int,
        default=1024,
        help="Maximum sequence length"
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size per device"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )
    
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps for learning rate scheduler"
    )
    
    return parser.parse_args()

class SFTDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: 
                    continue
                self.examples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        q = item["question"]
        a = item["teacher_answer"]

        prompt = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        answer = f"{a}"
        full_text = prompt + answer

        enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        prompt_enc = self.tokenizer(prompt, truncation=True, max_length=self.max_length)

        input_ids = torch.tensor(enc["input_ids"])
        labels = input_ids.clone()

        prompt_len = len(prompt_enc["input_ids"])
        labels[:prompt_len] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return input_ids, labels


def train(args):
    """Main training function."""
    print("="*70)
    print("Knowledge Distillation Training")
    print("="*70)
    print(f"Cache File: {args.cache_file}")
    print(f"Student Checkpoint: {args.student_checkpoint}")
    print(f"Tokenizer ID: {args.tokenizer_id}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Model Type: {args.model_type}")
    print(f"Max Length: {args.max_length}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Gradient Accumulation: {args.gradient_accumulation_steps}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print("="*70)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        mixed_precision="bf16"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_cls = SpikingLLM if args.model_type == "spiking" else RegularLLM
    
    model = model_cls(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            num_layers=args.num_layers, 
            max_seq_len=args.max_length,
        )

    if os.path.exists(args.student_checkpoint):
        state_dict = torch.load(args.student_checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        print("✅ Pretrained weights loaded.")

    model = model.to(torch.bfloat16)

    dataset = SFTDataset(args.cache_file, tokenizer, args.max_length)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    num_training_steps = len(train_loader) * args.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    optimizer.zero_grad(set_to_none=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        valid_steps = 0

        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch}", 
            disable=not accelerator.is_local_main_process
        )

        for step, (input_ids, labels) in enumerate(progress_bar):
            if (labels == -100).all():
                continue

            with accelerator.accumulate(model):
                logits, _ = model(input_ids.long())

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

                # Capture loss value for logging BEFORE cleanup
                current_loss = loss.item()

                if torch.isnan(loss):
                    print(f"⚠️ NaN loss at step {step}, skipping")
                    del loss, logits, shift_logits, shift_labels
                    torch.cuda.empty_cache()
                    continue

                accelerator.backward(loss)

                # 1. Delete heavy tensors to free GPU memory
                del loss
                del logits
                del shift_logits
                del shift_labels
                
                # 2. Optimizer step
                optimizer.step()
                lr_scheduler.step()
                
                # 3. Zero gradients only on sync (last accumulation step)
                if accelerator.sync_gradients:
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                    # Logging and stats update only at the end of accumulation
                    epoch_loss += current_loss
                    valid_steps += 1

                    if step % 100 == 0 and accelerator.is_main_process:
                        print(
                            f" Step {step} | Loss: {current_loss:.4f} | "
                            f"LR: {lr_scheduler.get_last_lr()[0]:.2e}"
                        )

        avg_epoch_loss = epoch_loss / max(1, valid_steps)

        if accelerator.is_main_process:
            print(f"📉 Epoch {epoch} avg loss: {avg_epoch_loss:.4f}")

            os.makedirs(args.output_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)

            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_path = f"{args.output_dir}/{args.model_type}_1b_gsm8k_best.bin"
                torch.save(unwrapped.state_dict(), best_path)
                print(f"🏆 New best model saved → {best_path}")

    if accelerator.is_main_process:
        final_path = f"{args.output_dir}/{args.model_type}_1b_gsm8k_final.bin"
        unwrapped = accelerator.unwrap_model(model)
        torch.save(unwrapped.state_dict(), final_path)
        print(f"🏁 Final model saved → {final_path}")


def main():
    """Main execution function."""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()