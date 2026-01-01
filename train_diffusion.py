"""
Training script for character-level discrete diffusion model with SNN
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.diffusion.transformer import DiffusionTransformer
from models.diffusion.layers import DiffusionConfig
from torch.optim import AdamW

# Import Muon from the provided implementation
from torch.optim import Muon

class DiscreteNoiseSchedule:
    """
    Simple noise schedule for discrete diffusion.
    At each timestep, we have a probability of replacing a token with a random token.
    """

    def __init__(self, num_timesteps, vocab_size):
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size

        # Linear schedule: probability of corruption increases linearly
        self.corruption_probs = torch.linspace(0.0, 0.95, num_timesteps)

    def add_noise(self, x_0, t):
        """
        Add noise to clean tokens x_0 at timestep t
        Args:
            x_0: Clean tokens, shape (B, T)
            t: Timestep indices, shape (B,)
        Returns:
            x_t: Noisy tokens at timestep t
        """
        B, T = x_0.shape
        device = x_0.device

        # Get corruption probability for each sample (index on CPU, then move to device)
        corruption_prob = self.corruption_probs[t.cpu()].to(device)  # (B,)

        # Create mask: which tokens to corrupt
        mask = torch.rand(B, T, device=device) < corruption_prob.unsqueeze(1)  # (B, T)

        # Generate random tokens
        random_tokens = torch.randint(0, self.vocab_size, (B, T), device=device)

        # Replace masked positions with random tokens
        x_t = torch.where(mask, random_tokens, x_0)

        return x_t


def get_data_loader(data_path, batch_size, seq_len, device, num_workers=0):
    """
    Data loader for text data (supports single file or directory)
    Args:
        data_path: Path to text file or directory containing text files
        batch_size: Batch size
        seq_len: Sequence length
        device: Device to load data on
        num_workers: Number of worker threads (0 for single-threaded)
    Returns:
        generator: Data generator
        num_batches: Total number of batches (for epoch calculation)
    """
    import os
    
    # Check if path is directory or file
    if os.path.isdir(data_path):
        # Load all .txt files from directory
        text_files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.txt')])
        print(f"Loading {len(text_files)} text files from {data_path}")
        text = ""
        for txt_file in text_files:
            print(f"  - Loading: {os.path.basename(txt_file)}")
            with open(txt_file, "r", encoding="utf-8") as f:
                text += f.read() + "\n\n"  # Add spacing between books
    else:
        # Load single file
        print(f"Loading single file: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()

    print(f"Total characters loaded: {len(text):,}")
    
    # Convert to tokens (simple ASCII encoding)
    tokens = torch.tensor([min(ord(c), 127) for c in text], dtype=torch.long)
    print(f"Total tokens: {len(tokens):,}")

    # Calculate number of complete batches
    total_samples = len(tokens) // seq_len
    num_batches = total_samples // batch_size
    
    # Trim to exact batch size
    tokens = tokens[: num_batches * batch_size * seq_len]
    tokens = tokens.view(batch_size, -1)
    
    print(f"Batches per epoch: {num_batches:,}")
    print(f"Training samples: {num_batches * batch_size:,}")

    # Generator function
    def data_generator():
        while True:
            for i in range(0, tokens.size(1) - seq_len, seq_len):
                batch = tokens[:, i : i + seq_len].to(device)
                yield batch

    return data_generator(), num_batches


def train_step(model, x_0, noise_schedule, optimizer_muon, optimizer_adamw):
    """
    Single training step
    Args:
        model: SNNDiffusionTransformer model
        x_0: Clean tokens, shape (B, T)
        noise_schedule: Noise schedule object
        optimizer_muon: Muon optimizer for embeddings
        optimizer_adamw: AdamW optimizer for the rest
    Returns:
        loss: Training loss
    """
    B, T = x_0.shape
    device = x_0.device

    # Sample random timesteps
    t = torch.randint(0, noise_schedule.num_timesteps, (B,), device=device)

    # Add noise to get x_t
    x_t = noise_schedule.add_noise(x_0, t)

    # Reset SNN memories before forward pass (important for SNNs!)
    model.reset_memories()

    # Forward pass: predict the original tokens
    # Single forward pass (like in your working train.py)
    logits = model(x_t, t)

    # Compute loss: cross-entropy between predicted and original tokens
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), x_0.view(-1), reduction="mean"
    )

    # Backward pass
    optimizer_muon.zero_grad()
    optimizer_adamw.zero_grad()
    loss.backward()
    
    # Clip gradients to prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Update parameters
    optimizer_muon.step()
    optimizer_adamw.step()

    return loss.item()


def train(
    model,
    data_loader,
    noise_schedule,
    optimizer_muon,
    optimizer_adamw,
    num_batches_per_epoch,
    num_epochs=1,
    sample_interval=500):
    """
    Main training loop for specified number of epochs
    Args:
        num_batches_per_epoch: Number of batches in one epoch
        num_epochs: Number of epochs to train
    """
    model.train()

    best_loss = float("inf")
    total_steps = num_batches_per_epoch * num_epochs

    pbar = tqdm(range(total_steps), desc="Training")
    for step in pbar:
        # Calculate current epoch
        current_epoch = (step // num_batches_per_epoch) + 1
        batch_in_epoch = (step % num_batches_per_epoch) + 1
        
        # Get batch (keep as long integers for embedding layer)
        x_0 = next(data_loader)

        # Training step
        loss = train_step(model, x_0, noise_schedule, optimizer_muon, optimizer_adamw)

        # Track best loss
        if loss < best_loss:
            best_loss = loss

        # Update progress bar with more info
        pbar.set_postfix({
            "epoch": f"{current_epoch}/{num_epochs}",
            "batch": f"{batch_in_epoch}/{num_batches_per_epoch}",
            "loss": f"{loss:.4f}", 
            "best": f"{best_loss:.4f}",
            "ppl": f"{torch.exp(torch.tensor(loss)).item():.2f}"  # Perplexity
        })

        # Sample generation
        if (step + 1) % sample_interval == 0:
            model.eval()
            with torch.no_grad():
                samples = model.sample(
                    batch_size=1,
                    seq_len=256,
                    num_steps=None,  # Use all timesteps
                    temperature=1.0,
                    device=model.get_device(),
                )
                # Decode samples
                text = "".join([chr(min(int(c), 127)) for c in samples[0]])
                tqdm.write(f"\n--- Sample at epoch {current_epoch}, step {step + 1} ---")
                tqdm.write(text[:500])  # Show first 500 chars
                tqdm.write("--- End sample ---\n")
            model.train()
    
    print(f"\nTraining completed! {num_epochs} epoch(s), {total_steps} steps total")
    print(f"Final loss: {loss:.4f}, Best loss: {best_loss:.4f}")


def print_model_details(model, config):
    """Print detailed model statistics"""
    print("\n" + "="*70)
    print("MODEL DETAILS")
    print("="*70)
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count embedding parameters
    embedding_params = sum(p.numel() for p in model.token_emb.parameters()) + \
                      sum(p.numel() for p in model.time_emb.parameters())
    
    # Architecture type
    arch_type = "SNN" if config.use_snn else "Regular"
    if config.use_moe:
        arch_type += " + MoE"
    else:
        arch_type += " Dense"
    
    print(f"Architecture Type:       {arch_type}")
    print(f"Using SNNs:              {'Yes' if config.use_snn else 'No (Regular activations)'}")
    
    # Handle MoE vs Dense architecture differently
    if config.use_moe:
        # MoE: Calculate active params (non-expert + active experts per layer)
        params_per_expert = sum(p.numel() for p in model.blocks[0].mlp.experts[0].parameters())
        total_expert_params = params_per_expert * config.num_experts * config.n_layer
        active_expert_params = params_per_expert * config.top_k * config.n_layer
        non_expert_params = total_params - total_expert_params
        active_params = non_expert_params + active_expert_params
        
        print(f"Total Parameters:        {total_params:>15,} ({total_params/1e6:.2f}M)")
        print(f"Trainable Parameters:    {trainable_params:>15,} ({trainable_params/1e6:.2f}M)")
        print(f"Active Parameters:       {active_params:>15,} ({active_params/1e6:.2f}M)")
        print(f"\nParameter Breakdown:")
        print(f"  Non-Expert Params:     {non_expert_params:>15,} ({non_expert_params/1e6:.2f}M)")
        print(f"  Total Expert Params:   {total_expert_params:>15,} ({total_expert_params/1e6:.2f}M)")
        print(f"  Active Expert Params:  {active_expert_params:>15,} ({active_expert_params/1e6:.2f}M)")
        print(f"  Embedding Params:      {embedding_params:>15,} ({embedding_params/1e6:.2f}M)")
    else:
        # Dense: All parameters are active
        print(f"Total Parameters:        {total_params:>15,} ({total_params/1e6:.2f}M)")
        print(f"Trainable Parameters:    {trainable_params:>15,} ({trainable_params/1e6:.2f}M)")
        print(f"\nParameter Breakdown:")
        print(f"  Embedding Params:      {embedding_params:>15,} ({embedding_params/1e6:.2f}M)")
        print(f"  Other Params:          {total_params - embedding_params:>15,} ({(total_params - embedding_params)/1e6:.2f}M)")
    
    # Model architecture
    print(f"\nArchitecture:")
    print(f"  Embedding Dimension:   {config.n_embd:>15,}")
    print(f"  Number of Layers:      {config.n_layer:>15,}")
    print(f"  Number of Heads:       {config.n_head:>15,}")
    print(f"  Sequence Length:       {config.sequence_len:>15,}")
    print(f"  Vocab Size:            {config.vocab_size:>15,}")
    if config.use_moe:
        print(f"  Number of Experts:     {config.num_experts:>15,}")
        print(f"  Active Experts (top-k):{config.top_k:>15,}")
    print(f"  Max Timesteps:         {config.max_timesteps:>15,}")
    
    # Data types
    print(f"\nData Types:")
    sample_param = next(model.parameters())
    param_dtype = sample_param.dtype
    print(f"  Dtype: {param_dtype}")
    print(f"  Device: {sample_param.device}")
    
    # Memory estimate based on actual dtype
    bytes_per_param = 2 if param_dtype in [torch.float16, torch.bfloat16] else 4
    param_memory_mb = total_params * bytes_per_param / (1024**2)
    
    print(f"\nEstimated Memory ({param_dtype}):")
    print(f"  Total Parameters:      {param_memory_mb:>15.2f} MB ({param_memory_mb/1024:>6.2f} GB)")
    
    # Only show active memory for MoE models
    if config.use_moe:
        active_memory_mb = active_params * bytes_per_param / (1024**2)
        print(f"  Active Parameters:     {active_memory_mb:>15.2f} MB ({active_memory_mb/1024:>6.2f} GB)")
    
    print("="*70 + "\n")


def main():
    # Training configuration
    batch_size = 32
    block_size = 256
    num_epochs = 1  # Train for 1 epoch (no max step limit)
    eval_interval = 500
    learning_rate = 1e-3
    n_embd = 896
    n_head = 16
    n_layer = 10
    num_steps = 16  # Number of diffusion steps
    
    # Architecture configuration
    use_snn = True  # USE REGULAR ACTIVATIONS (set to True for SNN)
    use_moe = False  # Whether to use Mixture of Experts
    num_experts = 8  # Number of experts in MoE (only used if use_moe=True)
    top_k = 2  # Number of experts to activate (only used if use_moe=True)

    # Configuration
    config = DiffusionConfig(
        sequence_len=block_size,
        vocab_size=128,  # ASCII
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        max_timesteps=num_steps,  # Number of diffusion steps
        use_snn=use_snn,  # Use SNN or regular activations
        use_moe=use_moe,  # Use MoE or dense architecture
        num_experts=num_experts,
        top_k=top_k,
        beta=0.90,  # SNN decay parameter (only used if use_snn=True)
    )

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Model
    model = SNNDiffusionTransformer(config).to(device)
    model.init_weights()
    
    # Convert to bfloat16 for faster training and reduced memory
    model = model.to(torch.bfloat16)
    print(f"Model dtype: {next(model.parameters()).dtype}")

    # Print detailed model information
    print_model_details(model, config)

    # Noise schedule
    noise_schedule = DiscreteNoiseSchedule(
        num_timesteps=config.max_timesteps, vocab_size=config.vocab_size
    )

    # Separate parameters for different optimizers
    embedding_params = list(model.token_emb.parameters()) + list(model.time_emb.parameters())
    other_params = [p for n, p in model.named_parameters() if not any(n.startswith(ep) for ep in ["token_emb", "time_emb"])]

    # Optimizers
    # Use Andrej Karpathy's Muon implementation for embeddings
    optimizer_muon = Muon(embedding_params, lr=learning_rate)
    optimizer_adamw = AdamW(other_params, lr=learning_rate, weight_decay=0.01)

    # Data loader - Load all Harry Potter books
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    data_loader, num_batches = get_data_loader(
        data_path="C:/Users/Cihan/Desktop/llamaindex/text-diff-snn/harry-potter-books",
        batch_size=batch_size,
        seq_len=config.sequence_len,
        device=device,
    )
    print("="*70 + "\n")

    # Train
    print("Starting training for {} epoch(s)...\n".format(num_epochs))
    train(
        model=model,
        data_loader=data_loader,
        noise_schedule=noise_schedule,
        optimizer_muon=optimizer_muon,
        optimizer_adamw=optimizer_adamw,
        num_batches_per_epoch=num_batches,
        num_epochs=num_epochs,
        sample_interval=eval_interval,
    )
    from datetime import datetime
    # Save model
    torch.save(model.state_dict(), f"C:/Users/Cihan/Desktop/llamaindex/text-diff-snn/snn_diffusion_model-{use_snn}-{use_moe}-{datetime.now().strftime('%Y%m%d%H%M%S')}.pt")
    print(f"Model saved to C:/Users/Cihan/Desktop/llamaindex/text-diff-snn/snn_diffusion_model-{use_snn}-{use_moe}-{datetime.now().strftime('%Y%m%d%H%M%S')}.pt")


if __name__ == "__main__":
    main()