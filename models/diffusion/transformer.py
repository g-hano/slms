import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from ..transformer.layers import norm, Block

class DiffusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token and time embeddings (include mask token in vocab)
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.time_emb = nn.Embedding(config.diffusion_steps, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Output head to predict denoised tokens
        self.output_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # SNN components for output head
        # Get surrogate gradient function
        spike_grad_fn = getattr(surrogate, config.spike_grad)()
        self.spike = snn.Leaky(beta=config.beta, threshold=config.threshold, 
                                spike_grad=spike_grad_fn, init_hidden=True)
        self.num_steps = config.num_steps

        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 2
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # Zero out output head weights
        torch.nn.init.zeros_(self.output_head.weight)
        # Zero out c_proj weights in all blocks
        for block in self.blocks:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # Init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.token_emb.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )  # add batch and head dims
        return cos, sin

    def get_device(self):
        return self.token_emb.weight.device

    def forward(self, x_t, t):
        """
        Forward pass for diffusion model
        Args:
            x_t: Noisy tokens at timestep t, shape (B, T)
            t: Timestep indices, shape (B,)
        Returns:
            logits: Predicted token logits, shape (B, T, vocab_size)
        """
        B, T = x_t.size()

        # Get embeddings
        x = self.token_emb(x_t)  # (B, T, n_embd)
        t_emb = self.time_emb(t)  # (B, n_embd)

        # Add time embedding to all positions
        x = x + t_emb.unsqueeze(1)  # broadcast time embedding across sequence
        x = norm(x)

        # Get rotary embeddings
        assert T <= self.cos.size(1)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])

        # Forward through transformer blocks
        for block in self.blocks:
            x = block(x, cos_sin)
        x = norm(x)

        # Predict denoised tokens
        logits = self.output_head(x)  # (B, T, vocab_size)
        
        # Apply SNN if enabled
        mem = self.spike.init_leaky()
        spike_acc = torch.zeros_like(logits)
        
        for _ in range(self.num_steps):
            spk, mem = self.spike(logits, mem)
            spike_acc += spk
        
        # Normalize by number of steps
        logits = spike_acc / self.num_steps
            
        return logits

    @torch.inference_mode()
    def sample_topk(
        self,
        batch_size,
        seq_len,
        k,
        num_steps=None,
        temperature=1.0,
        device=None,
        context_tokens=None,
    ):
        """
        Generate samples using top-K parallel decoding (LLaDA baseline).
        At each step, decode exactly K tokens with highest confidence.

        Args:
            batch_size: Number of samples to generate
            seq_len: Length of sequences to generate
            k: Number of tokens to decode per step
            num_steps: Maximum number of denoising steps
            temperature: Sampling temperature
            device: Device to generate on
            context_tokens: Optional context tokens for conditioning, shape (batch_size, context_len)
        Returns:
            samples: Generated token sequences, shape (batch_size, seq_len)
        """
        if device is None:
            device = self.get_device()
        if num_steps is None:
            num_steps = seq_len  # Maximum possible steps

        # Start from all mask tokens
        x = torch.full(
            (batch_size, seq_len),
            self.config.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        # If context tokens provided, set them in the first context_len positions
        if context_tokens is not None:
            context_len = context_tokens.size(1)
            x[:, :context_len] = context_tokens.to(device)

        # Track which positions are still masked
        masked_positions = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=device
        )
        if context_tokens is not None:
            masked_positions[:, :context_len] = False

        # Decode step by step
        for step in range(num_steps):
            # Check if all tokens are decoded
            if not masked_positions.any():
                break

            # Create timestep (use step as proxy for timestep)
            t_batch = torch.full((batch_size,), step, device=device, dtype=torch.long)
            t_batch = torch.clamp(t_batch, 0, self.config.diffusion_steps - 1)

            # Predict tokens
            logits = self.forward(x, t_batch)

            # Get confidence scores (max probability for each position)
            probs = F.softmax(logits / temperature, dim=-1)
            confidences, predicted_tokens = torch.max(probs, dim=-1)  # (B, T)

            # Mask out already-decoded positions
            confidences = confidences.masked_fill(~masked_positions, -float("inf"))

            # Select top-K positions per batch
            k_actual = min(k, masked_positions.sum(dim=1).max().item())
            _, topk_indices = torch.topk(confidences, k=k_actual, dim=1)  # (B, K)

            # Update the top-K positions
            for b in range(batch_size):
                for idx in topk_indices[b]:
                    if masked_positions[b, idx]:
                        x[b, idx] = predicted_tokens[b, idx]
                        masked_positions[b, idx] = False

        return x

    @torch.inference_mode()
    def sample_confidence(
        self,
        batch_size,
        seq_len,
        confidence_threshold=0.95,
        num_steps=None,
        temperature=1.0,
        device=None,
        context_tokens=None,
    ):
        """
        Generate samples using confidence-aware parallel decoding (Fast-dLLM).
        At each step, decode all tokens whose confidence exceeds a threshold.

        Args:
            batch_size: Number of samples to generate
            seq_len: Length of sequences to generate
            confidence_threshold: Threshold τ for token acceptance
            num_steps: Maximum number of denoising steps
            temperature: Sampling temperature
            device: Device to generate on
            context_tokens: Optional context tokens for conditioning, shape (batch_size, context_len)
        Returns:
            samples: Generated token sequences, shape (batch_size, seq_len)
        """
        if device is None:
            device = self.get_device()
        if num_steps is None:
            num_steps = seq_len  # Maximum possible steps

        # Start from all mask tokens
        x = torch.full(
            (batch_size, seq_len),
            self.config.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        # If context tokens provided, set them in the first context_len positions
        if context_tokens is not None:
            context_len = context_tokens.size(1)
            x[:, :context_len] = context_tokens.to(device)

        # Track which positions are still masked
        masked_positions = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=device
        )
        if context_tokens is not None:
            masked_positions[:, :context_len] = False

        # Decode step by step
        for step in range(num_steps):
            # Check if all tokens are decoded
            if not masked_positions.any():
                break

            # Create timestep (use step as proxy for timestep)
            t_batch = torch.full((batch_size,), step, device=device, dtype=torch.long)
            t_batch = torch.clamp(t_batch, 0, self.config.diffusion_steps - 1)

            # Predict tokens
            logits = self.forward(x, t_batch)

            # Get confidence scores (max probability for each position)
            probs = F.softmax(logits / temperature, dim=-1)
            confidences, predicted_tokens = torch.max(probs, dim=-1)  # (B, T)

            # Select positions above threshold (only among masked positions)
            above_threshold = (confidences >= confidence_threshold) & masked_positions

            # Ensure at least one token is decoded per batch if any remain masked
            for b in range(batch_size):
                if masked_positions[b].any() and not above_threshold[b].any():
                    # Decode the highest confidence masked token
                    masked_confidences = confidences[b].clone()
                    masked_confidences[~masked_positions[b]] = -float("inf")
                    best_idx = torch.argmax(masked_confidences)
                    above_threshold[b, best_idx] = True

            # Update positions above threshold
            x = torch.where(above_threshold, predicted_tokens, x)
            masked_positions = masked_positions & ~above_threshold

        return x

    @torch.inference_mode()
    def sample(
        self,
        batch_size,
        seq_len,
        num_steps=None,
        temperature=1.0,
        device=None,
        context_tokens=None,
        method="confidence",
        k=None,
        confidence_threshold=0.95,
    ):
        """
        Generate samples using parallel decoding methods.

        Args:
            batch_size: Number of samples to generate
            seq_len: Length of sequences to generate
            num_steps: Maximum number of denoising steps
            temperature: Sampling temperature
            device: Device to generate on
            context_tokens: Optional context tokens for conditioning, shape (batch_size, context_len)
            method: Decoding method - 'topk' or 'confidence'
            k: Number of tokens per step (for 'topk' method)
            confidence_threshold: Confidence threshold τ (for 'confidence' method)
        Returns:
            samples: Generated token sequences, shape (batch_size, seq_len)
        """
        if method == "topk":
            if k is None:
                k = max(1, seq_len // 10)  # Default: decode 10% per step
            return self.sample_topk(
                batch_size, seq_len, k, num_steps, temperature, device, context_tokens
            )
        elif method == "confidence":
            return self.sample_confidence(
                batch_size,
                seq_len,
                confidence_threshold,
                num_steps,
                temperature,
                device,
                context_tokens,
            )
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def configure_optimizers(self, weight_decay=0.1, learning_rate=1e-3, 
                            use_muon=True, adamw_params=None, muon_params=None):
        """
        Configure optimizers for the model.
        
        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            use_muon: Whether to use Muon optimizer for 2D parameters
            adamw_params: Additional parameters for AdamW optimizer
            muon_params: Additional parameters for Muon optimizer
            
        Returns:
            optimizer: Configured optimizer
        """
        if adamw_params is None:
            adamw_params = {}
        if muon_params is None:
            muon_params = {}
            
        # Separate parameters into groups
        decay_params = []
        no_decay_params = []
        muon_params_list = []
        
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
                
            # Skip biases and layernorms for weight decay
            if name.endswith('.bias') or 'norm' in name.lower():
                no_decay_params.append(param)
            # Use Muon for 2D parameters (weight matrices)
            elif use_muon and param.dim() == 2 and 'emb' not in name.lower():
                muon_params_list.append(param)
            else:
                decay_params.append(param)
        
        # Create parameter groups
        optimizer_groups = []
        
        # Add Muon parameters
        if use_muon and muon_params_list:
            muon_group = {
                'params': muon_params_list,
                'lr': learning_rate,
                'weight_decay': weight_decay,
                **muon_params
            }
            optimizer_groups.append(muon_group)
        
        # Add AdamW parameters with weight decay
        if decay_params:
            decay_group = {
                'params': decay_params,
                'lr': learning_rate,
                'weight_decay': weight_decay,
                **adamw_params
            }
            optimizer_groups.append(decay_group)
        
        # Add AdamW parameters without weight decay
        if no_decay_params:
            no_decay_group = {
                'params': no_decay_params,
                'lr': learning_rate,
                'weight_decay': 0.0,
                **adamw_params
            }
            optimizer_groups.append(no_decay_group)
        
        # Create optimizer
        if use_muon and muon_params_list:
            # Try to import Muon optimizer
            try:
                from torch.optim import Muon
                muon_optimizer = Muon([g for g in optimizer_groups if 'params' in g and 
                                      any(p.dim() == 2 for p in g['params'])])
                
                # Create AdamW for remaining parameters
                remaining_params = []
                for group in optimizer_groups:
                    if 'params' in group and not any(p.dim() == 2 for p in group['params']):
                        remaining_params.extend(group['params'])
                
                if remaining_params:
                    adamw_optimizer = torch.optim.AdamW(remaining_params, lr=learning_rate, 
                                                        weight_decay=weight_decay, **adamw_params)
                    # Return a tuple of optimizers
                    return muon_optimizer, adamw_optimizer
                else:
                    return muon_optimizer
            except ImportError:
                print("Warning: Muon optimizer not available. Using AdamW for all parameters.")
                use_muon = False
        
        # Fallback to AdamW for all parameters
        return torch.optim.AdamW(optimizer_groups, lr=learning_rate, 
                                weight_decay=weight_decay, **adamw_params)

