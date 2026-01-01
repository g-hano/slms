# Heavily modified version of nanochat gpt.py to do diffusion
# https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
#
# Config is based on hyperparameters from Karpathy's "Let's build GPT" video
# https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
#
# Tokenizer is simple ascii mapping

"""
Simple Character-Level Discrete Diffusion Transformer with Spiking Neural Networks
Major changes from nanochat/gpt.py:
- Bidirectional attention instead of Causal (no kvcache)
- Time step conditioning added (time embeddings)
- Replace autoregressive generation with topk and confidence-aware parallel decoding
- Removed MQA/GQA (n_kv_head), simplified to standard multi-head attention
- Removed optimizer setup, FLOPs estimation, and embedding dtype casting
- Added Spiking Neural Network components using snntorch
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate


@dataclass
class DiffusionConfig:
    sequence_len: int = 256
    vocab_size: int = 128  # Full ASCII (0-127), where 0 is reserved for mask
    mask_token_id: int = 0  # NUL character used as mask token
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    diffusion_steps: int = 128
    context_len: int = 16  # Number of prefix tokens that are never masked
    
    # SNN parameters
    use_snn: bool = False  # Whether to use SNN components
    num_steps: int = 10  # Number of time steps for SNN simulation
    beta: float = 0.95  # Leak rate for leaky integrate-and-fire neuron
    threshold: float = 1.0  # Firing threshold for neurons
    spike_grad: str = "arctan"  # Surrogate gradient function


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last time into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)  # re-assemble
    out = out.to(x.dtype)  # ensure input/output dtypes match
    return out


class BidirectionalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        
    # SNN components
        # Get surrogate gradient function
        spike_grad_fn = getattr(surrogate, config.spike_grad)()
        self.spike = snn.Leaky(beta=config.beta, threshold=config.threshold, 
                                spike_grad=spike_grad_fn, init_hidden=True)
        self.num_steps = config.num_steps

    def forward(self, x, cos_sin):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)  # QK norm
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # (B, T, H, D) -> (B, H, T, D)

        # Bidirectional attention - no causal masking
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Re-assemble the heads and project back
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        
        mem = self.spike.init_leaky()
        spike_acc = torch.zeros_like(y)
        
        for _ in range(self.num_steps):
            spk, mem = self.spike(y, mem)
            spike_acc += spk
        
        # Normalize by number of steps
        y = spike_acc / self.num_steps
            
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        
        # SNN components
        # Get surrogate gradient function
        spike_grad_fn = getattr(surrogate, config.spike_grad)()
        self.spike = snn.Leaky(beta=config.beta, threshold=config.threshold, 
                                spike_grad=spike_grad_fn, init_hidden=True)
        self.num_steps = config.num_steps

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        
        # Apply SNN if enabled
        mem = self.spike.init_leaky()
        spike_acc = torch.zeros_like(x)
        
        for _ in range(self.num_steps):
            spk, mem = self.spike(x, mem)
            spike_acc += spk
        
        # Normalize by number of steps
        x = spike_acc / self.num_steps
            
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = BidirectionalAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin)
        x = x + self.mlp(norm(x))
        return x


def encode_text(text):
    """Convert text to vocab indices using direct ASCII mapping"""
    tokens = torch.tensor([min(ord(c), 127) for c in text], dtype=torch.long)
    return tokens


def decode_tokens(tokens):
    """Convert vocab indices to text using direct ASCII mapping"""
    text = "".join([chr(int(t)) for t in tokens])
    return text