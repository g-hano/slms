import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from .layers import SpikingSwiGLU, SpikingGroupedSlidingAttention, PreRMSNorm, RegularGroupedSlidingAttention, SwiGLU

class RegularTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, intermediate_size=None, dropout=0.0,
                 max_seq_len=2048, rope_theta_local=1e4, rope_theta_global=1e6,
                 window_size=128, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.norm1 = PreRMSNorm(d_model, dtype=dtype, device=device)
        self.norm2 = PreRMSNorm(d_model, dtype=dtype, device=device)
        self.attn = RegularGroupedSlidingAttention(
            d_model, n_heads, n_kv_heads, dropout=dropout, max_seq_len=max_seq_len,
            rope_theta_local=rope_theta_local, rope_theta_global=rope_theta_global,
            window_size=window_size, dtype=dtype, device=device
        )
        hidden_dim = intermediate_size or d_model * 4
        self.ffn = SwiGLU(d_model, hidden_dim, dtype=dtype, device=device)

    def forward(self, x, use_cache=False, past_key_value=None, layer_idx=0):
        if self.training and not use_cache:
            return checkpoint(self._forward_impl, x, use_cache, past_key_value, layer_idx, use_reentrant=False)
        return self._forward_impl(x, use_cache, past_key_value, layer_idx)

    def _forward_impl(self, x, use_cache, past_key_value, layer_idx):
        x_norm = self.norm1(x)
        attn_out, present_kv = self.attn(x_norm, use_cache=use_cache, past_key_value=past_key_value, layer_idx=layer_idx)
        x = x + attn_out
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        return x, present_kv

class RegularLLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_kv_heads, num_layers,
                 max_seq_len=2048, rope_theta_local=1e4, rope_theta_global=1e6,
                 window_size=128, dropout=0.0, intermediate_size=None,
                 dtype=torch.bfloat16, device="cuda"):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model, dtype=dtype, device=device)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            RegularTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                max_seq_len=max_seq_len,
                rope_theta_local=rope_theta_local,
                rope_theta_global=rope_theta_global,
                window_size=window_size,
                dtype=dtype, 
                device=device
            )
            for _ in range(num_layers)
        ])

        self.final_norm = PreRMSNorm(d_model, dtype=dtype, device=device)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, dtype=dtype, device=device)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('o_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)  

    def get_num_params(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, use_cache=False, past_key_values=None):
        """
        input_ids: [batch, seq_len]
        past_key_values: list of (k, v) tuples per layer if use_cache=True
        """
        x = self.token_emb(input_ids)  # [batch, seq_len, d_model]
        x = self.dropout(x)

        presents = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = layer(x, use_cache=use_cache, past_key_value=past_kv, layer_idx=i)
            presents.append(present_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        return logits, presents

class SpikingTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, intermediate_size=None, dropout=0.0,
                 max_seq_len=2048, rope_theta_local=1e4, rope_theta_global=1e6,
                 window_size=128, beta=0.95, num_steps=3,
                 dtype=torch.bfloat16, device="cuda"):
        super().__init__()

        self.norm1 = PreRMSNorm(d_model, dtype=dtype, device=device)
        self.norm2 = PreRMSNorm(d_model, dtype=dtype, device=device)

        self.attn = SpikingGroupedSlidingAttention(
            d_model, n_heads, n_kv_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            rope_theta_local=rope_theta_local,
            rope_theta_global=rope_theta_global,
            window_size=window_size, beta=beta,
            num_steps=num_steps,
            dtype=dtype, device=device
        )

        hidden_dim = intermediate_size or d_model * 4
        self.ffn = SpikingSwiGLU(input_dim=d_model, hidden_dim=hidden_dim, beta=beta, num_steps=num_steps, dtype=dtype, device=device)

        self.attn_gate = nn.Parameter(torch.ones(1, dtype=dtype, device=device))
        self.ffn_gate = nn.Parameter(torch.ones(1, dtype=dtype, device=device))

    def forward(self, x, use_cache=False, past_key_value=None, layer_idx=0):
        #if self.training and not use_cache:
        #    return checkpoint(self._forward_impl, x, use_cache, past_key_value, layer_idx, use_reentrant=False)
        return self._forward_impl(x, use_cache, past_key_value, layer_idx)

    def _forward_impl(self, x, use_cache, past_key_value, layer_idx):
        x_norm = self.norm1(x)
        attn_out, present_kv = self.attn(x_norm, use_cache=use_cache, past_key_value=past_key_value, layer_idx=layer_idx)
        x = x + self.attn_gate * attn_out
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.ffn_gate * ffn_out
        return x, present_kv

class SpikingLLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_kv_heads, num_layers,
                 max_seq_len=2048, rope_theta_local=1e4, rope_theta_global=1e6,
                 window_size=128, dropout=0.0, beta=0.95, num_steps=3, intermediate_size=None,
                 dtype=torch.bfloat16, device="cuda"):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model, dtype=dtype, device=device)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            SpikingTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                max_seq_len=max_seq_len,
                rope_theta_local=rope_theta_local,
                rope_theta_global=rope_theta_global,
                window_size=window_size,
                beta=beta, num_steps=num_steps,
                dtype=dtype, 
                device=device
            )
            for _ in range(num_layers)
        ])

        self.final_norm = PreRMSNorm(d_model, dtype=dtype, device=device)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, dtype=dtype, device=device)
        self.lm_head.weight = self.token_emb.weight
        
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight') or pn.endswith('o_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * num_layers))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())

    def forward(self, input_ids, use_cache=False, past_key_values=None):
        """
        input_ids: [batch, seq_len]
        past_key_values: list of (k, v) tuples per layer if use_cache=True
        """
        x = self.token_emb(input_ids)  # [batch, seq_len, d_model]
        x = self.dropout(x)

        presents = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = layer(x, use_cache=use_cache, past_key_value=past_kv, layer_idx=i)
            presents.append(present_kv)

        x = self.final_norm(x)
        x = x.to(self.lm_head.weight.dtype)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        return logits, presents