import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn

def swish(x1):
    return x1 * torch.sigmoid(x1)

class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim*2, dtype=dtype, device=device) # times(2) because will chunk into 2
        self.out = nn.Linear(hidden_dim, input_dim, dtype=dtype, device=device)
    
    def forward(self, x):
        x = x.to(self.fc.weight.dtype)
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        swh = swish(x1)
        gated = swh * x2
        return self.out(gated)

class SpikingSwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, beta=0.95, num_steps=5, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.swiglu = SwiGLU(input_dim, hidden_dim, dtype=dtype, device=device)
        self.lif = snn.Leaky(beta=beta, learn_beta=True)
        self.num_steps = num_steps
        
        self.spike_scale = nn.Parameter(torch.tensor(1.0, dtype=dtype, device=device))
        self.register_buffer('running_mean', torch.tensor(1.0, dtype=dtype, device=device))
        self.momentum = 0.9
    
    def forward(self, x):
        x = self.swiglu(x)

        mem = self.lif.init_leaky()
        spike_acc = torch.zeros_like(x)

        #for _ in range(self.num_steps):
        # Not using for loop for ANE
        # Iteration 1
        spk, mem = self.lif(x, mem)
        spike_acc += spk
        # Iteration 2
        spk, mem = self.lif(x, mem)
        spike_acc += spk
        # Iteration 3
        spk, mem = self.lif(x, mem)
        spike_acc += spk
        # Iteration 4
        spk, mem = self.lif(x, mem)
        spike_acc += spk
        # Iteration 5
        spk, mem = self.lif(x, mem)
        spike_acc += spk
        
        spike_rate = spike_acc / self.num_steps
        with torch.no_grad():
            if self.training:
                current_mean = x.abs().mean()
                self.running_mean.mul_(self.momentum).add_(current_mean, alpha=1-self.momentum)
        
        # Use running mean for stable scaling
        output = spike_rate * self.running_mean * self.spike_scale
        return output

class SurrogateSpike(torch.autograd.Function):
    """
    Surrogate gradient function for spikes to enable backpropagation.
    Forward: Heaviside step function
    Backward: Sigmoid derivative approximation
    """
    gamma = 4.0  # Controls steepness of surrogate gradient
    
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # Sigmoid derivative as surrogate gradient
        temp = SurrogateSpike.gamma * input
        temp = torch.clamp(temp, -10, 10)  # Prevent overflow
        temp = temp / (SurrogateSpike.gamma * (1.0 + torch.abs(temp))) ** 2
        return grad_input * temp

class SurrogateLIF(nn.Module):
    def __init__(self, beta=0.9, threshold=1.0, reset_mechanism='subtract'):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta))  # Learnable decay constant
        self.threshold = threshold
        self.reset_mechanism = reset_mechanism
        self.surrogate_spike = SurrogateSpike.apply
    
    def forward(self, x, mem=None):
        if mem is None:
            mem = torch.zeros_like(x)
        
        # Leaky integration
        mem = self.beta * mem + x
        
        # Spike generation with surrogate gradient
        spk = self.surrogate_spike(mem - self.threshold)
        
        # Reset mechanism
        if self.reset_mechanism == 'subtract':
            mem = mem - spk * self.threshold
        elif self.reset_mechanism == 'zero':
            mem = mem * (1 - spk)
        
        return spk, mem

class ImprovedSpikingSwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, beta=0.9, num_steps=5, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.swiglu = SwiGLU(input_dim, hidden_dim, dtype=dtype, device=device)
        self.lif = SurrogateLIF(beta=beta)
        self.num_steps = num_steps
        self.spike_scale = nn.Parameter(torch.tensor(0.1, dtype=dtype, device=device))
    
    def forward(self, x):
        # Standard SwiGLU activation
        swiglu_out = self.swiglu(x)
        
        # Spiking dynamics
        mem = None
        spike_sum = torch.zeros_like(swiglu_out)
        
        #for _ in range(self.num_steps):
        # Not using for loop for ANE
        # Iteration 1
        spk, mem = self.lif(swiglu_out, mem)
        spike_sum += spk
        # Iteration 2
        spk, mem = self.lif(swiglu_out, mem)
        spike_sum += spk
        # Iteration 3
        spk, mem = self.lif(swiglu_out, mem)
        spike_sum += spk
        # Iteration 4
        spk, mem = self.lif(swiglu_out, mem)
        spike_sum += spk
        # Iteration 5
        spk, mem = self.lif(swiglu_out, mem)
        spike_sum += spk
        
        # Simple rate-based output
        spike_rate = spike_sum / self.num_steps
        return spike_rate * self.spike_scale + swiglu_out * 0.1  # Residual connection

def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    # [seq_len, head_dim//2] -> [seq_len, head_dim]
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    
    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin

    return q_rotated, k_rotated

class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len, theta=10_000, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # frequency matrix
        # θ_i = 1 / (theta^(2i/head_dim)) for i = 0, 1, ..., head_dim//2 - 1
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=dtype, device=device).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

        self._precompute_cossin(max_seq_len)
    
    def _precompute_cossin(self, seq_len):
        pos = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, x, seq_len=None):
        seq_len = seq_len or x.shape[-2]
        if seq_len > self.max_seq_len:
            self._precompute_cossin(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def get_local_attn_mask(seq_len, window_size, device="cuda", kv_len=None):
    kv_len = kv_len or seq_len

    q_pos = torch.arange(seq_len, device=device).unsqueeze(1)
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)

    causal_mask = q_pos >= k_pos
    window_mask = (q_pos - k_pos) <= window_size
    combined_mask = causal_mask & window_mask
    return combined_mask

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim=None, dropout=0.0, bias=False, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim or d_model // n_heads

        assert n_heads % n_kv_heads == 0, f"{n_heads=} must be divisible by {n_kv_heads=}"
        self.n_groups = n_heads // n_kv_heads

        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, n_heads*self.head_dim, bias=bias, dtype=dtype, device=device)
        self.k_proj = nn.Linear(d_model, n_kv_heads*self.head_dim, bias=bias, dtype=dtype, device=device)
        self.v_proj = nn.Linear(d_model, n_kv_heads*self.head_dim, bias=bias, dtype=dtype, device=device)
        self.o_proj = nn.Linear(n_heads*self.head_dim, d_model, bias=bias, dtype=dtype, device=device)
        
        self.dropout = dropout
        self.dtype = dtype
        self.device = device

    def forward(self, x, mask=None, past_key_value=None, use_cache=False, rope=None):
        batch_size, seq_len, _ = x.shape
        x = x.to(self.q_proj.weight.dtype)

        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        if rope is not None:
            cos, sin = rope(q, seq_len)
            q, k = apply_rope(q, k, cos, sin)
        
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        present_key_value = (k.clone(), v.clone()) if use_cache else None

        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=2)
            v = v.repeat_interleave(self.n_groups, dim=2)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash Attention Optimization
        # Eğer maske yoksa (Global Attn) -> Flash Attention Causal Kernel kullanılır
        # Eğer maske varsa (Sliding Window) -> Maskeli SDPA kullanılır
        
        is_causal = False
        attn_mask = mask
        
        if mask is None:
            is_causal = True # Global attention için causal flag'i aç
        else:
            # Maske şekillendirme [B, 1, Q, K] veya [1, 1, Q, K]
            if mask.dtype == torch.bool:
                # F.sdpa bool maskeyi destekler ama float maske bazen daha stabildir
                pass 
            else:
                pass

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads*self.head_dim)
        return self.o_proj(out), present_key_value
    
class RegularGroupedSlidingAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim=None, dropout=0.0, 
                 max_seq_len=2048, rope_theta_local=1e4, rope_theta_global=1e6, window_size=128,
                 dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        
        self.local_rope = RoPE(head_dim or d_model//n_heads, max_seq_len, theta=rope_theta_local, dtype=dtype, device=device)
        self.global_rope = RoPE(head_dim or d_model//n_heads, max_seq_len, theta=rope_theta_global, dtype=dtype, device=device)

        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, dropout, dtype=dtype, device=device)
        self.window_size = window_size

    def get_causal_mask(self, seq_len, device, is_global, kv_len=None):
        kv_len = kv_len or seq_len

        if is_global:
            return None #torch.tril(torch.ones(seq_len, kv_len, dtype=torch.bool, device=device))
        else:
            idxs_q = torch.arange(seq_len, device=device).view(-1, 1)
            idxs_k = torch.arange(kv_len, device=device).view(1, -1)
            # FIX: Causal AND Window
            causal_mask = idxs_q >= idxs_k
            window_mask = (idxs_q - idxs_k) <= self.window_size
            return causal_mask & window_mask

    def forward(self, x, use_cache=False, past_key_value=None, layer_idx=0):
        batch_size, seq_len, _ = x.shape
        kv_len = seq_len + (past_key_value[0].size(2) if past_key_value else 0)
        is_global = (layer_idx % 6 == 5)
        
        mask = self.get_causal_mask(seq_len, x.device, is_global, kv_len=kv_len)
        if mask is not None:
             mask = mask.unsqueeze(0).unsqueeze(0)

        rope = self.global_rope if is_global else self.local_rope
        return self.attn(x, mask=mask, past_key_value=past_key_value, use_cache=use_cache, rope=rope)

class SpikingGroupedSlidingAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim=None, dropout=0.0, 
                 max_seq_len=2048, rope_theta_local=1e4, rope_theta_global=1e6, window_size=128, beta=0.95,
                 num_steps=5, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        
        self.local_rope = RoPE(head_dim or d_model//n_heads, max_seq_len, theta=rope_theta_local, dtype=dtype, device=device)
        self.global_rope = RoPE(head_dim or d_model//n_heads, max_seq_len, theta=rope_theta_global, dtype=dtype, device=device)

        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, dropout, dtype=dtype, device=device)
        self.spike = snn.Leaky(beta=beta, learn_beta=True)
        self.num_steps = num_steps
        self.spike_scale = nn.Parameter(torch.ones(1, dtype=dtype, device=device))
        self.window_size = window_size

    def get_causal_mask(self, seq_len, device, is_global, kv_len=None):
        kv_len = kv_len or seq_len

        if is_global:
            return None #torch.tril(torch.ones(seq_len, kv_len, dtype=torch.bool, device=device))
        else:
            idxs_q = torch.arange(seq_len, device=device).view(-1, 1)
            idxs_k = torch.arange(kv_len, device=device).view(1, -1)
            causal_mask = idxs_q >= idxs_k
            window_mask = (idxs_q - idxs_k) <= self.window_size
            return causal_mask & window_mask

    def forward(self, x, use_cache=False, past_key_value=None, layer_idx=0):
        batch_size, seq_len, _ = x.shape
        kv_len = seq_len + (past_key_value[0].size(2) if past_key_value else 0)
        is_global = (layer_idx % 6 == 5)
        
        mask = self.get_causal_mask(seq_len, x.device, is_global, kv_len=kv_len)
        if mask is not None:
             mask = mask.unsqueeze(0).unsqueeze(0)

        rope = self.global_rope if is_global else self.local_rope
        out, present_kv = self.attn(x, mask=mask, past_key_value=past_key_value, use_cache=use_cache, rope=rope)
        
        mem = self.spike.init_leaky()
        spike_acc = torch.zeros_like(out)

        #for _ in range(self.num_steps):
        # Not using for loop for ANE
        # Iteration 1
        spk, mem = self.spike(out, mem)
        spike_acc += spk
        # Iteration 2
        spk, mem = self.spike(out, mem)
        spike_acc += spk
        # Iteration 3
        spk, mem = self.spike(out, mem)
        spike_acc += spk
        # Iteration 4
        spk, mem = self.spike(out, mem)
        spike_acc += spk
        # Iteration 5
        spk, mem = self.spike(out, mem)
        spike_acc += spk
        
        return (spike_acc / self.num_steps) * out.abs().mean() * self.spike_scale, present_kv
    
class PreRMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
        self.eps = eps
        self.d_model = d_model
    
    def forward(self, x):
        input_dtype = x.dtype
        input_device = x.device
        
        if x.device != self.weight.device:
            x = x.to(self.weight.device)
        
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        
        result = self.weight * x_normed
        return result.to(dtype=input_dtype, device=input_device)
