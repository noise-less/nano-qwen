import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path
from safetensors import safe_open
from dataclasses import dataclass

from accelerate import load_checkpoint_and_dispatch


@dataclass
class ModelConfig:
    n_embed: int
    n_heads: int
    n_kv_heads: int
    n_layer: int
    n_mlp: int  # dense MLP intermediate size

    n_vocab: int
    tie_word_embeddings: bool

    rope_theta: float
    rms_norm_eps: float

    # MoE parameters
    d_head: Optional[int] = None
    n_experts: Optional[int] = None
    n_experts_per_token: Optional[int] = None
    n_moe_mlp: Optional[int] = None

    @classmethod
    def from_pretrained(cls, hf_config: dict):
        # Map from hugging face transformers config names
        return cls(
            n_embed=hf_config["hidden_size"],
            n_heads=hf_config["num_attention_heads"],
            n_kv_heads=hf_config["num_key_value_heads"],
            n_layer=hf_config["num_hidden_layers"],
            n_mlp=hf_config["intermediate_size"],
            n_vocab=hf_config["vocab_size"],
            tie_word_embeddings=hf_config["tie_word_embeddings"],
            rope_theta=hf_config["rope_theta"],
            rms_norm_eps=hf_config["rms_norm_eps"],
            d_head=hf_config.get("head_dim"),
            n_experts=hf_config.get("num_experts"),
            n_experts_per_token=hf_config.get("num_experts_per_tok"),
            n_moe_mlp=hf_config.get("moe_intermediate_size"),
        )


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Use explicit d_head if provided, otherwise calculate
        d = (
            config.d_head
            if config.d_head is not None
            else (config.n_embed // config.n_heads)
        )
        t = config.rope_theta
        r = torch.arange(0, d, 2)
        self.register_buffer("inv_freq", 1.0 / (t ** (r / d)).float(), persistent=False)

        # MRoPE section: hardcoded for all Qwen3-VL models
        self.mrope_section = [24, 20, 20]

    def forward(self, x, position_ids):
        # Expand 2D position_ids to 3D for text-only (matches official implementation)
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # Compute frequencies for each dimension (T, H, W)
        # inv_freq: [head_dim // 2]
        # position_ids: [3, B, T]
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()  # [3, B, 1, T]

        # Compute freqs: [3, B, T, head_dim // 2]
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)

        # Apply interleaved MRoPE
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)

        # Concatenate and compute cos/sin
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """Apply interleaved MRoPE to 3D rotary embeddings.

        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.

        Args:
            freqs: [3, B, T, head_dim // 2]
            mrope_section: [24, 20, 20]
        Returns:
            freqs_t: [B, T, head_dim // 2]
        """
        freqs_t = freqs[0]  # Start with temporal dimension
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


class DenseAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads

        self.n_embed = config.n_embed
        self.d_head = config.d_head
        self.n_kv_embed = config.n_kv_heads * config.d_head
        self.n_q_embed = config.n_heads * config.d_head

        self.q_proj = nn.Linear(self.n_embed, self.n_q_embed, bias=False)
        self.k_proj = nn.Linear(self.n_embed, self.n_kv_embed, bias=False)
        self.v_proj = nn.Linear(self.n_embed, self.n_kv_embed, bias=False)
        self.o_proj = nn.Linear(self.n_q_embed, self.n_embed, bias=False)

        self.q_norm = RMSNorm(self.d_head, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.d_head, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)

        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        if self.n_kv_heads < self.n_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_q_embed)
        y = self.o_proj(y)
        return y

    @staticmethod
    def _apply_rotary_pos_emb(q, k, cos, sin):
        if cos.dim() == 4:
            # shape [B, 3, T, D] -> multi-modal
            cos = DenseAttention._process_rotary_component(cos)
            sin = DenseAttention._process_rotary_component(sin)
        else:
            # shape [B, T, D] -> text-only
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_embed = (q * cos) + (DenseAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (DenseAttention._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _process_rotary_component(x):
        # Split into sections and select appropriate indices
        sections = x.split([16, 24, 24, 16, 24, 24], dim=-1)
        processed = [m[i % 3] for i, m in enumerate(sections)]
        # Combine and add dimension
        return torch.cat(processed, dim=-1).unsqueeze(1)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class MoeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_embed = config.n_embed

        self.d_head = (
            config.d_head
            if config.d_head is not None
            else (config.n_embed // config.n_heads)
        )

        self.q_proj = nn.Linear(self.n_embed, self.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(self.n_embed, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(self.n_embed, self.n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.d_head, self.n_embed, bias=False)

        self.q_norm = RMSNorm(self.d_head, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.d_head, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        # Apply normalization to q and k before RoPE (Qwen3 specific)
        q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)

        q, k = self._apply_rotary_pos_emb(q, k, cos, sin)

        if self.n_kv_heads < self.n_heads:
            num_repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(num_repeat, dim=1)
            v = v.repeat_interleave(num_repeat, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_head)
        y = self.o_proj(y)
        return y

    @staticmethod
    def _apply_rotary_pos_emb(q, k, cos, sin):
        if cos.dim() == 4:
            # shape [B, 3, T, D] -> multi-modal
            cos = MoeAttention._process_rotary_component(cos)
            sin = MoeAttention._process_rotary_component(sin)
        else:
            # shape [B, T, D] -> text-only
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        q_embed = (q * cos) + (MoeAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (MoeAttention._rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _process_rotary_component(x):
        # Split into sections and select appropriate indices
        sections = x.split([16, 24, 24, 16, 24, 24], dim=-1)
        processed = [m[i % 3] for i, m in enumerate(sections)]
        # Combine and add dimension
        return torch.cat(processed, dim=-1).unsqueeze(1)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, n_embed, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embed))
        self.variance_epsilon = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x.to(input_dtype)


class DenseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.up_proj = nn.Linear(config.n_embed, config.n_mlp, bias=False)
        self.down_proj = nn.Linear(config.n_mlp, config.n_embed, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts_per_token = config.n_experts_per_token
        self.n_experts = config.n_experts
        self.gate = nn.Linear(config.n_embed, config.n_experts, bias=False)

        # Expert layers with proper naming to match checkpoint
        self.experts = nn.ModuleList()
        for _ in range(config.n_experts):
            expert = nn.Module()
            expert.gate_proj = nn.Linear(config.n_embed, config.n_moe_mlp, bias=False)
            expert.up_proj = nn.Linear(config.n_embed, config.n_moe_mlp, bias=False)
            expert.down_proj = nn.Linear(config.n_moe_mlp, config.n_embed, bias=False)
            self.experts.append(expert)

    def forward(self, x):
        scores = self.gate(x)  # (b, seq_len, n_experts)
        topk_scores, topk_indices = torch.topk(scores, self.n_experts_per_token, dim=-1)
        topk_probs = torch.softmax(topk_scores, dim=-1)

        expert_outputs = []
        for e in range(self.n_experts):
            expert = self.experts[e]
            hidden = F.silu(expert.gate_proj(x)) * expert.up_proj(x)
            out = expert.down_proj(hidden)
            expert_outputs.append(out.unsqueeze(-2))
        expert_outputs = torch.cat(expert_outputs, dim=-2)  # (b, t, n_experts, emb_dim)

        gating_probs = torch.zeros_like(scores)

        for i in range(self.n_experts_per_token):
            indices = topk_indices[..., i : i + 1]
            prob = topk_probs[..., i : i + 1]
            gating_probs.scatter_(dim=-1, index=indices, src=prob)
        gating_probs = gating_probs.unsqueeze(-1)  # (b, t, n_experts, 1)

        # Weighted sum over experts
        y = (gating_probs * expert_outputs).sum(dim=-2)
        return y


class DenseBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embed, eps = config.n_embed, config.rms_norm_eps
        self.input_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.self_attn = DenseAttention(config)
        self.post_attention_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.mlp = DenseMLP(config)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embed, eps = config.n_embed, config.rms_norm_eps
        self.input_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.self_attn = MoeAttention(config)
        self.post_attention_layernorm = RMSNorm(n_embed=n_embed, eps=eps)

        # Use MoE if experts are configured, otherwise regular MLP
        if config.n_experts and config.n_experts > 0:
            self.mlp = MoEMLP(config)
        else:
            self.mlp = DenseMLP(config)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class DenseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.n_vocab, config.n_embed)
        self.rotary_emb = RotaryEmbedding(config)

        self.layers = nn.ModuleList(DenseBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)

    def forward(self, x, position_ids):
        cos, sin = self.rotary_emb(x, position_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return x


class Qwen3Dense(nn.Module):
    """Qwen3 dense model - text-only version"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.language_model = DenseModel(config)

        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embed, config.n_vocab, bias=False)

    def _get_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(T, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        return position_ids

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.language_model.embed_tokens(input_ids)
        position_ids = self._get_position_ids(input_ids)
        x = self.language_model(x=x, position_ids=position_ids)

        if self.lm_head is None:
            logits = torch.matmul(x, self.language_model.embed_tokens.weight.T)
        else:
            logits = self.lm_head(x)
        return logits


class Qwen3(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3Dense(config)

    def forward(self, input_ids: torch.Tensor):
        x = self.model(input_ids)
        return x

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1,
        stop_tokens: list = None,
    ):
        if stop_tokens is None:
            # <|im_end|>, <|im_start|>, <|endoftext|>
            stop_tokens = [151645, 151644, 151643]

        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids=input_ids)
                last_logits = logits[:, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                next_token = probs.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # Check if we hit a stop token
                if next_token.item() in stop_tokens:
                    break

        return input_ids

    @classmethod
    def from_pretrained(cls, weights_path: str, device_map: str = "auto"):
        model_path = Path(weights_path)

        with open(model_path / "config.json", "r") as f:
            hf_config = json.load(f)
        config = ModelConfig.from_pretrained(hf_config["text_config"])

        model = cls(config)

        # Use accelerate to load weights efficiently without loading all to RAM
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=str(model_path),
            device_map=device_map,
            no_split_module_classes=["DenseBlock"],
            dtype=torch.bfloat16,
        )

        return model


class MoEModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.n_vocab, config.n_embed)
        self.rotary_emb = RotaryEmbedding(config)

        # Use Qwen3MoeBlock with proper attention and MoE
        self.layers = nn.ModuleList(MoEBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)

        # Store config for convenience
        self.config = config

    def forward(self, x, position_ids):
        cos, sin = self.rotary_emb(x, position_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return x


class Qwen3MoE(nn.Module):
    """Qwen3 MoE model - text-only version with mixture of experts"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = MoEModel(config)

        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embed, config.n_vocab, bias=False)

    def _get_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(T, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
        return position_ids

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.model.embed_tokens(input_ids)
        position_ids = self._get_position_ids(input_ids)
        x = self.model(x=x, position_ids=position_ids)

        if self.lm_head is None:
            logits = torch.matmul(x, self.model.embed_tokens.weight.T)
        else:
            logits = self.lm_head(x)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 1,
        stop_tokens: list = None,
        stream: bool = False,
    ):
        if stop_tokens is None:
            # <|im_end|>, <|im_start|>, <|endoftext|>
            stop_tokens = [151645, 151644, 151643]

        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(input_ids=input_ids)
                last_logits = logits[:, -1, :]
                probs = F.softmax(last_logits, dim=-1)
                next_token = probs.argmax(dim=-1, keepdim=True)
                input_ids = torch.cat([input_ids, next_token], dim=1)

                # If streaming, yield the new token
                if stream:
                    yield next_token.item()

                # Check if we hit a stop token
                if next_token.item() in stop_tokens:
                    break

        # If not streaming, return the full input_ids
        if not stream:
            return input_ids

    @classmethod
    def get_config_class(cls):
        return ModelConfig

    @classmethod
    def from_pretrained(cls, repo_id: str, device_map: str = "auto"):
        from .util import load_pretrained_model

        return load_pretrained_model(cls, repo_id, device_map=device_map)


# Configuration key mapping for loading HuggingFace pretrained language models
# Maps: HuggingFace config key -> tiny-qwen config key
HF_TO_LM_CONFIG = {
    "hidden_size": "n_embed",
    "num_attention_heads": "n_heads",
    "num_key_value_heads": "n_kv_heads",
    "num_hidden_layers": "n_layer",
    "intermediate_size": "n_mlp",
    "rope_theta": "rope_theta",
    "rms_norm_eps": "rms_norm_eps",
    "vocab_size": "n_vocab",
    "tie_word_embeddings": "tie_word_embeddings",
    "head_dim": "d_head",
    # MoE parameters
    "num_experts": "n_experts",
    "num_experts_per_tok": "n_experts_per_token",
    "moe_intermediate_size": "n_moe_mlp",
}

# Weight key mapping for loading HuggingFace pretrained language model weights
# Maps: HuggingFace component name -> tiny-qwen component name
HF_TO_LM_WEIGHTS = {}
