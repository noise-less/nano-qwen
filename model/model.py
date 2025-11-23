import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path
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
            n_embed=hf_config["text_config"]["hidden_size"],
            n_heads=hf_config["text_config"]["num_attention_heads"],
            n_kv_heads=hf_config["text_config"]["num_key_value_heads"],
            n_layer=hf_config["text_config"]["num_hidden_layers"],
            n_mlp=hf_config["text_config"]["intermediate_size"],
            n_vocab=hf_config["text_config"]["vocab_size"],
            tie_word_embeddings=hf_config["tie_word_embeddings"],
            rope_theta=hf_config["text_config"]["rope_theta"],
            rms_norm_eps=hf_config["text_config"]["rms_norm_eps"],
            d_head=hf_config["text_config"].get("head_dim"),
            n_experts=hf_config["text_config"].get("num_experts"),
            n_experts_per_token=hf_config["text_config"].get("num_experts_per_tok"),
            n_moe_mlp=hf_config["text_config"].get("moe_intermediate_size"),
        )


class RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_head
        t = config.rope_theta
        r = torch.arange(0, d, 2)
        self.register_buffer("inv_freq", 1.0 / (t ** (r / d)).float(), persistent=False)

        self.mrope_section = [24, 20, 20]

    def forward(self, x, position_ids):
        inv_freq = self.inv_freq.to(dtype=torch.float32, device=x.device)
        inv_freq_expanded = inv_freq[None, None, :, None].expand(
            3, position_ids.shape[1], -1, 1
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)

        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(x.dtype)
        sin = emb.sin().to(x.dtype)
        return cos, sin

    def apply_interleaved_mrope(self, freqs, mrope_section):
        """
        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT], preserving frequency continuity.
        """
        freqs_t = freqs[0]  # Start with temporal dimension
        for dim, offset in enumerate((1, 2), start=1):  # H, W
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.n_kv_heads = config.n_kv_heads
        self.n_embed = config.n_embed

        self.q_proj = nn.Linear(self.n_embed, self.n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(self.n_embed, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(self.n_embed, self.n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.d_head, self.n_embed, bias=False)

        self.q_norm = RMSNorm(self.d_head, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.d_head, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

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
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (SelfAttention._rotate_half(q) * sin)
        k_embed = (k * cos) + (SelfAttention._rotate_half(k) * sin)
        return q_embed, k_embed

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


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_embed, eps = config.n_embed, config.rms_norm_eps
        self.input_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.self_attn = SelfAttention(config)
        self.post_attention_layernorm = RMSNorm(n_embed=n_embed, eps=eps)
        self.mlp = MoEMLP(config) if config.n_experts else DenseMLP(config)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.n_vocab, config.n_embed)
        self.rotary_emb = RotaryEmbedding(config)

        self.layers = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)

    def forward(self, x, position_ids):
        x = self.embed_tokens(x)
        cos, sin = self.rotary_emb(x, position_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return x


class Qwen3VL(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Wrap in a model container to match checkpoint structure
        self.model = nn.Module()
        self.model.language_model = Model(config)
        self.model.lm_head = None
        if not config.tie_word_embeddings:
            self.model.lm_head = nn.Linear(config.n_embed, config.n_vocab, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        position_ids = self._get_position_ids(input_ids)
        x = self.model.language_model(x=input_ids, position_ids=position_ids)
        if self.model.lm_head is None:
            logits = torch.matmul(x, self.model.language_model.embed_tokens.weight.T)
        else:
            logits = self.model.lm_head(x)
        return logits

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
        config = ModelConfig.from_pretrained(hf_config)

        model = cls(config)

        # Use accelerate to load weights efficiently without loading all to RAM
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=str(model_path),
            device_map=device_map,
            no_split_module_classes=["Block"],
            dtype=torch.bfloat16,
        )

        return model

    def _get_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        position_ids = torch.arange(T, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(3, B, -1)
        return position_ids


# class Qwen3(nn.Module):
#     def __init__(self, config: ModelConfig):
#         super().__init__()
#         self.config = config
#         self.model = Qwen3Dense(config)

#         # Match lm_head dtype with embeddings for checkpoints stored in float32
#         if self.model.lm_head is not None:
#             target_dtype = self.model.language_model.embed_tokens.weight.dtype
#             self.model.lm_head = self.model.lm_head.to(target_dtype)

#     def forward(self, input_ids: torch.Tensor):
#         x = self.model(input_ids)
#         return x

#     def generate(
#         self,
#         input_ids: torch.Tensor,
#         max_new_tokens: int = 1,
#         stop_tokens: list = None,
#     ):
#         if stop_tokens is None:
#             # <|im_end|>, <|im_start|>, <|endoftext|>
#             stop_tokens = [151645, 151644, 151643]

#         self.eval()
#         with torch.no_grad():
#             for _ in range(max_new_tokens):
#                 logits = self.forward(input_ids=input_ids)
#                 last_logits = logits[:, -1, :]
#                 probs = F.softmax(last_logits, dim=-1)
#                 next_token = probs.argmax(dim=-1, keepdim=True)
#                 input_ids = torch.cat([input_ids, next_token], dim=1)

#                 # Check if we hit a stop token
#                 if next_token.item() in stop_tokens:
#                     break

#         return input_ids

#     @classmethod
#     def from_pretrained(cls, weights_path: str, device_map: str = "auto"):
#         model_path = Path(weights_path)

#         with open(model_path / "config.json", "r") as f:
#             hf_config = json.load(f)
#         config = ModelConfig.from_pretrained(hf_config)

#         model = cls(config)

#         # Use accelerate to load weights efficiently without loading all to RAM
#         model = load_checkpoint_and_dispatch(
#             model,
#             checkpoint=str(model_path),
#             device_map=device_map,
#             no_split_module_classes=["DenseBlock"],
#             dtype=torch.bfloat16,
#         )

#         return model


# class MoEModel(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.embed_tokens = nn.Embedding(config.n_vocab, config.n_embed)
#         self.rotary_emb = RotaryEmbedding(config)

#         # Use Qwen3MoeBlock with proper attention and MoE
#         self.layers = nn.ModuleList(MoEBlock(config) for _ in range(config.n_layer))
#         self.norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)

#         # Store config for convenience
#         self.config = config

#     def forward(self, x, position_ids):
#         cos, sin = self.rotary_emb(x, position_ids)
#         for layer in self.layers:
#             x = layer(x, cos, sin)
#         x = self.norm(x)
#         return x


# class Qwen3MoE(nn.Module):
#     def __init__(self, config: ModelConfig):
#         super().__init__()
#         self.config = config
#         self.model = MoEModel(config)

#         self.lm_head = None
#         if not config.tie_word_embeddings:
#             self.lm_head = nn.Linear(config.n_embed, config.n_vocab, bias=False)

#     def _get_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
#         B, T = input_ids.shape
#         device = input_ids.device
#         position_ids = torch.arange(T, dtype=torch.long, device=device)
#         position_ids = position_ids.unsqueeze(0).expand(B, -1)
#         return position_ids

#     def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
#         target_dtype = self.model.embed_tokens.weight.dtype
#         x = self.model.embed_tokens(input_ids).to(target_dtype)
#         position_ids = self._get_position_ids(input_ids)
#         x = self.model(x=x, position_ids=position_ids)

#         x = x.to(target_dtype)
#         if self.lm_head is None:
#             logits = torch.matmul(x, self.model.embed_tokens.weight.T.to(target_dtype))
#         else:
#             logits = self.lm_head(x)
#         return logits

#     def generate(
#         self,
#         input_ids: torch.Tensor,
#         max_new_tokens: int = 1,
#         stop_tokens: list = None,
#         stream: bool = False,
#     ):
#         if stop_tokens is None:
#             # <|im_end|>, <|im_start|>, <|endoftext|>
#             stop_tokens = [151645, 151644, 151643]

#         self.eval()
#         with torch.no_grad():
#             for _ in range(max_new_tokens):
#                 logits = self.forward(input_ids=input_ids)
#                 last_logits = logits[:, -1, :]
#                 probs = F.softmax(last_logits, dim=-1)
#                 next_token = probs.argmax(dim=-1, keepdim=True)
#                 input_ids = torch.cat([input_ids, next_token], dim=1)

#                 # If streaming, yield the new token
#                 if stream:
#                     yield next_token.item()

#                 # Check if we hit a stop token
#                 if next_token.item() in stop_tokens:
#                     break

#         # If not streaming, return the full input_ids
#         if not stream:
#             return input_ids

#     @classmethod
#     def get_config_class(cls):
#         return ModelConfig

#     @classmethod
#     def from_pretrained(cls, repo_id: str, device_map: str = "auto"):
#         from .util import load_pretrained_model

#         return load_pretrained_model(cls, repo_id, device_map=device_map)
