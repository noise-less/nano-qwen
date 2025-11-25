import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path
from dataclasses import dataclass

from accelerate import load_checkpoint_and_dispatch

from .vision import VisionEncoder, VisionConfig


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
        llm_config = hf_config["text_config"]
        return cls(
            n_embed=llm_config["hidden_size"],
            n_heads=llm_config["num_attention_heads"],
            n_kv_heads=llm_config["num_key_value_heads"],
            n_layer=llm_config["num_hidden_layers"],
            n_mlp=llm_config["intermediate_size"],
            n_vocab=llm_config["vocab_size"],
            tie_word_embeddings=hf_config["tie_word_embeddings"],
            rope_theta=llm_config["rope_theta"],
            rms_norm_eps=llm_config["rms_norm_eps"],
            d_head=llm_config.get("head_dim"),
            n_experts=llm_config.get("num_experts"),
            n_experts_per_token=llm_config.get("num_experts_per_tok"),
            n_moe_mlp=llm_config.get("moe_intermediate_size"),
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
        """[TTT...HHH...WWW] -> [THWTHWTHW...TT]"""
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


class MoEExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_experts
        self.hidden_size = config.n_embed
        self.expert_dim = config.n_moe_mlp

        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.expert_dim, self.hidden_size)
        )

    def forward(self, x: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        gate_up = torch.einsum("th,ehq->teq", x, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)
        expert_outputs = torch.einsum("teq,eqh->teh", F.silu(gate) * up, self.down_proj)
        weighted = expert_outputs * routing_weights.unsqueeze(-1)
        return weighted.sum(dim=1)


class MoEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.n_embed
        self.expert_dim = config.n_moe_mlp
        self.num_experts = config.n_experts
        self.top_k = config.n_experts_per_token
        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = MoEExperts(config)

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        hidden = x.reshape(-1, self.hidden_size)

        router_logits = self.gate(hidden)
        routing_weights = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        routed = torch.zeros_like(router_logits)
        topk_weights = topk_weights.to(router_logits.dtype)
        routed.scatter_(1, topk_indices, topk_weights)
        routed = routed / (routed.sum(dim=-1, keepdim=True) + 1e-9)
        routed = routed.to(hidden.dtype)

        expert_out = self.experts(hidden, routed)
        combined = expert_out.view(bsz, seq_len, self.hidden_size)

        return combined, router_logits.to(hidden.dtype)


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
        mlp_out = self.mlp(self.post_attention_layernorm(x))
        if isinstance(mlp_out, tuple):
            mlp_out = mlp_out[0]
        x = x + mlp_out
        return x


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.n_vocab, config.n_embed)
        self.rotary_emb = RotaryEmbedding(config)

        self.layers = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.n_embed, eps=config.rms_norm_eps)

    def forward(
        self, x, position_ids, visual_pos_masks=None, deepstack_visual_embeds=None
    ):
        # x can be either token IDs (int) or embeddings (float)
        if x.dtype in (torch.long, torch.int):
            x = self.embed_tokens(x)

        cos, sin = self.rotary_emb(x, position_ids)
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x, cos, sin)
            # Add deepstack visual features residually at early layers
            if deepstack_visual_embeds is not None:
                visual_embed = None
                if isinstance(deepstack_visual_embeds, dict):
                    visual_embed = deepstack_visual_embeds.get(layer_idx)
                elif layer_idx < len(deepstack_visual_embeds):
                    visual_embed = deepstack_visual_embeds[layer_idx]

                if visual_embed is not None:
                    x = self._deepstack_process(
                        x,
                        visual_pos_masks,
                        visual_embed,
                    )
        x = self.norm(x)
        return x

    def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
        """Add visual features residually to language model hidden states."""
        if visual_pos_masks is None:
            return hidden_states
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks, :] = (
            hidden_states[visual_pos_masks, :] + visual_embeds
        )
        return hidden_states


class Qwen3VL(nn.Module):
    def __init__(
        self, config: ModelConfig, vision_config: Optional[VisionConfig] = None
    ):
        super().__init__()
        self.config = config
        self.vision_config = vision_config

        self.model = nn.Module()
        self.model.language_model = Model(config)
        self.lm_head = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.n_embed, config.n_vocab, bias=False)

        if vision_config is not None:
            self.model.visual = VisionEncoder(vision_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_embeds = self.model.language_model.embed_tokens(input_ids)

        # Process vision inputs if provided
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if pixels is not None and self.vision_config is not None:
            pixels = pixels.to(input_embeds.dtype)
            image_embeds, deepstack_visual_embeds = self.model.visual(
                pixels=pixels, d_image=d_image
            )

            image_pad_token = getattr(self.config, "image_token_id", 151655)
            image_pad_mask = input_ids == image_pad_token
            visual_pos_masks = image_pad_mask
            image_pad_mask_expanded = image_pad_mask.unsqueeze(-1).expand_as(
                input_embeds
            )
            input_embeds = input_embeds.masked_scatter(
                image_pad_mask_expanded, image_embeds
            )

            if deepstack_visual_embeds:
                deepstack_visual_embeds = {
                    layer_idx: embed
                    for layer_idx, embed in zip(
                        self.vision_config.deepstack_visual_indexes,
                        deepstack_visual_embeds,
                    )
                }

        position_ids = self._get_position_ids(input_ids=input_ids, d_image=d_image)

        x = self.model.language_model(
            x=input_embeds,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        logits = (
            x @ self.model.language_model.embed_tokens.weight.T
            if self.lm_head is None
            else self.lm_head(x)
        )

        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        pixels: Optional[torch.Tensor] = None,
        d_image: Optional[torch.Tensor] = None,
        max_new_tokens: int = 1,
        stop_tokens: list = None,
    ):
        if stop_tokens is None:
            # <|im_end|>, <|im_start|>, <|endoftext|>
            stop_tokens = [151645, 151644, 151643]

        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.forward(
                    input_ids=input_ids, pixels=pixels, d_image=d_image
                )
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

        # Load vision config if present
        vision_config = None
        if "vision_config" in hf_config:
            vision_config = VisionConfig.from_pretrained(hf_config)

        model = cls(config, vision_config=vision_config)

        # Use accelerate to load weights efficiently without loading all to RAM
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=str(model_path),
            device_map=device_map,
            no_split_module_classes=["Block", "VisionBlock"],
            dtype=torch.bfloat16,
        )

        return model

    def _get_position_ids(
        self, input_ids: torch.Tensor, d_image: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device

        # Text-only case: simple sequential position IDs
        if d_image is None:
            position_ids = torch.arange(T, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(3, B, -1)
            return position_ids

        vision_start_id = getattr(self.config, "vision_start_token_id", 151652)
        vision_end_id = getattr(self.config, "vision_end_token_id", 151653)
        image_token_id = getattr(self.config, "image_token_id", 151655)
        spatial_merge_size = self.vision_config.spatial_merge_size

        position_ids = torch.zeros(3, B, T, dtype=torch.long, device=device)

        for b in range(B):
            seq = input_ids[b]
            text_counter = 0
            image_counter = 0
            i = 0

            while i < T:
                token_id = seq[i].item()

                if token_id == vision_start_id:
                    position_ids[:, b, i] = text_counter
                    text_counter += 1
                    i += 1
                    continue

                if token_id == vision_end_id:
                    position_ids[:, b, i] = text_counter
                    text_counter += 1
                    image_counter += 1
                    i += 1
                    continue

                if token_id == image_token_id:
                    if d_image is None:
                        raise ValueError(
                            "image_grid_thw must be provided for image tokens."
                        )
                    t_img, h_img, w_img = d_image[image_counter]
                    h_img = (h_img // spatial_merge_size).item()
                    w_img = (w_img // spatial_merge_size).item()
                    t_img = t_img.item()
                    image_tokens = t_img * h_img * w_img
                    base = text_counter

                    for img_idx in range(image_tokens):
                        if i + img_idx >= T:
                            break
                        t_pos = img_idx // (h_img * w_img)
                        remaining = img_idx % (h_img * w_img)
                        h_pos = remaining // w_img
                        w_pos = remaining % w_img

                        position_ids[0, b, i + img_idx] = base
                        position_ids[1, b, i + img_idx] = h_pos + base
                        position_ids[2, b, i + img_idx] = w_pos + base

                    i += image_tokens
                    text_counter = base + 1
                    continue

                position_ids[:, b, i] = text_counter
                text_counter += 1
                i += 1

        return position_ids
