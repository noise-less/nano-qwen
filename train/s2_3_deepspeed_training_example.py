import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from torch.utils.data import Dataset
from deepspeed.pipe import PipelineModule, LayerSpec
from deepspeed.runtime.pipe.topology import PipeModelDataParallelTopology as Topology

"""
if you are reading this, congrats. you've came far! :)

this script is a toy example of how to do Pipeline Parallelism (PP) with DeepSpeed.

once LLMs go beyond ~8B, we cannot use lightning for training anymore like we did in s2_1 and s2_2 since models won't fit in one GPU. Most intuitive course of action is just to split the transformer blocks onto multiple GPUs and have input go thru each like water through a pipe (PP).

my goal for libraries s.a. lightning or deepspeed is to glue model <> data <> compute together so i don't have to worry about distributed training code, however, even with current setup there are alot more moving pieces than i expeceted and i can't find any other frameworks i can use instead. seems like they either is too opinionated about what data format should look like (e.g. LLaMA-Factory), only support certain model implementation (e.g. trl), or only support one distribution strategy (e.g. lightning, align-anything). so far, deepspeed seems to be closest to what i'm looking for.

i could also use PeFT to reduce memory that way we can just stick to lightning. However, it is unclear to me what task should i choose to do so. pretty much all material online explains what PeFT IS and not WHEN to use it (e.g. format changing -> PeFT, adding new knowledge -> Full Fine-tuning, etc). it seems the internet only go as far as PeFT reducing catastrophic forgetting and overfitting.

will revisit once training bigger models becomes necessary again. 
"""

torch.backends.cuda.matmul.allow_tf32 = True  # ok for toy demo


class ToyTextDataset(Dataset):
    def __init__(self, num_samples=10_000, seq_len=128, vocab_size=50257):
        self.n = num_samples
        self.seq_len = seq_len
        self.vocab = vocab_size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = torch.randint(
            low=0, high=self.vocab, size=(self.seq_len,), dtype=torch.long
        )
        y = x.clone()
        return x, y


class TokenPosEmbed(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, hidden_size)
        self.pos = nn.Embedding(max_len, hidden_size)

    def forward(self, input_ids):
        b, s = input_ids.shape
        pos = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s)
        return self.tok(input_ids) + self.pos(pos)


class Block(nn.Module):
    """
    NOT a real Transformer block—just a placeholder MLP+residual+norm
    that preserves [B,S,H] shapes to demonstrate pipeline wiring.
    """

    def __init__(self, hidden_size, mlp_ratio=4):
        super().__init__()
        inner = hidden_size * mlp_ratio
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, inner)
        self.fc2 = nn.Linear(inner, hidden_size)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = F.gelu(h)
        h = self.fc2(h)
        return x + h


class LMHead(nn.Module):
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        x = self.norm(x)
        return self.proj(x)  # (B, S, V)


def lm_loss_fn(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1)
    )


def build_pipeline_model(
    vocab_size, hidden_size, n_layers, max_len, pp_size, world_size
):
    # pipeline topology: dp x mp x pp (we use mp=1)
    assert world_size % pp_size == 0, "WORLD_SIZE must be divisible by --pp"
    dp_size = world_size // pp_size
    topo = Topology(num_pp=pp_size, num_mp=1, num_dp=dp_size)

    layers = [LayerSpec(TokenPosEmbed, vocab_size, hidden_size, max_len)]
    for _ in range(n_layers):
        layers.append(LayerSpec(Block, hidden_size))
    layers.append(LayerSpec(LMHead, hidden_size, vocab_size))

    pipe = PipelineModule(
        layers=layers,
        loss_fn=lm_loss_fn,
        num_stages=pp_size,
        topology=topo,
        partition_method="parameters",  # simple automatic partitioning
        activation_checkpoint_interval=0,
    )
    return pipe


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_size", type=int, default=50257)
    p.add_argument("--hidden_size", type=int, default=768)
    p.add_argument("--n_layers", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--num_samples", type=int, default=50_000)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--pp", type=int, default=2, help="pipeline parallel size")
    p.add_argument("--micro_batch", type=int, default=4, help="per-GPU micro batch")
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--fp16", action="store_true")
    args = p.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # dataset
    dataset = ToyTextDataset(
        num_samples=args.num_samples, seq_len=args.seq_len, vocab_size=args.vocab_size
    )

    # pipeline model
    pipe = build_pipeline_model(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        max_len=args.seq_len,
        pp_size=args.pp,
        world_size=world_size,
    )

    ds_config = {
        "train_micro_batch_size_per_gpu": args.micro_batch,
        "gradient_accumulation_steps": args.grad_accum,
        "fp16": {"enabled": bool(args.fp16)},
        "zero_optimization": {"stage": 0},
        "steps_per_print": 10,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },
    }

    engine, _, _, _ = deepspeed.initialize(
        model=pipe,
        model_parameters=[p for p in pipe.parameters() if p.requires_grad],
        training_data=dataset,
        config=ds_config,
    )

    for step in range(1, args.steps + 1):
        loss = engine.train_batch()
        if engine.global_rank == 0 and step % 5 == 0:
            try:
                loss_val = float(loss)
            except Exception:
                loss_val = float(loss.item())
            print(f"[step {step}] loss={loss_val:.4f}")


if __name__ == "__main__":
    main()
