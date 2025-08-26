# lightning_qwen3v_projection.py
import os, math, argparse
import torch
import torch.nn.functional as F
import lightning as L
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from safetensors.torch import save_file

# FSDP plumbing
from functools import partial
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# your modules
from model.processor import Processor
from model.qwen3v import Qwen3V
from data.llava import LLaVAPretrainDataset


# -------------------- utils --------------------
def freeze_except_projection(model):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.projection.parameters():
        p.requires_grad = True
    model.visual.eval()
    model.model.eval()
    model.train()


def make_collate_fn(processor: Processor, max_seq_len: int = 2048):
    pad_id = processor.tokenizer.encode("<|endoftext|>").ids[0]
    img_pad_id = processor.image_pad_token_id

    def collate(batch):
        ids_list, labels_list, pixels_list, dimg_list = [], [], [], []
        for ex in batch:
            out = processor(messages=ex["messages"], device=None)
            ids = out["input_ids"].squeeze(0)
            pix = out["pixels"]
            dimg = out["d_image"]

            # avoid truncating through an image block
            if ids.numel() > max_seq_len:
                if (ids[max_seq_len:] == img_pad_id).any():
                    continue
                ids = ids[:max_seq_len]

            labels = ids.clone()
            labels[labels == img_pad_id] = -100
            ids_list.append(ids)
            labels_list.append(labels)
            if pix is not None:
                pixels_list.append(pix)
                dimg_list.append(dimg.squeeze(0))

        if not ids_list:
            return None

        input_ids = pad_sequence(ids_list, batch_first=True, padding_value=pad_id)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        labels[input_ids == pad_id] = -100

        pixels = torch.cat(pixels_list, dim=0) if pixels_list else None
        d_image = torch.stack(dimg_list, dim=0) if dimg_list else None
        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixels": pixels,
            "d_image": d_image,
        }

    return collate


def lm_ce_loss(logits, labels):
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
    )


# -------------------- LightningModule --------------------
class ProjectionOnlyLM(L.LightningModule):
    def __init__(
        self,
        model,
        lr=5e-4,
        weight_decay=0.01,
        total_steps=1000,
        warmup_steps=100,
        log_every=50,
        use_act_ckpt=False,
        block_cls=None,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.total_steps = max(1, int(total_steps))
        self.warmup_steps = max(1, int(warmup_steps))
        self.log_every = int(log_every)
        self.use_act_ckpt = use_act_ckpt
        self.block_cls = block_cls
        freeze_except_projection(self.model)

    def forward(self, input_ids, pixels=None, d_image=None):
        return self.model(input_ids=input_ids, pixels=pixels, d_image=d_image)

    def training_step(self, batch, batch_idx):
        if batch is None:
            return None
        logits = self(
            input_ids=batch["input_ids"],
            pixels=batch["pixels"],
            d_image=batch["d_image"],
        )
        loss = lm_ce_loss(logits, batch["labels"])
        if (self.global_step + 1) % self.log_every == 0:
            self.log(
                "train/loss", loss, prog_bar=True, on_step=True, rank_zero_only=True
            )
            ppl = torch.exp(loss.clamp(0, 20))
            self.log("train/ppl", ppl, on_step=True, rank_zero_only=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.model.projection.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        tsteps, wwarm = self.total_steps, self.warmup_steps

        def lr_lambda(step: int):
            if step < wwarm:
                return step / max(1, wwarm)
            prog = (step - wwarm) / max(1, tsteps - wwarm)
            return 0.5 * (1.0 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


# -------------------- Callbacks --------------------
class SaveProjectionSafetensors(L.Callback):
    def __init__(self, path: str = "projection.safetensors"):
        super().__init__()
        self.path = path

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: ProjectionOnlyLM):
        if trainer.is_global_zero:
            state = {
                f"projection.{k}": v.detach().cpu()
                for k, v in pl_module.model.projection.state_dict().items()
            }
            save_file(state, self.path)
            print(f"[epoch {trainer.current_epoch}] saved {self.path}")


class PrintEpochSummary(L.Callback):
    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: ProjectionOnlyLM):
        if trainer.is_global_zero:
            loss = trainer.callback_metrics.get("train/loss")
            ppl = trainer.callback_metrics.get("train/ppl")
            print(
                f"Epoch {trainer.current_epoch+1}/{trainer.max_epochs} | "
                f"loss={float(loss) if loss is not None else 'n/a'} | "
                f"ppl={float(ppl) if ppl is not None else 'n/a'}"
            )


# -------------------- DataModule --------------------
class VLDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset,
        processor: Processor,
        batch_size=16,
        max_seq_len=1024,
        num_workers=4,
    ):
        super().__init__()
        self.dataset = dataset
        self.processor = processor
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_workers = num_workers
        self.collate_fn = make_collate_fn(processor, max_seq_len=max_seq_len)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(self.num_workers > 0),
        )


# -------------------- main --------------------
def main():
    # helpful for Tensor Cores perf
    torch.set_float32_matmul_precision("high")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--precision", type=str, default="bf16-mixed")
    ap.add_argument("--strategy", type=str, default="ddp", choices=["ddp", "fsdp"])
    ap.add_argument("--proj_out", type=str, default="projection.safetensors")
    ap.add_argument("--model_variant", type=str, default="8B")
    ap.add_argument("--vision_repo", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--text_repo", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--processor_repo", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--cache_dir", type=str, default="./cache")
    args = ap.parse_args()

    # 1) Build model & processor on CPU (ensure your factory DOES NOT .to('cuda') internally)
    model = Qwen3V.from_pretrained_components(
        model_variant=args.model_variant,
        vision_model_repo=args.vision_repo,
        text_model_repo=args.text_repo,
    )
    processor = Processor(
        repo_id=args.processor_repo, vision_config=model.vision_config
    )

    # 2) Data
    dataset = LLaVAPretrainDataset(cache_dir=args.cache_dir)
    steps_per_epoch = max(1, math.ceil(len(dataset) / args.batch_size))
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(100, int(0.02 * total_steps))
    dm = VLDataModule(
        dataset=dataset,
        processor=processor,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
    )

    # 3) Figure out the Transformer block class (for wrapping/checkpointing)
    #    Assumes your blocks live at model.model.layers
    Block = type(model.model.model.layers[0])

    # 4) Lightning module
    use_act_ckpt = args.strategy == "fsdp"
    lit = ProjectionOnlyLM(
        model=model,
        lr=args.lr,
        weight_decay=args.weight_decay,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        log_every=10,
        use_act_ckpt=use_act_ckpt,
        block_cls=Block,
    )

    # 5) Strategy
    if args.strategy == "fsdp":
        auto_wrap = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        strategy = FSDPStrategy(
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy, transformer_layer_cls={Block}
            ),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            limit_all_gathers=True,
            activation_checkpointing_policy={Block},
        )

    else:
        strategy = "ddp"

    # 6) Trainer
    trainer = L.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy=strategy,
        precision=args.precision,
        max_epochs=args.epochs,
        accumulate_grad_batches=args.grad_accum,
        log_every_n_steps=10,
        callbacks=[SaveProjectionSafetensors(args.proj_out), PrintEpochSummary()],
        enable_progress_bar=True,
    )

    trainer.fit(lit, datamodule=dm)


"""
PYTHONPATH=. python train/s2_qwen3v.py \
    --devices 2 \
    --batch_size 16 \
    --epochs 1 \
    --grad_accum 1 \
    --max_seq_len 1024 \
    --lr 2e-3 \
    --weight_decay 0.01 \
    --num_workers 4 \
    --precision bf16-mixed \
    --strategy ddp \
    --proj_out projection.safetensors \
    --model_variant 4B \
    --vision_repo Qwen/Qwen2.5-VL-7B-Instruct \
    --text_repo Qwen/Qwen3-4B-Instruct-2507 \
    --processor_repo Qwen/Qwen2.5-VL-7B-Instruct \
    --cache_dir ./cache
"""

if __name__ == "__main__":
    main()
