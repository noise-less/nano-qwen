# lightning_train_projection.py
import math
import torch
import lightning as L
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from model.qwen3v import Qwen3V
from model.processor import Processor
from data.llava import LLaVAPretrainDataset

batch_size = 16
max_seq_len = 1024
epochs = 1
grad_accum = 1
lr = 5e-4
weight_decay = 0.01

devices = 2
strategy = "ddp"
precision = "bf16-mixed"
model_variant = "4B"


# ---------- utils ----------
def freeze_except_projection(model: Qwen3V):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.projection.parameters():
        p.requires_grad = True
    # keep backbone deterministic (no dropout), but still allow grad flow
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
            ids = out["input_ids"].squeeze(0)  # (T,)
            pix = out["pixels"]  # (Npatch, D) or None
            dimg = out["d_image"]  # (1,3) or None

            # Avoid truncating inside an image block
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
                dimg_list.append(dimg.squeeze(0))  # -> (3,)

        if not ids_list:
            return None  # drop batch

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
    # logits: (B, T, V), labels: (B, T)
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    return nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
    )


# ---------- LightningModule ----------
class ProjectionOnlyLitModule(L.LightningModule):
    def __init__(
        self,
        model: Qwen3V,
        lr=5e-4,
        weight_decay=0.01,
        steps_total: int = 1000,
        warmup_steps: int = 100,
        log_every: int = 50,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.steps_total = max(1, steps_total)
        self.warmup_steps = max(1, warmup_steps)
        self.log_every = log_every

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

        # log every N steps
        if (self.global_step + 1) % self.log_every == 0:
            avg_loss = loss.detach()
            ppl = math.exp(float(max(min(avg_loss, 20.0), 0.0)))
            self.log_dict(
                {
                    "train/loss": avg_loss,
                    "train/ppl": ppl,
                    "train/step": float(self.global_step + 1),
                },
                prog_bar=True,
                on_step=True,
                on_epoch=False,
                rank_zero_only=True,
            )
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.model.projection.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(
                1, self.steps_total - self.warmup_steps
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }


# ---------- Callback: save only projection each epoch ----------
class SaveProjectionCallback(Callback):
    def __init__(self, ckpt_path="proj_only.pt"):
        super().__init__()
        self.ckpt_path = ckpt_path

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            state = {
                "projection": {
                    k: v.detach().cpu()
                    for k, v in pl_module.model.projection.state_dict().items()
                }
            }
            torch.save(state, self.ckpt_path)
            print(
                f"[epoch {trainer.current_epoch}] saved projection to {self.ckpt_path}"
            )


# ---------- DataModule ----------
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

    def _collate_drop_none(self, batch):
        out = self.collate_fn(batch)
        return out if out is not None else None

    def train_dataloader(self):
        # We drop batches that return None via a small wrapper
        def collate(batch):
            out = self._collate_drop_none(batch)
            if out is None:
                # return a tiny dummy that will be skipped in step (rare)
                return None
            return out

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=(self.num_workers > 0),
        )


if __name__ == "__main__":
    # Load model & processor
    text_model_repo = f"Qwen/Qwen3-{model_variant}"
    vision_model_repo = "Qwen/Qwen2.5-VL-7B-Instruct"

    model = Qwen3V.from_pretrained_components(
        model_variant=model_variant,
        vision_model_repo=vision_model_repo,
        text_model_repo=text_model_repo,
    )
    processor = Processor(repo_id=text_model_repo, vision_config=model.vision_config)

    # Data
    dataset = LLaVAPretrainDataset(cache_dir="./cache")

    # Schedule math (rough; Lightning also knows steps_per_epoch but we keep your cosine)
    steps_per_epoch = max(1, math.ceil(len(dataset) / batch_size))
    total_steps = steps_per_epoch * epochs
    warmup = max(100, int(0.02 * total_steps))

    lit = ProjectionOnlyLitModule(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        steps_total=total_steps,
        warmup_steps=warmup,
        log_every=50,
    )
    dm = VLDataModule(
        dataset,
        processor,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_workers=4,
    )

    # (Optional) W&B logger; comment out if not using W&B
    logger = WandbLogger(project="qwen3v-projection", log_model=False)

    trainer = L.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        precision=precision,  # bf16 if supported
        max_epochs=epochs,
        accumulate_grad_batches=grad_accum,
        logger=logger,
        callbacks=[
            SaveProjectionCallback(ckpt_path="proj_only.pt"),
            ModelCheckpoint(save_last=True, save_top_k=0),
        ],
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    trainer.fit(lit, datamodule=dm)
