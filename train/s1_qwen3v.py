import math
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedDataParallelKwargs

from model.processor import Processor
from model.qwen3v import Qwen3V
from data.llava import LLaVAPretrainDataset


def freeze_except_projection(model: Qwen3V):
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
            ids = out["input_ids"].squeeze(0)  # (T,)
            pix = out["pixels"]  # (Npatch, D) or None
            dimg = out["d_image"]  # (1,3) or None

            # Avoid truncating inside an image block
            if ids.numel() > max_seq_len:
                if (ids[max_seq_len:] == img_pad_id).any():
                    continue
                ids = ids[:max_seq_len]

            # labels: ignore image-pad positions now; right-pad later
            labels = ids.clone()
            labels[labels == img_pad_id] = -100

            ids_list.append(ids)
            labels_list.append(labels)
            if pix is not None:
                pixels_list.append(pix)
                dimg_list.append(dimg.squeeze(0))  # -> (3,)

        if not ids_list:
            return None

        input_ids = pad_sequence(ids_list, batch_first=True, padding_value=pad_id)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        # IMPORTANT: ignore right-padding in the loss too
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
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
    )


def train_projection_only(
    model: Qwen3V,
    processor: Processor,
    dataset,
    batch_size=8,
    max_seq_len=1024,
    lr=5e-4,
    weight_decay=0.01,
    num_epochs=1,
    grad_accum_steps=1,
    amp=True,
    num_workers=4,
    ckpt_path="proj_only.pt",
):
    # --- Accelerate setup (with W&B logging) ---
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum_steps,
        mixed_precision="bf16" if amp else "no",
        log_with="wandb",
        kwargs_handlers=[ddp_kwargs],
    )
    accelerator.init_trackers(
        project_name="qwen3v-projection",
        config=dict(
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            lr=lr,
            weight_decay=weight_decay,
            epochs=num_epochs,
            grad_accum_steps=grad_accum_steps,
            amp=amp,
        ),
    )

    freeze_except_projection(model)

    collate_fn = make_collate_fn(processor, max_seq_len=max_seq_len)

    def _collate_drop_none(batch):
        out = collate_fn(batch)
        return out or {"_skip": True}

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_drop_none,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )

    optim = torch.optim.AdamW(
        model.projection.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    # Rough steps/epoch (will be a bit off if many batches are skipped)
    steps_per_epoch = max(1, math.ceil(len(dataset) / batch_size))
    total_steps = steps_per_epoch * num_epochs
    warmup = max(100, int(0.02 * total_steps))

    def lr_sched(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_sched)

    # Prepare everything for distributed
    model, optim, loader, scheduler = accelerator.prepare(
        model, optim, loader, scheduler
    )

    running = 0.0
    global_step = 0

    for epoch in range(num_epochs):
        for batch in loader:
            if "_skip" in batch:
                continue

            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                pixels = batch["pixels"]
                d_image = batch["d_image"]

                logits = model(input_ids=input_ids, pixels=pixels, d_image=d_image)
                loss = lm_ce_loss(logits, labels)

                accelerator.backward(loss)
                optim.step()
                optim.zero_grad(set_to_none=True)
                scheduler.step()

                running += loss.item()
                global_step += 1

                if global_step % 50 == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    avg_loss = running / 50
                    try:
                        ppl = math.exp(max(min(avg_loss, 20.0), 0.0))
                    except OverflowError:
                        ppl = float("inf")

                    accelerator.log(
                        {
                            "train/loss": avg_loss,
                            "train/ppl": ppl,
                            "train/lr": lr_now,
                            "train/step": global_step,
                        },
                        step=global_step,
                    )
                    if accelerator.is_local_main_process:
                        print(
                            f"step {global_step}/{total_steps} | loss {avg_loss:.4f} | lr {lr_now:.2e}"
                        )
                    running = 0.0

        # Save projection weights each epoch (rank-0 only)
        if accelerator.is_main_process:
            state = {"projection": accelerator.get_state_dict(model)["projection"]}
            accelerator.save(state, ckpt_path)
            print(f"Saved projection checkpoint to {ckpt_path}")

    accelerator.end_training()


if __name__ == "__main__":
    # Run with:
    #   accelerate config
    #   accelerate launch -m train.s1_qwen3v

    # Model + processor
    model = Qwen3V(model_variant="1.7B")
    processor = Processor(repo_id="Qwen/Qwen3-1.7B", vision_config=model.vision_config)

    # Data
    pretrain_dataset = LLaVAPretrainDataset(cache_dir="./cache")

    # Train
    train_projection_only(
        model=model,
        processor=processor,
        dataset=pretrain_dataset,
        batch_size=16,
        max_seq_len=1024,
        lr=5e-4,
        weight_decay=0.01,
        num_epochs=1,
        grad_accum_steps=1,
        amp=True,
        num_workers=4,
        ckpt_path="proj_only.pt",
    )
