import torch
import math

from model.processor import Processor
from model.qwen3v import Qwen3V

from data.llava import LLaVAPretrainDataset
from torch.nn.utils.rnn import pad_sequence


def freeze_except_projection(model: Qwen3V):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.projection.parameters():
        p.requires_grad = True
    # keep the frozen parts in eval mode to disable dropout etc.
    model.visual.eval()
    model.model.eval()
    model.train()


def make_collate_fn(processor: Processor, max_seq_len: int = 2048):
    # Pick a safe pad id; using <|endoftext|> as a no-op pad token.
    pad_id = processor.tokenizer.encode("<|endoftext|>").ids[0]
    img_pad_id = processor.image_pad_token_id

    def collate(batch):
        input_ids_list = []
        labels_list = []
        pixels_list = []
        d_image_list = []

        for ex in batch:
            # Processor builds the text+image sequence with <|image_pad|> block(s)
            out = processor(messages=ex["messages"], device=None)

            ids = out["input_ids"].squeeze(0)  # (T,)
            pix = out["pixels"]  # (num_patches, patch_dim) or None
            dimg = out["d_image"]  # (num_images, 3) or None

            # Truncation policy: right-truncate only if necessary.
            # IMPORTANT: we want to avoid chopping *inside* the image pad block.
            # For simplicity, if truncation would hit image pads, just skip this sample.
            if ids.numel() > max_seq_len:
                # naive check: if any image pad token lies beyond max_seq_len, skip example
                if (ids[max_seq_len:] == img_pad_id).any():
                    continue
                ids = ids[:max_seq_len]

            input_ids_list.append(ids)

            # Build labels: next-token prediction on all text tokens.
            # We ignore loss on image pad positions (set to -100 after shift).
            labels = ids.clone()
            labels[labels == img_pad_id] = -100
            labels_list.append(labels)

            if pix is not None:
                pixels_list.append(pix)  # concat later along dim=0
                # out["d_image"] is shape (num_images, 3); LLaVA pretrain has exactly 1
                d_image_list.append(dimg.squeeze(0))  # -> (3,)

        # Dynamic right padding to batch max length
        if len(input_ids_list) == 0:
            # rare case: all skipped due to truncation; let DataLoader retry
            return None

        input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_id
        )  # (B, T*)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

        # pixel stream: concat all images in the batch (order must match the order
        # of <|image_pad|> blocks scanned row-major by masked_scatter)
        pixels = (
            torch.cat(pixels_list, dim=0) if pixels_list else None
        )  # (sum_patches, patch_dim)
        d_image = torch.stack(d_image_list, dim=0) if d_image_list else None  # (B, 3)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixels": pixels,
            "d_image": d_image,
        }

    return collate


def lm_ce_loss(logits, labels):
    # logits: (B, T, V), labels: (B, T)
    # shift for next-token prediction
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    return torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
    )


def train_projection_only(
    model: Qwen3V,
    processor: Processor,
    dataset,
    device="cuda",
    batch_size=8,
    max_seq_len=2048,
    lr=5e-4,
    weight_decay=0.01,
    num_epochs=1,
    grad_accum_steps=1,
    amp=True,
    num_workers=4,
    ckpt_path="proj_only.pt",
):
    freeze_except_projection(model)
    model.to(device)

    collate_fn = make_collate_fn(processor, max_seq_len=max_seq_len)

    from torch.utils.data import DataLoader

    def _collate_drop_none(batch):
        out = collate_fn(batch)
        # If the collate returned None (e.g., every sample in that mini-batch was skipped),
        # return an empty batch so DataLoader can continue; we’ll guard in the loop.
        return out or {"skip": True}

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

    # Only projection parameters will be optimized
    optim = torch.optim.AdamW(
        model.projection.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    # Cosine scheduler with linear warmup (tiny and simple)
    total_steps = max(1, (len(loader) // max(1, grad_accum_steps)) * num_epochs)
    warmup = max(100, int(0.02 * total_steps))  # 2% or at least 100 steps

    def lr_sched(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_sched)

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    global_step = 0
    running = 0.0
    model.train()  # remember: only projection is actually in train mode

    for epoch in range(num_epochs):
        for batch in loader:
            if "skip" in batch:
                continue

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            pixels = batch["pixels"]
            d_image = batch["d_image"]

            if pixels is not None:
                pixels = pixels.to(device, non_blocking=True)
            if d_image is not None:
                d_image = d_image.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(input_ids=input_ids, pixels=pixels, d_image=d_image)
                loss = lm_ce_loss(logits, labels) / grad_accum_steps

            scaler.scale(loss).backward()
            running += loss.item() * grad_accum_steps

            if (global_step + 1) % grad_accum_steps == 0:
                # projection-only grad clip (a little safety)
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.projection.parameters(), 1.0)

                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()

            global_step += 1

            if global_step % 50 == 0:
                print(
                    f"[epoch {epoch}] step {global_step} | loss {running/50:.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
                )
                running = 0.0

        # save projection weights each epoch
        torch.save(
            {
                "projection": model.projection.state_dict(),
                "optim": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
            },
            ckpt_path,
        )
        print(f"Saved projection checkpoint to {ckpt_path}")


if __name__ == "__main__":
    model = Qwen3V(model_variant="1.7B").cuda()

    processor = Processor(
        repo_id="Qwen/Qwen3-1.7B-Instruct",
        vision_config=model.vision_config,
    )
    pretrain_dataset = LLaVAPretrainDataset(cache_dir="./cache")

    # messages = pretrain_dataset[0]["messages"]

    # inputs = processor(messages=messages, device="cuda")
    # print("Input shapes:")
    # print(f"input_ids: {inputs['input_ids'].shape}")
    # print(f"pixels: {inputs['pixels']}")
    # print(f"d_image: {inputs['d_image']}")

    # # Test forward pass
    # model.eval()
    # with torch.no_grad():
    #     logits = model(
    #         input_ids=inputs["input_ids"],
    #         pixels=inputs["pixels"],
    #         d_image=inputs["d_image"],
    #     )
    #     print(f"Output logits shape: {logits.shape}")

    train_projection_only(
        model=model,
        processor=processor,
        dataset=pretrain_dataset,
        device="cuda",
        batch_size=8,
        max_seq_len=1024,  # these samples are short; 1024 is plenty
        lr=5e-4,
        weight_decay=0.01,
        num_epochs=1,
        grad_accum_steps=1,
        amp=True,
        num_workers=4,
        ckpt_path="proj_only.pt",
    )
