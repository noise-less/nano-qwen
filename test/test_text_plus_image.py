import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import snapshot_download
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)

from model.model import Qwen3VL
from model.processor import Processor

DEFAULT_IMAGE_PATH = Path("test/data/test-img-1.jpg")
DEFAULT_PROMPT = "what is in the image?"


def _is_moe_model(model_name: str) -> bool:
    return "A3B" in model_name


def _move_to_device(batch: dict, device: torch.device) -> dict:
    for key in ("input_ids", "pixels", "d_image", "pixel_values", "image_grid_thw"):
        tensor = batch.get(key)
        if tensor is not None and hasattr(tensor, "to"):
            batch[key] = tensor.to(device)
    return batch


def run_text_plus_image_generation(
    model_name: str,
    max_new_tokens: int,
    device: str = "cuda",
    image_path: Path = DEFAULT_IMAGE_PATH,
) -> dict:
    requested_device = torch.device(
        device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
    )

    weights_path = snapshot_download(repo_id=model_name, cache_dir=".cache")
    our_model = Qwen3VL.from_pretrained(weights_path=weights_path, device_map="auto")
    our_model.eval()
    our_processor = Processor.from_pretrained(model_name)

    hf_processor = AutoProcessor.from_pretrained(model_name)
    hf_model_cls = (
        Qwen3VLMoeForConditionalGeneration
        if _is_moe_model(model_name)
        else Qwen3VLForConditionalGeneration
    )
    if requested_device.type == "cuda":
        hf_model = hf_model_cls.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
    else:
        hf_model = hf_model_cls.from_pretrained(model_name)
        hf_model.to(requested_device)
    hf_model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": str(image_path)},
                {"type": "text", "text": DEFAULT_PROMPT},
            ],
        }
    ]

    our_inputs = our_processor(messages, add_generation_prompt=True)
    our_inputs = _move_to_device(our_inputs, requested_device)
    our_output_ids = our_model.generate(
        input_ids=our_inputs["input_ids"],
        pixels=our_inputs.get("pixels"),
        d_image=our_inputs.get("d_image"),
        max_new_tokens=max_new_tokens,
    )
    our_outputs = hf_processor.batch_decode(
        our_output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    image = Image.open(image_path)
    hf_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": DEFAULT_PROMPT},
            ],
        }
    ]

    hf_inputs = hf_processor.apply_chat_template(
        hf_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    hf_inputs = _move_to_device(hf_inputs, requested_device)
    hf_output_ids = hf_model.generate(
        **hf_inputs,
        max_new_tokens=max_new_tokens,
    )
    hf_outputs = hf_processor.batch_decode(
        hf_output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return {"ours": our_outputs, "transformers": hf_outputs}


def main():
    parser = argparse.ArgumentParser(description="Run text+image generation test.")
    parser.add_argument("model_name", type=str, help="HuggingFace model identifier.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on (e.g., cuda or cpu).",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="Path to an image file to include in the prompt.",
    )
    args = parser.parse_args()

    outputs = run_text_plus_image_generation(
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        image_path=args.image_path,
    )
    for idx, output in enumerate(outputs["ours"]):
        print(f"[ours][text+image][{args.model_name}][sample-{idx}]: {output}")
    for idx, output in enumerate(outputs["transformers"]):
        print(f"[transformers][text+image][{args.model_name}][sample-{idx}]: {output}")


if __name__ == "__main__":
    main()
