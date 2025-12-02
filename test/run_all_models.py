import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from test_text_only import run_text_only_generation
from test_text_plus_image import run_text_plus_image_generation

DEFAULT_MODELS = [
    "Qwen3-VL-2B-Instruct",
    "Qwen3-VL-2B-Thinking",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-4B-Thinking",
    "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-8B-Thinking",
    "Qwen3-VL-30B-A3B-Instruct",
    "Qwen3-VL-30B-A3B-Thinking",
    "Qwen3-VL-32B-Instruct",
    "Qwen3-VL-32B-Thinking",
]


def build_repo_id(model_suffix: str) -> str:
    return f"Qwen/{model_suffix}"


def run_suite(
    model_suffixes: list[str],
    max_new_tokens: int,
    device: str,
    image_path: Path,
):
    for suffix in model_suffixes:
        repo_id = build_repo_id(suffix)
        print(f"\n=== Running text-only generation for {repo_id} ===")
        text_only_outputs = run_text_only_generation(
            model_name=repo_id,
            max_new_tokens=max_new_tokens,
            device=device,
        )
        for idx, output in enumerate(text_only_outputs["ours"]):
            print(f"[ours][text-only][{repo_id}][sample-{idx}]: {output}")
        for idx, output in enumerate(text_only_outputs["transformers"]):
            print(f"[transformers][text-only][{repo_id}][sample-{idx}]: {output}")

        print(f"\n=== Running text+image generation for {repo_id} ===")
        text_plus_image_outputs = run_text_plus_image_generation(
            model_name=repo_id,
            max_new_tokens=max_new_tokens,
            device=device,
            image_path=image_path,
        )
        for idx, output in enumerate(text_plus_image_outputs["ours"]):
            print(f"[ours][text+image][{repo_id}][sample-{idx}]: {output}")
        for idx, output in enumerate(text_plus_image_outputs["transformers"]):
            print(
                f"[transformers][text+image][{repo_id}][sample-{idx}]: {output}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL generation smoke tests across model variants."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Model suffixes to evaluate (prefix Qwen/ added automatically).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate for each request.",
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
        default=Path("test/data/test-img-1.jpg"),
        help="Image path to use for the text+image generation test.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_suite(
        model_suffixes=args.models,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        image_path=args.image_path,
    )


if __name__ == "__main__":
    main()

