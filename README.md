<p align="left">
    English | <a href="README_CN.md">中文</a>
</p>

<p align="center">
    <img src="data/chat.jpg" alt="Tiny Qwen Interactive Chat">
</p>

## ✨ Tiny Qwen

A minimal, easy-to-read PyTorch re-implementation of `Qwen3-VL`, supporting both text + vision as well as dense and mixture of experts.

If you find Hugging Face code verbose and challenging to interpret, this repo is for you!

Join my [Discord channel](https://discord.gg/sBNnqP9gaY) for more discussion!

For `Qwen3` text-only version and `Qwen2.5 VL` of this repo, see [this branch](https://github.com/Emericen/tiny-qwen/tree/legacy/qwen2_5).

## 🦋 Quick Start

I recommend using `uv` and creating a virtual environment:

```bash
pip install uv && uv venv

# activate the environment
source .venv/bin/activate # Linux/macOS
.venv\Scripts\activate # Windows

# install dependencies
uv pip install -r requirements.txt
```

Launch the interactive chat:

```bash
python run.py
```

**Note:** `Qwen3` is text-only. Use `@path/to/image.jpg` to reference images with `Qwen2.5-VL`.

```
USER: @data/test-img-1.jpg tell me what you see in this image?
✓ Found image: data/test-img-1.jpg
ASSISTANT: The image shows a vibrant sunflower field with a close-up of a sunflower...
```

## 📝 Code Examples

Using `Qwen3VL` class in code:

```python
from PIL import Image
from huggingface_hub import snapshot_download
from model.model import Qwen3VL
from model.processor import Processor

image = Image.open("test/data/test-img-1.jpg")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "What's on this image?"},
        ],
    },
]

model_name = "Qwen/Qwen3-VL-4B-Instruct"
weights = snapshot_download(repo_id=model_name, cache_dir=".cache")
model = Qwen3VL.from_pretrained(weights_path=weights, device_map="auto")
processor = Processor.from_pretrained(model_name)

device = next(model.parameters()).device
inputs = processor(messages, add_generation_prompt=True, device=device)

output_ids = model.generate(**inputs, max_new_tokens=64)
print(processor.tokenizer.decode(output_ids[0].tolist()))

print("Streaming output:", end=" ", flush=True)
for token_id in model.generate_stream(**inputs, max_new_tokens=64):
    print(processor.tokenizer.decode([token_id]), end="", flush=True)
print()
```
