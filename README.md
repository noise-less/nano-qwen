<p align="left">
    English | <a href="README_CN.md">中文</a>
</p>

<p align="center">
    <img src="test/data/banner-1.png" alt="Tiny Qwen Interactive Chat">
</p>

## ✨ Tiny Qwen

A minimal, easy-to-read PyTorch re-implementation of of `Qwen3-VL`. Supports text and vision as well as dense and mixture of experts.

For `Qwen3` (text-only) and `Qwen2.5 VL` support, see [this branch](https://github.com/Emericen/tiny-qwen/tree/legacy/qwen2_5). 

For `DeepSeek R1`, see [this repo](https://github.com/Emericen/tiny-deepseek-r1).

Join my [Discord channel](https://discord.gg/sBNnqP9gaY) for more discussion!

## 🎇 Quick Start

Create a virtual environment:

```bash
pip install uv 
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Launch the interactive chat:

```bash
python run.py
```

**Note:** Use `@relative/path/to/image.jpg` to reference images.

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
