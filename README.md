# Nano Qwen

A minimal, PyTorch re‑implementation of **Qwen3‑VL**, supporting both text and vision. Includes dense and MoE variants.

For older `Qwen3` (text‑only) and `Qwen2.5‑VL` support, visit the legacy branch.

---

## 🚀 Quick Start

Create and activate a virtual environment:

```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Run the chat interface:

```bash
python run.py
```

---

## 📘 Code Example

Using the `Qwen3VL` model directly:

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

---

## 📌 Disclaimer

This project is a modified and rebranded fork of **Emericen/tiny-qwen**.
