import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.model import Qwen3VL
from transformers import AutoProcessor
from huggingface_hub import snapshot_download


def test_text_only_generation():
    model_name = "Qwen/Qwen3-VL-4B-Instruct"
    weights_path = snapshot_download(repo_id=model_name, cache_dir=".cache")
    model = Qwen3VL.from_pretrained(weights_path=weights_path, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_name)

    query = "What is the purpose of filing your hand for rock climbing?"
    messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    output_ids = model.generate(input_ids=inputs.input_ids, max_new_tokens=16)
    output_text = processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print(output_text)

    # expected = "user\nWhat is the purpose of filing your hand for rock climbing?\nassistant\nFiling your hand for rock climbing — specifically, **filing your fingers and"
    # assert output_text == [expected], f"Expected: {expected}\nGot: {output_text[0]}"
    # print("========== Test Passed ==========")


if __name__ == "__main__":
    test_text_only_generation()
