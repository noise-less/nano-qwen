import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
from model.model import Qwen3VL
from transformers import AutoProcessor
from huggingface_hub import snapshot_download


def test_text_plus_image_generation():
    model_name = "Qwen/Qwen3-VL-4B-Instruct"
    weights_path = snapshot_download(repo_id=model_name, cache_dir=".cache")
    model = Qwen3VL.from_pretrained(weights_path=weights_path, device_map="auto")
    processor = AutoProcessor.from_pretrained(model_name)

    image_1 = Image.open("test/data/test-img-1.jpg")
    image_2 = Image.open("test/data/test-img-2.jpg")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_1},
                {"type": "text", "text": "what is in the image?"},
                # {"type": "image", "image": image_2},
                # {"type": "text", "text": "now what about this image? answer in 2 sentences, one for each image"},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Extract vision inputs
    # pixel_values shape: [num_patches, patch_dim] where patch_dim = 3*temporal*spatial*spatial
    # image_grid_thw shape: [num_images, 3] where 3 = [t, h, w] dimensions
    pixels = inputs.get("pixel_values", None)
    d_image = inputs.get("image_grid_thw", None)

    output_ids = model.generate(
        input_ids=inputs.input_ids,
        pixels=pixels,
        d_image=d_image,
        max_new_tokens=16,
    )
    output_text = processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )  # ['The image shows a close-up of a sunflower in a field. A bee']

    print(output_text)

    # expected = "user\nwhat is in the image?\nassistant\nThe image shows a sunflower field with a sunflower with a bee on it"
    # assert output_text == [expected], f"Expected: {expected}\nGot: {output_text[0]}"
    # print("========== Test Passed ==========")


if __name__ == "__main__":
    test_text_plus_image_generation()
