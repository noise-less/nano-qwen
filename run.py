import os
import time
from typing import List, Dict

import torch
from prompt_toolkit import PromptSession, print_formatted_text, HTML
from prompt_toolkit.shortcuts import choice
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.filters import is_done
from transformers import AutoProcessor
from huggingface_hub import snapshot_download

from model.model import Qwen3VL

ASCII_LOGO = """
██╗    ████████╗██╗███╗   ██╗██╗   ██╗    ██████╗ ██╗    ██╗███████╗███╗   ██╗
╚██╗   ╚══██╔══╝██║████╗  ██║╚██╗ ██╔╝   ██╔═══██╗██║    ██║██╔════╝████╗  ██║
 ╚██╗     ██║   ██║██╔██╗ ██║ ╚████╔╝    ██║   ██║██║ █╗ ██║█████╗  ██╔██╗ ██║
 ██╔╝     ██║   ██║██║╚██╗██║  ╚██╔╝     ██║▄▄ ██║██║███╗██║██╔══╝  ██║╚██╗██║
██╔╝      ██║   ██║██║ ╚████║   ██║      ╚██████╔╝╚███╔███╔╝███████╗██║ ╚████║
╚═╝       ╚═╝   ╚═╝╚═╝  ╚═══╝   ╚═╝       ╚══▀▀═╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝
"""

STARTING_HELP_TEXT = """
Welcome to Tiny-Qwen Interactive Chat!

Tips:
1. /help for more information.
2. /exit or Ctrl+C to exit.
"""

ALL_MODELS = [
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-2B-Thinking",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-4B-Thinking",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-8B-Thinking",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Thinking",
    "Qwen/Qwen3-VL-32B-Instruct",
    "Qwen/Qwen3-VL-32B-Thinking",
]


_session = PromptSession()


def log_help_message():
    log_text(
        """
<b>Commands:</b>
  <b>/help</b>    Show this message again
  <b>/switch</b>  Change to a different model
  <b>/reset</b>   Clear the current conversation history
  <b>/exit</b>    Quit the CLI
"""
    )


def log_startup_banner():
    log_text(ASCII_LOGO)
    log_text(STARTING_HELP_TEXT)
    log_help_message()


def pick_model() -> str:
    model = prompt_user_choice("Select a Qwen3-VL model", ALL_MODELS + ["Exit"])
    if model == "Exit":
        return ""
    return model


def load_model_and_processor(repo_id: str) -> tuple[Qwen3VL, AutoProcessor, str]:
    log_text(f"<i>Downloading weights for <b>{repo_id}</b>...</i>")
    weights_path = snapshot_download(repo_id=repo_id, cache_dir=".cache")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_text(f"<i>Loading model on <b>{device}</b>...</i>")
    model = Qwen3VL.from_pretrained(weights_path=weights_path, device_map="auto")
    model = model.to(device)
    processor = AutoProcessor.from_pretrained(repo_id)
    return model, processor, device


def build_prompt_message(content: str) -> Dict:
    return {"role": "user", "content": [{"type": "text", "text": content}]}


def build_assistant_message(content: str) -> Dict:
    return {"role": "assistant", "content": [{"type": "text", "text": content}]}


def generate_completion(
    messages: List[Dict],
    model: Qwen3VL,
    processor: AutoProcessor,
    device: str,
    max_new_tokens: int,
) -> str:
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(device)
    output_ids = model.generate(
        input_ids=inputs.input_ids,
        max_new_tokens=max_new_tokens,
    )
    decoded = processor.batch_decode(
        output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded[0]


def prompt_user_query(keywords: list[str] = [], default: str = "") -> str:
    word_completer = WordCompleter(keywords, ignore_case=True)
    if not keywords:
        word_completer = None

    bindings = KeyBindings()

    @bindings.add("enter")
    def _(event):
        """Submit on Enter"""
        event.current_buffer.validate_and_handle()

    @bindings.add("c-j")
    def _(event):
        """Newline on Ctrl+J"""
        event.current_buffer.insert_text("\n")

    print_formatted_text("")
    user_input = _session.prompt(
        HTML("<b>></b> "),
        completer=word_completer,
        cursor=CursorShape.BLINKING_BLOCK,
        multiline=True,
        prompt_continuation=lambda width, _, __: " " * width,
        key_bindings=bindings,
        default=default,
    )
    return user_input.strip()


def prompt_user_choice(prompt: str, options: list[str]) -> str:
    style = Style.from_dict(
        {
            "frame.border": "white",
            "selected-option": "bold",
            "bottom-toolbar": "#ffffff bg:#333333 noreverse",
        }
    )

    result = choice(
        message=HTML(f"<u>{prompt}</u>:"),
        options=[(option, option) for option in options],
        style=style,
        bottom_toolbar=HTML(
            " Press <b>[Up]</b>/<b>[Down]</b> to select, <b>[Enter]</b> to accept."
        ),
        show_frame=~is_done,
    )
    return result


def log_text(text: str):
    print_formatted_text("")
    print_formatted_text(HTML(text))


def render_datetime(name: str) -> str:
    timestamp = time.strftime("%Y%m%d%H%M")
    sanitized_name = name.lower().replace(" ", "_")
    return f"{timestamp}_{sanitized_name}"


def list_text_files(folder_path: str) -> list[str]:
    files = []
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            files.append(os.path.join(folder_path, file))
    return files


def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


def main():
    clear_console()
    log_startup_banner()

    model_repo = pick_model()
    if not model_repo:
        log_text("Exiting...")
        return

    try:
        model, processor, device = load_model_and_processor(model_repo)
    except Exception as load_error:
        log_text(f"<red>Failed to load model: {load_error}</red>")
        return

    history: List[Dict] = []
    max_new_tokens = 64
    default_input = ""

    while True:
        user_input = prompt_user_query(
            keywords=["/help", "/switch", "/reset", "/exit"], default=default_input
        )
        default_input = ""

        if not user_input:
            continue
        if user_input == "/exit":
            log_text("Goodbye!")
            break
        if user_input == "/help":
            log_help_message()
            continue
        if user_input == "/reset":
            history.clear()
            log_text("<i>Conversation history cleared.</i>")
            continue
        if user_input == "/switch":
            model_repo = pick_model()
            if not model_repo:
                log_text("Goodbye!")
                break
            try:
                model, processor, device = load_model_and_processor(model_repo)
                history.clear()
            except Exception as load_error:
                log_text(f"<red>Failed to load model: {load_error}</red>")
                continue
            log_text(f"<i>Switched to <b>{model_repo}</b>.</i>")
            continue

        history.append(build_prompt_message(user_input))
        log_text("<i>Generating response...</i>")
        try:
            response = generate_completion(
                messages=history,
                model=model,
                processor=processor,
                device=device,
                max_new_tokens=max_new_tokens,
            )
        except Exception as generation_error:
            log_text(f"<red>Generation failed: {generation_error}</red>")
            history.pop()
            continue

        assistant_text = response.strip()
        history.append(build_assistant_message(assistant_text))
        log_text(f"<b>ASSISTANT:</b>\n{assistant_text}")


if __name__ == "__main__":
    main()
