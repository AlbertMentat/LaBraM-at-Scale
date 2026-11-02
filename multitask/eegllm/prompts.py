"""Prompt formatting for Qwen3 (post-trained hybrid-thinking checkpoints) in non-thinking mode.

The string reproduced here equals
    tokenizer.apply_chat_template([system, user], add_generation_prompt=True, enable_thinking=False)
i.e. the assistant turn starts with an empty `<think>\n\n</think>\n\n` block (verified in tools/smoke_bench.py).
EEG tokens are K copies of the existing special token `<|vision_pad|>` wrapped by `<|vision_start|>` /
`<|vision_end|>`; their input embeddings are replaced by the projector outputs, so no vocabulary is added.
"""
from __future__ import annotations

import re

EEG_START = '<|vision_start|>'
EEG_PAD = '<|vision_pad|>'
EEG_END = '<|vision_end|>'
IM_START = '<|im_start|>'
IM_END = '<|im_end|>'
SYSTEM_PROMPT = 'You are an expert assistant for EEG analysis.'


def user_message(instruction: str, classes: list[str], num_eeg_tokens: int) -> str:
    return (f'EEG: {EEG_START}{EEG_PAD * num_eeg_tokens}{EEG_END}\n'
            f'{instruction}\n'
            f'Options: {", ".join(classes)}.\n'
            f'Answer with exactly one option.')


def prefix_text(instruction: str, classes: list[str], num_eeg_tokens: int) -> str:
    """Everything before the answer (generation prompt, thinking disabled)."""
    return (f'{IM_START}system\n{SYSTEM_PROMPT}{IM_END}\n'
            f'{IM_START}user\n{user_message(instruction, classes, num_eeg_tokens)}{IM_END}\n'
            f'{IM_START}assistant\n<think>\n\n</think>\n\n')


def chat_messages(instruction: str, classes: list[str], num_eeg_tokens: int) -> list[dict]:
    return [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_message(instruction, classes, num_eeg_tokens)}]


def target_text(class_name: str) -> str:
    """Answer = class name followed by the end-of-turn token (scored and trained on)."""
    return f'{class_name}{IM_END}'


_STRIP = re.compile(r'^[\s"\'`*(\[]+|[\s"\'`*.)\]!]+$')


def parse_generation(text: str, classes: list[str]) -> int:
    """Exact-match parse of a free generation; returns class index or -1 (unparseable)."""
    t = text.split(IM_END)[0]
    t = t.replace('<|endoftext|>', '')
    t = re.sub(r'<think>.*?</think>', '', t, flags=re.S)
    t = _STRIP.sub('', t.strip())
    t = t.casefold()
    for i, c in enumerate(classes):
        if t == c.casefold():
            return i
    return -1
