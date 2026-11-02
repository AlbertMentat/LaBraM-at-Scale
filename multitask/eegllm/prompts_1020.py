"""Multi-template prompts for the 23 Omni23 tasks (1020).

8 templates per task.  The six 1019 tasks keep their hand-written 1019 paraphrases; the other 17 get 8 templates
built from the task instruction (template 0 = the canonical instruction from configs/tasks_omni23_1020.yaml) and a
per-task short description/question.  Training shuffles template + option order; evaluation uses template 0 with
canonical option order (plus per-template test BAcc for mean +- std).
"""
from __future__ import annotations

from .prompts import EEG_END, EEG_PAD, EEG_START, IM_END, IM_START, SYSTEM_PROMPT
from .prompts_1019 import TEMPLATES as T1019

# task -> (short description, question)
_DESC = {
    'AD65': ('dementia diagnosis from resting-state EEG', 'Which diagnostic group does this subject belong to?'),
    'PD31': ("Parkinson's disease detection", "Does this EEG come from a healthy control or a Parkinson's patient?"),
    'BCIC_IV_1': ('two-class motor imagery decoding', 'Which of the two imagined movements does this trial show?'),
    'BCIC_IV_2A': ('four-class motor imagery decoding', 'Which movement was imagined in this trial?'),
    'SHU_MI': ('left versus right hand motor imagery', 'Which hand did the subject imagine moving?'),
    'BETA_SSVEP': ('SSVEP speller target identification', 'Which flickering target was the subject looking at?'),
    'Monitoring_ErrP': ('error-related potential detection', 'Was the observed feedback correct or erroneous?'),
    'LEMON_gender': ('sex classification from resting-state EEG', 'Is the subject female or male?'),
    'LEMON_extraversion': ('extraversion classification from resting-state EEG',
                           "Is the subject's extraversion score low or high?"),
    'FACED': ('nine-class emotion recognition', 'Which emotion was elicited during this segment?'),
    'SEED_IV': ('four-class emotion recognition', 'Which emotion was the subject experiencing?'),
    'SEED_V': ('five-class emotion recognition', 'Which emotion was the subject experiencing?'),
    'SEED_VII': ('seven-class emotion recognition', 'Which emotion was the subject experiencing?'),
    'ISRUC_S2': ('sleep staging', 'Which sleep stage does this epoch belong to?'),
    'ISRUC_S3': ('sleep staging', 'Which sleep stage does this epoch belong to?'),
    'SleepEDF': ('sleep staging', 'Which sleep stage does this epoch belong to?'),
    'EEGDenoiseNet': ('EEG artifact identification', 'Is this segment clean EEG or an artifact?'),
}


def build_templates(instructions: dict[str, str]) -> dict[str, list[str]]:
    out = {}
    for task, instr in instructions.items():
        if task in T1019:
            out[task] = list(T1019[task])
            continue
        desc, q = _DESC[task]
        out[task] = [
            instr,
            q,
            f'Classify this EEG segment ({desc}).',
            f'You are analysing an EEG recording for {desc}. {q}',
            f'Based on the EEG signal: {q[0].lower() + q[1:]}',
            f'{desc[0].upper() + desc[1:]}: choose the correct label for this EEG segment.',
            f'Look at the EEG and answer the question. {q}',
            f'EEG {desc}. Which option best describes this segment?',
        ]
    return out


TEMPLATES: dict[str, list[str]] = {}


def init_templates(specs) -> None:
    TEMPLATES.clear()
    TEMPLATES.update(build_templates({t: s.instruction for t, s in specs.items()}))


def num_templates(task: str) -> int:
    return len(TEMPLATES[task])


def prefix_text_v(task: str, template: int, options: list[str], num_eeg_tokens: int) -> str:
    user = (f'EEG: {EEG_START}{EEG_PAD * num_eeg_tokens}{EEG_END}\n'
            f'{TEMPLATES[task][template]}\n'
            f'Options: {", ".join(options)}.\n'
            f'Answer with exactly one option.')
    return (f'{IM_START}system\n{SYSTEM_PROMPT}{IM_END}\n'
            f'{IM_START}user\n{user}{IM_END}\n'
            f'{IM_START}assistant\n<think>\n\n</think>\n\n')
