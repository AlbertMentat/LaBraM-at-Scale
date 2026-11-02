"""Multi-template prompts (1019): 8 instruction paraphrases per task + option-order shuffling.

Template 0 with canonical option order equals the 1018 prompt (prompts.prefix_text), so canonical-template results
are directly comparable.  Answers are always the canonical class names.
"""
from __future__ import annotations

from .prompts import EEG_END, EEG_PAD, EEG_START, IM_END, IM_START, SYSTEM_PROMPT

TEMPLATES: dict[str, list[str]] = {
    'TUAB': [
        "Task: clinical EEG screening. Decide whether this EEG recording is normal or abnormal.",
        "Is this clinical EEG recording normal or abnormal?",
        "Classify the EEG recording above as normal or abnormal.",
        "You are reviewing a routine clinical EEG. Is the recording normal or abnormal?",
        "Based on the EEG signal, would a neurologist read this recording as normal or abnormal?",
        "Abnormal EEG detection: label this recording.",
        "Determine if this scalp EEG shows abnormal findings.",
        "Screen this EEG segment: normal or abnormal?",
    ],
    'TUEV': [
        "Task: EEG event classification. Identify the type of event in this EEG segment.",
        "Which type of EEG event does this segment contain?",
        "Classify the clinical EEG event shown in this segment.",
        "Label this short EEG segment with its event type.",
        "An EEG reader annotated this segment. What event type did they mark?",
        "Identify the epileptiform discharge, artifact or background activity in this segment.",
        "What kind of event is present in this EEG epoch?",
        "EEG event annotation: choose the event type for this segment.",
    ],
    'TUSL': [
        "Task: EEG slowing detection. Classify this EEG segment as background, seizure, or slowing.",
        "Does this EEG segment show background activity, a seizure, or slowing?",
        "Classify the activity in this EEG segment.",
        "Label this clinical EEG segment by its activity type.",
        "Distinguish seizure from slowing and background in this EEG segment.",
        "What activity does this EEG segment contain?",
        "An EEG reader annotated this segment. Which label did they assign?",
        "EEG activity classification: choose the label for this segment.",
    ],
    'HMC': [
        "Task: sleep staging. Identify the sleep stage of this EEG epoch.",
        "Which sleep stage does this polysomnography epoch belong to?",
        "Score the sleep stage of this EEG epoch.",
        "Classify this sleep EEG epoch into its sleep stage.",
        "A sleep technician scored this epoch. Which stage did they assign?",
        "Sleep scoring: label this epoch.",
        "What is the sleep stage during this EEG epoch?",
        "Determine the AASM sleep stage of this EEG segment.",
    ],
    'SEED': [
        "Task: emotion recognition. Identify the emotion the subject felt while watching a film clip during this EEG.",
        "Which emotion was the subject experiencing during this EEG recording?",
        "The subject watched an emotional film clip. Classify the emotion from the EEG.",
        "Recognize the emotional state from this EEG segment.",
        "What was the emotional valence of the film clip the subject was watching?",
        "Emotion classification: label this EEG segment.",
        "Decode the subject's emotion from this EEG.",
        "Identify whether the subject felt a negative, neutral or positive emotion.",
    ],
    'EEGMAT': [
        "Task: mental workload detection. Decide whether the subject was resting or performing mental arithmetic during this EEG.",
        "Was the subject resting or doing mental arithmetic during this EEG?",
        "Classify the cognitive state of the subject during this EEG segment.",
        "Detect mental workload: is the subject at rest or calculating?",
        "Determine the task the subject was performing during this EEG.",
        "Mental arithmetic detection: label this EEG segment.",
        "Is this EEG recorded during rest or during a mental calculation task?",
        "Identify the subject's condition from the EEG: resting or mental arithmetic.",
    ],
}


def num_templates(task: str) -> int:
    return len(TEMPLATES[task])


def prefix_text_v(task: str, template: int, options: list[str], num_eeg_tokens: int) -> str:
    instr = TEMPLATES[task][template]
    user = (f'EEG: {EEG_START}{EEG_PAD * num_eeg_tokens}{EEG_END}\n'
            f'{instr}\n'
            f'Options: {", ".join(options)}.\n'
            f'Answer with exactly one option.')
    return (f'{IM_START}system\n{SYSTEM_PROMPT}{IM_END}\n'
            f'{IM_START}user\n{user}{IM_END}\n'
            f'{IM_START}assistant\n<think>\n\n</think>\n\n')
