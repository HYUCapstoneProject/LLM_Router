"""Structure-preserving input contract for the new integrated dataset.

The legacy operational normalizer remains unchanged for old feedback artifacts.
"""

from __future__ import annotations

import re

from ..shared.structured_text import clean_text, structured_duplicate_key

__all__ = ["clean_text", "clean_xroute_text", "compatible_near_duplicate", "structured_duplicate_key"]


def clean_xroute_text(value: object) -> str:
    text = clean_text(value)
    match = re.fullmatch(
        r"\[System Instruction\]\n(.*?)\n\n\[User Query\]\n(.*)", text, re.DOTALL
    )
    if not match:
        return text
    system, query = match.groups()
    # Only known output-format wrappers may be discarded. Unknown instructions
    # and task constraints (e.g. function body only, provided context) survive.
    if (
        system
        == "Answer the following multiple-choice question by selecting the correct option (A, B, C, or D). You MUST put your final answer letter in a parenthesis."
    ):
        system = ""
    system = system.replace(
        " Make sure to put the answer (and only answer) inside \\boxed{}.", ""
    )
    system = system.replace(
        " Put your code in this format: [BEGIN] <Your Code> [Done]", ""
    )
    system = system.replace(
        " Put the code in this format: [BEGIN] <Your Code - function body only> [Done]",
        "",
    )
    return clean_text((system + "\n\n" if system else "") + query)


def compatible_near_duplicate(left: str, right: str) -> bool:
    def signature(text: str) -> tuple:
        return (
            tuple(re.findall(r"[=<>+*/%^|&~!\-]+|\d+(?:\.\d+)?", text)),
            tuple(re.findall(r"\b(?:not|never|except|without)\b", text.casefold())),
            tuple(
                len(line) - len(line.lstrip())
                for line in text.splitlines()
                if line.strip()
            ),
        )

    return signature(left) == signature(right)
