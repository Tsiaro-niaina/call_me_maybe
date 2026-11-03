"""Function calling with constrained decoding"""

try:
    from .decoder import Decoder
    from .loader import DataError, load_functions, load_prompts, save_results
    from .models import FunctionCall, FunctionDefinition, PromptItem
    from .prompt import build_prompt, describe
    from .rules import (
        CHARACTERS_AFTER_VALUE,
        INTEGER_COMPLETE,
        INTEGER_START,
        NUMBER_COMPLETE,
        NUMBER_START,
        OPENING_QUOTE,
        read_string_token,
    )
    from .vocabulary import Vocabulary
except KeyboardInterrupt as key:
    print(f"Program interrupted: {key}")

__all__ = [
    "build_prompt",
    "describe",
    "Decoder",
    "DataError",
    "FunctionCall",
    "FunctionDefinition",
    "PromptItem",
    "CHARACTERS_AFTER_VALUE",
    "INTEGER_COMPLETE",
    "INTEGER_START",
    "NUMBER_COMPLETE",
    "NUMBER_START",
    "OPENING_QUOTE",
    "read_string_token",
    "Vocabulary",
    "load_functions",
    "load_prompts",
    "save_results"
]
