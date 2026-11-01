"""Prompt construction (ChatML format, thinking disabled)."""

from .models import FunctionDefinition

NAME_START = '{"name": "'


def describe(function: FunctionDefinition) -> str:
    """Return a one-line description of a function."""
    params = ", ".join(
        f"{name}: {spec.type}" for name, spec in function.parameters.items()
    )
    return f"- {function.name}({params}): {function.description}"


def build_prompt(functions: list[FunctionDefinition], request: str) -> str:
    """Build the full prompt, ready for the assistant to answer.

    Args:
        functions: Available functions.
        request: The user's natural language request.

    Returns:
        The prompt text (without the forced JSON start).
    """
    listing = "\n".join(describe(f) for f in functions)
    system = (
        "You are a function calling assistant. Choose the single most "
        "appropriate function for the user's request and give its "
        "arguments.\n\nAvailable functions:\n"
        f"{listing}\n\n"
        'Answer only with JSON: {"name": <function name>, '
        '"parameters": {<arguments>}}'
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{request}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
