from typing import Any

from pydantic import BaseModel


class TypeSpec(BaseModel):
    """Type descriptor of a parameter or of a return value."""

    type: str


class FunctionDefinition(BaseModel):
    """A function the LLM can choose to call."""

    name: str
    description: str
    parameters: dict[str, TypeSpec]
    returns: TypeSpec


class PromptItem(BaseModel):
    """One natural language request to process."""

    prompt: str


class FunctionCall(BaseModel):
    """One line of the output file."""

    prompt: str
    name: str
    parameters: dict[str, Any]
