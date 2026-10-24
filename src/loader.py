"""Loading and saving of the JSON files, with clear error messages."""

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .models import FunctionCall, FunctionDefinition, PromptItem

T = TypeVar("T", bound=BaseModel)


class DataError(Exception):
    """Raised when an input or output file cannot be handled."""


def _read_json(path: str) -> Any:
    """Read and parse a JSON file.

    Args:
        path: Path of the file.

    Returns:
        The decoded JSON content.

    Raises:
        DataError: If the file is missing, unreadable or not valid JSON.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise DataError(f"file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise DataError(
            f"invalid JSON in {path} "
            f"(line {exc.lineno}, column {exc.colno}): {exc.msg}"
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise DataError(f"cannot read {path}: {exc}") from exc


def _parse_list(path: str, data: Any, model: type[T]) -> list[T]:
    """Validate a JSON array entry by entry with a pydantic model.

    Args:
        path: File path (for error messages).
        data: Decoded JSON content.
        model: Pydantic model each entry must satisfy.

    Returns:
        The list of validated models.

    Raises:
        DataError: If data is not an array or an entry is invalid.
    """
    if not isinstance(data, list):
        raise DataError(f"{path}: expected a JSON array")
    items: list[T] = []
    for index, entry in enumerate(data):
        try:
            items.append(model.model_validate(entry))
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
                for e in exc.errors()
            )
            raise DataError(
                f"{path}: invalid entry #{index}: {details}"
            ) from exc
    return items


def load_functions(path: str) -> list[FunctionDefinition]:
    """Load the function definitions.

    Args:
        path: Path of functions_definition.json.

    Returns:
        The validated, non-empty list of function definitions.

    Raises:
        DataError: On any problem with the file content.
    """
    functions = _parse_list(path, _read_json(path), FunctionDefinition)
    if not functions:
        raise DataError(f"{path}: no function defined")
    names = [f.name for f in functions]
    if len(set(names)) != len(names):
        raise DataError(f"{path}: duplicate function names")
    return functions


def load_prompts(path: str) -> list[PromptItem]:
    """Load the prompts to process.

    Args:
        path: Path of function_calling_tests.json.

    Returns:
        The validated list of prompts.

    Raises:
        DataError: On any problem with the file content.
    """
    return _parse_list(path, _read_json(path), PromptItem)


def save_results(path: str, results: list[FunctionCall]) -> None:
    """Write the results file, creating parent directories if needed.

    Args:
        path: Output file path.
        results: Function calls to write.

    Raises:
        DataError: If the file cannot be written.
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [r.model_dump() for r in results],
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.write("\n")
    except OSError as exc:
        raise DataError(f"cannot write {path}: {exc}") from exc
