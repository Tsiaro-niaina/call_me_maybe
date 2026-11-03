"""Vocabulary of the model: token id -> text (byte-level BPE)."""

import json

from pydantic import BaseModel

from .loader import DataError


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 mapping: byte value -> printable unicode character.

    Returns:
        Dictionary giving the stand-in character of each byte value.
    """
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    code_points = byte_values[:]
    next_extra_code_point = 256
    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            code_points.append(next_extra_code_point)
            next_extra_code_point += 1
    return {
        byte_value: chr(code_point)
        for byte_value, code_point in zip(byte_values, code_points)
    }


class Vocabulary(BaseModel):
    """Text of every token, by token id.

    Tokens that do not decode to complete UTF-8 text (pieces of a multi-byte
    character) are left out: the decoder never produces them.
    """

    token_texts: dict[int, str]

    @classmethod
    def from_file(cls, path: str) -> "Vocabulary":
        """Build the vocabulary from a vocab.json file.

        Args:
            path: Path returned by get_path_to_vocab_file().

        Returns:
            The vocabulary.

        Raises:
            DataError: If the file cannot be read or parsed.
        """
        try:
            with open(path, encoding="utf-8") as vocab_file:
                raw_vocabulary: dict[str, int] = json.load(vocab_file)
        except (OSError, json.JSONDecodeError) as error:
            raise DataError(
                f"cannot load vocabulary {path}: {error}"
            ) from error
        byte_of_character = {
            character: byte_value
            for byte_value, character in bytes_to_unicode().items()
        }
        token_texts: dict[int, str] = {}
        for spelling, token_id in raw_vocabulary.items():
            try:
                token_bytes = bytes(
                    byte_of_character[character] for character in spelling
                )
                token_text = token_bytes.decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                continue
            if token_text:
                token_texts[token_id] = token_text
        return cls(token_texts=token_texts)
