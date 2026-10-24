"""Vocabulary of the model: token id <-> text (byte-level BPE)."""

import json

from pydantic import BaseModel

from .loader import DataError

STRING_ESCAPES = frozenset('"\\/bfnrt')
NUMBER_CHARS = frozenset("-.0123456789")
START_CHARS = (",", "}", '"')


def bytes_to_unicode() -> dict[int, str]:
    """GPT-2 mapping: byte value -> printable unicode character.

    Returns:
        Dictionary giving the stand-in character of each byte.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def scan_string_token(text: str, pending: bool) -> tuple[bool, bool]:
    """Check a token used as the content of a JSON string.

    Args:
        text: Text of the token.
        pending: True if the previous token ended with a lone backslash.

    Returns:
        (valid, pending_after). A raw quote or control character is invalid
        (a closing quote is handled separately).
    """
    for ch in text:
        if pending:
            if ch not in STRING_ESCAPES:
                return False, False
            pending = False
        elif ch == "\\":
            pending = True
        elif ch == '"' or ord(ch) < 0x20:
            return False, False
    return True, pending


def find_closing(text: str, pending: bool) -> tuple[str, str] | None:
    """Split a token at its first unescaped quote.

    Args:
        text: Token text.
        pending: True if the previous token ended with a lone backslash.

    Returns:
        (content before the quote, text after the quote), or None if the
        token has no usable closing quote.
    """
    for pos, ch in enumerate(text):
        if pending:
            if ch not in STRING_ESCAPES:
                return None
            pending = False
        elif ch == "\\":
            pending = True
        elif ch == '"':
            return text[:pos], text[pos + 1:]
        elif ord(ch) < 0x20:
            return None
    return None


class Vocabulary(BaseModel):
    """Token table built from vocab.json, with precomputed token classes.

    Only tokens that decode to complete UTF-8 text are kept.
    """

    id_to_str: dict[int, str]
    str_to_ids: dict[str, list[int]]
    starts: dict[str, list[int]]
    string_ok: dict[bool, list[int]]
    closers: dict[bool, dict[int, tuple[str, str]]]
    number_tokens: list[tuple[int, str]]
    space_ids: list[int]
    spaced_numbers: list[tuple[int, str]]

    @classmethod
    def from_file(cls, path: str) -> "Vocabulary":
        """Build the table from a vocab.json file.

        Args:
            path: Path returned by get_path_to_vocab_file().

        Returns:
            The vocabulary.

        Raises:
            DataError: If the file cannot be read or parsed.
        """
        try:
            with open(path, encoding="utf-8") as f:
                raw: dict[str, int] = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise DataError(f"cannot load vocabulary {path}: {exc}") from exc
        char_to_byte = {c: b for b, c in bytes_to_unicode().items()}
        id_to_str: dict[int, str] = {}
        str_to_ids: dict[str, list[int]] = {}
        starts: dict[str, list[int]] = {c: [] for c in START_CHARS}
        string_ok: dict[bool, list[int]] = {False: [], True: []}
        closers: dict[bool, dict[int, tuple[str, str]]] = {
            False: {},
            True: {},
        }
        number_tokens: list[tuple[int, str]] = []
        space_ids: list[int] = []
        spaced_numbers: list[tuple[int, str]] = []
        for token, token_id in raw.items():
            try:
                text = bytes(char_to_byte[c] for c in token).decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                continue
            if not text:
                continue
            id_to_str[token_id] = text
            str_to_ids.setdefault(text, []).append(token_id)
            for ch in START_CHARS:
                if text.startswith(ch):
                    starts[ch].append(token_id)
            for pending in (False, True):
                if scan_string_token(text, pending)[0]:
                    string_ok[pending].append(token_id)
                split = find_closing(text, pending)
                if split is not None:
                    closers[pending][token_id] = split
            if set(text) <= NUMBER_CHARS:
                number_tokens.append((token_id, text))
            if text == " ":
                space_ids.append(token_id)
            elif text[0] == " " and set(text[1:]) <= NUMBER_CHARS:
                spaced_numbers.append((token_id, text[1:]))
        return cls(
            id_to_str=id_to_str,
            str_to_ids=str_to_ids,
            starts=starts,
            string_ok=string_ok,
            closers=closers,
            number_tokens=number_tokens,
            space_ids=space_ids,
            spaced_numbers=spaced_numbers,
        )

    def ids_starting_with(self, char: str) -> list[int]:
        """Ids of the tokens starting with char (',', '}' or '"')."""
        return self.starts.get(char, [])
