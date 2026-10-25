"""Constrained decoding: logits masking driven by the expected structure."""

import json
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from .models import FunctionDefinition
from .prompt import NAME_START
from .vocab import Vocabulary, scan_string_token

MAX_STRING_TOKENS = 150
MAX_NUMBER_TOKENS = 25
NUMBER_ACCEPTING = frozenset({"zero", "int", "frac"})


def step_number(state: str, ch: str, integer: bool) -> str | None:
    """Advance the JSON number automaton by one character.

    Args:
        state: Current state (start, minus, zero, int, dot, frac).
        ch: Next character.
        integer: If True, a decimal point is forbidden.

    Returns:
        The new state, or None if the character is not allowed.
    """
    digit = ch in "0123456789"
    if state == "start":
        if ch == "-":
            return "minus"
        if ch == "0":
            return "zero"
        if digit:
            return "int"
    elif state == "minus":
        if ch == "0":
            return "zero"
        if digit:
            return "int"
    elif state in ("zero", "int"):
        if state == "int" and digit:
            return "int"
        if ch == "." and not integer:
            return "dot"
    elif state in ("dot", "frac"):
        if digit:
            return "frac"
    return None


def advance_number(state: str, text: str, integer: bool) -> str | None:
    """Advance the number automaton by a whole token (None if invalid)."""
    current: str | None = state
    for ch in text:
        if current is None:
            return None
        current = step_number(current, ch, integer)
    return current


class Decoder(BaseModel):
    """Generates text token by token under constraints."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any
    vocab: Vocabulary

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token ids."""
        ids: list[int] = self.model.encode(text).tolist()[0]
        return ids

    def pick(self, ids: list[int], allowed: list[int]) -> int:
        """Run one forward pass and return the best allowed token.

        Logits of every token outside `allowed` are set to -inf.

        Args:
            ids: Current context.
            allowed: Ids of the tokens that keep the output valid.

        Returns:
            The id of the allowed token with the highest logit.
        """
        if len(allowed) == 1:
            return allowed[0]
        logits = np.array(
            self.model.get_logits_from_input_ids(ids), dtype=np.float32
        )
        masked = np.full_like(logits, -np.inf)
        masked[allowed] = logits[allowed]
        return int(np.argmax(masked))

    def choose_name(self, ids: list[int], names: list[str]) -> str:
        """Let the LLM pick one of the given words (function names, ...).

        Args:
            ids: Context ending right where the word must start.
            names: Valid words.

        Returns:
            One of `names`.

        Raises:
            ValueError: If no token can continue any word.
        """
        work = list(ids)
        text = ""
        while True:
            can_stop = text in names
            live = [
                n[len(text):] for n in names
                if n.startswith(text) and n != text
            ]
            if not live:
                return text
            if len(live) == 1 and not can_stop:
                return text + live[0]  # forced: no LLM call needed
            valid = {x[:k] for x in live for k in range(1, len(x) + 1)}
            allowed = [
                i for s in valid for i in self.vocab.str_to_ids.get(s, [])
            ]
            if can_stop:  # a shorter word is complete: closing quote allowed
                allowed += self.vocab.ids_starting_with('"')
            if not allowed:
                raise ValueError(f"no valid token after {text!r}")
            best = self.pick(work, allowed)
            piece = self.vocab.id_to_str[best]
            if can_stop and piece.startswith('"'):
                return text
            text += piece
            work.append(best)

    def generate_string(
        self, ids: list[int], follow: str
    ) -> tuple[str, list[int]]:
        """Generate the content of a JSON string (opening quote included).

        Args:
            ids: Context ending right after the opening quote.
            follow: Skeleton text that comes after the closing quote.

        Returns:
            (raw JSON content, still escaped; context with the content).
        """
        work = list(ids)
        text = ""
        pending = False
        for _ in range(MAX_STRING_TOKENS):
            closers = self.vocab.closers[pending]
            extra = [
                i for i, (_head, rest) in closers.items()
                if follow.startswith(rest)
            ]
            best = self.pick(work, self.vocab.string_ok[pending] + extra)
            if best in closers:  # content + closing quote (+ skeleton)
                head = closers[best][0]
                tail = self.encode(head) if head else []
                return text + head, work + tail
            piece = self.vocab.id_to_str[best]
            pending = scan_string_token(piece, pending)[1]
            text += piece
            work.append(best)
        if pending:  # cut off inside an escape: drop the lone backslash
            text = text[:-1]
        return text, work

    def generate_number(
        self, ids: list[int], integer: bool, delimiter: str
    ) -> tuple[str, list[int]]:
        """Generate a JSON number.

        The context ends right after the colon: the model chooses the space
        itself, since ' -' is one BPE token while a positive number is ' '
        followed by digits.

        Args:
            ids: Context ending right after the colon.
            integer: If True, no decimal point is allowed.
            delimiter: Skeleton character that follows (',' or '}').

        Returns:
            (number text, context with the digits).
        """
        work = list(ids)
        text = ""
        state = "start"
        started = False  # True once the leading space has been chosen
        for _ in range(MAX_NUMBER_TOKENS + 1):
            accepting = state in NUMBER_ACCEPTING
            if not started:
                allowed = self.vocab.space_ids + [
                    i for i, t in self.vocab.spaced_numbers
                    if advance_number("start", t, integer) is not None
                ]
            else:
                allowed = [
                    i for i, t in self.vocab.number_tokens
                    if advance_number(state, t, integer) is not None
                ]
                if accepting:  # the number may end here
                    allowed += self.vocab.ids_starting_with(delimiter)
            best = self.pick(work, allowed)
            piece = self.vocab.id_to_str[best]
            if started and accepting and piece.startswith(delimiter):
                break
            if not started:
                started = True
                piece = piece[1:]  # drop the leading space
            new_state = advance_number(state, piece, integer)
            if new_state is None:
                break
            state = new_state
            text += piece
            work.append(best)
        return text, work

    def generate_call(
        self, prompt: str, functions: list[FunctionDefinition]
    ) -> tuple[str, dict[str, Any]]:
        """Choose a function and fill its arguments.

        Args:
            prompt: Full prompt (see prompt), without the JSON start.
            functions: Available functions.

        Returns:
            (function name, typed arguments).

        Raises:
            ValueError: On an unsupported parameter type or a dead end.
        """
        by_name = {f.name: f for f in functions}
        ids = self.encode(prompt + NAME_START)
        name = self.choose_name(ids, list(by_name))
        params = by_name[name].parameters
        keys = list(params)
        values: dict[str, Any] = {}
        pending = name + '", "parameters": {'
        for idx, key in enumerate(keys):
            last = idx == len(keys) - 1
            kind = params[key].type
            pending += ("" if idx == 0 else ", ") + json.dumps(key) + ":"
            if kind == "string":
                pending += ' "'
            ids = ids + self.encode(pending)
            pending = ""
            if kind == "string":
                nxt = "}}" if last else f", {json.dumps(keys[idx + 1])}: "
                text, ids = self.generate_string(ids, nxt)
                values[key] = json.loads(f'"{text}"')
                pending = '"'
            elif kind in ("number", "integer"):
                text, ids = self.generate_number(
                    ids, kind == "integer", "}" if last else ","
                )
                values[key] = int(text) if kind == "integer" else float(text)
            elif kind == "boolean":
                word = self.choose_name(ids, [" true", " false"])
                values[key] = word == " true"
                pending = word
            else:
                raise ValueError(f"unsupported parameter type: {kind}")
        return name, values
