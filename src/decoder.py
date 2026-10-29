"""Constrained decoding: the model only writes what the schema allows."""

import json
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict

from .models import FunctionDefinition
from .prompt import NAME_START
from .rules import (
    CHARACTERS_AFTER_VALUE,
    INTEGER_COMPLETE,
    INTEGER_START,
    NUMBER_COMPLETE,
    NUMBER_START,
    read_string_token,
)
from .vocab import Vocabulary

MAX_STRING_TOKENS = 150
MAX_NUMBER_TOKENS = 25
BOOLEAN_WORDS = [" true", " false", "true", "false"]


class Decoder(BaseModel):
    """Generates a function call token by token under constraints."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Any
    vocab: Vocabulary

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token ids."""
        token_ids: list[int] = self.model.encode(text).tolist()[0]
        return token_ids

    def first_valid_token(
        self, context_ids: list[int], is_valid: Callable[[str], bool]
    ) -> tuple[int, str]:
        """Return the most likely token whose text is valid.

        The tokens are examined from the most to the least likely, and the
        first valid one is kept. This gives the same result as setting the
        logits of the invalid tokens to -inf and taking the argmax, but only
        the few tokens that are examined have to be checked.

        Args:
            context_ids: Token ids written so far (prompt included).
            is_valid: Tells whether a token text keeps the output valid.

        Returns:
            (token id, token text) of the chosen token.

        Raises:
            ValueError: If no token is valid.
        """
        logits = np.array(
            self.model.get_logits_from_input_ids(context_ids),
            dtype=np.float32,
        )
        for token_id in np.argsort(logits)[::-1]:
            token_text = self.vocab.token_texts.get(int(token_id))
            if token_text is not None and is_valid(token_text):
                return int(token_id), token_text
        raise ValueError("no token can continue the output")

    def generate_word(
        self, context_ids: list[int], words: list[str], end_characters: str
    ) -> str:
        """Let the model choose one word among `words`.

        Args:
            context_ids: Context ending right where the word must start.
            words: Allowed words.
            end_characters: Characters that follow the word in the skeleton.
                The model ends a word by writing one of them.

        Returns:
            One of `words`.
        """
        context_ids = list(context_ids)
        word_so_far = ""
        while True:
            possible_words = [
                word for word in words if word.startswith(word_so_far)
            ]
            if len(possible_words) == 1:
                return possible_words[0]  # no need to ask the model

            def is_valid(token_text: str) -> bool:
                if token_text[0] in end_characters:
                    return word_so_far in words
                return any(
                    word.startswith(word_so_far + token_text)
                    for word in possible_words
                )

            token_id, token_text = self.first_valid_token(
                context_ids, is_valid
            )
            if token_text[0] in end_characters:
                return word_so_far
            word_so_far += token_text
            context_ids.append(token_id)

    def generate_number(
        self, context_ids: list[int], is_integer: bool
    ) -> tuple[str, list[int]]:
        """Generate a JSON number, with an optional leading space.

        Args:
            context_ids: Context ending right after the colon.
            is_integer: If True, no decimal point is allowed.

        Returns:
            (number text, context with the number written).

        Raises:
            ValueError: If the number is not finished after the token limit.
        """
        number_start = INTEGER_START if is_integer else NUMBER_START
        number_complete = INTEGER_COMPLETE if is_integer else NUMBER_COMPLETE
        context_ids = list(context_ids)
        number_text = ""
        for _ in range(MAX_NUMBER_TOKENS):

            def is_valid(token_text: str) -> bool:
                if token_text[0] in CHARACTERS_AFTER_VALUE:
                    return number_complete.fullmatch(number_text) is not None
                new_text = number_text + token_text
                return number_start.fullmatch(new_text) is not None

            token_id, token_text = self.first_valid_token(
                context_ids, is_valid
            )
            if token_text[0] in CHARACTERS_AFTER_VALUE:
                break  # the model ended the number
            number_text += token_text
            context_ids.append(token_id)
        if number_complete.fullmatch(number_text) is None:
            raise ValueError(f"number not finished: {number_text!r}")
        return number_text, context_ids

    def generate_string(
        self, context_ids: list[int]
    ) -> tuple[str, list[int]]:
        """Generate the content of a JSON string (still escaped).

        Args:
            context_ids: Context ending right after the opening quote.

        Returns:
            (string content, context with the content written).

        Raises:
            ValueError: If the string is not closed after the token limit.
        """
        context_ids = list(context_ids)
        string_text = ""
        after_backslash = False
        for _ in range(MAX_STRING_TOKENS):

            def is_valid(token_text: str) -> bool:
                reading = read_string_token(token_text, after_backslash)
                if reading is None:
                    return False
                content, is_end, _ = reading
                if string_text == "" and content.startswith(" "):
                    # The model writes ' *' for '*': a value never starts
                    # with a space glued to text. A value made only of
                    # spaces stays possible.
                    return is_end and content.strip(" ") == ""
                return True

            token_id, token_text = self.first_valid_token(
                context_ids, is_valid
            )
            reading = read_string_token(token_text, after_backslash)
            if reading is None:  # cannot happen: the token was checked
                raise ValueError(f"invalid string token: {token_text!r}")
            content, is_end, after_backslash = reading
            string_text += content
            if is_end:
                # The closing token is not kept: it also holds the end of
                # the skeleton, which is written separately.
                if content:
                    context_ids += self.encode(content)
                return string_text, context_ids
            context_ids.append(token_id)
        raise ValueError(f"string longer than {MAX_STRING_TOKENS} tokens")

    def generate_call(
        self, prompt: str, functions: list[FunctionDefinition]
    ) -> tuple[str, dict[str, Any]]:
        """Choose a function and fill its arguments.

        Args:
            prompt: Full prompt (see build_prompt), without the JSON start.
            functions: Available functions.

        Returns:
            (function name, typed arguments).

        Raises:
            ValueError: On an unsupported parameter type or a dead end.
        """
        function_by_name = {function.name: function for function in functions}
        context_ids = self.encode(prompt + NAME_START)
        function_name = self.generate_word(
            context_ids, list(function_by_name), '"'
        )
        parameters = function_by_name[function_name].parameters
        argument_values: dict[str, Any] = {}
        # Skeleton text not yet written in the context.
        skeleton_text = function_name + '", "parameters": {'
        for position, parameter_name in enumerate(parameters):
            parameter_type = parameters[parameter_name].type
            if position > 0:
                skeleton_text += ", "
            skeleton_text += json.dumps(parameter_name) + ":"
            if parameter_type == "string":
                skeleton_text += ' "'
            context_ids = context_ids + self.encode(skeleton_text)
            skeleton_text = ""
            if parameter_type == "string":
                string_text, context_ids = self.generate_string(context_ids)
                argument_values[parameter_name] = json.loads(
                    f'"{string_text}"'
                )
                skeleton_text = '"'
            elif parameter_type in ("number", "integer"):
                number_text, context_ids = self.generate_number(
                    context_ids, parameter_type == "integer"
                )
                if parameter_type == "integer":
                    argument_values[parameter_name] = int(number_text)
                else:
                    argument_values[parameter_name] = float(number_text)
            elif parameter_type == "boolean":
                boolean_text = self.generate_word(
                    context_ids, BOOLEAN_WORDS, CHARACTERS_AFTER_VALUE
                )
                argument_values[parameter_name] = (
                    boolean_text.strip() == "true"
                )
                skeleton_text = boolean_text
            else:
                raise ValueError(
                    f"unsupported parameter type: {parameter_type}"
                )
        return function_name, argument_values
