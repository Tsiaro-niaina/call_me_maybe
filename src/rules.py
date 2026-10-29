"""Rules telling which texts are valid JSON values."""

import re

STRING_ESCAPE_LETTERS = '"\\/bfnrt'
CHARACTERS_AFTER_VALUE = ",}"

# A number may start with one space: BPE writes ' -' as a single token.
NUMBER_START = re.compile(r" ?-?((0|[1-9][0-9]*)(\.[0-9]*)?)?")
NUMBER_COMPLETE = re.compile(r" ?-?(0|[1-9][0-9]*)(\.[0-9]+)?")
INTEGER_START = re.compile(r" ?-?(0|[1-9][0-9]*)?")
INTEGER_COMPLETE = re.compile(r" ?-?(0|[1-9][0-9]*)")


def read_string_token(
    token_text: str, after_backslash: bool
) -> tuple[str, bool, bool] | None:
    """Read a token as a part of a JSON string.

    Args:
        token_text: Text of the token.
        after_backslash: True if the previous token ended with a lone
            backslash, which still waits for its escape letter.

    Returns:
        None if the token cannot be part of the string. Otherwise a tuple
        (content, is_end, after_backslash): the characters that belong to
        the string, whether the token holds the closing quote, and whether
        a lone backslash is now waiting for its escape letter.
    """
    content = ""
    for position, character in enumerate(token_text):
        if after_backslash:
            if character not in STRING_ESCAPE_LETTERS:
                return None
            after_backslash = False
        elif character == "\\":
            after_backslash = True
        elif character == '"':
            text_after_quote = token_text[position + 1:]
            if text_after_quote[:1] and (
                text_after_quote[0] not in CHARACTERS_AFTER_VALUE
            ):
                return None
            return content, True, False
        elif ord(character) < 32:
            return None
        content += character
    return content, False, after_backslash
