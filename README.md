*This project has been created as part of the 42 curriculum by rmandimb.*

# call me maybe

Function calling with a small language model (Qwen/Qwen3-0.6B) and
**constrained decoding**.

## Description

Given a natural language request such as `"What is the sum of 2 and 3?"`,
the program does not answer the question. It returns the function that
should be called and its typed arguments:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

A 0.6B model is unreliable when it is only prompted to write JSON. Here the
JSON is never "hoped for": at every step, only the tokens that keep the
output valid and schema-compliant can be chosen, so the result is valid by
construction. The model only decides **which function** to call and **what
the argument values are**.

The function is chosen by the LLM (no regex, keyword or other heuristic),
the generated values are never corrected afterwards, and nothing in the
code depends on the provided example files.

## Instructions

Requirements: Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
make install        # uv sync
make run            # run with the default files
make debug          # run under pdb
make lint           # flake8 + mypy
make lint-strict    # flake8 + mypy --strict
make clean          # remove caches
```

Direct usage (run it from the repository root):

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

All three options are optional and default to the paths above.

- The result is written to the output file (the directory is created if
  needed) **and** printed on stdout. Progress and warnings go to stderr, so
  `uv run python -m src > out.json` gives a clean JSON file.
- Exit code: `0` on success, `1` on an error or if a prompt could not be
  processed, `130` on Ctrl+C.
- The first run downloads the model (about 1.2 GB) into the Hugging Face
  cache. Set `HF_HOME` to move that cache if your home directory is small.
- `torch` is installed from the PyTorch CPU index (see `pyproject.toml`), so
  `uv sync` needs network access to `download.pytorch.org`.

Project layout:

```
src/
  __main__.py   CLI, main loop, error handling
  loader.py     reading/validating the JSON files, writing the result
  models.py     pydantic models (definitions, prompts, results)
  prompt.py     prompt construction
  vocab.py      vocabulary: token id -> text
  rules.py      which texts are valid JSON numbers and strings
  decoder.py    constrained decoding
llm_sdk/        provided SDK (unmodified)
data/input/     example input files
```

## Algorithm explanation

### 1. Vocabulary (`vocab.py`)

`vocab.json` is a byte-level BPE table: its keys are not plain text (`Ġ` is a
space, `Ċ` a newline, ...). Each key is converted back to its real string with
the inverse of the GPT-2 `bytes_to_unicode` mapping, which gives a dictionary
`token id -> text`. Tokens that do not decode to complete UTF-8 text (pieces
of multi-byte characters) are left out.

A token is not a character: `"}}`, `",` or `)",` are single tokens, and BPE
glues a space to the next characters (` -`, ` true`, ` "*`). A token is
therefore never judged alone, but on **the text it would add** to the output.

### 2. Prompt and skeleton (`prompt.py`, `decoder.py`)

The prompt uses the ChatML format of Qwen, with the list of functions (name,
parameter names and types, description) and the request. It ends with the
empty `<think>\n\n</think>\n\n` block that the Qwen3 template inserts when
thinking is disabled. The program writes the start of the answer itself, so
the model never gets to write its own `<think>` block: without the empty
block the results changed (see Challenges).

The JSON skeleton is built by the program, not by the model:

```
{"name": "<LLM choice>", "parameters": {"<key>":<LLM value>, "<key>":<LLM value>}}
```

Fixed parts (`{"name": "`, the key names taken from the definition, the
separators) are appended to the context without calling the model. Each key
is written up to the colon only: the leading space and the opening quote of
the value are chosen by the model, because BPE merges them with the next
characters (` -3` is ` -` + `3` but `7` is ` ` + `7`; `"*"` is one chunk).

### 3. Choosing the next token (`Decoder.first_valid_token`)

The model returns one logit (score) per token of the vocabulary. The tokens
are examined from the most to the least likely, and the first one whose text
keeps the output valid is chosen. This gives exactly the same result as
setting the logits of the invalid tokens to `-inf` and taking the argmax, but
only the few tokens that are examined have to be checked, so no per-step
classification of the whole vocabulary is needed.

### 4. Function name (`generate_word`)

A token is valid if the text written so far plus the token is still the
beginning of at least one function name. When a single name remains, it is
completed without calling the model. The model ends a name by writing the
closing quote.

### 5. Parameter values, by type (`rules.py`, `decoder.py`)

- **`number` / `integer`**: four regular expressions describe the valid
  beginnings and the complete forms of a JSON number (one optional leading
  space, optional minus, no leading zero, a digit after the decimal point).
  A token is valid if the text stays a valid beginning, so `007`, `1.5.2` or
  `.5` can never be produced. The model ends the number by writing a token
  that starts with `,` or `}`, accepted only when the number is complete.
  `integer` forbids the decimal point.
- **`string`**: the model writes the opening quote, then the content. Any
  character is allowed except a raw quote and control characters, and after
  a backslash only a valid escape letter is allowed (`\\`, `\"`, `\n`...), so
  a regex such as `\d+` is written `\\d+` and decodes correctly with
  `json.loads`. The string ends with the first token that contains an
  unescaped quote, if what follows the quote is empty or starts with `,` or
  `}`. The characters before the quote are kept; the rest of the token is
  dropped and the skeleton is written separately. A value cannot start with
  a space glued to text (the model writes ` *` for `*`), but a value made only
  of spaces is allowed.
- **`boolean`**: the model chooses between ` true` and ` false` (single tokens
  with their leading space), like a function name.

Other types are reported as an error for that prompt.

### 6. Output

Results are validated with pydantic and written as a JSON array of
`{prompt, name, parameters}` objects (numbers as floats, e.g. `2.0`).

## Design decisions

- **Hand-built skeleton instead of a generic JSON automaton.** The schema is
  known, so the model only decides at the value positions: the structure
  cannot be wrong, there are fewer model calls, and the code is shorter.
- **Rules on text, not on tokens.** Validity is computed from the string a
  token adds, with regular expressions and one small string reader, instead
  of tables precomputed over the whole vocabulary.
- **Follow the natural tokenization.** The context the model sees is the one
  BPE would produce for the same JSON, which matters for small models.
- **Short prompt.** The SDK has no KV cache (and private attributes are
  forbidden), so every generated token recomputes the whole context: the cost
  grows with the prompt length.
- **No post-correction.** The values written by the model are returned as they
  are.
- **pydantic everywhere** (models, vocabulary, decoder), as required.
- **Errors are never fatal.** Missing or invalid files, wrong JSON shapes and
  model-loading problems end with a clear message on stderr. If one prompt
  fails, it is reported and skipped, and the others are still written.
- **Only the public SDK is used**: `encode`, `get_logits_from_input_ids` and
  `get_path_to_vocab_file`.

## Performance analysis

Measured on CPU only (no GPU), on the 11 provided prompts:

| Metric | Result |
|---|---|
| Valid JSON, schema-compliant | 11/11 |
| Correct function | 11/11 |
| Correct arguments | XX/11 |
| Run time, personal laptop | XX s |
| Run time, school computer | XX s |

The substitution results were checked by applying the generated regex to the
source string and comparing with the expected text, since several regexes are
valid for the same request.

Observations:

- Time is dominated by forward passes (about 2 s each on the laptop). Every
  generated string costs about one pass per token, so the substitution prompts
  (three strings) are the slowest.
- A prompt with extra rules and examples made the run about 2.3x slower
  (about 500 s) with the same score.
- A shorter prompt saved about 17% of the time but dropped the score from
  10/11 to 9/11.
- Removing the empty `<think>` block also changed the results (the
  replacement for "asterisks" became `*****`). Small models are very
  sensitive to the exact prompt.
- Validity is not a statistical property: it comes from the constraints, not
  from the model.

## Challenges faced

- **Tokens that glue content and the closing quote.** The first version only
  allowed a string to end with a token *starting* with a quote. The real
  vocabulary has tokens like `)",`. The constraint forbade the model's
  favourite token and forced a wrong one, producing the regex `([0-9]+)|`. It
  was found by logging, at each step, the model's unconstrained choice next
  to the constrained one.
- **Spaces and BPE.** A negative number came out positive (`-3` became `3`)
  because the skeleton forced a lone space after the colon, which BPE never
  produces before a minus sign. Later, a replacement came out as ` *` (with a
  space) because the opening quote was forced alone, while BPE merges it with
  a symbol. The skeleton now stops at the colon and the model writes the
  space and the opening quote itself.
- **Run time.** No KV cache and a growing prompt exceeded the 5 minutes limit.
  Solved by keeping the prompt short and not calling the model when a single
  choice remains.
- **JSON escaping of regexes.** A single backslash is not valid JSON, so only
  valid escapes are accepted after a backslash, and the value is decoded with
  `json.loads`.
- **Limits of a 0.6B model.** The model sometimes copies a word from the
  request instead of the symbol it stands for. Constrained decoding
  guarantees the form of the output, not the correctness of the values.
- **Disk space.** The model and package caches were moved out of the home
  directory with `HF_HOME` and `UV_CACHE_DIR`.

### Known limitations

- **Values come from a 0.6B model.** The constraints guarantee the form of
  the output, not the correctness of the values.
- **Quotes inside a string.** The model tends to copy them without escaping,
  and an unescaped quote then looks like the end of the string:
  `Convert 'say "hi"' to uppercase` gives `say ` instead of `say "hi"`.
- **Leading spaces.** A value that starts with a space followed by text cannot
  be generated (a value made only of spaces can).
- **Prompt sensitivity.** Small changes in the prompt change the results, in
  both directions.
- **Speed.** There is no KV cache, so the run time grows with the number of
  prompts and with the prompt length. On a slow CPU the 5 minutes limit is
  tight.
- **Characters** split across several tokens (some emojis and rare
  characters) cannot be generated inside a string, and `\uXXXX` escapes are
  not produced.
- **Caps**: strings are limited to 150 tokens and numbers to 25 tokens (a
  longer one is reported as an error for that prompt).
- **Types**: `number`, `integer`, `string` and `boolean` are supported. Other
  types (arrays, objects, null) are reported as an error for that prompt.

## Testing strategy

- **Input handling**: missing file, directory instead of a file, invalid
  JSON, JSON that is not an array, missing keys, empty function list and
  empty prompt list all give a clear error without a traceback.
- **Provided prompts**: a throwaway script generated the 11 calls and compared
  them with expected values (applying the regex for substitutions).
- **Mask debugging**: a trace of each step showing where the constraint
  overrides the model's own preference, used to find the closing-quote bug.
- **Personal test set** (not part of the repository): 14 prompts with 6 other
  functions (`integer`, `boolean`, negative and decimal numbers, an empty
  string, an accent, escaped quotes), used during development. It revealed
  the leading-space bug on negative numbers.
- **Static checks**: `flake8` and `mypy` through `make lint`.
- **Clean-clone check**: clone the repository and run only `uv sync` then
  `uv run python -m src`.

## Example usage

```console
$ make run
[1/11] fn_add_numbers (...)
...
[11/11] fn_substitute_string_with_regex (...)
Total: ...
[
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  },
  ...
]
```

The time printed after each prompt depends on the machine.

```console
$ uv run python -m src --input missing.json
Error: file not found: missing.json
```

## Resources

- Qwen3 model card: https://huggingface.co/Qwen/Qwen3-0.6B
- about constrained decoding: https://www.zeroentropy.dev/concepts/constrained-decoding/
- everything about tokenization: https://www.openxcell.com/blog/llm-tokenization
- JSON specification, RFC 8259: https://www.rfc-editor.org/rfc/rfc8259
- Python `re` module: https://docs.python.org/3/library/re.html


### Use of AI

An AI assistant (Claude) was used as a tutor and pair-programming partner
for:

- gave me a global plan on where to start and the architecture
- shortcut to explain every concept that I needed to learn
- reasoning about bugs from the logs (the closing-quote and leading-space
  issues above), about the run time, and about simplifying the decoder.
