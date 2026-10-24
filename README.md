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
JSON is never "hoped for": at every step the logits of the tokens that would
break the structure or the schema are set to `-inf`, so the output is valid
and schema-compliant by construction. The model only decides **which
function** to call and **what the argument values are**.

The function is chosen by the LLM (no regex, keyword or other heuristic),
and nothing in the code depends on the provided example files.

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
  vocab.py      vocabulary and token classes
  decoder.py    constrained decoding
llm_sdk/        provided SDK (unmodified)
data/input/     example input files
```

## Algorithm explanation

### 1. Vocabulary (`vocab.py`)

`vocab.json` is a byte-level BPE table: its keys are not plain text (`Ġ` is a
space, `Ċ` a newline, ...). Each key is converted back to its real string with
the inverse of the GPT-2 `bytes_to_unicode` mapping. Tokens that do not decode
to complete UTF-8 text (pieces of multi-byte characters) are dropped.

The important point is that a token is not a character: `"}}`, `",` or `)",`
are single tokens. A token is therefore never validated on its own, but on
the **text it would add** to the output.

Once, at load time, the tokens are classified so that no per-step scan of the
~150k tokens is needed:

- `string_ok[pending]`: tokens that can appear inside a JSON string (no raw
  quote, no control character, valid escapes). `pending` tells whether the
  previous token ended with a lone backslash.
- `closers[pending]`: tokens that contain the first unescaped quote, split in
  *(content before the quote, text after it)*, e.g. `)",` -> `(")", ",")`.
- `number_tokens`: tokens made only of `-`, `.` and digits.
- `str_to_ids`: text -> token ids, to turn "valid next strings" into ids.

### 2. Skeleton and prompt (`prompt.py`, `decoder.py`)

The prompt is written in ChatML (Qwen format) with an empty `<think>` block
so that Qwen3 does not start reasoning. It contains the list of functions
(name, parameter names and types, description) and the request.

The JSON skeleton is built by the program, not by the model:

```
{"name": "<LLM choice>", "parameters": {"<key>": <LLM value>, "<key>": ...}}
```

Fixed parts (`{"name": "`, key names taken from the definition, separators)
are appended to the context directly, with no forward pass.

### 3. Function name

At each step, the allowed tokens are those whose text extends the current
text towards at least one function name. Every other logit is set to `-inf`
and the argmax is taken. As soon as a single name remains, it is completed
without calling the model.

### 4. Parameter values (by type from `functions_definition.json`)

- **`number` / `integer`**: a small automaton
  (`start -> minus/zero/int -> dot -> frac`) validates the number character by
  character, so `007`, `1.` or `.5` can never be produced. A token is allowed
  if the whole token keeps the automaton valid. The number can end only in an
  accepting state, and the model ends it by choosing a token that starts with
  the delimiter (`,` or `}`). `integer` forbids the decimal point. The forced skeleton stops right after
  the colon and the model picks the leading space itself: BPE writes `-3` as
  `' -'` + `'3'` but `7` as `' '` + `'7'`, so forcing a lone space would push
  the model towards digits.
- **`string`**: any token valid inside a JSON string is allowed. The string
  ends when the model picks a *closing token*: a token containing the first
  unescaped quote whose remaining text is a prefix of the skeleton that
  follows (`",`, `"}}`, `)",`...). The content before the quote is kept, and
  the closing token itself is replaced by the forced skeleton. Escapes are
  enforced (`\\`, `\"`, `\n`...), so a regex such as `\d+` must be written
  `\\d+` and decodes correctly with `json.loads`.
- **`boolean`**: the model chooses between ` true` and ` false` (single
  tokens with their leading space) with the same mechanism as function names.

Other types are reported as an error for that prompt.

### 5. Output

Results are validated with pydantic and written as a JSON array of
`{prompt, name, parameters}` objects (numbers as floats, e.g. `2.0`).

### Speed-ups

- Forced tokens (skeleton, unique function name) never call the model.
- The model is skipped whenever only one token is allowed.
- Token classes are precomputed once; "valid next strings" are turned into
  ids with a dictionary lookup.

## Design decisions

- **Hand-built skeleton instead of a generic JSON automaton.** The schema is
  known, so the model only decides at the value positions. It is simpler,
  faster (fewer forward passes) and the structure cannot be wrong.
- **Constraints on text, not on tokens.** Because of BPE merges, validity is
  computed from the string a token adds.
- **Short prompt.** The SDK has no KV cache (and private attributes are
  forbidden), so every generated token recomputes the whole context: the cost
  grows with the prompt length. A long prompt with rules and examples doubled
  the run time without improving the score (see below).
- **pydantic everywhere** (models, vocabulary, decoder), as required.
- **Errors are never fatal.** Missing or invalid files, wrong JSON shapes and
  model-loading problems end with a clear message on stderr. If one prompt
  fails, it is reported and skipped, and the others are still written.
- **Only the public SDK is used**: `encode`, `get_logits_from_input_ids` and
  `get_path_to_vocab_file`.

## Performance analysis

Measured on CPU only (no GPU), on the 11 provided prompts and on a personal
set of 14 prompts with 6 other functions:

| Metric | Provided prompts (11) | Personal set (14) |
|---|---|---|
| Valid JSON, schema-compliant | 11/11 | 14/14 |
| Correct function | 11/11 | 14/14 |
| Correct arguments | 10/11 (about 91%) | 13/14 (about 93%) |
| Run time, personal laptop | 220-260 s | about 180 s |

The substitution results were checked by applying the generated regex to the
source string and comparing with the expected text. The one failure is the
prompt "Replace all vowels ... with asterisks": the regex is correct but the
model writes the word `asterisk` instead of `*`.

Observations:

- Time is dominated by forward passes (about 2 s each on the laptop).
  Simple prompts take 7-20 s, substitution prompts 30-55 s because they
  generate three strings.
- A prompt with extra rules and examples made the run about 2.3x slower
  (about 500 s) with the same score.
- Shortening the prompt further saved about 17% of the time but dropped the
  score to 9/11: small models are very sensitive to the prompt wording.
- Validity is not a statistical property: it comes from the masks, not from
  the model.

### Known limitations

- **Values come from a 0.6B model.** The masks guarantee the form of the
  output, not the correctness of the values. In "Replace all vowels ... with
  asterisks" the model writes the word `asterisk` instead of `*`. There is no
  post-correction on purpose: fixing values with rules would be a heuristic.
- **Quotes inside a string.** The model tends to copy them without escaping,
  and an unescaped quote then looks like the end of the string:
  `Convert 'say "hi"' to uppercase` gives `say ` instead of `say "hi"`.
- **Prompt sensitivity.** Small changes in the prompt wording change the
  results, in both directions (see above).
- **Speed.** There is no KV cache, so the run time grows with the number of
  prompts and with the prompt length. On a slow CPU the 5 minutes limit is
  tight for the 11 provided prompts.
- **Characters** split across several tokens (some emojis and rare
  characters) cannot be generated inside a string, and `\uXXXX` escapes are
  not produced.
- **Caps**: strings are limited to 150 tokens and numbers to 25 tokens.
- **Types**: `number`, `integer`, `string` and `boolean` are supported. Other
  types (arrays, objects, null) are reported as an error for that prompt.

## Challenges faced

- **Tokens that glue content and the closing quote.** The first version only
  allowed a string to end with a token *starting* with `"`. The real
  vocabulary has tokens like `)",`. The mask forbade the model's favourite
  token and forced a wrong one, producing the regex `([0-9]+)|`. It was
  found by logging, for each step, the model's unconstrained choice next to
  the constrained one, and fixed with the `closers` table.
- **Spaces and BPE.** A negative number came out positive (`-3` became `3`)
  because the skeleton forced a lone space after the colon, which BPE never
  produces before a minus sign. The model now chooses the space itself.
- **Run time.** No KV cache and a growing prompt exceeded the 5 minutes limit.
  Solved by keeping the prompt short, forcing deterministic tokens and
  skipping the model when only one token is possible.
- **JSON escaping of regexes.** A single backslash is not valid JSON, so the
  mask only accepts valid escapes after a backslash, and the value is decoded
  with `json.loads`.
- **Limits of a 0.6B model.** The model sometimes copies a word from the
  request instead of the symbol it stands for. Constrained decoding
  guarantees the form of the output, not the correctness of the values.
- **Disk space.** The model and package caches were moved out of the home
  directory with `HF_HOME` and `UV_CACHE_DIR`.

## Testing strategy

- **Input handling**: missing file, directory instead of a file, invalid
  JSON, JSON that is not an array, missing keys, empty function list and
  empty prompt list all give a clear error without a traceback.
- **Provided prompts**: a throwaway script generated the 11 calls and compared
  them with expected values (applying the regex for substitutions, since
  several regexes are valid).
- **Mask debugging**: a trace of each step showing where the mask overrides the
  model's own preference, used to find the closing-quote bug.
- **Personal test set**: 14 prompts with 6 other functions (`integer`,
  `boolean`, negative and decimal numbers, an empty string, an accent, escaped
  quotes) and expected outputs, run locally and not part of the repository. It
  revealed the leading-space bug on negative numbers. The remaining failure is
  the unescaped quote case described above.
- **Static checks**: `flake8` and `mypy` through `make lint`.
- **Clean-clone check**: clone the repository and run only `uv sync` then
  `uv run python -m src`.

## Example usage

```console
$ make run
[1/11] fn_add_numbers (16.1s)
...
[11/11] fn_substitute_string_with_regex (35.7s)
Total: 230s
[
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {"name": "shrek"}
  },
  ...
]
```

```console
$ uv run python -m src --input missing.json
Error: file not found: missing.json
```

## Resources

- Qwen3 model card: https://huggingface.co/Qwen/Qwen3-0.6B
- B. T. Willard, R. Louf, *Efficient Guided Generation for Large Language
  Models*, 2023 (arXiv:2307.09702)
- R. Sennrich et al., *Neural Machine Translation of Rare Words with Subword
  Units*, 2016 (BPE)
- A. Radford et al., *Language Models are Unsupervised Multitask Learners*,
  2019 (GPT-2, byte-level BPE)
- JSON specification, RFC 8259: https://www.rfc-editor.org/rfc/rfc8259
- pydantic documentation: https://docs.pydantic.dev
- uv documentation: https://docs.astral.sh/uv/

### Use of AI

An AI assistant (Claude) was used as a tutor and pair-programming partner
for:

- reasoning about bugs from the logs (for instance the closing-quote issue
  above) and about the run time.
- redaction of README.md
