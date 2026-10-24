import argparse
import json
import sys
import time

from llm_sdk import Small_LLM_Model

from .decoder import Decoder
from .loader import DataError, load_functions, load_prompts, save_results
from .models import FunctionCall
from .prompt import build_prompt
from .vocab import Vocabulary

DEFAULT_FUNCTIONS = "data/input/functions_definition.json"
DEFAULT_INPUT = "data/input/function_calling_tests.json"
DEFAULT_OUTPUT = "data/output/function_calling_results.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        The parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Translate natural language prompts into function calls.",
    )
    parser.add_argument("--functions_definition", default=DEFAULT_FUNCTIONS)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Load the data, process every prompt and write the results.

    Args:
        args: Parsed command line arguments.

    Returns:
        0 if every prompt was processed, 1 otherwise.

    Raises:
        DataError: On a problem with the files or the model.
    """
    functions = load_functions(args.functions_definition)
    prompts = load_prompts(args.input)
    try:
        model = Small_LLM_Model()
        vocab_path = model.get_path_to_vocab_file()
    except Exception as exc:
        raise DataError(f"cannot load the model: {exc}") from exc
    decoder = Decoder(model=model, vocab=Vocabulary.from_file(vocab_path))

    results: list[FunctionCall] = []
    failures = 0
    started = time.time()
    for index, item in enumerate(prompts, start=1):
        step = time.time()
        try:
            name, params = decoder.generate_call(
                build_prompt(functions, item.prompt), functions
            )
        except (ValueError, RuntimeError) as exc:
            failures += 1
            print(f"Warning: prompt #{index} skipped: {exc}", file=sys.stderr)
            continue
        results.append(
            FunctionCall(prompt=item.prompt, name=name, parameters=params)
        )
        print(
            f"[{index}/{len(prompts)}] {name} ({time.time() - step:.1f}s)",
            file=sys.stderr,
        )
    print(f"Total: {time.time() - started:.0f}s", file=sys.stderr)
    save_results(args.output, results)
    print(
        json.dumps(
            [r.model_dump() for r in results], indent=2, ensure_ascii=False
        )
    )
    print(
        f"{len(results)} result(s) written to {args.output}", file=sys.stderr
    )
    return 1 if failures else 0


def main() -> int:
    """Run the program without ever crashing with a traceback.

    Returns:
        Process exit code (0 on success, 1 on error, 130 on Ctrl+C).
    """
    args = parse_args()
    try:
        return run(args)
    except DataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
