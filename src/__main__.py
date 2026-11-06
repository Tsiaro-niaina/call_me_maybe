"""Entry point of the function calling tool."""

import argparse
import json
import sys
import textwrap
import time
from llm_sdk import Small_LLM_Model
from .decoder import Decoder
from .loader import DataError, load_functions, load_prompts, save_results
from .models import FunctionCall
from .prompt import build_prompt
from .vocabulary import Vocabulary


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
    parser.add_argument("--functions_definition",
                        default="data/input/functions_definition.json")
    parser.add_argument("--input",
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output",
                        default="data/output/function_calling_results.json")
    return parser.parse_args(argv)


def format_function_call(function_call: FunctionCall) -> str:
    """Return a function call as indented JSON, ready to sit in an array.

    Args:
        function_call: The call to format.

    Returns:
        The JSON text, indented by two spaces like the elements of an array.
    """
    call_text = json.dumps(
        function_call.model_dump(), indent=2, ensure_ascii=False
    )
    return textwrap.indent(call_text, "  ")


def run(args: argparse.Namespace) -> int:
    """Load the data, process every prompt and write the results.

    Each result is printed as soon as it is generated, so that the output on
    stdout is a JSON array that grows one element at a time.

    Args:
        args: Parsed command line arguments.

    Returns:
        0 if every prompt was processed, 1 otherwise.

    Raises:
        DataError: On a problem with the files or the model.
    """
    functions = load_functions(args.functions_definition)
    prompt_items = load_prompts(args.input)
    try:
        model = Small_LLM_Model()
        vocab_path = model.get_path_to_vocab_file()
    except Exception as error:
        raise DataError(f"cannot load the model: {error}") from error
    decoder = Decoder(model=model, vocab=Vocabulary.from_file(vocab_path))

    function_calls: list[FunctionCall] = []
    failed_prompts = 0
    start_time = time.time()
    print("[", flush=True)
    for prompt_number, prompt_item in enumerate(prompt_items, start=1):
        try:
            function_name, argument_values = decoder.generate_call(
                build_prompt(functions, prompt_item.prompt), functions
            )
        except (ValueError, RuntimeError) as error:
            failed_prompts += 1
            print(
                f"Warning: prompt #{prompt_number} skipped: {error}",
                file=sys.stderr,
            )
            continue
        function_call = FunctionCall(
            prompt=prompt_item.prompt,
            name=function_name,
            parameters=argument_values,
        )
        if function_calls:
            print(",")
        print(format_function_call(function_call), end="", flush=True)
        function_calls.append(function_call)
    print("\n]" if function_calls else "]")
    print(f"Total: {time.time() - start_time:.0f}s", file=sys.stderr)
    save_results(args.output, function_calls)
    print(
        f"{len(function_calls)} result(s) written to {args.output}",
        file=sys.stderr,
    )
    return 1 if failed_prompts else 0


def main() -> int:
    """Run the program without ever crashing with a traceback.

    Returns:
        Process exit code (0 on success, 1 on error, 130 on Ctrl+C).
    """
    args = parse_args()
    try:
        return run(args)
    except DataError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Unexpected error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
