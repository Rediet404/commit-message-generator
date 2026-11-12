"""Extraction and loading of code-docstring pairs for fine-tuning.

Implements the requirement:
"Adaptation: Use as is first; then fine-tune on code-docstring pairs"
"Data: CodeSearchNet pairs, or commits from public repositories"
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
from pathlib import Path
from typing import Iterable, Sequence


# Curated starter code-docstring pairs (representative of CodeSearchNet Python pairs)
STARTER_PAIRS = [
    {
        "function_name": "add_vectors",
        "code": "def add_vectors(v1, v2):\n    return [x + y for x, y in zip(v1, v2)]",
        "docstring": "Compute element-wise sum of two vectors.",
        "source": "curated_starter",
    },
    {
        "function_name": "find_primes",
        "code": "def find_primes(limit):\n    primes = []\n    for num in range(2, limit):\n        if all(num % d != 0 for d in range(2, int(num**0.5) + 1)):\n            primes.append(num)\n    return primes",
        "docstring": "Return all prime numbers up to the specified limit.",
        "source": "curated_starter",
    },
    {
        "function_name": "parse_header",
        "code": "def parse_header(header_line):\n    key, _, value = header_line.partition(':')\n    return key.strip().lower(), value.strip()",
        "docstring": "Parse an HTTP header line into a normalized key-value tuple.",
        "source": "curated_starter",
    },
    {
        "function_name": "flatten_nested",
        "code": "def flatten_nested(items):\n    result = []\n    for item in items:\n        if isinstance(item, list):\n            result.extend(flatten_nested(item))\n        else:\n            result.append(item)\n    return result",
        "docstring": "Recursively flatten a nested list structure into a flat list.",
        "source": "curated_starter",
    },
    {
        "function_name": "validate_email",
        "code": "def validate_email(address):\n    return bool(re.match(r'^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$', address))",
        "docstring": "Validate whether the string adheres to a standard email format.",
        "source": "curated_starter",
    },
    {
        "function_name": "load_config",
        "code": "def load_config(file_path):\n    with open(file_path, 'r', encoding='utf-8') as stream:\n        return json.load(stream)",
        "docstring": "Read and deserialize JSON configuration from the given file path.",
        "source": "curated_starter",
    },
    {
        "function_name": "clamp_value",
        "code": "def clamp_value(val, min_val, max_val):\n    return max(min_val, min(val, max_val))",
        "docstring": "Constrain a value within a specified minimum and maximum range.",
        "source": "curated_starter",
    },
    {
        "function_name": "to_camel_case",
        "code": "def to_camel_case(snake_str):\n    parts = snake_str.split('_')\n    return parts[0] + ''.join(word.capitalize() for word in parts[1:])",
        "docstring": "Convert a snake_case string into camelCase format.",
        "source": "curated_starter",
    },
]


def remove_docstring(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    """Remove docstring from an AST function node in-place to prevent target leakage."""
    if (
        fn_node.body
        and isinstance(fn_node.body[0], ast.Expr)
        and isinstance(fn_node.body[0].value, ast.Constant)
        and isinstance(fn_node.body[0].value.value, str)
    ):
        fn_node.body.pop(0)
    if not fn_node.body:
        fn_node.body = [ast.Pass()]


def extract_pairs_from_code(source_code: str, source_label: str = "source") -> list[dict[str, str]]:
    """Extract (code_without_docstring, docstring) pairs from Python source code."""
    pairs: list[dict[str, str]] = []
    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return pairs

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            if not docstring:
                continue

            # First non-empty line of docstring
            first_line = docstring.strip().splitlines()[0].strip()
            if not first_line or len(first_line) < 5:
                continue

            # Clone function and remove docstring to produce clean code input
            node_copy = copy.deepcopy(node)
            remove_docstring(node_copy)
            ast.fix_missing_locations(node_copy)

            try:
                code_text = ast.unparse(node_copy).strip()
            except Exception:
                code_text = f"def {node.name}(): pass"

            pairs.append({
                "function_name": node.name,
                "code": code_text,
                "docstring": first_line,
                "source": source_label,
            })

    return pairs


def extract_pairs_from_directory(root_dir: Path) -> list[dict[str, str]]:
    """Scan directory recursively for Python files and extract code-docstring pairs."""
    all_pairs: list[dict[str, str]] = []
    for py_file in root_dir.rglob("*.py"):
        if any(part.startswith((".", "__pycache__", "venv", ".venv")) for part in py_file.parts):
            continue
        try:
            content = py_file.read_text(encoding="utf-8")
            all_pairs.extend(extract_pairs_from_code(content, source_label=str(py_file)))
        except Exception:
            continue
    return all_pairs


def load_code_docstring_dataset(path: Path | None = None) -> list[dict[str, str]]:
    """Load code-docstring pairs from a JSONL file, or return starter pairs if None/missing."""
    if path is None or not path.exists():
        return list(STARTER_PAIRS)

    records: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Support both CodeSearchNet format ("code" or "func_code", "docstring" or "doc")
            code = data.get("code") or data.get("func_code") or data.get("function") or ""
            docstring = data.get("docstring") or data.get("doc") or data.get("summary") or ""
            if code and docstring:
                records.append({
                    "function_name": data.get("function_name", "func"),
                    "code": code,
                    "docstring": docstring,
                    "source": data.get("source", str(path)),
                })
    return records or list(STARTER_PAIRS)


def save_code_docstring_dataset(records: Iterable[dict[str, str]], output_path: Path) -> None:
    """Save code-docstring pairs to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Directory of Python code to extract pairs from")
    parser.add_argument("--output", type=Path, default=Path("data/processed/code_docstring_pairs.jsonl"))
    parser.add_argument("--include-starter", action="store_true", default=True)
    args = parser.parse_args(argv)

    pairs: list[dict[str, str]] = []
    if args.include_starter:
        pairs.extend(STARTER_PAIRS)
    if args.source_dir and args.source_dir.is_dir():
        extracted = extract_pairs_from_directory(args.source_dir)
        pairs.extend(extracted)
        print(f"Extracted {len(extracted)} pairs from {args.source_dir}")

    save_code_docstring_dataset(pairs, args.output)
    print(f"Saved {len(pairs)} code-docstring pairs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
