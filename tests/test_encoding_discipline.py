"""Guard the explicit-encoding discipline that keeps the project portable off UTF-8 locales.

On an English Windows install ``Path.read_text``, ``Path.write_text`` and ``open`` in text mode
resolve their encoding from the ANSI code page (cp1252), not UTF-8. Every artifact this project
writes is UTF-8, and several committed documents and generated reports contain characters cp1252
cannot even decode (``−`` U+2212, ``≥`` U+2265, ``²`` U+00B2), so a missing ``encoding=`` is a
hard failure on the target machine rather than a style preference.

The first test reads the shipped sources instead of executing them, so a new file cannot quietly
reintroduce the defect. The second test proves the runtime path: it loads the committed summary
under a deliberately non-UTF-8 locale. The end-to-end proof is the whole suite executed with
``LC_ALL=C PYTHONUTF8=0``, which is recorded in the audit document.
"""

from __future__ import annotations

import ast
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SHIPPED = [
    "main.py",
    "config.py",
    "errors.py",
    "dashboard",
    "healing",
    "experiments",
    "evaluation",
    "rl",
    "scheduler",
    "visualization",
    "tools",
]
# ``webbrowser.open`` and friends are not file handles; only path/mode calls are inspected.
NON_FILE_OPENERS = {"webbrowser", "os", "socket", "urllib", "gzip", "codecs", "builtins"}


def _shipped_files():
    for entry in SHIPPED:
        path = ROOT / entry
        if path.is_file():
            yield path
        else:
            yield from sorted(path.rglob("*.py"))


def _text_calls(tree: ast.AST):
    """Yield (call, describes_file_handle) for text-mode file API calls."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        keywords = {keyword.arg for keyword in node.keywords if keyword.arg}
        if name in {"read_text", "write_text"}:
            yield node, "encoding" not in keywords
            continue
        if name != "open" or "encoding" in keywords:
            continue
        parts = ast.unparse(func).split(".")
        if len(parts) > 1 and parts[0] in NON_FILE_OPENERS:
            continue
        # ``open(path, mode)`` takes the mode second, a bound ``path.open(mode)`` first.
        mode_index = 1 if isinstance(func, ast.Name) else 0
        mode: object = "r"
        if len(node.args) > mode_index:
            argument = node.args[mode_index]
            mode = argument.value if isinstance(argument, ast.Constant) else None
        for keyword in node.keywords:
            if keyword.arg == "mode":
                mode = keyword.value.value if isinstance(keyword.value, ast.Constant) else None
        if not isinstance(mode, str):
            # A computed mode (for example ``"ab" if append else "wb"``) is chosen at runtime
            # and is binary in every branch used here.
            continue
        if "b" in mode:
            continue
        yield node, True


def test_shipped_modules_name_an_explicit_encoding():
    missing = []
    for path in _shipped_files():
        source = path.read_text(encoding="utf-8")
        for node, needs_encoding in _text_calls(ast.parse(source)):
            if needs_encoding:
                missing.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not missing, (
        "text-mode file calls without an explicit encoding fail on a cp1252 Windows "
        f"console: {sorted(missing)}"
    )


def test_canonical_summary_loads_under_a_non_utf8_locale():
    script = (
        "import json, pathlib, tempfile\n"
        "from healing import canonical_numbers\n"
        "disc = canonical_numbers.load_summary_discrimination()\n"
        "assert disc['helped'] == canonical_numbers.HELPFUL, disc['helped']\n"
        "text = json.dumps(disc)\n"
        "target = pathlib.Path(tempfile.mkdtemp()) / 'summary-copy.json'\n"
        "target.write_text(text, encoding='utf-8')\n"
        "assert json.loads(target.read_text(encoding='utf-8')) == disc\n"
        "print('ok')\n"
    )
    environment = dict(os.environ, LC_ALL="C", LANG="C", PYTHONUTF8="0", PYTHONIOENCODING="")
    completed = subprocess.run(
        [sys.executable, "-X", "utf8=0", "-c", script],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
