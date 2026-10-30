"""Built-in local tools. Each one is a plain function plus a JSON schema and a tier."""

import ast
import operator
import re
import subprocess
from datetime import datetime
from pathlib import Path

from agent import config
from agent.tools import BLOCKED, CONFIRM, tool

SECRET_PATTERNS = (".ssh", ".env", "id_rsa", ".aws", "keychain", "credentials", ".netrc")
DANGEROUS_SHELL = [
    r"\brm\s+-\w*[rf]",
    r"\bsudo\b",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"curl[^|]*\|\s*(ba|z)?sh",
    r"\bchmod\s+(-R\s+)?777",
    r">\s*/dev/",
    r"\bshutdown\b",
    r"\breboot\b",
    r":\(\)\s*\{",
]


def _inside_workspace(path: str) -> Path | None:
    p = (config.WORKSPACE / path).resolve()
    return p if p.is_relative_to(config.WORKSPACE.resolve()) else None


def _workspace_path(path: str) -> Path:
    """Resolve a path inside the workspace.

    Args:
        path (str): Path relative to the workspace.

    Returns:
        Path: The resolved absolute path.

    Raises:
        PermissionError: If the path leaves the workspace. The guard normally blocks this first.
    """
    p = _inside_workspace(path)
    if p is None:
        raise PermissionError(f"'{path}' is outside the workspace")
    return p


def _path_guard(args: dict) -> str | None:
    path = args.get("path", ".")
    if any(s in path.lower() for s in SECRET_PATTERNS):
        return f"'{path}' looks like a secrets file"
    if _inside_workspace(path) is None:
        return f"'{path}' is outside the workspace {config.WORKSPACE}"
    return None


def _shell_guard(args: dict) -> str | None:
    for pattern in DANGEROUS_SHELL:
        if re.search(pattern, args["command"]):
            return f"command matches a dangerous pattern ({pattern})"
    return None


@tool(
    "get_time",
    "Get the current local date and time.",
    {"properties": {}},
    examples=["what time is it", "what's today's date"],
)
def get_time() -> str:
    """Return the current local date and time.

    Returns:
        str: For example "Wednesday 2026-10-28 14:05".
    """
    return datetime.now().strftime("%A %Y-%m-%d %H:%M")


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


@tool(
    "calculator",
    "Evaluate an arithmetic expression such as '12 * (3 + 4)'.",
    {"properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    examples=["what is 15% of 80", "multiply 17 by 23"],
)
def calculator(expression: str) -> float:
    """Evaluate an arithmetic expression without calling eval().

    Args:
        expression (str): Numbers with + - * / % and **, and unary minus.

    Returns:
        float: The value of the expression.

    Raises:
        ValueError: If the expression has anything else in it.
    """

    def ev(n):  # walk the syntax tree ourselves: never eval() model output
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp):
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp):
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("only numbers and + - * / % ** are allowed")

    return ev(ast.parse(expression, mode="eval").body)


@tool(
    "save_note",
    "Remember something for the user: a fact, appointment, decision, reminder or note. Saves it to long-term memory.",
    {
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["text"],
    },
    examples=[
        "remember that my dentist is on Friday",
        "note down that Ravi's number changed",
        "don't let me forget the demo",
    ],
)
def save_note(text: str, tags: list[str] | None = None) -> str:
    """Save a note to long-term memory.

    Args:
        text (str): What to remember.
        tags (list[str], optional): Labels stored with the note. Defaults to None.

    Returns:
        str: "saved", or a message that the junk filter rejected the note.
    """
    from agent.memory import episodic

    stored = episodic.store(text, kind="note", meta={"tags": tags or []})
    return "saved" if stored else "not saved (filtered as junk or duplicate)"


@tool(
    "search_memory",
    "Recall what the user said, noted, decided, saw or had to do earlier (voice notes, action items, past screens).",
    {"properties": {"query": {"type": "string", "minLength": 1}}, "required": ["query"]},
    examples=["what did I say about Priya", "what did I decide about the venue", "what do I need to do tomorrow"],
)
def search_memory(query: str) -> list[dict]:
    """Search long-term memory.

    Args:
        query (str): What to look for.

    Returns:
        list[dict]: The best matches, each with `text`, `kind` and `when`.
    """
    from agent.memory import episodic

    return [{"text": m["text"], "kind": m["kind"], "when": m["when"]} for m in episodic.search(query)]


@tool(
    "list_files",
    "List files in a folder of the agent workspace.",
    {"properties": {"path": {"type": "string"}}},
    guard=_path_guard,
    examples=["what's in my notes folder", "list my files"],
)
def list_files(path: str = ".") -> list[str]:
    """List the entries of a folder in the workspace.

    Args:
        path (str, optional): Folder relative to the workspace. Defaults to ".".

    Returns:
        list[str]: Sorted entry names.
    """
    return sorted(p.name for p in _workspace_path(path).iterdir())


@tool(
    "read_file",
    "Read a text file from the agent workspace.",
    {"properties": {"path": {"type": "string"}}, "required": ["path"]},
    guard=_path_guard,
    examples=["open hello.txt", "show me what todo.md says"],
)
def read_file(path: str) -> str:
    """Read the start of a text file in the workspace.

    Args:
        path (str): File relative to the workspace.

    Returns:
        str: The first 4,000 characters of the file.
    """
    return _workspace_path(path).read_text()[:4000]


@tool(
    "write_file",
    "Write a text file in the agent workspace.",
    {"properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    tier=CONFIRM,
    guard=_path_guard,
    examples=["save this text as todo.txt", "create a file called plan.md"],
)
def write_file(path: str, content: str) -> str:
    """Write a text file in the workspace, creating folders as needed.

    Args:
        path (str): File relative to the workspace.
        content (str): Text to write.

    Returns:
        str: A short confirmation.
    """
    p = _workspace_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} characters to {path}"


@tool(
    "run_shell",
    "Run a shell command in the agent workspace and return its output.",
    {"properties": {"command": {"type": "string"}}, "required": ["command"]},
    tier=CONFIRM,
    guard=_shell_guard,
    examples=["run ls", "run the tests with pytest"],
)
def run_shell(command: str) -> str:
    """Run a shell command in the workspace. The tool is in the confirm tier, so the user approves it first.

    Args:
        command (str): The command line.

    Returns:
        str: The last 4,000 characters of stdout and stderr.
    """
    r = subprocess.run(command, shell=True, cwd=config.WORKSPACE, capture_output=True, text=True, timeout=30)
    return (r.stdout + r.stderr)[-4000:]


@tool(
    "delete_file",
    "Delete a file.",
    {"properties": {"path": {"type": "string"}}, "required": ["path"]},
    tier=BLOCKED,
    examples=["delete old.txt", "remove that file"],
)
def delete_file(path: str) -> str:
    """Delete a file. The tool is in the blocked tier and never runs.

    Args:
        path (str): File relative to the workspace.

    Raises:
        RuntimeError: Always.
    """
    raise RuntimeError("unreachable: blocked tier")


config.WORKSPACE.mkdir(parents=True, exist_ok=True)
