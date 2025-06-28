from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable, Dict

from duckduckgo_search import DDGS
import requests
from bs4 import BeautifulSoup


class Tool:
    """Base callable tool."""

    def __init__(self, name: str, description: str, fn: Callable[[str], str]):
        self.name = name
        self.description = description
        self.fn = fn

    def __call__(self, arg: str) -> str:
        return self.fn(arg)


# ---------------------------------------------------------------------------
# Individual tool implementations
# ---------------------------------------------------------------------------

def shell_tool(command: str) -> str:
    """Execute *read-only* shell commands and capture output."""
    try:
        completed = subprocess.run(
            ["powershell", "-Command", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
        output = completed.stdout.strip()
        if len(output.encode("utf-8")) > 24000:
            output = output[:8000] + "\n... (output truncated) ...\n" + output[-8000:]
        return output
    except Exception as exc:  # noqa: BLE001
        return f"[shell error] {exc}"


def fs_read_tool(path: str) -> str:
    """Read text files (<128 KB) relative to CWD."""
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        return f"[fs error] File not found: {file_path}"
    if file_path.stat().st_size > 128_000:
        return "[fs error] File too large (>128 KB)."
    return file_path.read_text(encoding="utf-8", errors="ignore")


def fs_write_tool(path_content: str) -> str:
    """Write content to a text file relative to CWD."""
    try:
        lines = path_content.split("\n", 1)
        if len(lines) < 2:
            return (
                "[fs error] Invalid argument format. "
                "Expected path on the first line and content on subsequent lines."
            )

        path_str, content = lines
        file_path = Path(path_str.strip()).expanduser().resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        return f"[fs success] File written to {file_path}"
    except Exception as exc:
        return f"[fs error] {exc}"


def web_search_tool(query: str) -> str:
    """Search the web using DuckDuckGo."""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=5)]
        return json.dumps(results)
    except Exception as exc:
        return f"[web_search error] {exc}"


def browse_web_page_tool(url: str) -> str:
    """Fetch and parse the content of a web page."""
    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text()
    except Exception as exc:
        return f"[browse_web_page error] {exc}"


# Registry of available tools
TOOLS: Dict[str, Tool] = {
    "shell": Tool("shell", "Run Windows Master Administrator Powershell commands", shell_tool),
    "fs_read": Tool("fs_read", "Read a small text file", fs_read_tool),
    "fs_write": Tool(
        "fs_write",
        "Write content to a text file. The first line of the argument must be the file path, and the rest is the content to write.",
        fs_write_tool,
    ),
    "web_search": Tool("web_search", "Search the web with a query", web_search_tool),
    "browse_web_page": Tool("browse_web_page", "Get the text content of a web page", browse_web_page_tool),
}
