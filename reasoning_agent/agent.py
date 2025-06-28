from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Dict, List

import openai
import tiktoken
from dotenv import load_dotenv
from rich.console import Console

from .tools import TOOLS

# ---------------------------------------------------------------------------
# Environment & OpenAI setup
# ---------------------------------------------------------------------------
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

console = Console()

# ---------------------------------------------------------------------------
# Agent logic and manifest
# ---------------------------------------------------------------------------
AGENT_LOGIC = textwrap.dedent(
    """
    You are a planning agent that helps break down tasks into smaller steps and reason about the current state.
    Your role is to:
    1. Analyze the current state and history.
    2. Evaluate progress towards the ultimate goal.
    3. Identify potential challenges or roadblocks.
    4. Formulate a plan and decide the next concrete action to take.

    Your output must always be a single JSON object.

    If the task is not yet complete, you must include a tool call in your response. Your JSON response should include both your full plan and the next tool to use.
    - The `tool` and `arg` fields specify the next action to take.
    - The other fields are for your planning and reasoning process.

    Example Response (when a tool needs to be run):
    {{
        "state_analysis": "I need to read the contents of three files.",
        "progress_evaluation": "10% - Just starting, need to read the first file.",
        "challenges": "The files might not exist.",
        "next_steps": ["Read poem1.txt", "Read poem2.txt", "Read poem3.txt", "Combine content", "Count words"],
        "reasoning": "I will start by reading the first file to make progress on the task.",
        "tool": "fs_read",
        "arg": "poem1.txt"
    }}

    When the entire task is complete and no more tools are needed, provide the final answer in the `answer` field.

    Example Response (when the task is complete):
    {{
        "state_analysis": "All files have been read, combined, and the words have been counted.",
        "progress_evaluation": "100% - The task is complete.",
        "challenges": "None.",
        "next_steps": [],
        "reasoning": "The final word count has been determined, so the task is finished.",
        "answer": "The total word count is 42."
    }}
    """
)

TOOL_MANIFEST = {name: tool.description for name, tool in TOOLS.items()}


class ReasoningAgent:
    """LLM-driven agent that keeps its own reasoning & execution trace."""

    def __init__(self, model: str = MODEL, max_turns: int = 10):
        self.model = model
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []
        self.last_fs_snapshot: str = ""
        self.last_shell_output: str = ""
        self.report_path = Path("query_report.md")
        self.last_context: List[Dict[str, str]] = []
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        with self.report_path.open("w", encoding="utf-8") as f:
            f.write("# Agent Query Report\n\n")

    def _format_message_for_report(self, message: Dict[str, str], is_fs_unchanged: bool = False) -> str:
        role = message["role"]
        content = message["content"]

        colors = {
            "system": "#e1f5fe",
            "user": "#e8f5e9",
            "assistant": "#f5f5f5",
        }
        color = colors.get(role, "#ffffff")

        processed_content = ""
        if role == "assistant":
            if content.startswith("[tool_result]"):
                color = "#fffde7"
                processed_content = f"<b>Tool Result:</b><pre>{content[len('[tool_result]'):].strip()}</pre>"
            elif content.startswith("[filesystem]"):
                if is_fs_unchanged:
                    shell_part = ""
                    if "[shell_output]" in content:
                        shell_part = content.split("[shell_output]", 1)[1]
                        return (
                            '<div style="background-color: #fff8e1; padding: 2px; margin: 0; border-radius: 5px; font-family: monospace;">'
                            f'<b>ASSISTANT</b><br/><b>Shell Output:</b><pre>{shell_part.strip()}</pre></div>'
                        )
                    else:
                        return ""
                color = "#fff8e1"
                processed_content = f"<b>Filesystem:</b><pre>{content[len('[filesystem]'):].strip()}</pre>"
            else:
                try:
                    payload = json.loads(content)
                    explanation = payload.get("explanation", "")
                    tool = payload.get("tool")
                    arg = payload.get("arg")
                    answer = payload.get("answer")

                    processed_content = f"<b>Thought:</b> {explanation}<br/>"
                    if tool:
                        processed_content += f"<b>Tool:</b> {tool} | <b>Arg:</b> {arg}"
                    if answer:
                        processed_content += f"<b>Answer:</b> {answer}"
                except (json.JSONDecodeError, TypeError):
                    processed_content = f"<pre>{content}</pre>"
        else:
            processed_content = f"<pre>{content}</pre>"

        return (
            f'<div style="background-color: {color}; padding: 2px; margin: 0; border-radius: 5px; font-family: monospace;">'
            f'<b>{role.upper()}</b><br/>{processed_content}</div>'
        )

    @staticmethod
    def _filesystem_snapshot(limit: int = 20) -> str:
        from datetime import datetime

        output = []
        entries = sorted(Path.cwd().iterdir())[:limit]
        for entry in entries:
            try:
                stat = entry.stat()
                size = stat.st_size
                mtime = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                if entry.is_dir():
                    output.append(f"drw-r--r-- {size:>10} {mtime} {entry.name}/")
                else:
                    output.append(f"-rw-r--r-- {size:>10} {mtime} {entry.name}")
            except OSError:
                output.append(f"?-????-?? ? {'?':>10} ? ? {entry.name}")
        return "\n".join(output)

    def _build_context(self, user_msg: str) -> List[Dict[str, str]]:
        script_path = Path(__file__).resolve()
        agent_logic_prompt = AGENT_LOGIC.format(cwd=Path.cwd(), script_path=script_path)
        context: List[Dict[str, str]] = [
            {"role": "system", "content": agent_logic_prompt},
            {"role": "system", "content": f"TOOL MANIFEST:\n{json.dumps(TOOL_MANIFEST, indent=2)}"},
        ]
        context.extend(self.history)

        current_fs_snapshot = self._filesystem_snapshot()
        env_parts = [f"[filesystem]\n{current_fs_snapshot}"]
        if self.last_shell_output:
            env_parts.append(f"[shell_output]\n{self.last_shell_output}")

        context.append({"role": "assistant", "content": "\n".join(env_parts)})
        context.append({"role": "user", "content": user_msg})
        return context

    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        num_tokens = sum(len(self.encoding.encode(msg["content"])) for msg in messages)
        with self.report_path.open("a", encoding="utf-8") as f:
            f.write(f"### Turn {len(self.history) // 2 + 1} ({num_tokens} tokens)\n\n")
            f.write("<h4>CONTEXT DIFF</h4>\n")
            diff_messages = messages[len(self.last_context):]
            new_fs_snapshot = self._filesystem_snapshot()
            is_fs_unchanged = new_fs_snapshot == self.last_fs_snapshot
            self.last_fs_snapshot = new_fs_snapshot
            for msg in diff_messages:
                f.write(self._format_message_for_report(msg, is_fs_unchanged=is_fs_unchanged))
            self.last_context = messages
            f.write("\n\n")

        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
        )
        assistant_response = response.choices[0].message.content.strip()

        with self.report_path.open("a", encoding="utf-8") as f:
            f.write("<h4>LLM ANSWER</h4>\n")
            try:
                payload = json.loads(assistant_response)
                explanation = payload.get("explanation", "")
                tool = payload.get("tool")
                arg = payload.get("arg")
                answer = payload.get("answer")

                content = f"<b>Thought:</b> {explanation}<br/>"
                if tool:
                    content += f"<b>Tool:</b> {tool} | <b>Arg:</b> {arg}"
                if answer:
                    content += f"<b>Answer:</b> {answer}"

                f.write(
                    f'<div style="background-color: #fce4ec; padding: 2px; margin: 0; border-radius: 5px; font-family: monospace;">{content}</div>'
                )
            except json.JSONDecodeError:
                f.write(
                    f'<div style="background-color: #fce4ec; padding: 2px; margin: 0; border-radius: 5px; font-family: monospace;">'
                    f'<pre>{assistant_response}</pre></div>'
                )
            f.write("\n\n---\n\n")

        return assistant_response

    def run(self, user_input: str) -> None:
        with self.report_path.open("a", encoding="utf-8") as f:
            f.write(f"**User Query:** `{user_input}`\n\n")

        turn = 0
        pending_user_msg = user_input
        while turn < self.max_turns:
            turn += 1
            context = self._build_context(pending_user_msg)
            assistant_raw = self._call_llm(context)

            try:
                payload = json.loads(assistant_raw)
            except json.JSONDecodeError:
                console.print("[red]✗ Invalid JSON, retrying.")
                self.history.append({"role": "assistant", "content": assistant_raw})
                self.history.append({"role": "user", "content": "Your last response was not valid JSON. Please correct it."})
                continue

            if "state_analysis" in payload:
                console.print(f"🤔 [bold]State Analysis:[/bold] {payload.get('state_analysis', 'N/A')}")
                console.print(f"📈 [bold]Progress:[/bold] {payload.get('progress_evaluation', 'N/A')}")
                console.print(f"챌 [bold]Challenges:[/bold] {payload.get('challenges', 'N/A')}")
                console.print(f"🚀 [bold]Next Steps:[/bold] {payload.get('next_steps', 'N/A')}")
                console.print(f"🧠 [bold]Reasoning:[/bold] {payload.get('reasoning', 'N/A')}")

            if "tool" in payload and payload["tool"]:
                tool_name = payload["tool"]
                arg = payload.get("arg", "")

                console.print(f"⚙️  Running [bold cyan]{tool_name}[/bold cyan]: [dim]'{arg}'[/dim]")

                tool = TOOLS.get(tool_name)
                if tool is None:
                    self.last_shell_output = f"[error] Unknown tool: {tool_name}"
                else:
                    result = tool(arg)
                    self.last_shell_output = result
                    if result:
                        console.print(f"↪️  [dim]{result}[/dim]")

                self.history.append({"role": "assistant", "content": assistant_raw})
                self.history.append({"role": "assistant", "content": f"[tool_result]\n{self.last_shell_output}"})
                pending_user_msg = "(continue)"
                continue

            if "answer" in payload:
                console.print(f"✅ [bold green]Answer:[/bold green] {payload['answer']}")
                break

            if "state_analysis" in payload:
                pending_user_msg = "(continue)"
                self.history.append({"role": "assistant", "content": assistant_raw})
                self.history.append({"role": "user", "content": "Your plan is noted. Please provide a `tool` to execute next or a final `answer`."})
                continue

            console.print(f"[red]✗ Unrecognized payload: {payload}")
            break
