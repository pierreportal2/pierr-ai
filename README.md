# Ultimate AI Agent

A minimal yet extensible agent framework that demonstrates iterative reasoning using OpenAI's chat models. The project has been split into a small package for easier extension.

## Installation

```bash
pip install openai python-dotenv rich tiktoken duckduckgo_search beautifulsoup4 requests
```

Create a `.env` file containing your `OPENAI_API_KEY` and optionally `OPENAI_MODEL`.

## Usage

Run the agent with an initial prompt:

```bash
python -m reasoning_agent "What's inside the current directory?"
```

If no prompt is supplied, the agent starts in interactive mode.

## Project Architecture

```
reasoning_agent/
├── __init__.py          # Package exports
├── __main__.py          # CLI entry point (run with `python -m reasoning_agent`)
├── agent.py             # `ReasoningAgent` class implementation
└── tools.py             # Tool definitions used by the agent
main.py                  # Convenience entry point
README.md                # Project documentation
```

The `ReasoningAgent` keeps a reasoning history, uses tools from `tools.py`,
and records a detailed report of each conversation in `query_report.md`.
