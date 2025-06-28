from .agent import ReasoningAgent, console


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the reasoning agent in conversational mode."
    )
    parser.add_argument(
        "prompt",
        nargs="*",
        help="Initial user prompt for the agent. If not provided, the agent will start in interactive mode.",
    )
    args = parser.parse_args()

    agent = ReasoningAgent()

    initial_prompt = " ".join(args.prompt) if args.prompt else ""
    if initial_prompt:
        agent.run(initial_prompt)

    console.print("[bold green]Entered conversational mode. Type 'exit' or 'quit' to end.[/bold green]")
    while True:
        try:
            user_prompt = console.input("▶ ")
            if user_prompt.lower() in ["exit", "quit"]:
                break
            if not user_prompt:
                continue
            agent.run(user_prompt)
        except KeyboardInterrupt:
            break

    console.print("[bold red]Exiting agent.[/bold red]")


if __name__ == "__main__":
    main()
