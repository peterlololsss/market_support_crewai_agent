from __future__ import annotations


def run() -> None:
    """Project CLI entrypoint."""
    print(
        "Run the FastAPI service with: "
        "uvicorn market_support_crewai_agent.server.main:app --reload"
    )


if __name__ == "__main__":
    run()
