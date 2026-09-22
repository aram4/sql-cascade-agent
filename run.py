"""Interactive text-to-SQL agent — ask follow-up questions in terminal."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from setup_sample_db import create, DB_PATH
from src.agent import run_question


def main():
    if not os.path.exists(DB_PATH):
        create()

    print(f"Database: {DB_PATH}")
    print("Type your question (or 'quit' to exit)")
    print("-" * 60)

    history = []

    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        result = run_question(question=question, db_path=DB_PATH, history=history)

        print(f"SQL: {result['sql']}")
        if result["error"]:
            print(f"Error: {result['error']}")
        print(f"Answer: {result['answer']}")

        history.append({
            "question": question,
            "sql": result["sql"],
            "answer": result["answer"],
        })


if __name__ == "__main__":
    main()
