"""Run a single question through the text-to-SQL agent."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from setup_sample_db import create, DB_PATH
from src.agent import run_question


def main():
    if not os.path.exists(DB_PATH):
        create()

    question = "What are the names of employees in the Engineering department who earn more than 100000?"

    print(f"Question: {question}")
    print(f"Database: {DB_PATH}")
    print("-" * 60)

    result = run_question(question=question, db_path=DB_PATH)

    print(f"Generated SQL: {result['sql']}")
    print(f"Retries: {result['retries']}")
    if result["error"]:
        print(f"Error: {result['error']}")
    else:
        print(f"Result: {json.dumps(result['result'], indent=2)}")


if __name__ == "__main__":
    main()
