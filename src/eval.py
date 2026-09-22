"""Eval harness: score agent-generated SQL against gold SQL by comparing result sets."""

import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.agent import run_question
from src.eval_questions import SAMPLE_EVAL
from setup_sample_db import create, DB_PATH


def execute_gold_sql(db_path: str, sql: str) -> dict:
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()
        return {"columns": columns, "rows": rows, "error": ""}
    except Exception as e:
        return {"columns": [], "rows": [], "error": str(e)}


def results_match(generated: dict, gold: dict) -> bool:
    if not generated or not gold:
        return False
    gen_rows = set(tuple(r) for r in generated.get("rows", []))
    gold_rows = set(tuple(r) for r in gold.get("rows", []))
    return gen_rows == gold_rows


def run_eval(db_path: str, questions: list[dict], use_sandbox: bool = False, model: str = None) -> dict:
    results = []
    correct = 0
    total = len(questions)

    for i, q in enumerate(questions):
        start = time.time()

        kwargs = {"question": q["question"], "db_path": db_path, "use_sandbox": use_sandbox, "summarize": False}
        if model:
            kwargs["model"] = model

        agent_result = run_question(**kwargs)
        latency = time.time() - start

        gold_result = execute_gold_sql(db_path, q["gold_sql"])
        match = results_match(agent_result.get("result"), gold_result)

        if match:
            correct += 1

        entry = {
            "question_id": q["question_id"],
            "question": q["question"],
            "gold_sql": q["gold_sql"],
            "generated_sql": agent_result.get("sql", ""),
            "match": match,
            "retries": agent_result.get("retries", 0),
            "error": agent_result.get("error", ""),
            "latency_s": round(latency, 2),
        }
        results.append(entry)

        status = "PASS" if match else "FAIL"
        print(f"  [{i+1}/{total}] {status} ({latency:.1f}s) — {q['question'][:60]}")

    accuracy = correct / total if total > 0 else 0
    avg_latency = sum(r["latency_s"] for r in results) / total if total > 0 else 0
    retry_rate = sum(1 for r in results if r["retries"] > 0) / total if total > 0 else 0

    summary = {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "avg_latency_s": round(avg_latency, 2),
        "retry_rate": round(retry_rate, 4),
        "results": results,
    }
    return summary


def main():
    if not os.path.exists(DB_PATH):
        create()

    use_sandbox = "--sandbox" in sys.argv
    model = None
    for arg in sys.argv:
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]

    print(f"Running eval: {len(SAMPLE_EVAL)} questions")
    print(f"Database: {DB_PATH}")
    print(f"Sandbox: {use_sandbox}")
    if model:
        print(f"Model: {model}")
    print("-" * 60)

    if use_sandbox:
        from src.sandbox import get_runner, shutdown_sandbox
        get_runner().start()

    try:
        summary = run_eval(DB_PATH, SAMPLE_EVAL, use_sandbox=use_sandbox, model=model)
    finally:
        if use_sandbox:
            shutdown_sandbox()

    print("-" * 60)
    print(f"Accuracy: {summary['accuracy']:.1%} ({summary['correct']}/{summary['total']})")
    print(f"Avg latency: {summary['avg_latency_s']}s")
    print(f"Retry rate: {summary['retry_rate']:.1%}")

    failures = [r for r in summary["results"] if not r["match"]]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print(f"  Q{f['question_id']}: {f['question'][:50]}")
            print(f"    Gold: {f['gold_sql'][:80]}")
            print(f"    Generated: {f['generated_sql'][:80]}")
            if f["error"]:
                print(f"    Error: {f['error'][:80]}")

    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "eval_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
