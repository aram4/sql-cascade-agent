"""BIRD benchmark eval harness: runs 250 questions against real databases on Modal."""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.agent import get_llm, SQLOutput, MAX_RETRIES
from src.sandbox import get_runner, shutdown_sandbox
from src.eval import results_match

BIRD_SLICE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "bird", "bird_slice.json")


def load_bird_slice() -> list[dict]:
    with open(BIRD_SLICE_PATH) as f:
        return json.load(f)


def generate_sql_for_question(question: str, schema: str, model: str = None) -> dict:
    """Generate SQL using Fireworks — runs locally, not on Modal."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_llm(model) if model else get_llm()

    messages = [
        SystemMessage(content=(
            "You are a SQL expert. Given a database schema and a question, "
            "write a SQLite query that answers the question.\n"
            "Rules:\n"
            "- Use only tables and columns from the schema\n"
            "- Use SQLite syntax\n"
            "- Return only the data requested, no extra columns\n"
            "- Use JOINs when data spans multiple tables\n"
            "- Use COLLATE NOCASE or LOWER() for string comparisons to handle case differences\n"
        )),
        HumanMessage(content=f"Schema:\n{schema}\n\nQuestion: {question}"),
    ]

    try:
        structured_llm = llm.with_structured_output(SQLOutput)
        response = structured_llm.invoke(messages)
        return {"sql": response.sql, "error": ""}
    except Exception as e:
        return {"sql": "", "error": str(e)}


def run_with_retries(question: str, schema: str, db_id: str, runner, model: str = None) -> dict:
    """Generate SQL and execute with retry loop."""
    sql = ""
    error = ""
    result = None
    retries = 0

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            retries = attempt

        gen = generate_sql_for_question(
            question if attempt == 0 else f"{question}\n\nPrevious SQL failed:\n{sql}\nError: {error}\nFix the query.",
            schema,
            model,
        )
        sql = gen["sql"]
        if gen["error"]:
            error = gen["error"]
            continue

        try:
            result = runner.run_on_volume(db_id, sql)
        except Exception as e:
            error = str(e)
            result = {"columns": [], "rows": [], "error": error}
            continue
        error = result.get("error", "")
        if not error:
            break

    return {"sql": sql, "result": result, "error": error, "retries": retries}


def run_bird_eval(questions: list[dict], model: str = None) -> dict:
    runner = get_runner()
    results = []
    correct = 0
    total = len(questions)

    schema_cache = {}

    for i, q in enumerate(questions):
        start = time.time()
        db_id = q["db_id"]

        if db_id not in schema_cache:
            schema_cache[db_id] = runner.get_schema(db_id)

        agent = run_with_retries(q["question"], schema_cache[db_id], db_id, runner, model)
        try:
            gold_result = runner.run_on_volume(db_id, q["SQL"])
        except Exception as e:
            gold_result = {"columns": [], "rows": [], "error": str(e)}
        latency = time.time() - start

        match = results_match(agent["result"], gold_result)
        if match:
            correct += 1

        entry = {
            "question_id": q.get("question_id", i),
            "db_id": db_id,
            "question": q["question"],
            "gold_sql": q["SQL"],
            "generated_sql": agent["sql"],
            "match": match,
            "retries": agent["retries"],
            "error": agent["error"],
            "latency_s": round(latency, 2),
        }
        results.append(entry)

        status = "PASS" if match else "FAIL"
        print(f"  [{i+1}/{total}] {status} ({latency:.1f}s) {db_id}: {q['question'][:50]}")

    accuracy = correct / total if total > 0 else 0
    avg_latency = sum(r["latency_s"] for r in results) / total if total > 0 else 0
    retry_rate = sum(1 for r in results if r["retries"] > 0) / total if total > 0 else 0

    return {
        "model": model or "default",
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "avg_latency_s": round(avg_latency, 2),
        "retry_rate": round(retry_rate, 4),
        "results": results,
    }


def main():
    model = None
    limit = None
    for arg in sys.argv[1:]:
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]
        elif arg.startswith("--limit="):
            limit = int(arg.split("=", 1)[1])

    questions = load_bird_slice()
    if limit:
        questions = questions[:limit]

    print(f"Running BIRD eval: {len(questions)} questions")
    if model:
        print(f"Model: {model}")
    dbs = set(q["db_id"] for q in questions)
    print(f"Databases: {len(dbs)} ({', '.join(sorted(dbs))})")
    print("-" * 60)

    try:
        summary = run_bird_eval(questions, model=model)
    finally:
        shutdown_sandbox()

    print("-" * 60)
    print(f"Accuracy: {summary['accuracy']:.1%} ({summary['correct']}/{summary['total']})")
    print(f"Avg latency: {summary['avg_latency_s']}s")
    print(f"Retry rate: {summary['retry_rate']:.1%}")

    failures = [r for r in summary["results"] if not r["match"]]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures[:10]:
            print(f"  [{f['db_id']}] {f['question'][:60]}")
            print(f"    Gold:      {f['gold_sql'][:80]}")
            print(f"    Generated: {f['generated_sql'][:80]}")
            if f["error"]:
                print(f"    Error: {f['error'][:80]}")
        if len(failures) > 10:
            print(f"  ... and {len(failures) - 10} more")

    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "bird_eval_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
