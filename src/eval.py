"""Eval harness: score agent-generated SQL against gold SQL by comparing result sets."""

import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.agent import run_question, judge_summary_call
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


FLOAT_TOLERANCE = 1e-6


def _normalize_value(v):
    """Normalize a single value for comparison."""
    if v is None:
        return None
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _sort_key(v):
    """Sort key that handles mixed types including None."""
    if v is None:
        return (0, "")
    if isinstance(v, (int, float)):
        return (1, v)
    return (2, str(v))


def _normalize_row(row: tuple) -> tuple:
    """Normalize a row: sort values to handle column-order independence."""
    return tuple(sorted((_normalize_value(v) for v in row), key=_sort_key))


def results_match(generated: dict, gold: dict) -> bool:
    """Compare result sets: order-insensitive, column-order-independent,
    NULL-aware, float-tolerant multiset comparison."""
    if not generated or not gold:
        return False

    gen_rows = generated.get("rows", [])
    gold_rows = gold.get("rows", [])

    if len(gen_rows) != len(gold_rows):
        return False

    def _row_sort_key(row):
        return tuple(_sort_key(v) for v in row)

    gen_normalized = sorted((_normalize_row(tuple(r)) for r in gen_rows), key=_row_sort_key)
    gold_normalized = sorted((_normalize_row(tuple(r)) for r in gold_rows), key=_row_sort_key)

    for gen_row, gold_row in zip(gen_normalized, gold_normalized):
        if len(gen_row) != len(gold_row):
            return False
        for gv, ev in zip(gen_row, gold_row):
            if gv is None and ev is None:
                continue
            if gv is None or ev is None:
                return False
            if isinstance(gv, float) and isinstance(ev, float):
                if abs(gv - ev) > FLOAT_TOLERANCE:
                    return False
            elif gv != ev:
                return False

    return True


def run_eval(db_path: str, questions: list[dict], use_sandbox: bool = False, model: str = None, judge_summary: bool = False) -> dict:
    """`judge_summary`, if set, additionally runs summarize_result (normally skipped in
    eval mode) and scores the English answer's faithfulness to the SQL result rows with
    an LLM judge. It's the one output here with no ground-truth string to diff against,
    so it's checked separately from — not folded into — execution accuracy."""
    results = []
    correct = 0
    judged = 0
    faithful_count = 0
    total = len(questions)

    for i, q in enumerate(questions):
        start = time.time()

        kwargs = {"question": q["question"], "db_path": db_path, "use_sandbox": use_sandbox, "summarize": judge_summary}
        if model:
            kwargs["model"] = model

        agent_result = run_question(**kwargs)
        latency = time.time() - start

        gold_result = execute_gold_sql(db_path, q["gold_sql"])
        match = results_match(agent_result.get("result"), gold_result)

        if match:
            correct += 1

        trace = agent_result.get("trace", [])
        faithful = None
        judge_reason = ""
        if judge_summary and not agent_result.get("error") and agent_result.get("result"):
            try:
                faithful, judge_reason, judge_span = judge_summary_call(
                    q["question"], agent_result["result"], agent_result["answer"]
                )
                trace = trace + [judge_span]
                judged += 1
                if faithful:
                    faithful_count += 1
            except Exception as e:
                judge_reason = f"judge pass failed: {e}"
                trace = trace + [{"stage": "judge_summary", "error": str(e)}]

        entry = {
            "question_id": q["question_id"],
            "question": q["question"],
            "gold_sql": q["gold_sql"],
            "generated_sql": agent_result.get("sql", ""),
            "match": match,
            "retries": agent_result.get("retries", 0),
            "error": agent_result.get("error", ""),
            "latency_s": round(latency, 2),
            "trace": trace,
            "faithful": faithful,
            "judge_reason": judge_reason,
        }
        results.append(entry)

        status = "PASS" if match else "FAIL"
        judge_tag = "" if faithful is None else (" | faithful" if faithful else " | UNFAITHFUL")
        print(f"  [{i+1}/{total}] {status} ({latency:.1f}s){judge_tag} — {q['question'][:60]}")

    accuracy = correct / total if total > 0 else 0
    latencies = [r["latency_s"] for r in results]
    avg_latency = sum(latencies) / total if total > 0 else 0
    retry_rate = sum(1 for r in results if r["retries"] > 0) / total if total > 0 else 0

    summary = {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "avg_latency_s": round(avg_latency, 2),
        "p50_latency_s": round(percentile(latencies, 50), 2),
        "p95_latency_s": round(percentile(latencies, 95), 2),
        "p99_latency_s": round(percentile(latencies, 99), 2),
        "retry_rate": round(retry_rate, 4),
        "routes": summarize_routes(results),
        "results": results,
    }
    if judge_summary:
        summary["faithfulness_rate"] = round(faithful_count / judged, 4) if judged else None
        summary["judged"] = judged
    return summary


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (the same method numpy's default and
    Excel's PERCENTILE.INC use). Works fine down to a single data point —
    needed since some routes only have a handful of calls in a small eval run."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize_routes(results: list[dict]) -> list[dict]:
    """Roll up per-stage trace spans across all questions into per-(role, model) stats —
    this is what "measure accuracy/cost per route" is computed from. Tracks every call's
    latency (not just a running sum) specifically so p50/p95/p99 can be computed, not just
    a mean that hides tail latency — see the card_games Q39 outlier (150s+ on one call
    while most take a few seconds) that an average alone would completely hide."""
    buckets: dict[tuple[str, str], dict] = {}

    for r in results:
        for span in r.get("trace", []):
            role = span.get("role", span.get("stage", "unknown"))
            model = span.get("model", "unknown")
            key = (role, model)
            bucket = buckets.setdefault(key, {
                "role": role, "model": model, "calls": 0, "errors": 0,
                "latencies_ms": [], "input_tokens": 0, "output_tokens": 0,
            })
            bucket["calls"] += 1
            if "error" in span:
                bucket["errors"] += 1
                continue
            bucket["latencies_ms"].append(span.get("latency_ms") or 0)
            bucket["input_tokens"] += span.get("input_tokens") or 0
            bucket["output_tokens"] += span.get("output_tokens") or 0

    routes = []
    for (role, model), b in buckets.items():
        calls = b["calls"]
        lat = b["latencies_ms"]
        routes.append({
            "role": role,
            "model": model,
            "calls": calls,
            "errors": b["errors"],
            "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else 0,
            "p50_latency_ms": round(percentile(lat, 50), 1),
            "p95_latency_ms": round(percentile(lat, 95), 1),
            "p99_latency_ms": round(percentile(lat, 99), 1),
            "max_latency_ms": round(max(lat), 1) if lat else 0,
            "input_tokens": b["input_tokens"],
            "output_tokens": b["output_tokens"],
        })
    return sorted(routes, key=lambda x: (x["role"], x["model"]))


def main():
    if not os.path.exists(DB_PATH):
        create()

    use_sandbox = "--sandbox" in sys.argv
    judge_summary = "--judge-summary" in sys.argv
    model = None
    for arg in sys.argv:
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]

    print(f"Running eval: {len(SAMPLE_EVAL)} questions")
    print(f"Database: {DB_PATH}")
    print(f"Sandbox: {use_sandbox}")
    print(f"Judge summary: {judge_summary}")
    if model:
        print(f"Model: {model}")
    print("-" * 60)

    if use_sandbox:
        from src.sandbox import get_runner, shutdown_sandbox
        get_runner().start()

    try:
        summary = run_eval(DB_PATH, SAMPLE_EVAL, use_sandbox=use_sandbox, model=model, judge_summary=judge_summary)
    finally:
        if use_sandbox:
            shutdown_sandbox()

    print("-" * 60)
    print(f"Accuracy: {summary['accuracy']:.1%} ({summary['correct']}/{summary['total']})")
    print(f"Latency: avg={summary['avg_latency_s']}s  p50={summary['p50_latency_s']}s  "
          f"p95={summary['p95_latency_s']}s  p99={summary['p99_latency_s']}s")
    print(f"Retry rate: {summary['retry_rate']:.1%}")
    if judge_summary:
        rate = summary["faithfulness_rate"]
        print(f"Summary faithfulness: {rate:.1%} ({summary['judged']} judged)" if rate is not None else "Summary faithfulness: n/a (nothing judged)")

    print("\nRoutes:")
    for route in summary["routes"]:
        print(
            f"  {route['role']:<10} {route['model']:<55} calls={route['calls']:<4} errors={route['errors']:<3}"
        )
        print(
            f"    latency(ms): avg={route['avg_latency_ms']:<8} p50={route['p50_latency_ms']:<8} "
            f"p95={route['p95_latency_ms']:<8} p99={route['p99_latency_ms']:<8} max={route['max_latency_ms']:<8} "
            f"tokens_in={route['input_tokens']} tokens_out={route['output_tokens']}"
        )

    failures = [r for r in summary["results"] if not r["match"]]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print(f"  Q{f['question_id']}: {f['question'][:50]}")
            print(f"    Gold: {f['gold_sql'][:80]}")
            print(f"    Generated: {f['generated_sql'][:80]}")
            if f["error"]:
                print(f"    Error: {f['error'][:80]}")

    if judge_summary:
        unfaithful = [r for r in summary["results"] if r["faithful"] is False]
        if unfaithful:
            print(f"\nUnfaithful summaries ({len(unfaithful)}):")
            for u in unfaithful:
                print(f"  Q{u['question_id']}: {u['question'][:50]}")
                print(f"    Reason: {u['judge_reason'][:100]}")

    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "eval_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
