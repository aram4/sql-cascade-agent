# SQL Cascade Agent

A text-to-SQL agent that routes between a small and large LLM to optimize cost vs. accuracy. Built on a slice of the BIRD benchmark (~200–300 questions), it uses a LangGraph agent with self-correction, executes queries in Modal sandboxes, and measures execution accuracy, cost, and latency.

## How it works

1. **Schema retrieval** — reads table definitions and sample rows from the target SQLite database (cached per session); pure SQLite introspection, no LLM call
2. **SQL generation** (`generate` role) — a strong model writes the SQL with structured output
3. **Validation** (`validate` role) — a cheap/fast model checks the candidate SQL against the schema before anything executes, rejecting hallucinated tables/columns or broken syntax without spending a sandbox round trip
4. **Execution** — runs the generated SQL in an isolated Modal sandbox with timeouts, or locally for interactive use
5. **Self-correction** — on a validator rejection or an execution error, feeds the reason back to the generator and retries (max 2 attempts)
6. **Summarization** (`summarize` role) — a cheap/fast model turns the result set into a plain-English answer (chat mode only; skipped in eval mode)
7. **Cascade routing** *(planned)* — tries the small model first within the `generate` role; escalates to the large model on execution error or empty result

## Model routing

Each pipeline stage above is a *role*, not a hardcoded model. `src/model_router.py` resolves the model for
a role at call time, in this order: an env var (`MODEL_ROUTE_GENERATE`, `MODEL_ROUTE_VALIDATE`,
`MODEL_ROUTE_SUMMARIZE`) → `model_routes.json` → a built-in default. Every LLM call returns a
`ModelResolution(role, model, source)`, which gets written onto a trace span alongside latency and
token usage — so the resolved model for every call is visible after the fact, not just configured up front.

```json
// model_routes.json
{
  "generate": "accounts/fireworks/models/qwen3p8-max",          // strong model, writes the SQL
  "validate": "accounts/fireworks/models/deepseek-v4p1-flash",  // cheap/fast, schema-grounded lint
  "summarize": "accounts/fireworks/models/deepseek-v4p1-flash"  // cheap/fast, presentation only
}
```

This is a config *cascade*, not an error-handling fallback: if a role is misconfigured (unknown role, blank
entry) resolution raises immediately. If a configured model's API call fails at runtime, the error
propagates — nothing here catches a failed call and silently retries with a different model. The one
explicit override is `run_question(..., model=...)` / `--model=` on the eval CLIs, which swaps the
`generate` role's model for that call only (the lever for the "small model vs. large model" comparison in
the build plan); it's recorded with `source="override"` on the span like everything else.

Every eval run (`src/eval.py`, `src/eval_bird.py`) rolls the per-question trace spans up into a `routes`
summary — calls, errors, avg latency, and input/output tokens per `(role, model)` pair — so accuracy and
cost can be compared per route, not just per overall run. Example from the sample eval:

```
Routes:
  generate   accounts/fireworks/models/qwen3p8-max          calls=15  avg_latency=2820ms  tokens_in=16534 tokens_out=2430
  validate   accounts/fireworks/models/deepseek-v4p1-flash   calls=15  avg_latency=1381ms  tokens_in=15443 tokens_out=1385
```

Token counts are from each provider's `usage_metadata`; converting to dollars just needs multiplying by your
current Fireworks per-token rate for that model, which isn't hardcoded here since it changes over time.

## Two modes, one graph

The agent runs the same LangGraph pipeline in two modes:

- **Chat mode** (`python3 run.py`) — interactive REPL with conversation history for follow-up questions. Returns both raw SQL results and a plain English answer.
- **Eval mode** (`summarize=False`) — stateless, no conversation history, no English summarization. Returns raw SQL and result sets for reproducible scoring against gold answers.

The English answer is a presentation layer only. Eval accuracy is measured by comparing result sets (rows returned), never prose.

## Eval methodology

- **Scoring**: order-insensitive, column-order-independent multiset comparison. NULLs must match exactly. Floats compared within 1e-6 tolerance. Strings normalized to lowercase/trimmed.
- **Temperature**: 0 (deterministic). Single pass per question, no k-run majority vote.
- **Dataset**: targeting a 250-question stratified slice of BIRD dev set across difficulty levels. Current sample eval (n=15) is for harness validation only — not for model comparison claims.

## Stack

- **LangGraph** — agent orchestration with typed state and bounded retry loops
- **LangChain** — unified LLM interface (structured output, message types, provider abstraction)
- **Fireworks AI** — LLM inference, routed per role (see Model routing below)
- **Modal** — sandboxed query execution with timeouts and parallel eval harness
- **BIRD benchmark** — evaluation dataset (SQLite databases + natural language questions)

## Quick start

```bash
pip install modal langgraph langchain-fireworks python-dotenv
python3 -m modal setup

# Add your Fireworks API key
echo "FIREWORKS_API_KEY=your_key" > .env

# Interactive chat mode
python3 run.py

# Run a single question with Modal sandbox (eval mode)
python3 -c "
from src.agent import run_question
result = run_question('How many employees?', 'data/sample/sample.sqlite', use_sandbox=True, summarize=False)
print(result['sql'], result['result'])
"
```
