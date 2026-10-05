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

## Tracing

Every stage, plus a `run_question` root span wrapping all of them, emits a real OpenTelemetry span via
`src/tracing.py` — trace/span IDs, parent/child nesting, and attributes (role, model, source, tokens,
retries). Spans print as JSON to the console, no collector needed. This is additive to the `routes` summary
above, not a replacement — that reads the lightweight `trace` dict on `AgentState` for aggregate
cost/accuracy; OTel is for inspecting one request's actual call shape.

**Why it's worth having:**
- Fireworks and Modal each only see their own slice — Fireworks doesn't know a `generate` call was followed
  by a `validate` rejection and a retry; Modal doesn't know which Fireworks call produced the SQL it ran. A
  shared `trace_id` across all three systems is the only way to see one causal chain for a single question.
- A retry shows up as a second `generate_sql` span, sibling to the first, under the same trace — not
  indistinguishable noise in a flat log.
- The `role`/`model` attributes are recorded off the actual call, not off config — so a routing bug (e.g.
  `summarize_result` silently resolving the `generate` role) would show up as a mismatched `model` under the
  wrong stage, which a config-only check can't catch.

**Upgrading later:** `ConsoleSpanExporter` is live-only — close the terminal, traces are gone. Since the
instrumentation only calls vendor-neutral OTel API (`tracer.start_as_current_span`, `span.set_attribute`),
getting persistence and a real UI is an exporter swap in `init_tracing()`, not a re-instrumentation:

```python
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import BatchSpanProcessor

provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
    endpoint=os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"],
    headers={"x-api-key": os.environ["OTEL_API_KEY"]},
)))
```

**Honeycomb** ingests OTLP natively — a good fit for the generic tracing view above. **Arize** is built for
LLM pipelines specifically (prompts/tokens as first-class fields via OpenInference on top of OTLP) — a
closer match for analyzing the generate/validate/summarize calls themselves. Either way, `agent.py` and
`eval_bird.py` don't change.

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
- **OpenTelemetry** — per-stage tracing (console exporter today; a vendor-neutral API means dropping in a
  real observability backend like Honeycomb or Arize later is an exporter swap, not a rewrite)

## Quick start

```bash
pip install modal langgraph langchain-fireworks python-dotenv opentelemetry-api opentelemetry-sdk
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
