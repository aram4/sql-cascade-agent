# SQL Cascade Agent

A text-to-SQL agent that routes between a small and large LLM to optimize cost vs. accuracy. Built on a slice of the BIRD benchmark (~200–300 questions), it uses a LangGraph agent with self-correction, executes queries in Modal sandboxes, and measures execution accuracy, cost, and latency.

## How it works

1. **Schema retrieval** — reads table definitions and sample rows from the target SQLite database (cached per session); pure SQLite introspection, no LLM call
2. **SQL generation** (`generate` role) — a strong model writes the SQL with structured output
3. **Validation** (`validate` role) — a cheap/fast model checks the candidate SQL against the schema before anything executes, rejecting hallucinated tables/columns or broken syntax without spending a sandbox round trip
4. **Execution** — runs the generated SQL in an isolated Modal sandbox with timeouts, or locally for interactive use
5. **Self-correction** — on a validator rejection or an execution error, feeds the reason back to the generator and retries (max 2 attempts)
6. **Summarization** (`summarize` role) — a cheap/fast model turns the result set into a plain-English answer (chat mode only; skipped in eval mode unless judged — see LLM-as-judge below)
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
  "summarize": "accounts/fireworks/models/deepseek-v4p1-flash", // cheap/fast, presentation only
  "judge": "accounts/fireworks/models/deepseek-v4p1-flash"      // cheap/fast, eval-only faithfulness check
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

## Router classifier

`src/router_features.py` + `src/train_router.py` are the one component in this project trained from
scratch on its own data, rather than a prompt to someone else's foundation model: a classical classifier
(logistic regression / gradient boosting) predicting, before either `generate` model is called, whether
the cheap one is likely sufficient or the question should escalate to the strong one.

Deliberately not an LLM: cascading only saves money if the routing decision itself is close to free. An
LLM-based router would add a real token/latency cost to every question just to decide not to escalate it,
defeating the point. The classifier runs on 15 hand-built features (question text/keywords, BIRD's own
difficulty label, schema size) in microseconds on CPU, no API call.

**Training data**: run both models over the same question set (`eval_bird.py --model=...`), then label
`needs_big = 1` only where the small model failed *and* the large model succeeded — i.e. escalating
demonstrably would have helped.

**Honest result so far**: of 125 BIRD questions, only 6 are `needs_big = 1`. Too little signal to train a
useful classifier yet — the highest-accuracy model (90.6%) has 0% recall on that class, i.e. it just learned
to always predict "small is fine." Expected failure mode for a rare, imbalanced label, not a bug. Next step:
more labeled data, or reframing as two separate per-model success predictions instead of one rare label.

## LLM-as-judge: summary faithfulness

Execution accuracy (comparing result sets) is deterministic and doesn't need an LLM judge — a row either
matches the gold rows or it doesn't. The one output with no ground truth to diff against is the English
answer from `summarize_result`: free text, no gold sentence to compare it to. That's the one place an LLM
judge actually earns its keep here, so it's the only thing judged.

- **What's checked:** not "is the SQL correct" — a separate `judge` role reads the question, the *actual*
  SQL result rows, and the English answer, and scores whether the answer is faithful to those rows (no
  hallucinated numbers, no dropped/misread data).
- **Opt-in, not default:** pass `--judge-summary` to either eval CLI. It's off by default because it adds
  two more model calls per question (`summarize_result` + `judge`) that eval mode normally skips — only pay
  for it when you want the faithfulness number.
- **Independent from accuracy, deliberately:** a question can fail execution accuracy (wrong SQL) and still
  be marked faithful, if the summary honestly describes the (wrong) rows that SQL happened to return. Those
  are two different failure modes — bad query vs. dishonest narration — and conflating them into one score
  would hide which one you're actually looking at.

```bash
python3 src/eval.py --judge-summary
python3 src/eval_bird.py --limit=50 --judge-summary
```

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

## Data quality (dbt)

`dbt/` runs automated integrity tests against the Postgres-migrated BIRD databases — `not_null`, `unique`,
and foreign-key `relationships` checks declared in `dbt/models/sources.yml` against the existing schemas
as dbt **sources**, not as transformation models. Deliberately not the usual staging/marts dbt pattern:
these schemas need to stay structurally faithful to BIRD's original tables for gold-SQL compatibility, so
dbt's job here is checking data quality, not reshaping data.

That distinction paid off immediately: `dbt test` catches real integrity gaps the migration script's FK
pass already skips loudly rather than silently (see `pg_migrate.py`) — but dbt quantifies the actual scope.
`satscores.cds → schools.cdscode` fails with **211 orphaned rows**, not the single instance spotted by eye
during migration. That's the concrete value of a real testing framework over manual spot-checking.

```bash
cd dbt
dbt test --profiles-dir .
```

Uses the same `agent_readonly` role as the agent itself — a tool that only ever checks data shouldn't need
write access either. `profiles.yml` references connection details via `env_var()`, nothing hardcoded.

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
- **Postgres** *(one database migrated so far — `california_schools`; 10 of 11 BIRD databases still
  SQLite-only)* — `src/pg_migrate.py` moves a BIRD SQLite database into local Postgres (schema translated,
  data copied, FKs applied as a second pass, read-only role granted for the agent); `src/pg_dialect.py`
  adapts SQLite-flavored gold SQL to run against it. `agent.py`'s execution path runs against Postgres
  when `pg_schema` is set on the agent state (see `_execute_sql_postgres`), SQLite/Modal otherwise.
- **dbt** — data-quality tests (not transformation models) against the migrated Postgres schemas; see
  Data quality (dbt) below

## Quick start

```bash
pip install modal langgraph langchain-fireworks python-dotenv opentelemetry-api opentelemetry-sdk "psycopg[binary]" scikit-learn joblib dbt-core dbt-postgres
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
