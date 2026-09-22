# SQL Cascade Agent

A text-to-SQL agent that routes between a small and large LLM to optimize cost vs. accuracy. Built on a slice of the BIRD benchmark (~200–300 questions), it uses a LangGraph agent with self-correction, executes queries in Modal sandboxes, and measures execution accuracy, cost, and latency.

## How it works

1. **Schema retrieval** — reads table definitions and sample rows from the target SQLite database (cached per session)
2. **SQL generation** — prompts a Fireworks-hosted model (DeepSeek v4 Flash or Pro) with structured output
3. **Execution** — runs the generated SQL in an isolated Modal sandbox with timeouts, or locally for interactive use
4. **Self-correction** — on failure, feeds the error back to the LLM and retries (max 2 attempts)
5. **Cascade routing** *(planned)* — tries the small model first; escalates to the large model on execution error or empty result

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
- **Fireworks AI** — LLM inference (DeepSeek v4 Flash as small, DeepSeek v4 Pro as large)
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
