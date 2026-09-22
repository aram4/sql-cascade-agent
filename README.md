# SQL Cascade Agent

A text-to-SQL agent that routes between a small and large LLM to optimize cost vs. accuracy. Built on a slice of the BIRD benchmark (~200–300 questions), it uses a LangGraph agent with self-correction, executes queries in Modal sandboxes, and measures execution accuracy, cost, and latency.

## How it works

1. **Schema retrieval** — reads table definitions and sample rows from the target SQLite database
2. **SQL generation** — prompts a Fireworks-hosted model (DeepSeek v4 Flash or Pro) with structured output
3. **Execution** — runs the generated SQL and checks for errors
4. **Self-correction** — on failure, retries with the error message (max 2 attempts)
5. **Cascade routing** — tries the small model first; escalates to the large model on execution error or empty result

## Stack

- **LangGraph** — agent orchestration with typed state and bounded retry loops
- **Fireworks AI** — LLM inference (DeepSeek v4 Flash as small, DeepSeek v4 Pro as large)
- **Modal** — sandboxed query execution and parallel eval harness
- **BIRD benchmark** — evaluation dataset (SQLite databases + natural language questions)

## Quick start

```bash
pip install modal langgraph langchain-fireworks python-dotenv
python3 -m modal setup

# Add your Fireworks API key
echo "FIREWORKS_API_KEY=your_key" > .env

# Run a sample question
python3 run.py
```
