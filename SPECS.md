# SQL Cascade Agent — Specs

## Overview

A text-to-SQL agent built on a slice of the BIRD benchmark (about 200–300 questions, not the full set). It routes between a small and a large open model on Fireworks, runs queries in Modal sandboxes, and reports accuracy, cost, and latency.

## Build Plan

| Day | Build | Interview Payoff |
|-----|-------|-----------------|
| 1 | LangGraph agent: schema retrieval → generate SQL → execute → self-correct (max 2 retries). Fireworks with structured output. Get one question working end to end. | Graph vs ReAct, bounded loops, typed state |
| 2 | Modal sandbox for query execution with timeouts. Eval harness fanned out on Modal over the BIRD slice, scoring execution accuracy. | Isolation of LLM-generated code; why evals come first |
| 3 | Run small model vs large model. Log cost, p50/p95 latency, retry rate. Hand-label about 30 failures into a taxonomy (bad join, hallucinated column, wrong filter). | Failure analysis, reading traces |
| 4 | Cascade router: small model first, escalate on execution error or empty result. Plot the cost/quality Pareto curve. | The headline: "X% cheaper at Y% of frontier accuracy" |
| 5 | README with architecture diagram, results table, chart, and "what didn't work / what's next" (fine-tuning on successful trajectories, self-hosted vLLM). Post it on achintya.dev. | A polished artifact to screen-share |

## Stretch Goal

If you have a spare half-day: serve the small model yourself with vLLM on Modal and compare cold start and latency against Fireworks. Worth doing if Modal or Baseten are live targets, skippable otherwise.
