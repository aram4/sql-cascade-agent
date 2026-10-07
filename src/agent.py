"""LangGraph text-to-SQL agent with self-correction loop."""

from __future__ import annotations

import sqlite3
import time
from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.model_router import get_llm_for_role, make_span
from src.tracing import init_tracing, get_tracer

load_dotenv()
init_tracing()
tracer = get_tracer()

MAX_RETRIES = 2

_schema_cache: dict[tuple[str, bool], str] = {}


class SQLOutput(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning about the query")
    sql: str = Field(description="The SQL query to execute")


class ValidationOutput(BaseModel):
    valid: bool = Field(description="Whether the SQL only uses tables/columns that exist in the schema and is syntactically sound")
    reason: str = Field(description="Why it's invalid; empty string if valid")


class JudgeOutput(BaseModel):
    faithful: bool = Field(description="Whether the English answer accurately reflects the SQL result rows, with no hallucinated or mischaracterized values")
    reason: str = Field(description="Why it's unfaithful; empty string if faithful")


class AgentState(BaseModel):
    question: str
    db_path: str
    schema: str = ""
    sql: str = ""
    result: Optional[Any] = None
    error: str = ""
    retries: int = 0
    done: bool = False
    answer: str = ""
    history: List[dict] = []
    use_sandbox: bool = False
    summarize: bool = True
    generate_model_override: str = ""
    trace: List[dict] = []
    pg_schema: str = ""  # set -> execute against Postgres instead of SQLite/Modal (schema name to run under)


ROW_PREVIEW_LIMIT = 50


def _format_rows(columns: list, rows: list) -> str:
    """Caps how many rows get stuffed into a summarize/judge prompt. A result set with
    hundreds of rows was overwhelming those roles' output budget before they ever got
    to their final answer — capping input size here fixes that at the source, instead
    of just raising the output ceiling and hoping the next large result set fits."""
    preview = rows[:ROW_PREVIEW_LIMIT]
    formatted = "\n".join(str(dict(zip(columns, row))) for row in preview)
    if len(rows) > ROW_PREVIEW_LIMIT:
        formatted += f"\n... and {len(rows) - ROW_PREVIEW_LIMIT} more rows (total {len(rows)} rows)"
    return formatted


def _usage_from_raw(raw) -> Optional[dict]:
    usage = getattr(raw, "usage_metadata", None)
    if not usage:
        return None
    return {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens")}


def retrieve_schema(state: AgentState) -> dict:
    # Postgres lowercases every identifier at migration time (see pg_migrate.py) — the
    # schema description shown to the generator has to match exactly, or it quotes a
    # mixed-case name copied straight from here and Postgres can't find the column.
    lowercase = bool(state.pg_schema)
    cache_key = (state.db_path, lowercase)

    with tracer.start_as_current_span("retrieve_schema") as span:
        span.set_attribute("db_path", state.db_path)
        cache_hit = cache_key in _schema_cache
        span.set_attribute("cache_hit", cache_hit)
        if cache_hit:
            return {"schema": _schema_cache[cache_key]}

        def ident(name: str) -> str:
            return name.lower() if lowercase else name

        conn = sqlite3.connect(state.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        span.set_attribute("table_count", len(tables))

        schema_parts = []
        for table in tables:
            cursor.execute(f"PRAGMA table_info('{table}')")
            columns = cursor.fetchall()
            col_defs = [f"  {ident(c[1])} {c[2]}{'  PRIMARY KEY' if c[5] else ''}" for c in columns]

            cursor.execute(f"PRAGMA foreign_key_list('{table}')")
            fks = cursor.fetchall()
            fk_defs = [f"  FOREIGN KEY ({ident(fk[3])}) REFERENCES {ident(fk[2])}({ident(fk[4])})" for fk in fks]

            cursor.execute(f"SELECT * FROM '{table}' LIMIT 3")
            sample_rows = cursor.fetchall()
            col_names = [ident(desc[0]) for desc in cursor.description]
            sample_str = "\n".join(
                "  " + str(dict(zip(col_names, row))) for row in sample_rows
            )

            schema_parts.append(
                f"CREATE TABLE {ident(table)} (\n"
                + ",\n".join(col_defs)
                + ("\n" + ",\n".join(fk_defs) if fk_defs else "")
                + "\n);\n"
                + f"-- Sample rows:\n{sample_str}"
            )

        conn.close()
        schema = "\n\n".join(schema_parts)
        _schema_cache[cache_key] = schema
        return {"schema": schema}


GENERATE_SYSTEM_PROMPT_SQLITE = (
    "You are a SQL expert. Given a database schema and a question, "
    "write a SQLite query that answers the question.\n"
    "Rules:\n"
    "- Use only tables and columns from the schema\n"
    "- Use SQLite syntax\n"
    "- Return only the data requested, no extra columns\n"
    "- Use JOINs when data spans multiple tables\n"
    "- Use COLLATE NOCASE or LOWER() for string comparisons to handle case differences\n"
)

GENERATE_SYSTEM_PROMPT_POSTGRES = (
    "You are a SQL expert. Given a database schema and a question, "
    "write a PostgreSQL query that answers the question.\n"
    "Rules:\n"
    "- Use only tables and columns from the schema\n"
    "- Use PostgreSQL syntax: double-quote identifiers that need quoting (never backticks), "
    "and use LIMIT n OFFSET m (never LIMIT m,n)\n"
    "- Table and column names were lowercased when this database was loaded — write "
    "identifiers in lowercase, whether quoted or not\n"
    "- Return only the data requested, no extra columns\n"
    "- Use JOINs when data spans multiple tables\n"
    "- Use LOWER() for string comparisons to handle case differences\n"
)

VALIDATE_SYSTEM_PROMPT = (
    "You check candidate SQLite queries before they run. Reject a query if it "
    "references a table or column that isn't in the schema, or if it's syntactically "
    "broken. Don't reject a query just because you'd have written it differently."
)

JUDGE_SYSTEM_PROMPT = (
    "You check whether an English answer accurately reflects a SQL query's result rows. "
    "Mark it unfaithful if it states a number, name, or fact not supported by the rows, "
    "drops or misreads a row, or asserts something the rows don't show. Don't mark it "
    "unfaithful just for phrasing, rounding, or omitting rows the question didn't ask for."
)


def generate_sql_call(schema: str, question: str, extra_context: str = "", override: str = "", retries: int = 0, dialect: str = "sqlite") -> tuple[str, dict]:
    """Role: generate. Shared by the LangGraph node and the BIRD eval harness."""
    llm, resolution = get_llm_for_role("generate", override=override or None)
    start = time.perf_counter()

    system_prompt = GENERATE_SYSTEM_PROMPT_POSTGRES if dialect == "postgres" else GENERATE_SYSTEM_PROMPT_SQLITE

    with tracer.start_as_current_span("generate_sql") as otel_span:
        otel_span.set_attribute("role", resolution.role)
        otel_span.set_attribute("model", resolution.model)
        otel_span.set_attribute("model_source", resolution.source)
        otel_span.set_attribute("retries", retries)
        otel_span.set_attribute("dialect", dialect)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Schema:\n{schema}\n\nQuestion: {question}{extra_context}"),
        ]

        structured_llm = llm.with_structured_output(SQLOutput, include_raw=True)
        response = structured_llm.invoke(messages)
        parsed = response["parsed"]
        if parsed is None:
            otel_span.set_attribute("error", True)
            raise RuntimeError(
                f"generate stage ({resolution.model}) returned unparseable output: {response.get('parsing_error')}"
            )

        usage = _usage_from_raw(response["raw"])
        if usage:
            otel_span.set_attribute("input_tokens", usage.get("input_tokens") or 0)
            otel_span.set_attribute("output_tokens", usage.get("output_tokens") or 0)

    span = make_span(resolution, "generate_sql", start, usage, retries=retries)
    return parsed.sql, span


def validate_sql_call(schema: str, question: str, sql: str, retries: int = 0) -> tuple[bool, str, dict]:
    """Role: validate. Shared by the LangGraph node and the BIRD eval harness."""
    llm, resolution = get_llm_for_role("validate")
    start = time.perf_counter()

    with tracer.start_as_current_span("validate_sql") as otel_span:
        otel_span.set_attribute("role", resolution.role)
        otel_span.set_attribute("model", resolution.model)
        otel_span.set_attribute("model_source", resolution.source)
        otel_span.set_attribute("retries", retries)

        messages = [
            SystemMessage(content=VALIDATE_SYSTEM_PROMPT),
            HumanMessage(content=f"Schema:\n{schema}\n\nQuestion: {question}\n\nCandidate SQL:\n{sql}"),
        ]

        structured_llm = llm.with_structured_output(ValidationOutput, include_raw=True)
        response = structured_llm.invoke(messages)
        parsed = response["parsed"]
        if parsed is None:
            otel_span.set_attribute("error", True)
            raise RuntimeError(
                f"validate stage ({resolution.model}) returned unparseable output: {response.get('parsing_error')}"
            )

        usage = _usage_from_raw(response["raw"])
        if usage:
            otel_span.set_attribute("input_tokens", usage.get("input_tokens") or 0)
            otel_span.set_attribute("output_tokens", usage.get("output_tokens") or 0)
        otel_span.set_attribute("valid", parsed.valid)

    span = make_span(resolution, "validate_sql", start, usage, retries=retries)
    return parsed.valid, parsed.reason, span


def judge_summary_call(question: str, result: dict, answer: str) -> tuple[bool, str, dict]:
    """Role: judge. Eval-only — scores whether the English answer is faithful to the
    actual SQL result rows. Not part of the interactive pipeline: it's a second
    opinion on the one non-deterministic, un-gradable-by-diff output this agent
    produces, not something a chat-mode user needs to pay latency for."""
    llm, resolution = get_llm_for_role("judge")
    start = time.perf_counter()

    formatted = _format_rows(result["columns"], result["rows"])

    with tracer.start_as_current_span("judge_summary") as otel_span:
        otel_span.set_attribute("role", resolution.role)
        otel_span.set_attribute("model", resolution.model)
        otel_span.set_attribute("model_source", resolution.source)

        messages = [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Question: {question}\n\n"
                f"SQL Result:\n{formatted}\n\n"
                f"English answer to check:\n{answer}"
            )),
        ]

        structured_llm = llm.with_structured_output(JudgeOutput, include_raw=True)
        response = structured_llm.invoke(messages)
        parsed = response["parsed"]
        if parsed is None:
            otel_span.set_attribute("error", True)
            raise RuntimeError(
                f"judge stage ({resolution.model}) returned unparseable output: {response.get('parsing_error')}"
            )

        usage = _usage_from_raw(response["raw"])
        if usage:
            otel_span.set_attribute("input_tokens", usage.get("input_tokens") or 0)
            otel_span.set_attribute("output_tokens", usage.get("output_tokens") or 0)
        otel_span.set_attribute("faithful", parsed.faithful)

    span = make_span(resolution, "judge_summary", start, usage)
    return parsed.faithful, parsed.reason, span


def generate_sql(state: AgentState) -> dict:
    error_context = ""
    if state.error:
        error_context = (
            f"\n\nYour previous SQL query failed:\n"
            f"Query: {state.sql}\n"
            f"Error: {state.error}\n"
            f"Fix the query based on this error."
        )

    history_context = ""
    if state.history:
        turns = "\n".join(
            f"Q: {h['question']}\nSQL: {h['sql']}\nAnswer: {h['answer']}"
            for h in state.history
        )
        history_context = f"\n\nConversation so far:\n{turns}\n\nUse this context to resolve references like 'they', 'those', 'that department', etc."

    sql, span = generate_sql_call(
        schema=state.schema,
        question=state.question,
        extra_context=f"{history_context}{error_context}",
        override=state.generate_model_override,
        retries=state.retries,
        dialect="postgres" if state.pg_schema else "sqlite",
    )
    return {"sql": sql, "trace": state.trace + [span]}


def validate_sql(state: AgentState) -> dict:
    valid, reason, span = validate_sql_call(state.schema, state.question, state.sql, retries=state.retries)
    trace = state.trace + [span]
    if valid:
        return {"error": "", "trace": trace}
    return {"error": f"SQL rejected by validator: {reason}", "trace": trace}


def route_after_validate(state: AgentState) -> str:
    return "check_result" if state.error else "execute_sql"


def execute_sql(state: AgentState) -> dict:
    with tracer.start_as_current_span("execute_sql") as span:
        span.set_attribute("use_sandbox", state.use_sandbox)
        span.set_attribute("pg_schema", state.pg_schema)
        span.set_attribute("sql", state.sql[:500])
        if state.pg_schema:
            result = _execute_sql_postgres(state)
        elif state.use_sandbox:
            result = _execute_sql_modal(state)
        else:
            result = _execute_sql_local(state)
        span.set_attribute("error", bool(result.get("error")))
        return result


def _execute_sql_postgres(state: AgentState) -> dict:
    import os
    import psycopg
    from psycopg import sql as pg_sql
    from src.pg_dialect import sqlite_to_postgres

    try:
        translated_sql = sqlite_to_postgres(state.sql)
        with psycopg.connect(os.environ["PG_AGENT_DSN"]) as conn:
            with conn.cursor() as cur:
                cur.execute(pg_sql.SQL("SET search_path TO {}").format(pg_sql.Identifier(state.pg_schema.lower())))
                cur.execute(translated_sql)
                columns = [desc[0] for desc in cur.description] if cur.description else []
                rows = cur.fetchall()
        return {"result": {"columns": columns, "rows": rows}, "error": ""}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _execute_sql_local(state: AgentState) -> dict:
    try:
        conn = sqlite3.connect(state.db_path)
        cursor = conn.cursor()
        cursor.execute(state.sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()
        return {"result": {"columns": columns, "rows": rows}, "error": ""}
    except Exception as e:
        return {"result": None, "error": str(e)}


def _execute_sql_modal(state: AgentState) -> dict:
    from src.sandbox import run_sandboxed

    try:
        with open(state.db_path, "rb") as f:
            db_bytes = f.read()
        result = run_sandboxed(db_bytes, state.sql)
        error = result.get("error", "")
        if error:
            return {"result": None, "error": error}
        return {"result": {"columns": result["columns"], "rows": result["rows"]}, "error": ""}
    except Exception as e:
        return {"result": None, "error": str(e)}


def summarize_result_call(question: str, result: dict) -> tuple[str, dict]:
    """Role: summarize. Shared by the LangGraph node and the BIRD eval harness's
    optional judge pass (which needs an English answer to judge in the first place)."""
    llm, resolution = get_llm_for_role("summarize")
    start = time.perf_counter()

    with tracer.start_as_current_span("summarize_result") as otel_span:
        otel_span.set_attribute("role", resolution.role)
        otel_span.set_attribute("model", resolution.model)
        otel_span.set_attribute("model_source", resolution.source)

        formatted = _format_rows(result["columns"], result["rows"])

        messages = [
            SystemMessage(content="You answer questions in plain English based on SQL query results. Be concise and direct. Don't forget that data will be uppercased in SQL"),
            HumanMessage(content=(
                f"Question: {question}\n\n"
                f"SQL Result:\n{formatted}\n\n"
                f"Answer the question in a natural sentence."
            )),
        ]
        response = llm.invoke(messages)
        usage = _usage_from_raw(response)
        if usage:
            otel_span.set_attribute("input_tokens", usage.get("input_tokens") or 0)
            otel_span.set_attribute("output_tokens", usage.get("output_tokens") or 0)

    span = make_span(resolution, "summarize_result", start, usage)
    return response.content, span


def summarize_result(state: AgentState) -> dict:
    if state.error or not state.result:
        return {"answer": f"Sorry, I couldn't answer that. Error: {state.error}"}

    answer, span = summarize_result_call(state.question, state.result)
    return {"answer": answer, "trace": state.trace + [span]}


def check_result(state: AgentState) -> dict:
    if state.error and state.retries < MAX_RETRIES:
        return {"retries": state.retries + 1}
    return {"done": True}


def should_retry(state: AgentState) -> str:
    if state.done:
        if state.summarize:
            return "summarize_result"
        return END
    return "generate_sql"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_schema", retrieve_schema)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("validate_sql", validate_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("check_result", check_result)
    graph.add_node("summarize_result", summarize_result)

    graph.set_entry_point("retrieve_schema")
    graph.add_edge("retrieve_schema", "generate_sql")
    graph.add_edge("generate_sql", "validate_sql")
    graph.add_conditional_edges("validate_sql", route_after_validate)
    graph.add_edge("execute_sql", "check_result")
    graph.add_conditional_edges("check_result", should_retry)
    graph.add_edge("summarize_result", END)

    return graph.compile()


def run_question(question: str, db_path: str, history: List[dict] = None, model: str = None, use_sandbox: bool = False, summarize: bool = True, pg_schema: str = "") -> dict:
    """`model`, if given, overrides only the `generate` role's model for this call
    (the lever used to compare small vs. large generation models in evals).
    The `validate` and `summarize` roles always resolve from model_routes.json.

    `pg_schema`, if given, executes against that Postgres schema instead of
    SQLite/Modal (see _execute_sql_postgres). `db_path` is still required in this
    mode — schema retrieval reads the local SQLite copy purely to describe the
    schema to the LLM; only execution moves to Postgres."""
    with tracer.start_as_current_span("run_question") as span:
        span.set_attribute("question", question)
        span.set_attribute("db_path", db_path)
        span.set_attribute("use_sandbox", use_sandbox)
        span.set_attribute("pg_schema", pg_schema)

        graph = build_graph()
        initial_state = AgentState(
            question=question,
            db_path=db_path,
            history=history or [],
            use_sandbox=use_sandbox,
            summarize=summarize,
            generate_model_override=model or "",
            pg_schema=pg_schema,
        )
        return graph.invoke(initial_state)
