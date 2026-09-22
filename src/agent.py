"""LangGraph text-to-SQL agent with self-correction loop."""

from __future__ import annotations

import os
import sqlite3
from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_fireworks import ChatFireworks
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

MAX_RETRIES = 2
DEFAULT_MODEL = "accounts/fireworks/models/deepseek-v4-flash-0731"

_schema_cache: dict[str, str] = {}


class SQLOutput(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning about the query")
    sql: str = Field(description="The SQL query to execute")


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


def get_llm(model: str = DEFAULT_MODEL) -> ChatFireworks:
    return ChatFireworks(
        model=model,
        api_key=os.environ["FIREWORKS_API_KEY"],
        temperature=0,
        max_tokens=1024,
    )


def retrieve_schema(state: AgentState) -> dict:
    if state.db_path in _schema_cache:
        return {"schema": _schema_cache[state.db_path]}

    conn = sqlite3.connect(state.db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    schema_parts = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info('{table}')")
        columns = cursor.fetchall()
        col_defs = [f"  {c[1]} {c[2]}{'  PRIMARY KEY' if c[5] else ''}" for c in columns]

        cursor.execute(f"PRAGMA foreign_key_list('{table}')")
        fks = cursor.fetchall()
        fk_defs = [f"  FOREIGN KEY ({fk[3]}) REFERENCES {fk[2]}({fk[4]})" for fk in fks]

        cursor.execute(f"SELECT * FROM '{table}' LIMIT 3")
        sample_rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        sample_str = "\n".join(
            "  " + str(dict(zip(col_names, row))) for row in sample_rows
        )

        schema_parts.append(
            f"CREATE TABLE {table} (\n"
            + ",\n".join(col_defs)
            + ("\n" + ",\n".join(fk_defs) if fk_defs else "")
            + "\n);\n"
            + f"-- Sample rows:\n{sample_str}"
        )

    conn.close()
    schema = "\n\n".join(schema_parts)
    _schema_cache[state.db_path] = schema
    return {"schema": schema}


def generate_sql(state: AgentState) -> dict:
    llm = get_llm()

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
        HumanMessage(content=(
            f"Schema:\n{state.schema}\n\n"
            f"Question: {state.question}"
            f"{history_context}"
            f"{error_context}"
        )),
    ]

    structured_llm = llm.with_structured_output(SQLOutput)
    response = structured_llm.invoke(messages)

    return {"sql": response.sql}


def execute_sql(state: AgentState) -> dict:
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


def summarize_result(state: AgentState) -> dict:
    if state.error or not state.result:
        return {"answer": f"Sorry, I couldn't answer that. Error: {state.error}"}

    llm = get_llm()
    rows = state.result["rows"]
    columns = state.result["columns"]
    formatted = "\n".join(str(dict(zip(columns, row))) for row in rows)

    messages = [
        SystemMessage(content="You answer questions in plain English based on SQL query results. Be concise and direct. Don't forget that data will be uppercased in SQL"),
        HumanMessage(content=(
            f"Question: {state.question}\n\n"
            f"SQL Result:\n{formatted}\n\n"
            f"Answer the question in a natural sentence."
        )),
    ]
    response = llm.invoke(messages)
    return {"answer": response.content}


def check_result(state: AgentState) -> dict:
    if state.error and state.retries < MAX_RETRIES:
        return {"retries": state.retries + 1}
    return {"done": True}


def should_retry(state: AgentState) -> str:
    if state.done:
        return "summarize_result"
    return "generate_sql"


def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("retrieve_schema", retrieve_schema)
    graph.add_node("generate_sql", generate_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("check_result", check_result)
    graph.add_node("summarize_result", summarize_result)

    graph.set_entry_point("retrieve_schema")
    graph.add_edge("retrieve_schema", "generate_sql")
    graph.add_edge("generate_sql", "execute_sql")
    graph.add_edge("execute_sql", "check_result")
    graph.add_conditional_edges("check_result", should_retry)
    graph.add_edge("summarize_result", END)

    return graph.compile()


def run_question(question: str, db_path: str, history: List[dict] = None, model: str = DEFAULT_MODEL) -> dict:
    global DEFAULT_MODEL
    original = DEFAULT_MODEL
    DEFAULT_MODEL = model

    graph = build_graph()
    initial_state = AgentState(question=question, db_path=db_path, history=history or [])
    final_state = graph.invoke(initial_state)

    DEFAULT_MODEL = original
    return final_state
