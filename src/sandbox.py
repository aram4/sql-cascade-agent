"""Modal sandbox for isolated SQL query execution with timeouts."""

import modal

app = modal.App("sql-sandbox")
volume = modal.Volume.from_name("bird-databases", create_if_missing=True)

QUERY_TIMEOUT = 30


@app.function(timeout=QUERY_TIMEOUT)
def execute_sql_sandboxed(db_bytes: bytes, sql: str) -> dict:
    """Execute SQL against a database passed as raw bytes."""
    import sqlite3
    import tempfile
    import os

    tmp_path = os.path.join(tempfile.mkdtemp(), "db.sqlite")
    with open(tmp_path, "wb") as f:
        f.write(db_bytes)

    try:
        conn = sqlite3.connect(tmp_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()
        return {"columns": columns, "rows": rows, "error": ""}
    except Exception as e:
        return {"columns": [], "rows": [], "error": str(e)}
    finally:
        os.unlink(tmp_path)


@app.function(timeout=QUERY_TIMEOUT, volumes={"/data": volume})
def execute_sql_on_volume(db_id: str, sql: str) -> dict:
    """Execute SQL against a BIRD database stored on the Modal Volume."""
    import sqlite3

    db_path = f"/data/{db_id}/{db_id}.sqlite"

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


@app.function(timeout=QUERY_TIMEOUT, volumes={"/data": volume})
def get_schema_from_volume(db_id: str) -> str:
    """Read schema from a BIRD database on the Modal Volume."""
    import sqlite3

    db_path = f"/data/{db_id}/{db_id}.sqlite"
    conn = sqlite3.connect(db_path)
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
    return "\n\n".join(schema_parts)


class SandboxRunner:
    """Keeps one Modal app alive for the duration of a session."""

    def __init__(self):
        self._context = None

    def start(self):
        if self._context is None:
            self._context = app.run()
            self._context.__enter__()

    def stop(self):
        if self._context is not None:
            self._context.__exit__(None, None, None)
            self._context = None

    def run(self, db_bytes: bytes, sql: str) -> dict:
        self.start()
        return execute_sql_sandboxed.remote(db_bytes, sql)

    def run_on_volume(self, db_id: str, sql: str) -> dict:
        self.start()
        return execute_sql_on_volume.remote(db_id, sql)

    def run_batch_on_volume(self, db_ids: list[str], queries: list[str]) -> list[dict]:
        self.start()
        return list(execute_sql_on_volume.map(db_ids, queries))

    def get_schema(self, db_id: str) -> str:
        self.start()
        return get_schema_from_volume.remote(db_id)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


_runner = None


def get_runner() -> SandboxRunner:
    global _runner
    if _runner is None:
        _runner = SandboxRunner()
    return _runner


def run_sandboxed(db_bytes: bytes, sql: str) -> dict:
    return get_runner().run(db_bytes, sql)


def shutdown_sandbox():
    global _runner
    if _runner is not None:
        _runner.stop()
        _runner = None
