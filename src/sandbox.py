"""Modal sandbox for isolated SQL query execution with timeouts."""

import modal

app = modal.App("sql-sandbox")
volume = modal.Volume.from_name("bird-databases", create_if_missing=True)

QUERY_TIMEOUT = 60


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
    """Calls deployed Modal functions — no ephemeral app context needed."""

    def __init__(self):
        self._exec_sandboxed = None
        self._exec_on_volume = None
        self._get_schema = None

    def _ensure_lookups(self):
        if self._exec_on_volume is None:
            self._exec_sandboxed = modal.Function.from_name("sql-sandbox", "execute_sql_sandboxed")
            self._exec_on_volume = modal.Function.from_name("sql-sandbox", "execute_sql_on_volume")
            self._get_schema = modal.Function.from_name("sql-sandbox", "get_schema_from_volume")

    def start(self):
        self._ensure_lookups()

    def stop(self):
        pass

    def run(self, db_bytes: bytes, sql: str) -> dict:
        self._ensure_lookups()
        return self._exec_sandboxed.remote(db_bytes, sql)

    def run_on_volume(self, db_id: str, sql: str) -> dict:
        self._ensure_lookups()
        return self._exec_on_volume.remote(db_id, sql)

    def run_batch_on_volume(self, db_ids: list[str], queries: list[str]) -> list[dict]:
        self._ensure_lookups()
        return list(self._exec_on_volume.map(db_ids, queries))

    def get_schema(self, db_id: str) -> str:
        self._ensure_lookups()
        return self._get_schema.remote(db_id)

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
