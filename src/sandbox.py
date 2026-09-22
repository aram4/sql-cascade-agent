"""Modal sandbox for isolated SQL query execution with timeouts."""

import modal

app = modal.App("sql-sandbox")

QUERY_TIMEOUT = 30


@app.function(timeout=QUERY_TIMEOUT)
def execute_sql_sandboxed(db_bytes: bytes, sql: str) -> dict:
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

    def run_batch(self, db_bytes: bytes, queries: list[str]) -> list[dict]:
        self.start()
        return list(execute_sql_sandboxed.map(
            [db_bytes] * len(queries),
            queries,
        ))

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
