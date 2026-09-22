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


def run_sandboxed(db_bytes: bytes, sql: str) -> dict:
    """Call the Modal function from outside a running Modal app."""
    with app.run():
        return execute_sql_sandboxed.remote(db_bytes, sql)
