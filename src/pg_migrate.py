"""SQLite -> Postgres migration for benchmark databases.

Reads a SQLite file's schema and data and recreates it in Postgres: tables
first (columns + primary keys only), data copied, then foreign keys added as
a separate pass. Doing FKs last sidesteps dependency-ordering entirely —
works regardless of how tangled a benchmark schema's references are.

All identifiers go through psycopg.sql.Identifier rather than manual string
quoting. BIRD schemas have identifiers with spaces, mixed case, and even a
literal "%" in at least one column name (california_schools) — raw %-style
string interpolation collides with psycopg's placeholder parser on that last
one; sql.Identifier composes safely regardless of what's in the name.
"""

from __future__ import annotations

import argparse
import os
import sqlite3

import psycopg
from psycopg import sql
from dotenv import load_dotenv

load_dotenv()

# SQLite's type affinity names -> a reasonable Postgres equivalent. SQLite is
# dynamically typed, so this is a best-effort mapping, not a guarantee every
# value round-trips perfectly for exotic declared types.
TYPE_MAP = {
    "INTEGER": "BIGINT",
    "INT": "BIGINT",
    "REAL": "DOUBLE PRECISION",
    "FLOAT": "DOUBLE PRECISION",
    "DOUBLE": "DOUBLE PRECISION",
    "NUMERIC": "NUMERIC",
    "DECIMAL": "NUMERIC",
    "BOOLEAN": "BOOLEAN",
    "BLOB": "BYTEA",
    "TEXT": "TEXT",
    "VARCHAR": "TEXT",
    "CHAR": "TEXT",
    "DATE": "TEXT",
    "DATETIME": "TEXT",
}


def _pg_type(sqlite_type: str) -> str:
    base = (sqlite_type or "TEXT").split("(")[0].strip().upper()
    return TYPE_MAP.get(base, "TEXT")


def list_tables(sconn: sqlite3.Connection) -> list[str]:
    cursor = sconn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
    return [row[0] for row in cursor.fetchall()]


def migrate_table(sconn: sqlite3.Connection, pconn: psycopg.Connection, table: str, schema: str) -> list[tuple]:
    """Creates `table` in Postgres and copies its rows. Returns its foreign
    keys (deferred — added in a second pass after every table exists).

    All Postgres-side identifiers are lowercased at creation time. BIRD's gold
    SQL references most columns unquoted (e.g. `T1.CDSCode`), relying on
    SQLite's case-insensitive matching — Postgres instead folds unquoted
    identifiers to lowercase, so preserving original mixed case here would
    make that same gold SQL fail to resolve. Lowercasing is also just normal
    Postgres convention, not a workaround peculiar to this migration."""
    cursor = sconn.cursor()
    cursor.execute(f"PRAGMA table_info('{table}')")
    columns = cursor.fetchall()  # (cid, name, type, notnull, dflt_value, pk)

    col_defs = []
    pk_cols = []
    for _, name, col_type, notnull, _, pk in columns:
        col_def = sql.SQL("{} {}").format(sql.Identifier(name.lower()), sql.SQL(_pg_type(col_type)))
        if notnull:
            col_def = sql.SQL("{} NOT NULL").format(col_def)
        col_defs.append(col_def)
        if pk:
            pk_cols.append(name.lower())

    if pk_cols:
        col_defs.append(sql.SQL("PRIMARY KEY ({})").format(sql.SQL(", ").join(map(sql.Identifier, pk_cols))))

    qualified = sql.Identifier(schema.lower(), table.lower())
    with pconn.cursor() as pcur:
        pcur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(qualified))
        pcur.execute(sql.SQL("CREATE TABLE {} (\n  {}\n)").format(qualified, sql.SQL(",\n  ").join(col_defs)))

    cursor.execute(f"PRAGMA foreign_key_list('{table}')")
    fks = cursor.fetchall()  # (id, seq, ref_table, from_col, to_col, ...)

    cursor.execute(f'SELECT * FROM "{table}"')
    col_names = [d[0].lower() for d in cursor.description]
    rows = cursor.fetchall()

    # Values are embedded as SQL literals (sql.Literal), not bound parameters —
    # some BIRD column names contain a literal "%" (california_schools), which
    # collides with psycopg's %s-style placeholder scanner when a query mixes
    # real bind params with unrelated "%" characters elsewhere in the same
    # rendered text. A fully-literal query (no params argument) skips that
    # scanner entirely, same reason CREATE TABLE above isn't affected.
    if rows:
        col_list = sql.SQL(", ").join(map(sql.Identifier, col_names))
        batch_size = 500
        with pconn.cursor() as pcur:
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i + batch_size]
                values_clause = sql.SQL(", ").join(
                    sql.SQL("({})").format(sql.SQL(", ").join(sql.Literal(v) for v in row))
                    for row in batch
                )
                pcur.execute(sql.SQL("INSERT INTO {} ({}) VALUES {}").format(qualified, col_list, values_clause))

    pconn.commit()
    print(f"  {table}: {len(columns)} columns, {len(rows)} rows")
    return [(table.lower(), fk[3].lower(), fk[2].lower(), fk[4].lower()) for fk in fks]  # (from_table, from_col, to_table, to_col)


def apply_foreign_keys(pconn: psycopg.Connection, schema: str, fks: list[tuple]) -> None:
    with pconn.cursor() as pcur:
        for i, (from_table, from_col, to_table, to_col) in enumerate(fks):
            qualified_from = sql.Identifier(schema.lower(), from_table)
            qualified_to = sql.Identifier(schema.lower(), to_table)
            constraint = sql.Identifier(f"fk_{from_table}_{from_col}_{i}")
            try:
                pcur.execute(
                    sql.SQL("ALTER TABLE {} ADD CONSTRAINT {} FOREIGN KEY ({}) REFERENCES {} ({})").format(
                        qualified_from, constraint, sql.Identifier(from_col), qualified_to, sql.Identifier(to_col)
                    )
                )
            except psycopg.Error as e:
                # Benchmark schemas sometimes declare FKs that don't form a clean
                # unique/PK target (duplicate values, composite mismatches). Skip
                # loudly rather than letting one bad FK abort the whole migration.
                pconn.rollback()
                print(f"  skipped FK {from_table}.{from_col} -> {to_table}.{to_col}: {e}")
                continue
    pconn.commit()


def migrate(sqlite_path: str, schema: str, pg_dsn: str = None) -> None:
    pg_dsn = pg_dsn or os.environ["PG_DSN"]
    sconn = sqlite3.connect(sqlite_path)
    pconn = psycopg.connect(pg_dsn)

    with pconn.cursor() as pcur:
        pcur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema.lower())))
    pconn.commit()

    tables = list_tables(sconn)
    print(f"Migrating {sqlite_path} -> schema {schema!r} ({len(tables)} tables)")

    all_fks = []
    for table in tables:
        all_fks.extend(migrate_table(sconn, pconn, table, schema))

    if all_fks:
        print(f"Applying {len(all_fks)} foreign keys...")
        apply_foreign_keys(pconn, schema, all_fks)

    sconn.close()
    pconn.close()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Migrate a SQLite benchmark database into Postgres.")
    parser.add_argument("sqlite_path", help="Path to the .sqlite file")
    parser.add_argument("--schema", required=True, help="Postgres schema name to create/overwrite")
    parser.add_argument("--pg-dsn", default=None, help="Postgres DSN (defaults to PG_DSN env var)")
    args = parser.parse_args()

    migrate(args.sqlite_path, args.schema, args.pg_dsn)


if __name__ == "__main__":
    main()
