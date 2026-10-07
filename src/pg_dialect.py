"""SQLite -> Postgres dialect translation for query *text* (schema/data migration
lives in pg_migrate.py — this is for gold SQL, and eventually generated SQL, that
need to run against the migrated databases).

pg_migrate.py lowercases every identifier at creation time (see its docstring for
why: Postgres folds unquoted identifiers to lowercase, so preserving BIRD's mixed
case would break the unquoted references its own gold SQL relies on). A
backtick-quoted span is the one place that lowercasing doesn't reach on its own —
it needs both the backticks swapped for double quotes AND its contents lowercased
to match. Everything outside a backtick span is left untouched; Postgres's default
unquoted-identifier folding handles the rest without needing a real SQL parser here.
"""

import re

_BACKTICK_RE = re.compile(r"`([^`]*)`")
# SQLite's `LIMIT offset, count` has no Postgres equivalent syntax — only
# `LIMIT count OFFSET offset` does the same thing. Case-insensitive, since
# gold SQL isn't consistent about keyword casing.
_LIMIT_OFFSET_RE = re.compile(r"\bLIMIT\s+(\d+)\s*,\s*(\d+)\b", re.IGNORECASE)


def sqlite_to_postgres(sql_text: str) -> str:
    sql_text = _LIMIT_OFFSET_RE.sub(lambda m: f"LIMIT {m.group(2)} OFFSET {m.group(1)}", sql_text)
    return _BACKTICK_RE.sub(lambda m: '"' + m.group(1).lower() + '"', sql_text)
