"""Silver frame vs a table definition, before the push. Severity: BLOCKER = the insert would fail,
WARNING = it would insert but not as expected, INFO = for information."""

import json
import math
import re
from collections import Counter

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Connection

from datetime import date, datetime

from . import config
from .prepare import convert
from .schema import TableDef, constant_default, has_db_default

INT_RANGES = {"smallint": 2**15 - 1, "integer": 2**31 - 1, "bigint": 2**63 - 1}
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$")
TRUE_FALSE = {"true", "false", "t", "f", "yes", "no", "y", "n", "1", "0", "on", "off"}
MAX_EXAMPLES = 3
KEY_CHUNK = 5000


# ── 2. Silver vs definition ───────────────────────────────────────────────────
def _is_null(v) -> bool:
    if v is None:
        return True
    if isinstance(v, (list, tuple, dict, np.ndarray)):
        return False
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _num(v):
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _value_problem(v, c: dict) -> str | None:
    """Why value v does not fit column c, else None."""
    dt, udt = (c["data_type"] or "").lower(), (c["udt_name"] or "").lower()
    is_list = isinstance(v, (list, tuple, np.ndarray))
    if dt == "array":
        return None                    # lists as they are; a single value becomes a one-element array (WARNING)
    if is_list or isinstance(v, dict):
        if dt in ("json", "jsonb"):
            return None
        if is_list and dt in ("character varying", "character", "text"):
            n = c.get("max_length")
            joined = config.LIST_SEPARATOR.join(str(x) for x in v)
            return f"longer than {n} characters" if n and len(joined) > int(n) else None
        return f"{type(v).__name__} value for a {c['type']} column"
    if dt in INT_RANGES:
        x = _num(v)
        if x is None or math.isnan(x) or math.isinf(x):
            return "not a number"
        if x != int(x):
            return "has decimals"
        return "out of range" if abs(x) > INT_RANGES[dt] else None
    if dt in ("numeric", "decimal"):
        x = _num(v)
        if x is None or math.isinf(x):
            return "not a number"
        p, s = c.get("numeric_precision"), c.get("numeric_scale") or 0
        if p and abs(round(x, int(s))) >= 10 ** (int(p) - int(s)):
            return f"too large for {c['type']}"
        return None
    if dt in ("real", "double precision"):
        return None if _num(v) is not None else "not a number"
    if dt == "boolean":
        if isinstance(v, (bool, np.bool_)):
            return None
        return None if str(v).strip().lower() in TRUE_FALSE else "not true/false"
    if dt == "date" or dt.startswith("timestamp") or dt.startswith("time"):
        if isinstance(v, (pd.Timestamp, np.datetime64)) or hasattr(v, "year"):
            return None
        if dt.startswith("time") and not dt.startswith("timestamp"):
            return None
        return None if pd.notna(pd.to_datetime(str(v), errors="coerce")) else "not a date/time"
    if dt in ("json", "jsonb"):
        if not isinstance(v, str):
            return None
        try:
            json.loads(v)              # NaN / Infinity are accepted here and reported as a WARNING
            return None
        except ValueError:
            return "not valid JSON"
    if dt in ("character varying", "character", "text"):
        n = c.get("max_length")
        return f"longer than {n} characters" if n and len(str(v)) > int(n) else None
    if dt == "uuid":
        return None if UUID_RE.match(str(v)) else "not a UUID"
    if dt == "user-defined" and c.get("enum_values"):
        return None if str(v) in c["enum_values"] else f"not one of {c['enum_values']}"
    return None


def _has_nan(s: str) -> bool:
    found = []
    try:
        json.loads(s, parse_constant=lambda x: found.append(x))
    except ValueError:
        return False
    return bool(found)


def _not_checked(c: dict) -> bool:
    dt = (c["data_type"] or "").lower()
    known = set(INT_RANGES) | {"numeric", "decimal", "real", "double precision", "boolean", "date", "json", "jsonb",
                               "character varying", "character", "text", "uuid", "array"}
    return not (dt in known or dt.startswith("timestamp") or dt.startswith("time")
                or (dt == "user-defined" and c.get("enum_values")))


def _short(v, n=60):
    s = str(v.tolist() if isinstance(v, np.ndarray) else v)
    return s if len(s) <= n else s[:n - 3] + "..."


def _key(v):
    """Hashable, comparable form of a key value."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return tuple(_key(x) for x in v)
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return int(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _as_text(v, d=None):
    if d is not None:
        try:
            v = convert(v, d)
        except (ValueError, TypeError, OverflowError):
            pass                               # reported by the type check; the database cast decides
    if v is None or _is_null(v):
        return None
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return str(v)


def check_violations(conn: Connection, t: TableDef, chk: dict, df: pd.DataFrame):
    """Rows of df that break a CHECK constraint, evaluated by the database on the values as they would be
    inserted. Returns (rows failing, examples) or a str saying why the check was not tested."""
    cols = chk.get("columns") or []
    if not cols or None in cols:
        return "constraint on no / unknown columns"
    defs = [t.column(c) for c in cols]
    for c, d in zip(cols, defs):
        if (d["data_type"] or "").lower() in ("array", "json", "jsonb"):
            return f"{c} is {d['data_type']}"
        if c not in df.columns and (d["default"] is not None or d["identity"] or d["generated"]):
            return f"{c} not in silver, filled by the database"
    expr = re.sub(r"^CHECK\s*", "", chk["definition"].strip())
    expr = re.sub(r"\s+(NOT VALID|NO INHERIT)\s*$", "", expr)
    values = [[_as_text(v, d) for v in df[c]] if c in df.columns else [None] * len(df)
              for c, d in zip(cols, defs)]
    combos = Counter(zip(*values))
    uniq = list(combos)
    args = ", ".join(f"CAST(:a{i} AS text[])" for i in range(len(cols)))
    names = ", ".join(f"dhms_a{i}" for i in range(len(cols)))
    casts = ", ".join(f'CAST(u.dhms_a{i} AS {d["type"]}) AS "{c}"' for i, (c, d) in enumerate(zip(cols, defs)))
    sql = (f"SELECT dhms_n FROM (SELECT u.dhms_n, {casts} FROM unnest(CAST(:n AS int[]), {args}) "
           f"AS u(dhms_n, {names})) v "
           f"WHERE ({expr}) IS FALSE")
    bad = []
    conn.execute(text("SAVEPOINT dhms_check"))
    try:
        for s in range(0, len(uniq), KEY_CHUNK):
            part = uniq[s:s + KEY_CHUNK]
            params = {"n": list(range(s, s + len(part)))}
            params.update({f"a{i}": [p[i] for p in part] for i in range(len(cols))})
            bad += [uniq[r[0]] for r in conn.execute(text(sql), params)]
        conn.execute(text("RELEASE SAVEPOINT dhms_check"))
    except Exception as e:
        conn.execute(text("ROLLBACK TO SAVEPOINT dhms_check"))
        return f"could not evaluate ({type(e).__name__}: {str(e).splitlines()[0][:120]})"
    rows = int(sum(combos[b] for b in bad))
    ex = [", ".join(f"{c}={v}" for c, v in zip(cols, b)) for b in bad[:MAX_EXAMPLES]]
    return rows, ex


def missing_parents(conn: Connection, t: TableDef, fk: dict, df: pd.DataFrame, pending: dict, mode: str):
    """Keys of a foreign key (one or more columns) in df that the parent will NOT hold at insert time.
    The parent at insert time = its database rows (minus this survey's rows that replace deletes, when the
    parent is part of this push) + its silver rows when it is pushed before this table (pending).
    A row with a NULL in any key column is not checked (Postgres MATCH SIMPLE).
    Returns (missing keys sorted — the value for one column, 'a | b' for several —, number of df rows holding
    one, description of where it was looked up)."""
    cols, refs = fk["columns"], fk["ref_columns"]

    def keys(frame, cs):
        return [None if any(_is_null(v) for v in row) else tuple(str(_key(v)) for v in row)
                for row in frame[cs].itertuples(index=False, name=None)]

    df_keys = keys(df, cols)
    vals = sorted({k for k in df_keys if k is not None})
    parent = (pending or {}).get(fk["ref_table"]) if fk["ref_schema"] == t.schema else None
    drop, drop_params = "", {}
    if parent is not None and mode == "replace" and parent[1] is not None:
        drop, drop_params = f" AND NOT ({parent[1][0]})", parent[1][1]
    found = set()
    if parent is not None and all(r in parent[0].columns for r in refs):
        found |= {k for k in keys(parent[0], refs) if k is not None}
    sel = ", ".join(f'"{r}"' for r in refs)
    tup = ", ".join(f'"{r}"::text' for r in refs)
    arrs = ", ".join(f"CAST(:v{i} AS text[])" for i in range(len(refs)))
    sql = (f'SELECT DISTINCT {sel} FROM "{fk["ref_schema"]}"."{fk["ref_table"]}" '
           f"WHERE ({tup}) IN (SELECT * FROM unnest({arrs})){drop}")
    for i in range(0, len(vals), KEY_CHUNK):
        part = vals[i:i + KEY_CHUNK]
        params = {**drop_params, **{f"v{j}": [k[j] for k in part] for j in range(len(refs))}}
        found |= {tuple(str(_key(v)) for v in r) for r in conn.execute(text(sql), params)}
    miss = [k for k in vals if k not in found]
    miss_s = set(miss)
    rows = sum(1 for k in df_keys if k is not None and k in miss_s)
    ref_txt = f"{fk['ref_table']}." + (refs[0] if len(refs) == 1 else f"({', '.join(refs)})")
    where = (f"{ref_txt} (database" +
             (", after replace deletes this survey's rows" if drop else "") +
             (", nor in its silver file pushed before this table)" if parent is not None else ")"))
    return [k[0] if len(k) == 1 else " | ".join(k) for k in miss], rows, where


def blank_missing_parents(conn: Connection, t: TableDef, df: pd.DataFrame, column: str, pending: dict,
                          mode: str):
    """Sets `column` to NULL on the rows whose parent (via the table's one-column foreign key on it) will not
    exist at insert time. Returns (frame, issue dicts without the table name)."""
    fk = next((f for f in t.foreign_keys if f["columns"] == [column]), None)
    if fk is None or column not in df.columns:
        return df, [{"severity": "WARNING", "column": column,
                     "issue": f"blank missing parent: no foreign key on {column} or column not in silver → skipped"}]
    miss, rows, where = missing_parents(conn, t, fk, df, pending, mode)
    if not miss:
        return df, []
    miss_s = {str(v) for v in miss}
    df = df.copy()
    df[column] = df[column].astype(object)
    df.loc[df[column].map(lambda v: not _is_null(v) and str(_key(v)) in miss_s), column] = None
    return df, [{"severity": "WARNING", "column": column, "rows": rows,
                 "issue": f"{column} set to NULL: parent not in {where} ({len(miss)} parent ids)",
                 "examples": "; ".join(_short(x) for x in miss[:MAX_EXAMPLES])}]


def compare(t: TableDef, df: pd.DataFrame | None, conn: Connection | None, db_checks: bool = True,
            scope=None, mode: str = "replace", pending=None):
    """Returns (column rows, issue rows). Issue severity: BLOCKER = the insert would fail;
    WARNING = it would insert but the data would not be what you expect; INFO = for information.
    scope = (SQL condition, params) selecting this survey's rows in the table (see survey.scope_sql);
    mode 'replace' deletes those rows before the insert, 'append' keeps them.
    pending = {table: (silver frame, scope)} of the tables this push inserts BEFORE this one, so a foreign
    key is checked against what the parent will hold at insert time: its rows in the database (minus the
    survey's rows that replace deletes) plus its silver rows."""
    pending = pending or {}
    cols, issues = [], []

    def issue(sev, column, what, n=None, examples=None):
        issues.append({"table": t.table, "severity": sev, "column": column, "issue": what,
                       "rows": n, "examples": "; ".join(_short(x) for x in (examples or []))})

    if df is None:
        issue("INFO", "", "no silver file for this table (this pipeline does not write it) → skipped")
        return cols, issues
    if not t.exists:
        issue("BLOCKER", "", "table does not exist in the database")
        return [{"table": t.table, "column": c["name"], "db_type": c["type"], "db_nullable": c["nullable"],
                 "db_default": c["default"], "in_silver": False} for c in t.columns], issues

    n = len(df)
    nulls = {c: df[c].map(_is_null) for c in df.columns}
    for c in t.columns:
        name = c["name"]
        row = {"table": t.table, "column": name, "db_type": c["type"], "db_nullable": c["nullable"],
               "db_default": c["default"], "db_identity": c["identity"], "db_generated": bool(c["generated"]),
               "in_silver": name in df.columns}
        if name not in df.columns:
            if c["generated"] or c["identity"] or c["default"] is not None:
                row["status"] = "filled by the database"
            elif not c["nullable"]:
                row["status"] = "BLOCKER: NOT NULL, no default, not in silver"
                issue("BLOCKER", name, "NOT NULL column with no default is missing in silver", n)
            else:
                row["status"] = "missing in silver → NULL"
                issue("WARNING", name, "column missing in silver → every row gets NULL", n)
            cols.append(row)
            continue
        s = df[name]
        nn = s[~nulls[name]]
        row.update({"silver_dtype": str(s.dtype), "silver_nulls": int(nulls[name].sum()),
                    "silver_example": _short(nn.iloc[0]) if len(nn) else None})
        problems = []
        if c["generated"]:
            problems.append("generated column: silver value must not be sent")
            issue("BLOCKER", name, "GENERATED ALWAYS column: the database computes it, silver must not send it", n)
        if c["identity"] == "ALWAYS":
            problems.append("identity ALWAYS: silver value rejected")
            issue("BLOCKER", name, "GENERATED ALWAYS AS IDENTITY: an explicit value is rejected "
                                   "(drop the column from the push or use OVERRIDING SYSTEM VALUE)", n)
        if not c["nullable"] and row["silver_nulls"]:
            if has_db_default(c):
                what = "identity" if c["default"] is None else f"default {c['default']}"
                problems.append(f"{row['silver_nulls']} NULL → database {what}")
                issue("WARNING", name, f"NULL in silver → the database {what} is written", row["silver_nulls"])
            else:
                problems.append(f"{row['silver_nulls']} NULL in NOT NULL column")
                issue("BLOCKER", name, "NULL values in a NOT NULL column without a default", row["silver_nulls"])
        if _not_checked(c):
            row["type_check"] = f"not checked ({c['type']})"
            issue("INFO", name, f"values not checked against type {c['type']}")
        else:
            bad = [(v, p) for v in nn for p in [_value_problem(v, c)] if p]
            row["type_check"] = "ok" if not bad else f"{len(bad)} bad"
            if bad:
                reasons = pd.Series([p for _, p in bad]).value_counts()
                for why, k in reasons.items():
                    ex = [v for v, p in bad if p == why][:MAX_EXAMPLES]
                    issue("BLOCKER", name, f"value does not fit {c['type']}: {why}", int(k), ex)
                problems.append(f"{len(bad)} values do not fit {c['type']}")
            dt = (c["data_type"] or "").lower()
            if dt in ("numeric", "decimal") and c.get("numeric_scale") is not None and len(nn):
                sc = int(c["numeric_scale"])
                x = pd.to_numeric(nn, errors="coerce").dropna()
                lost = int((x.round(sc) != x).sum())
                if lost:
                    issue("WARNING", name, f"{c['type']} keeps {sc} decimals: values will be rounded", lost,
                          x[x.round(sc) != x].head(MAX_EXAMPLES).tolist())
            if dt == "date" and len(nn):
                ts = pd.to_datetime(nn.astype(str), errors="coerce").dropna()
                cut = int((ts != ts.dt.normalize()).sum())
                if cut:
                    issue("WARNING", name, "date column: the time part will be dropped", cut,
                          nn.head(MAX_EXAMPLES).tolist())
        if (c["data_type"] or "").lower() == "array":
            single = [v for v in nn if not isinstance(v, (list, tuple, np.ndarray))]
            if single:
                issue("WARNING", name, "single value (not a list) → stored as a one-element array", len(single),
                      single[:MAX_EXAMPLES])
        if (c["data_type"] or "").lower() in ("json", "jsonb"):
            bad_const = [v for v in nn if isinstance(v, str) and _has_nan(v)]
            if bad_const:
                issue("WARNING", name, "NaN / Infinity inside the JSON (not valid for Postgres) → stored as null",
                      len(bad_const), bad_const[:MAX_EXAMPLES])
        if (c["data_type"] or "").lower() in ("character varying", "character", "text"):
            lists = [v for v in nn if isinstance(v, (list, tuple, np.ndarray))]
            if lists:
                issue("WARNING", name, f"silver lists stored as text joined with '{config.LIST_SEPARATOR}'",
                      len(lists), lists[:MAX_EXAMPLES])
        row["status"] = "; ".join(problems) if problems else "ok"
        cols.append(row)

    db_names = {c["name"] for c in t.columns}
    extra = [c for c in df.columns if c not in db_names]
    for c in extra:
        cols.append({"table": t.table, "column": c, "in_silver": True, "silver_dtype": str(df[c].dtype),
                     "status": "silver only: not in the database table (would not be pushed)"})
    if extra:
        issue("WARNING", ", ".join(extra), "silver columns not in the database table (dropped on push)", len(extra))

    # keys inside the silver file
    # NULL rules follow Postgres: a key with a NULL part never equals another key, unless NULLS NOT DISTINCT.
    # Primary key: NULLs are reported by the NOT NULL check above (BLOCKER, or WARNING when the database
    # default fills them), so they are left out here too.
    for kind, keys, nnd in ([("primary key", t.primary_key, False)] +
                            [(f"unique {u['name']}", u["columns"], u.get("nulls_not_distinct", False))
                             for u in t.uniques]):
        if not keys:
            continue
        missing = [k for k in keys if k not in df.columns]
        if missing:
            if kind == "primary key" and all(t.column(k) and (t.column(k)["identity"] or t.column(k)["default"])
                                             for k in missing):
                continue
            issue("BLOCKER" if kind == "primary key" else "WARNING", ", ".join(keys),
                  f"{kind} column(s) {missing} not in silver")
            continue
        k = df[keys].map(_key) if hasattr(df, "map") else df[keys].applymap(_key)
        if not nnd:
            null_part = (df[keys].map(_is_null) if hasattr(df, "map") else df[keys].applymap(_is_null)).any(axis=1)
            if kind == "primary key":
                fixed = [c for c in keys if constant_default(t.column(c)) and df[c].map(_is_null).sum() > 1]
                if fixed:
                    issue("WARNING", ", ".join(keys), f"NULL primary key values filled by a constant default "
                          f"({', '.join(fixed)}): those rows may collide", int(null_part.sum()))
            k = k[~null_part]
        dup = int(k.duplicated(keep=False).sum())
        if dup:
            issue("BLOCKER", ", ".join(keys), f"duplicate {kind} values inside the silver file", dup,
                  ["|".join("" if v is None or (isinstance(v, float) and v != v) else str(v) for v in row)
                   for row in k[k.duplicated(keep=False)].head(MAX_EXAMPLES).itertuples(index=False)])

    if t.primary_key is not None and not t.primary_key:
        issue("INFO", "", "table has no primary key: re-pushing the same UPC would add duplicate rows")

    if not (db_checks and conn is not None):
        return cols, issues
    fq = f'"{t.schema}"."{t.table}"'

    # CHECK constraints: the database evaluates each rule on the silver values (read only)
    for chk in t.checks:
        res = check_violations(conn, t, chk, df)
        if isinstance(res, str):
            issue("INFO", ", ".join(c for c in chk["columns"] if c), f"check {chk['name']} not tested: {res}")
        elif res[0]:
            issue("BLOCKER", ", ".join(chk["columns"]), f"check {chk['name']} fails: {chk['definition']}",
                  res[0], res[1])

    # foreign keys: silver values that the referenced table does not have
    for fk in t.foreign_keys:
        if not all(c in df.columns for c in fk["columns"]):
            continue
        col = ", ".join(fk["columns"])
        miss, rows, where = missing_parents(conn, t, fk, df, pending, mode)
        if miss:
            issue("BLOCKER", col, f"foreign key {fk['name']}: value not in {where}", rows, miss[:MAX_EXAMPLES])

    # rows of this survey already in the table, and primary keys that would collide
    if scope is None:
        issue("WARNING", "", "table has no UPC column: rows of this survey cannot be found (replace deletes nothing)")
        where, params = "FALSE", {}
    else:
        where, params = scope
        cnt = conn.execute(text(f"SELECT count(*) FROM {fq} WHERE {where}"), params).scalar()
        if mode == "replace":
            issue("INFO", "", f"rows of this survey already in the table: {cnt:,} → deleted before the insert", cnt)
        elif cnt:
            issue("WARNING", "", f"append: {cnt:,} rows of this survey are already in the table and stay "
                                 f"(the push adds {n:,} more)", cnt)
    for kind, keys in [("primary key", t.primary_key)] + [(f"unique {u['name']}", u["columns"]) for u in t.uniques]:
        if len(keys) != 1 or keys[0] not in df.columns:
            continue
        k = keys[0]
        vals = [str(_key(v)) for v in df[k] if not _is_null(v)]
        outside = f"NOT ({where})" if mode == "replace" else "TRUE"
        hit = 0
        for i in range(0, len(vals), KEY_CHUNK):
            hit += conn.execute(text(f'SELECT count(*) FROM {fq} WHERE "{k}"::text = ANY(:v) AND {outside}'),
                                {**params, "v": vals[i:i + KEY_CHUNK]}).scalar()
        if hit:
            issue("BLOCKER", k, f"{kind} values already in the table" +
                  (" on rows of ANOTHER survey (not deleted by replace)" if mode == "replace" else ""), hit)
    return cols, issues
