"""Silver files of one UPC: the dlm_kpi_new folder (or the UPC folder above it), local or s3://."""

import io
import os

import pandas as pd


def _s3_get(path: str) -> bytes | None:
    import boto3
    from botocore.exceptions import ClientError
    bucket, key = path[len("s3://"):].split("/", 1)
    try:
        return boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise


def silver_folder(path: str) -> str:
    """Accepts the dlm_kpi_new folder or the UPC folder above it."""
    path = path.rstrip("/\\")
    if path.lower().endswith("dlm_kpi_new"):
        return path
    if path.startswith("s3://"):
        return path + "/dlm_kpi_new"
    sub = os.path.join(path, "dlm_kpi_new")
    return sub if os.path.isdir(sub) else path


def filter_rows(df: pd.DataFrame, rules: dict) -> tuple[pd.DataFrame, list]:
    """Keeps the rows whose columns match rules {column: value} (no case / surrounding spaces).
       Returns (filtered frame, issue dicts without the table name)."""
    issues = []
    for col, value in rules.items():
        if col not in df.columns:
            issues.append({"severity": "BLOCKER", "column": col,
                           "issue": f"row filter {col} = '{value}': column not in silver, rows not filtered"})
            continue
        keep = df[col].astype("string").str.strip().str.lower() == str(value).strip().lower()
        keep = keep.fillna(False).astype(bool)
        dropped = int((~keep).sum())
        others = df.loc[~keep, col].astype("string").fillna("<NULL>").value_counts().head(5)
        issues.append({"severity": "INFO", "column": col, "rows": dropped,
                       "issue": f"row filter {col} = '{value}': kept {int(keep.sum()):,} of {len(df):,} rows",
                       "examples": "; ".join(f"{v} ({n:,})" for v, n in others.items())})
        df = df[keep].reset_index(drop=True)
    return df, issues


def read_silver(folder: str, name: str) -> tuple[pd.DataFrame | None, str]:
    """(DataFrame or None, path tried). Parquet preferred, csv if there is no parquet."""
    for ext in ("parquet", "csv"):
        p = f"{folder}/{name}.{ext}" if folder.startswith("s3://") else os.path.join(folder, f"{name}.{ext}")
        if folder.startswith("s3://"):
            data = _s3_get(p)
            if data is None:
                continue
            return (pd.read_parquet(io.BytesIO(data)) if ext == "parquet" else pd.read_csv(io.BytesIO(data))), p
        if os.path.exists(p):
            return (pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p, low_memory=False)), p
    return None, f"{folder}/{name}.parquet"
