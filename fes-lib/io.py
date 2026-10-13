import os
from glob import glob
from dataclasses import replace
import pandas as pd
from pathlib import Path
from .schema import Schema, SchemaError, GENERIC_SCHEMA 



# --------------------------------------------------------------------------
# Inventory discovery and header comparison
# --------------------------------------------------------------------------


def list_captures(paths_or_dir, pattern: str = "*.csv") -> list[str]:
    if isinstance(paths_or_dir, (str, os.PathLike)):
        paths = sorted(glob(os.path.join(str(paths_or_dir), pattern)))
    else:
        paths = [str(p) for p in paths_or_dir]
    if not paths:
        raise SchemaError(f"no capture files matched {paths_or_dir!r}")
    return paths


def read_header(path: str) -> list[str]:    
    return list(pd.read_csv(path, nrows=0).columns)


def compare_headers(paths) -> pd.DataFrame:
    paths = list(paths)
    reference = read_header(paths[0])
    ref_set = set(reference)

    rows = []
    for path in paths:
        header = read_header(path)
        hset = set(header)
        rows.append({
            "file": os.path.basename(path),
            "n_columns": len(header),
            "missing": sorted(ref_set - hset),
            "extra": sorted(hset - ref_set),
            "reordered": header != reference and hset == ref_set,
            "matches": hset == ref_set,
        })
    return pd.DataFrame(rows)

MISMATCH_PROTOCOLS = ("raise", "warn")

def infer_inventory(paths, on_mismatch: str = "raise") -> tuple[list[str], pd.DataFrame]:
    if on_mismatch not in MISMATCH_PROTOCOLS:
        raise ValueError(f"on_mismatch must be one of {MISMATCH_PROTOCOLS}")

    paths = list(paths)
    report = compare_headers(paths)
    reference = read_header(paths[0])

    if report["matches"].all():
        return reference, report

    bad = report[~report["matches"]]
    detail = "; ".join(
        f"{r.file}: missing={r.missing or '-'} extra={r.extra or '-'}"
        for r in bad.itertuples()
    )

    if on_mismatch == "raise":
        raise SchemaError(
            f"header mismatch against reference {os.path.basename(paths[0])} "
            f"({len(reference)} columns) -- {detail}. Fix the extraction"
        )

    if on_mismatch == "warn":
        print(f"WARNING: header mismatch -- {detail}")
        return reference, report
    
# --------------------------------------------------------------------------
# Add binary column 
# --------------------------------------------------------------------------


def _add_binary_label(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    if not schema.binary_col:
        return df
    df[schema.binary_col] = df[schema.label_col].map(schema.binary_from_label)
    return df


# --------------------------------------------------------------------------
# Schema resolution 
# --------------------------------------------------------------------------

def resolve_schema(paths_or_dir, schema: Schema = GENERIC_SCHEMA,
                   pattern: str = "*.csv", on_mismatch: str = "raise") -> tuple[Schema, pd.DataFrame]:
    """Bind a policy schema to a concrete corpus.

    Returns (resolved_schema, header_report).
    """
    paths = list_captures(paths_or_dir, pattern)
    columns, report = infer_inventory(paths, on_mismatch=on_mismatch)
    resolved = replace(schema, known_columns=tuple(columns))
    resolved.self_check()
    return resolved, report

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _load_capture(path: str, schema: Schema) -> pd.DataFrame:
    """Read one capture CSV, attach label + capture_id + facets

    `schema` must be resolved (see resolve_schema).
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    base = os.path.basename(path)
    df[schema.label_col] = schema.label_from_filename(base)
    df[schema.source_col] = base
    for facet, value in schema.facets_from_filename(base).items():
        df[facet] = value
    return df


def load_corpus(paths_or_dir, schema: Schema,
                pattern: str = "*.csv", binary_label: bool = False) -> pd.DataFrame:
    """
    Read every capture into one frame
    Call only after resolve_schema() is called
    """

    if not schema.is_resolved:
        raise SchemaError(
            "load_corpus needs a resolved schema: "
            "schema, _ = resolve_schema(directory, policy)"
        )
    
    paths = list_captures(paths_or_dir, pattern)
    frames = [_load_capture(p, schema) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    if binary_label:
        df = _add_binary_label(df, schema)
    df.attrs["schema"] = schema
    return df


# --------------------------------------------------------------------------
# Generate class imbalance report
# --------------------------------------------------------------------------
def class_balance(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """
    Rows and captures per label
    """
    schema = schema or df.attrs.get("schema") or GENERIC_SCHEMA
    g = df.groupby(schema.label_col)
    out = pd.DataFrame({
        "rows": g.size(),
        "captures": g[schema.source_col].nunique(),
    })
    out["row_share"] = out["rows"] / out["rows"].sum()
    out["imbalance_vs_largest"] = out["rows"].max() / out["rows"]
    if schema.binary_col and schema.binary_col in df.columns:
        out[schema.binary_col] = g[schema.binary_col].first()
    return out.sort_values("rows", ascending=False)



# --------------------------------------------------------------------------
# Save frame to CSV
# --------------------------------------------------------------------------

def save_corpus(df: pd.DataFrame, out_dir, name: str) ->str:
    """
    Save corpus to CSV
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    name = f"{name}_{timestamp}"
    path = out / f"{name}.csv"
    df.to_csv(path, index=False)
    return str(path)
