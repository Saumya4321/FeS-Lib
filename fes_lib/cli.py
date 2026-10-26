"""Terminal access to every public function in io.py and clean.py.

    python3 -m evse_features.cli <command> <directory> [options]
    python3 -m evse_features.cli --help          # list the commands
    python3 -m evse_features.cli <command> --help # one command's options

One command == one function. Nothing here calls more than one of your
functions in sequence; if you want check-ranges AND nan-report, run two
commands.

CHAINING
--------
Commands that return a FRAME (decode, drop-*, impute, normalize, load-corpus)
accept --out DIR and write the result there. Any command can then read that
file back with --from-csv FILE instead of a directory, so you can build a
pipeline one step at a time:

    ... decode    testdata/corpus --schema toy.yaml --out work
    ... impute    --from-csv work/decoded_*.csv --schema toy.yaml --out work
    ... normalize --from-csv work/imputed_*.csv --schema toy.yaml --out work
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)

from . import clean
from .io import (class_balance, compare_headers, infer_inventory, list_captures,
                 load_corpus, read_header, resolve_schema, save_corpus)
from .schema import GENERIC_SCHEMA, Role, Schema, SchemaError

# `decode` was called `convert_to_nan` in one revision. Accept either name so
# the CLI does not break the next time you rename it.
_decode = getattr(clean, "decode", None) or getattr(clean, "convert_to_nan")


# ---------------------------------------------------------------- plumbing

def _policy(args) -> Schema:
    """The schema BEFORE it is bound to a corpus (YAML, or the empty default).

    One wrinkle for --from-csv: a frame saved by a previous command already
    contains the binary_state column, and self_check() refuses a schema whose
    binary_col collides with a real column. So when reading a saved frame we
    clear binary_col -- the column is there, it just is not ours to add again.
    """
    policy = Schema.from_yaml(args.schema) if args.schema else GENERIC_SCHEMA
    if getattr(args, "from_csv", None) and policy.binary_col:
        policy = replace(policy, binary_col="",
                         categorical=policy.categorical + (policy.binary_col,))
    return policy


def _paths(args):
    """Where the data comes from: a directory, or one CSV via --from-csv."""
    if getattr(args, "from_csv", None):
        return [args.from_csv]
    if args.directory is None:
        raise SchemaError("give a directory, or --from-csv FILE")
    return args.directory


def _resolve(args):
    """resolve_schema on whatever --from-csv / directory points at."""
    return resolve_schema(_paths(args), _policy(args),
                          pattern=args.pattern, on_mismatch=args.on_mismatch)


def _frame(args, decode=True):
    """The frame every analysis command starts from.

    From a directory : resolve -> load_corpus -> decode
    From --from-csv  : resolve -> read_csv    -> decode   (no re-merge; the
                       file is assumed to be an already-merged corpus)

    `decode` is the default because NaN, ranges and dispersion are all
    meaningless while sentinels are still encoded as -1. Pass --raw to skip it.
    """
    schema, headers = _resolve(args)
    if getattr(args, "from_csv", None):
        df = pd.read_csv(args.from_csv)
        df.attrs["schema"] = schema
    else:
        df = load_corpus(args.directory, schema, pattern=args.pattern,
                         binary_label=not args.no_binary)
    # A file given with --from-csv was written by an earlier command, so it is
    # already decoded. Decoding again would add a second round of __missing
    # indicators. Pass --decode to force it.
    if getattr(args, "from_csv", None):
        decode = getattr(args, "decode", False)
    if decode and not getattr(args, "raw", False):
        df, _mask = _decode(df, schema)
    return schema, df, headers


def _emit(obj, args, title=None, save=True):
    """Print a frame/series/list, and save it if --csv was given.

    `save=False` for a secondary table (impute's fill report, decode's mask
    counts) so --csv keeps the main result rather than being overwritten.
    """
    if title:
        print(f"\n== {title} ==")
    if isinstance(obj, (list, tuple)):
        for item in obj:
            print(item)
        print(f"({len(obj)} items)")
        return
    if isinstance(obj, (pd.DataFrame, pd.Series)) and len(obj) == 0:
        print("(empty)")
    else:
        n = getattr(args, "rows", 0)
        shown = obj.head(n) if n and isinstance(obj, (pd.DataFrame, pd.Series)) else obj
        print(shown.to_string())
        if n and len(obj) > n:
            print(f"... {len(obj) - n} more rows (--rows 0 for all)")
    if save and getattr(args, "csv", None) and isinstance(obj, (pd.DataFrame, pd.Series)):
        obj.to_csv(args.csv, index=not isinstance(obj, pd.DataFrame)
                   or obj.index.name is not None)
        print(f"\ntable saved -> {args.csv}")


def _emit_frame(before: pd.DataFrame, after: pd.DataFrame, args, name: str):
    """Print the resulting frame, summarise what changed, and save it if --out.

    The frame itself is the point of these commands, so it is printed by
    default: --rows N of it (0 = all), --quiet for the summary only.
    """
    if not getattr(args, "quiet", False):
        _emit(after, args, "result")
    dropped = [c for c in before.columns if c not in after.columns]
    added = [c for c in after.columns if c not in before.columns]
    print("\n== what changed ==")
    print(f"rows    : {len(before)} -> {len(after)}")
    print(f"columns : {before.shape[1]} -> {after.shape[1]}")
    if dropped:
        print(f"dropped : {', '.join(dropped)}")
    if added:
        print(f"added   : {', '.join(added)}")
    print(f"NaN     : {int(before.isna().sum().sum())} -> {int(after.isna().sum().sum())}")
    if getattr(args, "out", None):
        print(f"saved   -> {save_corpus(after, args.out, args.name or name)}")
    else:
        print("(not saved -- add --out DIR to keep it)")


# ---------------------------------------------------------------- io.py

def cmd_list_captures(args):
    _emit(list_captures(_paths(args), args.pattern), args, "capture files")
    return 0


def cmd_read_header(args):
    _emit(read_header(args.file), args, f"columns in {args.file}")
    return 0


def cmd_compare_headers(args):
    _emit(compare_headers(list_captures(_paths(args), args.pattern)),
          args, "headers vs the first file")
    return 0


def cmd_infer_inventory(args):
    columns, table = infer_inventory(list_captures(_paths(args), args.pattern),
                                     on_mismatch=args.on_mismatch)
    _emit(table, args, "header agreement")
    _emit(columns, args, f"inventory ({len(columns)} columns)", save=False)
    return 0


def cmd_resolve_schema(args):
    schema, headers = _resolve(args)
    print(f"name      : {schema.name}")
    print(f"resolved  : {schema.is_resolved}")
    print(f"inventory : {len(schema.known_columns)} columns")
    print(f"label col : {schema.label_col}   source col: {schema.source_col}")
    print(f"binary col: {schema.binary_col or '(none)'}")
    _emit(headers, args, "files")
    return 0


def cmd_load_corpus(args):
    schema, _ = _resolve(args)
    df = load_corpus(args.directory, schema, pattern=args.pattern,
                     binary_label=not args.no_binary)
    print(f"{len(df)} rows x {df.shape[1]} columns from "
          f"{df[schema.source_col].nunique()} captures")
    _emit(df, args, "head")
    if args.out:
        print(f"\nsaved -> {save_corpus(df, args.out, args.name or 'corpus')}")
    return 0


def cmd_save_corpus(args):
    """load_corpus + save_corpus. The only command that pairs two functions,
    because save_corpus needs a frame and the CLI has no other way to hold one."""
    schema, df, _ = _frame(args, decode=not args.raw)
    print(f"saved -> {save_corpus(df, args.out, args.name or 'corpus')}")
    return 0


def cmd_class_balance(args):
    schema, df, _ = _frame(args, decode=False)
    _emit(class_balance(df, schema), args, "rows and captures per label")
    return 0


# ---------------------------------------------------------------- clean.py

def cmd_check_ranges(args):
    schema, df, _ = _frame(args)
    v = clean.check_ranges(df, schema)
    _emit(v, args, "range violations")
    if len(v):
        print("\n^ the EXTRACTOR produced these, the analysis did not")
    return 1 if len(v) else 0


def cmd_decode(args):
    schema, raw, _ = _frame(args, decode=False)
    typed, mask = _decode(raw, schema, add_missing_indicators=not args.no_indicators)
    _emit_frame(raw, typed, args, "decoded")
    _emit(mask.sum().sort_values(ascending=False), args,
          "values masked per column", save=False)
    return 0


def cmd_degenerate_columns(args):
    schema, df, _ = _frame(args)
    for kind, cols in clean.degenerate_columns(df, schema,
                                               max_missing=args.max_missing).items():
        print(f"{kind:14s} ({len(cols):3d}): {', '.join(cols) or '-'}")
    return 0


def cmd_drop_columns(args):
    schema, df, _ = _frame(args)
    out = clean.drop_columns(df, args.columns, schema, allow_group=args.allow_group)
    _emit_frame(df, out, args, "dropped")
    return 0


def cmd_drop_categorical(args):
    schema, df, _ = _frame(args)
    _emit_frame(df, clean.drop_categorical(df, schema), args, "no_categorical")
    return 0


def cmd_drop_binary(args):
    schema, df, _ = _frame(args)
    _emit_frame(df, clean.drop_binary(df, schema), args, "no_binary")
    return 0


def cmd_drop_non_numeric(args):
    schema, df, _ = _frame(args)
    _emit_frame(df, clean.drop_non_numeric(df, schema), args, "numeric_only")
    return 0


def cmd_nan_report(args):
    schema, df, _ = _frame(args)
    _emit(clean.nan_report(df, schema, by_label=args.by_label), args, "missingness")
    return 0


def cmd_drop_low_variance_numeric(args):
    schema, df, _ = _frame(args)
    out = clean.drop_low_variance_numeric(df, schema, threshold=args.threshold)
    _emit_frame(df, out, args, "high_variance")
    return 0


def cmd_dispersion_report(args):
    schema, df, _ = _frame(args)
    _emit(clean.dispersion_report(df, schema, bins=args.bins), args, "dispersion")
    return 0


def cmd_dispersion_by_label(args):
    schema, df, _ = _frame(args)
    rep = clean.dispersion_by_label(df, schema, min_rows=args.min_rows)
    if "warning" in rep.attrs:
        print("WARNING:", rep.attrs["warning"])
    _emit(rep, args, "dispersion within each class")
    return 0


def cmd_drop_nan_rows(args):
    schema, df, _ = _frame(args)
    out = clean.drop_nan_rows(df, schema, subset=args.subset or None,
                              min_present=args.min_present)
    _emit_frame(df, out, args, "rows_dropped")
    return 0


def cmd_impute(args):
    schema, df, _ = _frame(args)
    strategy = args.strategy
    if args.column_strategy:
        strategy = dict(pair.split("=", 1) for pair in args.column_strategy)
        bad = [v for v in strategy.values() if v not in clean._STRATEGIES]
        if bad:
            raise SchemaError(f"unknown strategy {bad}, pick from {clean._STRATEGIES}")
    filled, report = clean.impute(df, schema, strategy,
                                  const=args.const,
                                  max_missing=args.max_missing,
                                  fill_categorical=args.fill_categorical,
                                  default_strategy=args.default_strategy,
                                  skip_columns=args.skip or None)
    _emit(report, args, "what was filled", save=False)
    _emit_frame(df, filled, args, "imputed")
    return 0


def cmd_normalize(args):
    schema, df, _ = _frame(args)
    out = clean.normalize(df, schema, method=args.method)
    numeric = [c for c in df.columns
               if schema.role(c) is Role.NUMERIC and c not in schema.meta_columns]
    ruined = [c for c in numeric if out[c].isna().all() and not df[c].isna().all()]
    print(f"method  : {args.method}")
    print(f"numeric : {len(numeric)} columns")
    if ruined:
        print(f"RUINED  : {len(ruined)} became all-NaN -- {ruined}")
        print("          (zero spread, so the scaler divided by zero)")
    _emit_frame(df, out, args, "normalized")
    return 1 if ruined else 0

def cmd_columns(args):
    """Not an io/clean function -- just prints what the schema decided about
    each column, which is the fastest way to debug a YAML config."""
    schema, df, _ = _frame(args)
    rows = [{"column": c,
             "role": schema.role(c).value,
             "target": schema.is_target(c),
             "structural": schema.is_structural(c),
             "sentinels": schema.sentinels_for(c) or "",
             "validity_flag": schema.validity_flag_for(c) or "",
             "group": schema.group_of(c) or ""}
            for c in df.columns]
    _emit(pd.DataFrame(rows).sort_values(["role", "column"]), args, "columns")
    return 0

# ---------------------------------------------------------------- wiring
#
# One entry per function. `opts` adds that function's own keyword arguments;
# everything else is shared.

def _o(p, *args, **kw):
    p.add_argument(*args, **kw)


COMMANDS = [
    # name,                       handler,                       help, extra options
    ("list-captures",  cmd_list_captures,  "io: files matching the pattern", None),
    ("read-header",    cmd_read_header,    "io: column names of ONE csv", None),
    ("compare-headers", cmd_compare_headers, "io: do all files share a header?", None),
    ("infer-inventory", cmd_infer_inventory, "io: the column inventory + agreement table", None),
    ("resolve-schema", cmd_resolve_schema, "io: bind the YAML policy to this corpus", None),
    ("load-corpus",    cmd_load_corpus,    "io: merge every capture into one frame", None),
    ("save-corpus",    cmd_save_corpus,    "io: write the loaded frame to a csv", None),
    ("class-balance",  cmd_class_balance,  "io: rows and captures per label", None),

    ("decode",         cmd_decode,         "clean: sentinels -> NaN, + missing indicators",
     lambda p: _o(p, "--no-indicators", action="store_true",
                  help="do not add <col>__missing columns")),
    ("check-ranges",   cmd_check_ranges,   "clean: values outside the declared ranges", None),
    ("degenerate-columns", cmd_degenerate_columns,
     "clean: constant / all-missing / too-sparse columns",
     lambda p: _o(p, "--max-missing", type=float, default=0.98)),
    ("drop-columns",   cmd_drop_columns,   "clean: drop named columns",
     lambda p: (_o(p, "--columns", nargs="+", required=True, metavar="COL"),
                _o(p, "--allow-group", action="store_true",
                   help="permit dropping the capture_id/group column"))),
    ("drop-categorical", cmd_drop_categorical, "clean: drop every CATEGORICAL column", None),
    ("drop-binary",    cmd_drop_binary,    "clean: drop every BINARY column", None),
    ("drop-non-numeric", cmd_drop_non_numeric, "clean: keep NUMERIC (+ meta) only", None),
    ("nan-report",     cmd_nan_report,     "clean: missing values per column",
     lambda p: _o(p, "--by-label", action="store_true",
                  help="also the rate within each class")),
    ("drop-low-variance-numeric", cmd_drop_low_variance_numeric,
     "clean: drop numeric columns below a variance threshold",
     lambda p: _o(p, "--threshold", type=float, default=0.8)),
    ("dispersion-report", cmd_dispersion_report,
     "clean: how much each column varies (works on strings too)",
     lambda p: _o(p, "--bins", type=int, default=20)),
    ("dispersion-by-label", cmd_dispersion_by_label,
     "clean: the same, computed inside each class",
     lambda p: _o(p, "--min-rows", type=int, default=30)),
    ("drop-nan-rows",  cmd_drop_nan_rows,  "clean: drop rows with missing values",
     lambda p: (_o(p, "--subset", nargs="+", metavar="COL",
                   help="only test these columns"),
                _o(p, "--min-present", type=float, default=None,
                   help="keep a row if at least this fraction is present"))),
    ("impute",         cmd_impute,         "clean: fill missing values",
     lambda p: (_o(p, "--strategy", default="median", choices=list(clean._STRATEGIES)),
                _o(p, "--column-strategy", nargs="+", metavar="COL=STRAT",
                   help="per-column override, e.g. packets=zero rtt=mean"),
                _o(p, "--default-strategy", default="skip", choices=list(clean._STRATEGIES),
                   help="for columns not named in --column-strategy"),
                _o(p, "--const", type=float, default=None,
                   help="the value, when strategy=constant"),
                _o(p, "--fill-categorical", default=None,
                   help="value for CATEGORICAL columns (default: their mode)"),
                _o(p, "--max-missing", type=float, default=0.5,
                   help="leave a column alone if more than this fraction is missing"),
                _o(p, "--skip", nargs="+", default=[], metavar="COL"))),
    ("normalize",      cmd_normalize,      "clean: scale numeric columns",
     lambda p: _o(p, "--method", default="zscore", choices=["zscore", "minmax"])),

    ("columns",        cmd_columns,        "the role/policy the schema gave each column", None),
]

# commands that produce a frame worth keeping
SAVES = {"decode", "drop-columns", "drop-categorical", "drop-binary",
         "drop-non-numeric", "drop-low-variance-numeric", "drop-nan-rows",
         "impute", "normalize", "load-corpus"}
# commands that read the raw directory structure, not a single merged frame
DIR_ONLY = {"list-captures", "compare-headers", "infer-inventory",
            "resolve-schema", "load-corpus"}


def build_parser():
    ap = argparse.ArgumentParser(
        prog="python3 -m evse_features.cli",
        description="One subcommand per function in io.py and clean.py.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("CHAINING")[1])
    sub = ap.add_subparsers(dest="command", required=True, metavar="COMMAND")

    for name, handler, help_text, opts in COMMANDS:
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.set_defaults(func=handler)

        if name == "read-header":
            p.add_argument("file", help="one csv file")
        else:
            p.add_argument("directory", nargs="?", default=None,
                           help="folder of capture csvs (omit if --from-csv)")
            p.add_argument("--from-csv", metavar="FILE", default=None,
                           help="read one already-merged csv instead of a folder")
            p.add_argument("--schema", metavar="YAML", default=None,
                           help="format config; without it the generic schema "
                                "is used, which declares no roles or sentinels")
            p.add_argument("--pattern", default="*.csv")
            p.add_argument("--on-mismatch", default="raise", choices=["raise", "warn"])
            p.add_argument("--no-binary", action="store_true",
                           help="do not add the benign-vs-anomalous column")
            if name not in DIR_ONLY:
                p.add_argument("--raw", action="store_true",
                               help="skip decode; leave sentinels encoded")
                p.add_argument("--decode", action="store_true",
                               help="with --from-csv: decode it anyway "
                                    "(the file is assumed decoded already)")

        p.add_argument("--rows", type=int, default=25,
                       help="rows of the table to print (0 = all)")
        p.add_argument("--csv", metavar="FILE", default=None,
                       help="also write the printed table here")
        p.add_argument("--quiet", action="store_true",
                       help="print only the summary, not the frame/table")

        if name in SAVES or name == "save-corpus":
            p.add_argument("--out", metavar="DIR",
                           required=(name == "save-corpus"), default=None,
                           help="write the resulting FRAME to this folder")
            p.add_argument("--name", default=None, help="base filename")

        if opts:
            opts(p)

    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SchemaError as exc:
        print(f"\nSchemaError: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())