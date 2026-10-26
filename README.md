# FeS-Lib
+ Feature selection library
+ A library for loading, validating and cleaning directories of per-flow feature
CSVs extracted from network captures.
+ It was written for EVSE charging-station
traffic, but it knows nothing about EVSE, TCP or charging: everything
format-specific lives in a YAML config, so pointing it at a different extractor's
output means writing a new config, not editing code.

---

## 1. What problem it solves

### a) Eexploratory Data Analysis (EDA) on input data
You have a folder of CSVs, one per capture. Each row is a flow; each column is a
feature. The filename encodes the label (`..._Benign_...`, `..._icmp-flood_...`).
Before you can analyse any of it you have to answer a set of boring,
error-prone questions:

- Do all the files have the same columns? If not, which?
- Where does the label come from?
- Which cells are *missing* rather than *measured*? Extractors write `-1`,
  `-2`, `N/A`, `none` to mean "no value", and those are indistinguishable from
  real measurements until someone tells the code which is which.
- Which columns are identifiers, which are the target, which would leak it?
- Which columns are degenerate, and which only look degenerate because one class
  dominates the row count?

### b) Feature selection and ranking (TODO)
### c) Correlation analysis of features (TODO)
---

## 2. Layout

```
fen-lib/                      <- run everything from here
  fes_lib/
    __init__.py
    schema.py                 the Schema dataclass + YAML loader
    io.py                     find, validate, merge, save
    clean.py                  decode, inspect, drop, impute, normalise
    cli.py                    one subcommand per function
    schemas/
      tcp_stream.yaml         the config for the TCP-stream extractor
      template.yaml           a blank to copy
    data/
      Stream/                 real captures
      test_3/                 the test fixture + its schema
```

Requires Python 3.10+, `pandas`, `numpy`, `pyyaml`.

---

## 3. Five-minute start

From `fen-lib/`:

```bash
# does the data hold together?
python -m fes_lib.cli resolve-schema fes_lib/data/Stream --schema fes_lib/schemas/tcp_stream.yaml

# how many rows per class?
python -m fes_lib.cli class-balance fes_lib/data/Stream --schema fes_lib/schemas/tcp_stream.yaml

# what is missing, and is it missing in every class or just one?
python -m fes_lib.cli nan-report fes_lib/data/Stream --schema fes_lib/schemas/tcp_stream.yaml --by-label --rows 0

# which columns actually vary, measured per class?
python -m fes_lib.cli dispersion-by-label fes_lib/data/Stream --schema fes_lib/schemas/tcp_stream.yaml --rows 0
```

The same thing in Python:

```python
from fes_lib.schema import Schema
from fes_lib.io import resolve_schema, load_corpus, class_balance
from fes_lib import clean

policy          = Schema.from_yaml("fes_lib/schemas/tcp_stream.yaml")
schema, headers = resolve_schema("fes_lib/data/Stream", policy)
raw             = load_corpus("fes_lib/data/Stream", schema, binary_label=True)
typed, mask     = clean.decode(raw, schema)

print(class_balance(raw, schema))
print(clean.nan_report(typed, schema, by_label=True))
```

---

## 4. Python API

### `fes_lib.schema`

| Name | Purpose |
|---|---|
| `Schema.from_yaml(path)` | load a config. Validates sections and runs `self_check()`. |
| `Schema.from_dict(cfg)` | same, from an already-parsed mapping. |
| `GENERIC_SCHEMA` | empty policy: no roles, no sentinels, no labels. |
| `schema.role(col)` | the `Role` enum member. |
| `schema.is_target(col)` | is it `label_col` or `binary_col`? |
| `schema.is_structural(col)` | is it in `missingness.structural`? |
| `schema.meta_columns` | label + binary + source + facets. |
| `schema.numeric(cols)` | the NUMERIC subset. |
| `schema.sentinels_for(col)`, `validity_flag_for(col)`, `group_of(col)` | per-column policy lookups. |
| `schema.label_from_filename(name)` | raises if no pattern matches. |
| `schema.is_resolved` | has `resolve_schema` been run? |

`Schema` is a frozen dataclass. To change one field, use
`dataclasses.replace(schema, field=value)` — that is how `resolve_schema`
attaches the inventory.

### `fes_lib.io`

| Function | Returns |
|---|---|
| `list_captures(dir_or_paths, pattern="*.csv")` | sorted list of paths. Raises if empty. |
| `read_header(path)` | list of column names. |
| `compare_headers(paths)` | frame: file, n_columns, missing, extra, reordered, matches. |
| `infer_inventory(paths, on_mismatch="raise")` | `(columns, report)`. Raises or warns on mismatch. |
| `resolve_schema(dir, schema, pattern, on_mismatch)` | `(resolved_schema, header_report)`. |
| `load_corpus(dir, schema, pattern, binary_label=False)` | one frame, with `label`, `capture_id`, facets, optional `binary_state`. |
| `save_corpus(df, out_dir, name)` | writes `<out_dir>/<name>_<timestamp>.csv`, returns the path. |
| `class_balance(df, schema)` | rows, captures, row_share, imbalance_vs_largest per label. |

### `fes_lib.clean`

Every function takes a frame and returns a **new** frame. Nothing is modified in
place, and no source file is ever written to.

**Decode — run this first**

```python
typed, mask = clean.decode(df, schema, add_missing_indicators=True)
```

**Inspect** (return reports, not frames)

| Function | Answers |
|---|---|
| `check_ranges(df, schema)` | which values violate the declared ranges. A non-empty result means the *extractor* is wrong. |
| `degenerate_columns(df, schema, max_missing=0.98)` | `{constant, all_missing, near_all_missing}`. Reports only; drops nothing. |
| `nan_report(df, schema, by_label=False)` | per-column missingness, optionally per class. |
| `dispersion_report(df, schema, bins=20)` | `mode_share`, `entropy_norm`, `n_unique` for every role; `variance`/`range`/`iqr` for numeric only. |
| `dispersion_by_label(df, schema, min_rows=30)` | the same `mode_share`, computed inside each class, plus `min_mode_share` and `varies_in`. |

**Transform** (return frames)

| Function | Notes |
|---|---|
| `drop_columns(df, cols, schema, allow_group=False)` | raises on a meta column. Ignores names that aren't present. |
| `drop_categorical(df, schema)` / `drop_binary(df, schema)` / `drop_non_numeric(df, schema)` | role-based bulk drops. |
| `drop_nan_rows(df, schema, subset=None, min_present=None)` | `min_present=0.8` keeps rows that are ≥80% complete. |
| `drop_low_variance_numeric(df, schema, threshold=0.8)` | compares raw variances across columns, which is not a comparison — a column in different units ranks differently for that reason alone, 0/1 indicators can never exceed 0.25, and an all-NaN column survives because `NaN <= threshold` is `False`. Use `dispersion_report`'s `mode_share` to compare columns; keep this for `--threshold 0.0`. |
| `impute(df, schema, strategy, ...)` | returns `(frame, report)`. |
| `normalize(df, schema, method="zscore")` | `zscore` or `minmax`. |

### `impute` in detail

```python
filled, report = clean.impute(
    typed, schema,
    strategy="median",          # or a dict: {"packets": "zero", "rtt_ms": "mean"}
    default_strategy="skip",    # for columns the dict doesn't name
    const=None,                 # required when strategy == "constant"
    fill_categorical=None,      # default: the column's mode
    max_missing=0.5,            # above this, leave alone and report
    skip_columns=None,
)
```

Strategies: `median`, `mean`, `mode`, `zero`, `constant`, `skip`.

It never touches: targets, group columns, `__missing` indicators, and **anything
declared structural**. The report has one row per column it considered, saying
what it did and why — keep it; it is the record of what you fabricated.

---

## 5. The CLI

One subcommand per function, 24 of them:

```
io:    list-captures  read-header  compare-headers  infer-inventory
       resolve-schema  load-corpus  save-corpus  class-balance
clean: decode  check-ranges  degenerate-columns  drop-columns
       drop-categorical  drop-binary  drop-non-numeric  nan-report
       drop-low-variance-numeric  dispersion-report  dispersion-by-label
       drop-nan-rows  impute  normalize
extra: columns
```

```bash
python -m fes_lib.cli --help              # the list
python -m fes_lib.cli impute --help       # one command's options
```

### Try these

Against the test fixture, from `fen-lib/`. They are read-only — none of them
writes anything.

```bash
S=fes_lib/data/test_3/corpus
Y=--schema=fes_lib/data/test_3/toy_schema.yaml

# --- does the data hold together? -------------------------------------
python -m fes_lib.cli list-captures   $S
python -m fes_lib.cli compare-headers $S
python -m fes_lib.cli infer-inventory $S
python -m fes_lib.cli resolve-schema  $S $Y

# --- what am I looking at? --------------------------------------------
python -m fes_lib.cli load-corpus     $S $Y
python -m fes_lib.cli class-balance   $S $Y
python -m fes_lib.cli columns         $S $Y --rows 0

# --- turn sentinels into NaN ------------------------------------------
python -m fes_lib.cli decode          $S $Y --rows 5
python -m fes_lib.cli decode          $S $Y --quiet

# --- is anything wrong with the extraction? ---------------------------
python -m fes_lib.cli check-ranges    $S $Y
python -m fes_lib.cli check-ranges    fes_lib/data/test_3/bad_range $Y
python -m fes_lib.cli infer-inventory fes_lib/data/test_3/bad_header

# --- what is missing, and where? --------------------------------------
python -m fes_lib.cli nan-report      $S $Y --rows 0
python -m fes_lib.cli nan-report      $S $Y --by-label --rows 0
python -m fes_lib.cli degenerate-columns $S $Y

# --- which columns carry anything? ------------------------------------
python -m fes_lib.cli dispersion-report   $S $Y --rows 0
python -m fes_lib.cli dispersion-by-label $S $Y --rows 0
python -m fes_lib.cli dispersion-by-label $S $Y --min-rows 10 --rows 0

# --- thin the frame out (printed, not saved) --------------------------
python -m fes_lib.cli drop-categorical $S $Y --quiet
python -m fes_lib.cli drop-binary      $S $Y --quiet
python -m fes_lib.cli drop-non-numeric $S $Y --quiet
python -m fes_lib.cli drop-columns     $S $Y --columns src_ip stream_id --quiet
python -m fes_lib.cli drop-nan-rows    $S $Y --min-present 0.9 --quiet

# --- fill and scale ---------------------------------------------------
python -m fes_lib.cli impute    $S $Y --quiet
python -m fes_lib.cli impute    $S $Y --strategy mean --quiet
python -m fes_lib.cli impute    $S $Y --column-strategy packets=zero --default-strategy skip --quiet
python -m fes_lib.cli normalize $S $Y --quiet
python -m fes_lib.cli normalize $S $Y --method minmax --quiet

# --- keep a report, or keep the frame ---------------------------------
python -m fes_lib.cli nan-report $S $Y --rows 0 --csv missing.csv
python -m fes_lib.cli impute     $S $Y --out fes_lib/data/test_3/work --name filled
```

Swap `$S` and `$Y` for your own directory and config to run the same sequence on
real data. On Windows use Git Bash, or spell the paths out.

Shared options: `--schema`, `--pattern`, `--on-mismatch`, `--no-binary`,
`--raw`, `--from-csv`, `--rows N` (0 = all), `--quiet`, `--csv FILE`.

**Nothing is saved unless you ask.** Each command is its own process; the result
is printed and discarded. `--out DIR` writes the resulting *frame*;
`--csv FILE` writes the printed *table*. Your source CSVs are never modified.

Chaining is explicit, one step at a time:

```bash
python -m fes_lib.cli decode    data/Stream --schema schemas/tcp_stream.yaml --out work --name s1
python -m fes_lib.cli impute    --from-csv work/s1_<ts>.csv --schema schemas/tcp_stream.yaml --out work --name s2
python -m fes_lib.cli normalize --from-csv work/s2_<ts>.csv --schema schemas/tcp_stream.yaml --out work --name s3
```

`--from-csv` does not re-decode (the file already is), and clears `binary_col`
so a saved frame doesn't collide with `self_check`.

Exit codes: `0` fine · `1` ran but found a problem (range violations, a column
destroyed by normalisation) · `2` could not run.

`columns` has no function behind it — it prints the role, structural flag,
sentinels, validity flag and group the schema assigned to each column. It is the
fastest way to debug a config.

---

## 6. Recommended order

1. **`resolve-schema`** — do the files agree? If not, fix the extraction. Do not
   paper over it with `--on-mismatch warn`.
2. **`class-balance`** — how unbalanced is this? Everything after depends on the
   answer.
3. **`columns`** — did the config classify everything the way you meant?
4. **`decode`** — sentinels to NaN, indicators added.
5. **`check-ranges`** — any violation is an extractor bug, found before it
   becomes an analysis bug.
6. **`nan-report --by-label`** — which absences are data quality and which are
   signal.
7. **`dispersion-by-label`** — which columns carry anything, measured per class.
8. **`degenerate-columns`** — read the report; decide the drops yourself.
9. *then* split, *then* impute and normalise, inside the split.

Steps 1–8 are read-only. You can run them all and still have changed nothing.

---

## 7. Adapting it to a new extractor

1. Copy `schemas/template.yaml`.
2. `python -m fes_lib.cli read-header <one file>` and `infer-inventory <dir>` to
   see what you are dealing with.
3. Fill in `roles`, then `labels.patterns` (lowercase).
4. `python -m fes_lib.cli columns <dir> --schema <new.yaml> --rows 0` and check
   every column got the role you intended.
5. Add `sentinels` — the honest way to find them is `dispersion-report` and
   looking for implausible minimums.
6. Decide `missingness.structural`: for each sentinel column, ask whether the
   value *could* have existed. If not, it is structural.
7. `constraints.ranges` last, as a tripwire for future extractor changes.

No Python changes required at any step. If you find yourself wanting to edit
`schema.py` to support a new format, that is a sign the config is missing a
section — say so rather than hardcoding it.
