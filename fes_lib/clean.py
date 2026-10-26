import re
import numpy as np
import pandas as pd
from .schema import Schema, Role

# --------------------------------------------------------------------------
# Create different reports on the dataframe
# --------------------------------------------------------------------------

def check_ranges(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col in schema.range_exempt or schema.role(col) is not Role.NUMERIC:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        for pattern, lo, hi in schema.range_constraints:
            if re.match(pattern, col):
                bad = ((s < lo) | (s > hi)) & s.notna()
                if bad.any():
                    rows.append({"column": col, "rule": f"[{lo}, {hi}]",
                                 "n_violations": int(bad.sum()),
                                 "example": float(s[bad].iloc[0])}) # takes first bad value in column as example
    return pd.DataFrame(rows)





def degenerate_columns(df: pd.DataFrame, schema: Schema,
                       max_missing: float = 0.98) -> dict[str, list[str]]:
    """
    Columns to drop before correlation: constant, all-missing, or so sparse
    that any coefficient computed from them is noise.
    Note that this function does not drop columns, it just returns a report of 
    which columns are degenerate.
    """
    const, empty, sparse = [], [], []
    for col in df.columns:
        if col in schema.meta_columns:
            continue
        s = df[col]
        frac_missing = s.isna().mean()
        if frac_missing == 1.0:
            empty.append(col)
        elif frac_missing > max_missing:
            sparse.append(col)
        elif s.nunique(dropna=True) <= 1:
            const.append(col)
    return {"constant": const, "all_missing": empty, "near_all_missing": sparse}




def nan_report(df: pd.DataFrame, schema: Schema,
               by_label: bool = False) -> pd.DataFrame:
    """Per-column missingness.
 
    With by_label=True, also the rate within each class -- which is the more
    useful view: a column missing 100% in benign and 0% in scan traffic is not a
    data-quality problem, it is the signal.
    """
    cols = [c for c in df.columns if c not in schema.meta_columns]
    out = pd.DataFrame({
        "n_missing": df[cols].isna().sum(),
        "percent_missing": df[cols].isna().mean().mul(100).round(2),
        "n_present": df[cols].notna().sum(),
        "role": [schema.role(c).value for c in cols],
    })
    if by_label and schema.label_col in df.columns:
        per = df.groupby(schema.label_col)[cols].apply(
            lambda g: g.isna().mean().mul(100).round(1))
        out = out.join(per.T.add_prefix("percent_"))
    return out.sort_values("percent_missing", ascending=False)



def _mode_share(s: pd.Series) -> float:
    s = s.dropna()
    if s.empty:
        return np.nan
    return float(s.value_counts(normalize=True).iloc[0])

def dispersion_report(df: pd.DataFrame, schema: Schema, bins: int = 20) -> pd.DataFrame:
    """How much each column varies -- for every role, not just numeric.
 
    Variance is undefined for strings and incomparable across scales (a byte
    count varies by 1e9, a ratio by 0.01), so it cannot answer "is this column
    informative". These can:
 
      mode_share      fraction of rows holding the most common value. 1.0 means
                      constant; 0.999 means 10 rows in 10,000 differ. Defined
                      for every dtype and directly comparable across columns.
      entropy_norm    Shannon entropy / log(n_levels), in [0, 1]. 0 = constant,
                      1 = perfectly even spread. Numeric columns are binned
                      first. Comparable across roles.
      n_unique        level count (categorical) or distinct values (numeric).
      variance        kept for numeric columns, NaN elsewhere, because it is
                      still the right thing for a single column in its own units.
    """
    rows = []
    for col in df.columns:
        if col in schema.meta_columns:
            continue
        s = df[col].dropna()
        role = schema.role(col)
        if s.empty:
            rows.append({"column": col, "role": role.value, "n_unique": 0,
                         "mode_share": np.nan, "entropy_norm": np.nan,
                         "variance": np.nan, "range": np.nan, "iqr": np.nan, "note": "all missing"})
            continue
 
        if role is Role.NUMERIC and s.nunique() > bins:
            binned = pd.cut(s, bins=bins, duplicates="drop")
            counts = binned.value_counts(normalize=True)
        else:
            counts = s.value_counts(normalize=True)
 
        k = len(counts)
        entropy = -(counts * np.log(counts)).sum()
        rows.append({
            "column": col,
            "role": role.value,
            "n_unique": int(s.nunique()),
            "mode_share": round(float(s.value_counts(normalize=True).iloc[0]), 4),
            "entropy_norm": round(float(entropy / np.log(k)), 4) if k > 1 else 0.0,
            "variance": round(float(s.var()), 6) if role is Role.NUMERIC else np.nan,
            "range": round(float(s.max() - s.min()), 6) if role is Role.NUMERIC else np.nan,
            "iqr": round(float(s.quantile(0.75) - s.quantile(0.25)), 6) if role is Role.NUMERIC else np.nan,
            "note": "",
        })
    return pd.DataFrame(rows).sort_values("entropy_norm").reset_index(drop=True)


def dispersion_by_label(df: pd.DataFrame, schema: Schema,
                        min_rows: int = 30) -> pd.DataFrame:
    """mode_share computed separately within each class.
 
    The overall mode_share of a column is dominated by whichever class has most
    of the rows. On a 10270 : 4 : 1 corpus, a column that varies perfectly inside
    the benign captures still reads as 99.96% constant, because only 4 rows out
    of 10,275 are benign.
 
    `min_mode_share` is the smallest value across classes -- i.e. how much the
    column varies in the class where it varies most. That is the number to
    threshold on: a feature only needs to discriminate somewhere.
 
    Classes with fewer than `min_rows` rows are computed but flagged, because a
    mode_share from 4 observations is not an estimate of anything.
    """
    label = schema.label_col
    cols = [c for c in df.columns if c not in schema.meta_columns]
    counts = df[label].value_counts()
 
    per = df.groupby(label)[cols].agg(_mode_share).T
    per.columns = [f"mode_share_{c}" for c in per.columns]
 
    out = pd.DataFrame({
        "role": [schema.role(c).value for c in cols],
        "mode_share_overall": [_mode_share(df[c]) for c in cols],
    }, index=cols).join(per)
 
    share_cols = [c for c in out.columns if c.startswith("mode_share_")
                  and c != "mode_share_overall"]
    out["min_mode_share"] = out[share_cols].min(axis=1)
    out["varies_in"] = out[share_cols].lt(0.99).sum(axis=1)
 
    thin = [str(k) for k, v in counts.items() if v < min_rows]
    if thin:
        out.attrs["warning"] = (f"classes with <{min_rows} rows: {thin} -- their "
                                f"mode_share is not a reliable estimate")
    return out.sort_values("min_mode_share")
 
# --------------------------------------------------------------------------
# Methods to drop columns/rows based on criteria specified
# --------------------------------------------------------------------------


def drop_nan_rows(df: pd.DataFrame, schema: Schema, subset=None,
                  min_present: float | None = None) -> pd.DataFrame:
    """
    Drop rows with missing values.
 
    `subset` limits the test to those columns; `min_present` keeps a row if at
    least that fraction of the tested columns are present.
    """
    cols = list(subset) if subset is not None else [
        c for c in df.columns
        if not schema.is_target(c)
        and schema.role(c) is not Role.GROUP
        and not c.endswith("__missing")]
    if min_present is None:
        return df.dropna(subset=cols)
    thresh = int(np.ceil(min_present * len(cols)))
    return df.dropna(subset=cols, thresh=thresh)

def drop_columns(df: pd.DataFrame, columns: list[str], schema: Schema, allow_group: bool = False) -> pd.DataFrame:
    """
    Drop columns from a dataframe, ignoring any that are not present.
    """
    if allow_group is False:
        for c in columns:
            if c in schema.meta_columns:
                raise ValueError(f"Cannot drop meta column {c}")

            else:
                df = df.drop(columns=c, errors="ignore")

        return df

    else:
        for c in columns:
            if c == schema.label_col or c == schema.binary_col:
                raise ValueError(f"Cannot drop label column {c}")   
            else:
                df = df.drop(columns=c, errors="ignore")

        return df

def drop_low_variance_numeric(df: pd.DataFrame, schema: Schema, threshold: float = 0.8) -> pd.DataFrame:
    """Drop columns with variance below threshold."""
    cols = [c for c in df.columns if c not in schema.meta_columns and schema.role(c) is Role.NUMERIC]
    low_var_cols = df[cols].var()[df[cols].var() <= threshold].index.tolist()
    return drop_columns(df, low_var_cols, schema)


def drop_categorical(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """
    Drop all categorical columns from a dataframe.
    """
    cat_cols = [c for c in df.columns if schema.role(c) is Role.CATEGORICAL]
    return drop_columns(df, cat_cols, schema)

def drop_binary(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """
    Drop the binary column from a dataframe.
    """
    bin_cols = [c for c in df.columns if schema.role(c) is Role.BINARY]
    return drop_columns(df, bin_cols, schema)

def drop_non_numeric(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """
    Drop all non-numeric columns from a dataframe.
    """
    non_numeric_cols = [c for c in df.columns if schema.role(c) is not Role.NUMERIC and c not in schema.meta_columns]
    return drop_columns(df, non_numeric_cols, schema)

# --------------------------------------------------------------------------
# Methods to perform actions on columns
# --------------------------------------------------------------------------

def convert_to_nan(df: pd.DataFrame, schema: Schema,
           add_missing_indicators: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return (typed_frame, missing_mask).
 
    Sentinels become NaN. The mask is returned separately because *the fact that
    a value was absent is often the stronger signal than the value itself*.

    miss = (
    isna()
    OR sentinel
    OR validity_flag==0
)

Create feature__missing only if that feature has 
at least one missing/sentinel/invalid value somewhere in the dataframe.
    """
    out = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=[])
 
    for col in df.columns:
        if col in schema.meta_columns:
            continue
        role = schema.role(col)
 
        if role is Role.CATEGORICAL or role is Role.IDENTIFIER:
            s = out[col].astype("string").str.strip()
            miss = s.str.lower().isin([v.lower() for v in schema.string_sentinels])
            out[col] = s.where(~miss, np.nan)
        else:
            s = pd.to_numeric(out[col], errors="coerce")
            miss = s.isna()
            for sentinel in schema.sentinels_for(col):
                miss |= s.eq(sentinel)
            flag = schema.validity_flag_for(col)
            if flag is not None and flag in df.columns:
                miss |= pd.to_numeric(df[flag], errors="coerce").eq(0)
            out[col] = s.where(~miss, np.nan)
 
        if miss.any():
            mask[col] = miss
 
    if add_missing_indicators and len(mask.columns):
        ind = mask.astype(int).add_suffix("__missing")
        out = pd.concat([out, ind], axis=1)
 
    return out, mask 

_STRATEGIES = ("median", "mean", "mode", "zero", "constant", "skip")

def impute(df: pd.DataFrame, schema: Schema, strategy= "median", const: float | None = None,
           max_missing: float = 0.5,
           fill_categorical: str | None = None,
           default_strategy: str = "skip",
           skip_columns: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill missing values. Returns (frame, report of what was filled).
 
    strategy: "median" | "mean" | "zero" | "constant"| "mode" | for numeric columns.
    
    Columns missing more than `max_missing` are LEFT ALONE and reported. 
    
    Categorical columns are filled with `fill_categorical` if provided, 
    otherwise filled with the most frequent value.
 
    The `<col>__missing` indicators from decode() are never filled
    """
    names = [strategy] if isinstance(strategy, str) else list(strategy.values())
    bad = sorted({n for n in names if n not in _STRATEGIES})
    if bad:
        raise ValueError(f"unknown strategy name(s) {bad}; expected {list(_STRATEGIES)}")
 
 
    out = df.copy()
    rows = []
    cols = [c for c in df.columns
            if not schema.is_target(c)
            and (skip_columns is None or c not in skip_columns)
            and not schema.is_structural(c)
            and schema.role(c) is not Role.GROUP
            and not c.endswith("__missing")]
 
    for col in cols:
        frac = df[col].isna().mean()
        if frac == 0:
            continue
        role = schema.role(col)
 
        if frac > max_missing:
            rows.append({"column": col, "role": role.value, "percent_missing": round(frac*100, 2),
                         "action": "left as NaN", "value": None,
                         "note": f"above max_missing={max_missing}"})
            continue

        if isinstance(strategy, dict) and col in strategy:
            strategy_chosen = strategy.get(col, default_strategy)
        else:
            strategy_chosen = strategy if isinstance(strategy, str) else default_strategy

        if strategy_chosen == "skip":
            rows.append({"column": col, "role": role.value, "percent_missing": round(frac*100, 2),
                         "action": "left as NaN", "value": None,
                         "note": f"strategy=skip"})
            continue
    
        if role is Role.CATEGORICAL:
            if fill_categorical is None:
                value = df[col].mode(dropna=True).iloc[0]
            else:
                value = fill_categorical
        elif strategy_chosen == "median":
            value = df[col].median()
        elif strategy_chosen == "mean":
            value = df[col].mean()
        elif strategy_chosen == "mode":
            value = df[col].mode(dropna=True).iloc[0]
        elif strategy_chosen == "constant":
            if const is None:
                raise ValueError("const must be provided for constant strategy")
            value = const
        else:
            value = 0
 
        out[col] = df[col].fillna(value)
        rows.append({"column": col, "role": role.value, "percent_missing": round(frac*100, 2),
                     "action": f"filled ({strategy_chosen if role is not Role.CATEGORICAL else 'categorical'})",
                     "value": value, "note": ""})
 
    return out, pd.DataFrame(rows, columns=["column","role","percent_missing","action","value","note"])


def normalize(df: pd.DataFrame, schema: Schema, method: str = "zscore") -> pd.DataFrame:
    """Normalize numeric columns to zero mean and unit variance (z-score) or [0, 1] range (min-max)."""
    cols = [c for c in df.columns if schema.role(c) is Role.NUMERIC and not c in schema.meta_columns]
    out = df.copy()
    for col in cols:
        s = df[col]
        if method == "zscore":
            mean = s.mean()
            std = s.std()
            if std == 0 or pd.isna(std):
                continue          # leave constant columns alone
            out[col] = (s - mean) / std
        elif method == "minmax":
            min_val = s.min()
            max_val = s.max()
            if max_val == min_val or pd.isna(min_val):
                continue
            out[col] = (s - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"unknown normalization method {method}")
    return out


