

import re
import numpy as np
import pandas as pd
from .schema import Schema, Role


def feature_independence_report(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Report on how much each column varies independently of the others.
 
    For every column, compute the R^2 of a linear regression predicting it from
    all the other columns. 1.0 means perfectly predictable, 0.0 means no
    relationship. This is a measure of redundancy: if a column is perfectly
    predictable from others, it adds no information.
 
    Only numeric columns are tested; categorical columns are ignored.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
 
    cols = [c for c in df.columns if schema.role(c) is Role.NUMERIC and not c in schema.meta_columns]
    out = []
    for col in cols:
        X = df.drop(columns=col)
        y = df[col]
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        out.append({"column": col, "r2": r2})
    return pd.DataFrame(out).sort_values("r2", ascending=False)