import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

def fit_ridge(features_hist: pd.DataFrame, fwd_returns: pd.Series, alpha: float = 10.0) -> pd.Series:
    df = features_hist.dropna()
    common = df.index.intersection(fwd_returns.dropna().index)
    X = df.loc[common].values
    y = fwd_returns.loc[common].values
    if len(common) < 20:
        # Not enough data; default weights equal
        return pd.Series(1.0 / max(1, df.shape[1]), index=df.columns, name="weight")
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)
    w = pd.Series(model.coef_, index=df.columns, name="weight")
    return w

def score_current(features_now: pd.DataFrame, weights: pd.Series) -> pd.Series:
    cols = [c for c in weights.index if c in features_now.columns]
    X = features_now[cols].fillna(0.0)
    s = X.dot(weights.loc[cols])
    return s.sort_values(ascending=False).rename("score")
