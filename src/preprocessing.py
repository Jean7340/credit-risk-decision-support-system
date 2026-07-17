"""
Shared preprocessing utilities for the HELOC Credit Risk project.

This module is imported by BOTH the training notebook and the Streamlit app,
so the exact same special-value logic is applied at training time and at
inference time (no train/serve skew).

FICO HELOC data dictionary special codes:
    -9 : No Bureau Record or No Investigation
    -8 : No Usable / Valid Trades or Inquiries
    -7 : Condition Not Met (e.g. no delinquency has ever occurred)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SPECIAL_VALUES = [-7, -8, -9]

# Columns where -7 ("Condition Not Met") means the event NEVER happened,
# which is a LOW-risk signal (e.g. never delinquent). Filling these with the
# median would wrongly turn the safest applicants into "average" ones, so we
# instead fill with the maximum observed valid value (event infinitely long
# ago) and add an indicator flag.
NEVER_HAPPENED_COLS = ["MSinceMostRecentDelq", "MSinceMostRecentInqexcl7days"]


def drop_no_record_rows(df: pd.DataFrame, target_col: str = "RiskPerformance") -> pd.DataFrame:
    """Remove rows where ALL features are -9 (no bureau record at all).

    These rows carry no usable information and are conventionally dropped
    in published work on the FICO HELOC dataset (~588 rows).
    """
    feature_cols = [c for c in df.columns if c != target_col]
    all_minus9 = (df[feature_cols] == -9).all(axis=1)
    print(f"Dropped {int(all_minus9.sum())} rows with no bureau record "
          f"({all_minus9.mean():.1%} of the data)")
    return df.loc[~all_minus9].copy()


class SpecialValueTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer for the FICO special codes.

    Designed to be the FIRST step inside a Pipeline, so that the saved
    pipeline accepts raw feature values end-to-end:

        Pipeline([
            ("special", SpecialValueTransformer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),          # linear models only
            ("model",   LogisticRegression(...)),
        ])

    Behaviour
    ---------
    fit:
        * records the input feature names,
        * learns (from TRAINING data only) the fill value for -7 in the
          NEVER_HAPPENED_COLS (max of valid values),
        * records which columns contain -8/-9 so indicator columns are
          created consistently at inference time.
    transform:
        * adds <col>_NeverHappened indicator and fills -7 in those columns,
        * adds <col>_WasSpecial indicators for columns that had -8/-9,
        * converts all remaining special codes to NaN (downstream imputer
          fills them).
    """

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.feature_names_in_ = list(X.columns)

        self.never_fill_ = {}
        for col in NEVER_HAPPENED_COLS:
            if col in X.columns:
                valid = X.loc[~X[col].isin(SPECIAL_VALUES), col]
                self.never_fill_[col] = float(valid.max())

        self.special_indicator_cols_ = [
            c for c in X.columns if X[c].isin([-8, -9]).any()
        ]
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_).copy()

        # 1) -7 = "never happened" -> indicator + fill with train-time max
        for col, fill in self.never_fill_.items():
            X[col + "_NeverHappened"] = (X[col] == -7).astype(int)
            X[col] = X[col].replace(-7, fill)

        # 2) -8 / -9 = "information missing" -> indicator, learned on train
        for col in self.special_indicator_cols_:
            X[col + "_WasSpecial"] = X[col].isin([-8, -9]).astype(int)

        # 3) any remaining special codes -> NaN for the downstream imputer
        original = self.feature_names_in_
        X[original] = X[original].mask(X[original].isin(SPECIAL_VALUES), np.nan)

        return X

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_)
        names += [c + "_NeverHappened" for c in self.never_fill_]
        names += [c + "_WasSpecial" for c in self.special_indicator_cols_]
        return np.asarray(names, dtype=object)
