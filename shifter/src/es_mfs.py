"""Meta-feature extraction via PyMFE."""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pymfe.mfe import MFE


def compute_mfe(
    df_or_array: Union[pd.DataFrame, np.ndarray],
    features: Sequence[str],
    summary: Union[str, List[str]] = "mean",
    y: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    if isinstance(summary, str):
        summary = [summary]

    X = (df_or_array.select_dtypes(include=[np.number]).values.astype(np.float64)
         if isinstance(df_or_array, pd.DataFrame)
         else np.asarray(df_or_array, dtype=np.float64))
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfe = MFE(features=list(features), summary=summary)
        mfe.fit(X, y)
        names, values = mfe.extract()

    values = np.nan_to_num(np.array([float(v) for v in values], dtype=np.float64),
                           nan=0.0, posinf=1e10, neginf=-1e10)
    return values, list(names)
