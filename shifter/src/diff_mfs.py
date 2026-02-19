"""Differentiable meta-feature extraction via PyTorch.

Supports:
  - IQR via torch.quantile
  - MI via Gaussian approximation of Pearson correlation
  - Mean via torch.mean
  - Median via torch.quantile
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def iqr_torch(X: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Inter-quartile range per column (differentiable)."""
    return torch.quantile(X, 0.75, dim=dim) - torch.quantile(X, 0.25, dim=dim)


def mean_torch(X: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Mean per column (differentiable)."""
    return torch.mean(X, dim=dim)


def median_torch(X: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Median per column (differentiable)."""
    return torch.quantile(X, 0.5, dim=dim)


def mutual_info_gaussian(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Gaussian MI: MI(X,Y) = -0.5 * log(1 - rho^2)."""
    x_c = x - x.mean(dim=0, keepdim=True)
    y_c = y - y.mean(dim=0, keepdim=True)
    rho = (x_c * y_c).mean(dim=0) / (x_c.std(dim=0) * y_c.std(dim=0) + 1e-8)
    rho = torch.clamp(rho, -0.999, 0.999)
    return -0.5 * torch.log(1 - rho ** 2 + 1e-8)


def mutual_info_per_col(X: torch.Tensor) -> torch.Tensor:
    """Average pairwise Gaussian MI for each column."""
    n_cols = X.shape[1]
    if n_cols < 2:
        return torch.zeros(n_cols, device=X.device, dtype=X.dtype)
    mi = torch.zeros(n_cols, device=X.device, dtype=X.dtype)
    for i in range(n_cols):
        s = sum(mutual_info_gaussian(X[:, i:i+1], X[:, j:j+1]) for j in range(n_cols) if j != i)
        mi[i] = s / (n_cols - 1)
    return mi


def compute_diff_mfs(
    X: torch.Tensor,
    features: List[str],
    summary: Optional[str] = "mean",
) -> Tuple[torch.Tensor, List[str]]:
    """Compute differentiable meta-features on a tensor.

    Parameters
    ----------
    X        : [n_samples, n_features]
    features : list of "iq_range" / "mut_inf" / "mean" / "median"
    summary  : "mean" or None (per-column)

    Returns
    -------
    values : Tensor [m_dim]
    names  : list[str]
    """
    values: List[torch.Tensor] = []
    names: List[str] = []

    for feat in features:
        if feat == "iq_range":
            per_col = iqr_torch(X, dim=0)
        elif feat == "mut_inf":
            per_col = mutual_info_per_col(X)
        elif feat == "mean":
            per_col = mean_torch(X, dim=0)
        elif feat == "median":
            per_col = median_torch(X, dim=0)
        else:
            raise ValueError(f"Unknown differentiable meta-feature: {feat!r}")

        if summary is None:
            for idx in range(len(per_col)):
                values.append(per_col[idx:idx+1])
                names.append(f"{feat}.col{idx}")
        elif summary == "mean":
            values.append(per_col.mean().unsqueeze(0))
            names.append(f"{feat}.mean")
        else:
            raise ValueError(f"Unknown summary: {summary!r}")

    if not values:
        return torch.tensor([], device=X.device, dtype=X.dtype), []
    return torch.cat(values, dim=0), names
