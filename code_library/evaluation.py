"""
evaluation.py
-------------
Shared metric functions and comparison-table utilities for both domains.

Metrics
-------
rmse          Root mean squared error.
mae           Mean absolute error.
r2            Coefficient of determination.
nasa_score    Asymmetric NASA CMAPSS scoring function (penalises late
              predictions more heavily than early ones).

Table helpers
-------------
evaluate_cmapss   Returns a metric dict for one CMAPSS model run.
evaluate_naval    Returns a metric dict for one Naval model run (two targets).
make_comparison_table   Assemble a styled Pandas DataFrame for display.
plot_parity       Scatter plot of predicted vs. actual RUL / decay values.
plot_residuals    Residual histogram for a model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(r2_score(y_true, y_pred))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA asymmetric CMAPSS score.

    Mathematically:
        d_i = y_pred_i − y_true_i
        s_i = exp(−d_i / 13) − 1   if  d_i < 0  (early prediction, smaller penalty)
              exp( d_i / 10) − 1   if  d_i ≥ 0  (late prediction, larger penalty)
        S = Σ s_i

    A lower score is better. Perfect prediction → S = 0.
    """
    d = np.asarray(y_pred) - np.asarray(y_true)
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(scores.sum())


# ---------------------------------------------------------------------------
# Per-model evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_cmapss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "Model",
    split: str = "Test",
) -> dict:
    """Return an evaluation dict for one CMAPSS model / split combination."""
    y_pred_clipped = np.clip(y_pred, 0, None)   # RUL cannot be negative
    return {
        "Model": name,
        "Split": split,
        "RMSE": round(rmse(y_true, y_pred_clipped), 3),
        "MAE": round(mae(y_true, y_pred_clipped), 3),
        "R²": round(r2(y_true, y_pred_clipped), 4),
        "NASA Score": round(nasa_score(y_true, y_pred_clipped), 1),
    }


def evaluate_naval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str = "Model",
    split: str = "Test",
) -> dict:
    """Return an evaluation dict for one Naval model / split.

    Expects y_true / y_pred with shape (n, 2) where column 0 = kMc, column 1 = kMt.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "Model": name,
        "Split": split,
        "RMSE_kMc": round(rmse(y_true[:, 0], y_pred[:, 0]), 6),
        "MAE_kMc": round(mae(y_true[:, 0], y_pred[:, 0]), 6),
        "R²_kMc": round(r2(y_true[:, 0], y_pred[:, 0]), 5),
        "RMSE_kMt": round(rmse(y_true[:, 1], y_pred[:, 1]), 6),
        "MAE_kMt": round(mae(y_true[:, 1], y_pred[:, 1]), 6),
        "R²_kMt": round(r2(y_true[:, 1], y_pred[:, 1]), 5),
    }


# ---------------------------------------------------------------------------
# Comparison table builder
# ---------------------------------------------------------------------------

def make_comparison_table(
    records: list[dict],
    highlight: bool = True,
) -> pd.io.formats.style.Styler:
    """Build a colour-highlighted comparison DataFrame from a list of metric dicts.

    Parameters
    ----------
    records   : list of dicts produced by evaluate_cmapss or evaluate_naval.
    highlight : if True, green-highlight best values, red-highlight worst.

    Returns
    -------
    Pandas Styler object (display-ready in Jupyter).
    """
    df = pd.DataFrame(records)

    if not highlight:
        return df.style

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Determine whether a higher or lower value is better for each metric
    higher_is_better = {col: True for col in numeric_cols if col.startswith("R²")}
    lower_is_better = {
        col: True
        for col in numeric_cols
        if any(col.startswith(p) for p in ("RMSE", "MAE", "NASA"))
    }

    def _highlight(s: pd.Series) -> list[str]:
        col = s.name
        if col in higher_is_better:
            best_idx, worst_idx = s.idxmax(), s.idxmin()
        elif col in lower_is_better:
            best_idx, worst_idx = s.idxmin(), s.idxmax()
        else:
            return [""] * len(s)
        styles = [""] * len(s)
        styles[best_idx] = "background-color: #c6efce; color: #276221; font-weight: bold"
        styles[worst_idx] = "background-color: #ffc7ce; color: #9c0006; font-weight: bold"
        return styles

    styled = df.style.apply(_highlight, subset=numeric_cols)
    return styled


# ---------------------------------------------------------------------------
# Diagnostic visualisations
# ---------------------------------------------------------------------------

def plot_parity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predicted vs. Actual",
    xlabel: str = "Actual",
    ylabel: str = "Predicted",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Scatter plot of predicted vs. actual values with a 45° parity line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.4, s=12, color="#1C4E80", rasterized=True)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    return ax


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Distribution",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Histogram of residuals (y_pred − y_true)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    residuals = y_pred - y_true
    ax.hist(residuals, bins=50, color="#0E6B8A", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Residual (predicted − actual)")
    ax.set_ylabel("Count")
    ax.set_title(title)
    return ax


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    title: str = "Training Curves",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot MSE training and validation loss curves."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label="Train MSE", color="#1C4E80")
    if val_losses:
        ax.plot(val_losses, label="Val MSE", color="#F0A500")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(title)
    ax.legend()
    return ax
