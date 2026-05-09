"""
utils.py
Shared helpers: reproducibility, metrics, plotting, logging.
"""

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Survival metrics
# ---------------------------------------------------------------------------

def compute_cindex(risk_scores: np.ndarray,
                   times: np.ndarray,
                   events: np.ndarray) -> float:
    """
    Compute Harrell's concordance index.
    risk_scores: higher = higher risk (predicted hazard).
    events: 1 = event occurred, 0 = censored.
    """
    from lifelines.utils import concordance_index
    # lifelines expects: higher score → shorter survival, so pass -risk_scores
    # as the "predicted survival time"
    valid = ~np.isnan(risk_scores) & ~np.isnan(times)
    if valid.sum() == 0 or events[valid].sum() == 0:
        return float("nan")
    return concordance_index(times[valid], -risk_scores[valid], events[valid])


# ---------------------------------------------------------------------------
# Kaplan-Meier plotting
# ---------------------------------------------------------------------------

def plot_km_curves(risk_scores: np.ndarray,
                   times: np.ndarray,
                   events: np.ndarray,
                   title: str = "Kaplan-Meier Curves",
                   save_path: Path | None = None) -> None:
    """
    Plot KM curves for high- vs low-risk groups (median split) and run
    the log-rank test.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test

    median_risk = np.nanmedian(risk_scores)
    high_risk = risk_scores >= median_risk
    low_risk  = ~high_risk

    kmf_high = KaplanMeierFitter()
    kmf_low  = KaplanMeierFitter()
    kmf_high.fit(times[high_risk],  events[high_risk],  label="High risk")
    kmf_low.fit( times[low_risk],   events[low_risk],   label="Low risk")

    results = logrank_test(times[high_risk], times[low_risk],
                           events[high_risk], events[low_risk])
    p_val = results.p_value

    fig, ax = plt.subplots(figsize=(7, 5))
    kmf_high.plot_survival_function(ax=ax)
    kmf_low.plot_survival_function(ax=ax)
    ax.set_title(f"{title}  (log-rank p={p_val:.4f})")
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival probability")
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cox partial likelihood loss (Breslow approximation)
# ---------------------------------------------------------------------------

class CoxPHLoss(torch.nn.Module):
    """
    Negative Cox partial log-likelihood with Breslow approximation.
    Input:
        risk  : (N,) predicted log-hazard scores (higher = more risk)
        times : (N,) observed survival times
        events: (N,) event indicators (1 = event, 0 = censored)
    Expects the batch to be sorted in DESCENDING order of time externally,
    or we sort internally here.
    """

    def forward(self,
                risk: torch.Tensor,
                times: torch.Tensor,
                events: torch.Tensor) -> torch.Tensor:
        # Sort descending by time so risk set R(t_i) = all j with indices >= i
        order = torch.argsort(times, descending=True)
        risk   = risk[order]
        events = events[order]

        log_cumsum_exp = torch.logcumsumexp(risk, dim=0)
        # Only sum over event times
        loss = -torch.mean((risk - log_cumsum_exp)[events.bool()])
        return loss


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(path, index=False)
