"""Purged-CV report for a panel model on the training window.

uv run python skills/models/scripts/cv_report.py --model ridge
uv run python skills/models/scripts/cv_report.py --model gbm --data cache

Prints per-fold IC under purged, embargoed k-fold. Nothing here touches the hold-out.
"""

import argparse

import numpy as np
from sklearn.linear_model import Ridge

from quantskills.data import load_cached, synthetic_prices, to_wide
from quantskills.features import feature_panel, label_panel
from quantskills.models import cv_score, dropna_xy, fit_lgbm_cv, shap_importance
from quantskills.utils import holdout_date


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", choices=["ridge", "gbm"], default="ridge")
    p.add_argument("--data", choices=["synthetic", "cache"], default="synthetic")
    p.add_argument("--horizon", type=int, default=21)
    p.add_argument("--splits", type=int, default=4)
    p.add_argument("--test-fraction", type=float, default=0.3)
    a = p.parse_args()

    prices = load_cached() if a.data == "cache" else synthetic_prices(seed=0)
    dates = prices.index.get_level_values("date")
    train = prices[dates <= holdout_date(dates, a.test_fraction)]
    close = to_wide(train, "close")
    X, y = dropna_xy(feature_panel(close), label_panel(close, a.horizon))
    print(f"{len(X):,} training rows, {X.shape[1]} features, horizon {a.horizon}d\n")

    if a.model == "ridge":
        for alpha in (1.0, 100.0, 10_000.0):
            ics = cv_score(lambda alpha=alpha: Ridge(alpha=alpha), X, y, a.splits, a.horizon)
            print(f"ridge alpha={alpha:>8g}  fold IC {np.round(ics, 3)}  mean {np.mean(ics):+.4f}")
    else:
        booster, rep = fit_lgbm_cv(X, y, n_splits=a.splits, horizon=a.horizon)
        print(f"fold IC {np.round(rep['fold_ic'], 3)}")
        print(f"early-stopping rounds per fold {rep['fold_rounds']} -> {rep['rounds']}")
        print("\nmean |SHAP|:")
        print(shap_importance(booster, X).round(5).to_string())


if __name__ == "__main__":
    main()
