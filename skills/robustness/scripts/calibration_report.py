"""Calibration of a direction model's probabilities, scored walk-forward on the training window.

uv run python skills/robustness/scripts/calibration_report.py
uv run python skills/robustness/scripts/calibration_report.py --data cache --figure out.png

Model: logistic regression for P(stock beats the cross-sectional median over the
next 21 days), on the panel features. Probabilities come from models fitted only
on earlier dates (``oof_predictions``), never from the model that saw the label.
"""

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression

from quantskills.data import load_cached, synthetic_prices, to_wide
from quantskills.features import feature_panel, label_panel
from quantskills.models import dropna_xy, oof_predictions
from quantskills.robustness import calibration_summary, plot_reliability, reliability_table
from quantskills.utils import holdout_date


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", choices=["synthetic", "cache"], default="synthetic")
    p.add_argument("--horizon", type=int, default=21)
    p.add_argument("--C", type=float, default=0.01, help="inverse L2 strength")
    p.add_argument("--bins", type=int, default=10)
    p.add_argument("--figure", default=None, help="save a reliability diagram here")
    a = p.parse_args()

    prices = load_cached() if a.data == "cache" else synthetic_prices(seed=0)
    dates = prices.index.get_level_values("date")
    close = to_wide(prices[dates <= holdout_date(dates, 0.3)], "close")
    X, y = dropna_xy(feature_panel(close), label_panel(close, a.horizon, "up"))

    def make() -> LogisticRegression:
        return LogisticRegression(C=a.C, max_iter=1000)

    proba = oof_predictions(
        make, X, y, horizon=a.horizon, predict=lambda m, x: m.predict_proba(x)[:, 1]
    )
    yy = y.reindex(proba.index)
    # The same ranking, made overconfident: what an un-regularised model tends to do.
    logit = np.log(proba / (1 - proba))
    sharp = 1 / (1 + np.exp(-4 * logit))

    for label, q in (("logistic (walk-forward)", proba), ("same, overconfident x4", sharp)):
        s = calibration_summary(q, yy, a.bins)
        print(f"\n{label}: {len(q):,} forecasts")
        for k in ("brier", "brier_climatology", "brier_skill", "ece", "log_loss",
                  "reliability", "resolution"):  # fmt: skip
            print(f"  {k:>18}: {s[k]:+.4f}")
    print("\nReliability table (logistic):")
    print(reliability_table(proba, yy, a.bins).round(3).to_string(index=False))
    if a.figure:
        tables = {
            "logistic": reliability_table(proba, yy, a.bins),
            "overconfident x4": reliability_table(sharp, yy, a.bins),
        }
        plot_reliability(tables, a.figure, title="Walk-forward P(beat median, 21d)")
        print(f"saved {a.figure}")


if __name__ == "__main__":
    main()
