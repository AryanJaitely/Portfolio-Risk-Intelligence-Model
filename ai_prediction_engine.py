"""
PRISM - Module 7: AI Prediction Module
==========================================

Responsibility: predict a fund's NEXT-PERIOD daily return and
volatility from its own historical window statistics.

CORE RULE: this module predicts INPUTS ONLY (return, volatility).
It never touches weights, allocation, or ranking -- those stay
rule-based in Modules 8/9.

Approach (upgraded): Random Forest regression over a richer feature
vector, evaluated with proper walk-forward (expanding-window,
no-shuffle) validation. Reuses Module 6's rolling_windows() for
window construction (no duplicated logic).

Uses sklearn.ensemble.RandomForestRegressor. Unlike the previous
plain-linear-regression version, this is a black box in the sense
that coefficients aren't directly readable -- but it exposes
model.feature_importances_ instead, so predictions stay explainable
at the feature level even though the model is nonlinear.

Public API (unchanged from the linear-regression version, so every
other module -- prism_score.py, backtest_engine.py, etc. -- keeps
working without modification):

    predict_fund_return_and_volatility(returns, window_length=21, ...)
    predict_universe(returns_dict, window_length=21, ...)

New optional keyword arguments were added with backward-compatible
defaults (fund_age=0.0, hyperparameters, etc.) -- no existing caller
needs to change.
"""

import math

from sklearn.ensemble import RandomForestRegressor

from consistency_engine import rolling_windows


# ---------------------------------------------------------------------------
# 0. Feature names, in the exact order the model consumes them
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "prev_mean_return",   # Previous Window Mean Return
    "prev_volatility",    # Previous Window Volatility
    "rolling_sharpe",     # Rolling Sharpe Ratio (mean_k / vol_k, divide-by-zero safe)
    "momentum_20d",       # 20-trading-day momentum
    "momentum_60d",       # 60-trading-day momentum
    "fund_age",           # Static feature merged onto every sample
]

DEFAULT_RF_PARAMS = dict(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=3,
    random_state=42,
)


# ---------------------------------------------------------------------------
# 1. Per-window statistics (mean, volatility) -- same as the original module
# ---------------------------------------------------------------------------
def compute_window_stats(returns, window_length):
    """Per-window (mean, volatility) pairs. Time: O(T)."""
    windows = rolling_windows(returns, window_length)
    stats = []
    for w in windows:
        mean_k = sum(w) / len(w)
        var_k = sum((r - mean_k) ** 2 for r in w) / (len(w) - 1)
        stats.append((mean_k, var_k ** 0.5))
    return stats


# ---------------------------------------------------------------------------
# 2. New dynamic features: rolling Sharpe + momentum
# ---------------------------------------------------------------------------
def _safe_sharpe(mean_k, vol_k):
    """Mean/volatility ratio, protected against divide-by-zero."""
    if vol_k == 0:
        return 0.0
    return mean_k / vol_k


def _momentum(returns, end_idx, lookback):
    """
    Momentum over `lookback` trading days, ending at (and including)
    return index end_idx, computed from a LOG-return series.

    Since log returns are additive (r_t = ln(p_t/p_(t-1))), the sum of
    the last `lookback` log returns equals ln(p_end / p_(end-lookback)),
    so:

        Momentum = p_end / p_(end-lookback) - 1
                 = exp(sum of last `lookback` log returns) - 1

    This reproduces the NAV-ratio definition in the spec WITHOUT
    needing the raw NAV series -- only the log-return series this
    module already receives.

    If there isn't enough history yet (start_idx < 0), momentum is
    undefined; we return 0.0 (neutral) rather than raising, since this
    can legitimately happen for the earliest windows.

    Time complexity: O(lookback), i.e. O(1) for a fixed lookback.
    """
    start_idx = end_idx - lookback + 1
    if start_idx < 0:
        return 0.0
    window_sum = sum(returns[start_idx:end_idx + 1])
    return math.exp(window_sum) - 1.0


def compute_dynamic_features(returns, window_length):
    """
    Build the per-window dynamic feature rows:
        prev_mean_return, prev_volatility, rolling_sharpe,
        momentum_20d, momentum_60d

    (fund_age, the one STATIC feature, is merged in separately by
    build_feature_target_arrays() since it doesn't vary by window.)

    Returns
    -------
    list[dict], one dict per window, in chronological order.

    Time complexity: O(T) -- window statistics are O(T) total (as in
    the original module), and momentum is O(1) per window since
    lookback is fixed, so O(K) additional work for K windows.
    """
    stats = compute_window_stats(returns, window_length)
    rows = []
    for k, (mean_k, vol_k) in enumerate(stats):
        end_idx = (k + 1) * window_length - 1  # last return index in window k
        rows.append({
            "prev_mean_return": mean_k,
            "prev_volatility": vol_k,
            "rolling_sharpe": _safe_sharpe(mean_k, vol_k),
            "momentum_20d": _momentum(returns, end_idx, 20),
            "momentum_60d": _momentum(returns, end_idx, 60),
        })
    return rows


# ---------------------------------------------------------------------------
# 3. Fund Age (static feature, from fund_meta.csv-style data)
# ---------------------------------------------------------------------------
def compute_fund_age(fund_meta_row, current_year=None):
    """
    Fund Age = Current Year - Launch Year.

    Uses an existing "fund_age" column directly if the caller's
    fund_meta row already has one (pre-computed / cached); otherwise
    derives it from a "launch_year" column.

    Parameters
    ----------
    fund_meta_row : dict, one row as returned by
                    data_layer.load_fund_meta() (possibly with an
                    added "fund_age" or "launch_year" column)
    current_year : int, optional (defaults to today's calendar year)

    Time complexity: O(1)
    """
    if "fund_age" in fund_meta_row and fund_meta_row["fund_age"] not in (None, ""):
        return float(fund_meta_row["fund_age"])

    if "launch_year" not in fund_meta_row or fund_meta_row["launch_year"] in (None, ""):
        raise ValueError(
            "fund_meta_row has neither a usable 'fund_age' nor a 'launch_year' "
            "column -- cannot compute Fund Age."
        )

    if current_year is None:
        from datetime import date
        current_year = date.today().year

    return float(current_year - int(fund_meta_row["launch_year"]))


def load_fund_ages(fund_meta_rows, current_year=None):
    """
    Convenience helper: build a {fund_id: fund_age} dict from the
    rows returned by data_layer.load_fund_meta(), for use with
    predict_universe(..., fund_ages=...).

    Time complexity: O(n), n = number of funds
    """
    return {
        row["fund_id"]: compute_fund_age(row, current_year)
        for row in fund_meta_rows
    }


# ---------------------------------------------------------------------------
# 4. Supervised dataset construction (features at k-1 -> targets at k)
# ---------------------------------------------------------------------------
def build_feature_target_arrays(returns, window_length, fund_age=0.0):
    """
    Turn a fund's raw return series into a supervised learning
    dataset: window k-1's features predict window k's (mean, vol).

    Returns
    -------
    X : list[list[float]]         -- feature matrix, columns = FEATURE_NAMES
    y_return : list[float]        -- next-window mean return targets
    y_vol : list[float]           -- next-window volatility targets
    last_features : list[float]   -- feature row for the LATEST window,
                                      used to predict the still-unseen
                                      next window (no matching target yet)

    Never shuffled -- rows stay in chronological order throughout, so
    every downstream walk-forward split is automatically leak-free.

    Time complexity: O(T)
    """
    dyn_rows = compute_dynamic_features(returns, window_length)
    if len(dyn_rows) < 4:
        raise ValueError(
            f"Need at least 4 windows to fit a predictive model, got {len(dyn_rows)}. "
            f"Use more history or a smaller window_length."
        )

    def to_vector(row):
        return [
            row["prev_mean_return"],
            row["prev_volatility"],
            row["rolling_sharpe"],
            row["momentum_20d"],
            row["momentum_60d"],
            fund_age,
        ]

    X = [to_vector(dyn_rows[k - 1]) for k in range(1, len(dyn_rows))]
    y_return = [dyn_rows[k]["prev_mean_return"] for k in range(1, len(dyn_rows))]
    y_vol = [dyn_rows[k]["prev_volatility"] for k in range(1, len(dyn_rows))]
    last_features = to_vector(dyn_rows[-1])

    return X, y_return, y_vol, last_features


# ---------------------------------------------------------------------------
# 5. Walk-forward (time-series, no-shuffle) validation
# ---------------------------------------------------------------------------
def walk_forward_validate(X, y_return, y_vol, min_train_samples=3, rf_params=None):
    """
    Expanding-window walk-forward validation, exactly analogous to the
    calendar-year scheme in the spec (train on everything seen so far,
    test on the single next, never-before-seen period), just indexed
    by window number instead of calendar year -- the mechanism is
    identical: every fold's training set is a strict PREFIX of the
    data, so the model never sees the future.

    For each fold i (i = min_train_samples .. len(X)-1):
        train on X[0:i], y[0:i]
        predict/evaluate on X[i]  (a single, strictly-future sample)

    Parameters
    ----------
    X, y_return, y_vol : as produced by build_feature_target_arrays()
    min_train_samples : int, smallest training set allowed before the
                         first fold is evaluated (need enough rows for
                         a meaningful Random Forest fit)
    rf_params : dict, RandomForestRegressor hyperparameters (defaults
                to DEFAULT_RF_PARAMS if not given)

    Returns
    -------
    dict with:
        "mae_return", "rmse_return", "directional_accuracy",
        "mae_vol", "rmse_vol",
        "feature_importance_return" : list[(name, importance)], sorted desc
        "feature_importance_vol"    : list[(name, importance)], sorted desc
        "num_folds" : int

    If there isn't enough data for even one fold, all metrics come
    back as None and num_folds == 0 (caller decides how to handle
    that -- this function never raises for "too little data", since
    it's a diagnostic, not a hard requirement of prediction).

    Time complexity: O(F * fit_cost), F = number of folds. Each fold
    refits two small Random Forests, so this is the expensive part of
    the module by design -- walk-forward validation is inherently
    O(F) refits, there's no shortcut without leaking information.
    """
    params = dict(DEFAULT_RF_PARAMS if rf_params is None else rf_params)

    n = len(X)
    if n <= min_train_samples:
        return {
            "mae_return": None, "rmse_return": None, "directional_accuracy": None,
            "mae_vol": None, "rmse_vol": None,
            "feature_importance_return": [], "feature_importance_vol": [],
            "num_folds": 0,
        }

    return_errors, vol_errors = [], []
    correct_direction, total_direction = 0, 0
    importances_return_sum = [0.0] * len(FEATURE_NAMES)
    importances_vol_sum = [0.0] * len(FEATURE_NAMES)
    num_folds = 0

    for i in range(min_train_samples, n):
        X_train, y_ret_train, y_vol_train = X[:i], y_return[:i], y_vol[:i]
        X_test, y_ret_true, y_vol_true = [X[i]], y_return[i], y_vol[i]

        ret_model = RandomForestRegressor(**params).fit(X_train, y_ret_train)
        vol_model = RandomForestRegressor(**params).fit(X_train, y_vol_train)

        y_ret_pred = ret_model.predict(X_test)[0]
        y_vol_pred = max(vol_model.predict(X_test)[0], 0.0)

        return_errors.append((y_ret_pred, y_ret_true))
        vol_errors.append((y_vol_pred, y_vol_true))

        # Directional accuracy: did we get the SIGN of the next return right?
        total_direction += 1
        if (y_ret_pred >= 0) == (y_ret_true >= 0):
            correct_direction += 1

        for j, imp in enumerate(ret_model.feature_importances_):
            importances_return_sum[j] += imp
        for j, imp in enumerate(vol_model.feature_importances_):
            importances_vol_sum[j] += imp

        num_folds += 1

    def mae(pairs):
        return sum(abs(p - t) for p, t in pairs) / len(pairs)

    def rmse(pairs):
        return math.sqrt(sum((p - t) ** 2 for p, t in pairs) / len(pairs))

    def sorted_importance(sums):
        avg = [s / num_folds for s in sums]
        return sorted(zip(FEATURE_NAMES, avg), key=lambda kv: kv[1], reverse=True)

    return {
        "mae_return": mae(return_errors),
        "rmse_return": rmse(return_errors),
        "directional_accuracy": correct_direction / total_direction,
        "mae_vol": mae(vol_errors),
        "rmse_vol": rmse(vol_errors),
        "feature_importance_return": sorted_importance(importances_return_sum),
        "feature_importance_vol": sorted_importance(importances_vol_sum),
        "num_folds": num_folds,
    }


def print_evaluation_report(evaluation, fund_id=None):
    """
    Pretty-print a walk_forward_validate() result: MAE, RMSE,
    Directional Accuracy, and feature importances sorted descending.

    Time complexity: O(1) (fixed-size report)
    """
    label = f" for {fund_id}" if fund_id else ""
    print(f"=== Walk-forward evaluation{label} ({evaluation['num_folds']} folds) ===")
    if evaluation["num_folds"] == 0:
        print("  Not enough history for walk-forward validation.")
        return

    print(f"  Return  -> MAE: {evaluation['mae_return']:.6f}  "
          f"RMSE: {evaluation['rmse_return']:.6f}  "
          f"Directional Accuracy: {evaluation['directional_accuracy']:.2%}")
    print(f"  Vol     -> MAE: {evaluation['mae_vol']:.6f}  "
          f"RMSE: {evaluation['rmse_vol']:.6f}")

    print("  Feature importance (return model, sorted):")
    for name, imp in evaluation["feature_importance_return"]:
        print(f"    {name:<18} {imp:.4f}")

    print("  Feature importance (volatility model, sorted):")
    for name, imp in evaluation["feature_importance_vol"]:
        print(f"    {name:<18} {imp:.4f}")


# ---------------------------------------------------------------------------
# 6. Public API (unchanged signatures, richer internals)
# ---------------------------------------------------------------------------
def predict_fund_return_and_volatility(
    returns,
    window_length=21,
    fund_age=0.0,
    rf_params=None,
    evaluate=False,
    min_train_samples=3,
):
    """
    Random-Forest version of the original API. Fits:
        mean_k   ~ (mean_{k-1}, vol_{k-1}, sharpe_{k-1}, mom20_{k-1}, mom60_{k-1}, fund_age)
        vol_k    ~ (same features)
    on ALL available windows, then predicts window K+1 using the
    latest window's features. This mirrors the original module's
    "fit on everything available, predict the next unseen window"
    contract exactly -- callers (backtest_engine.py, prism_score.py)
    keep working unmodified.

    In addition (and this is new), if `evaluate=True` a proper
    walk-forward (expanding-window, no-shuffle) backtest of the model
    itself is run and reported under "evaluation" -- MAE, RMSE,
    Directional Accuracy, and sorted feature importances. This is
    purely diagnostic; it does not affect predicted_daily_return /
    predicted_daily_volatility, which are still fit on the full
    history as described above. Defaults to False because it refits
    O(K) extra Random Forests (K = number of windows) -- callers that
    invoke this per-fund inside a hot loop (e.g. backtest_engine.py's
    multi-seed backtest) should leave it off and call
    walk_forward_validate() directly, once, when they specifically
    want the Part 4 evaluation report.

    Parameters
    ----------
    returns : list[float], daily log returns, chronological
    window_length : int, trading days per window
    fund_age : float, static feature (years since launch); defaults to
               0.0 for backward compatibility with callers that don't
               pass fund metadata
    rf_params : dict, RandomForestRegressor hyperparameters (defaults
                to n_estimators=300, max_depth=8, min_samples_leaf=3,
                random_state=42)
    evaluate : bool, whether to also run walk-forward validation
    min_train_samples : int, smallest walk-forward training set

    Time complexity: O(T) for feature building + O(fit) for the final
    model + O(F * fit) for walk-forward evaluation if evaluate=True
    (F = number of folds); the evaluation is the dominant cost and can
    be disabled with evaluate=False for a fast production predict.
    """
    params = dict(DEFAULT_RF_PARAMS if rf_params is None else rf_params)

    X, y_return, y_vol, last_features = build_feature_target_arrays(
        returns, window_length, fund_age=fund_age
    )

    return_model = RandomForestRegressor(**params).fit(X, y_return)
    vol_model = RandomForestRegressor(**params).fit(X, y_vol)

    predicted_return = return_model.predict([last_features])[0]
    predicted_vol = max(vol_model.predict([last_features])[0], 0.0)  # volatility can't be negative

    result = {
        "predicted_daily_return": predicted_return,
        "predicted_daily_volatility": predicted_vol,
        "num_training_samples": len(X),
        "return_model_r2": return_model.score(X, y_return),
        "vol_model_r2": vol_model.score(X, y_vol),
        "feature_importance_return": sorted(
            zip(FEATURE_NAMES, return_model.feature_importances_),
            key=lambda kv: kv[1], reverse=True,
        ),
        "feature_importance_vol": sorted(
            zip(FEATURE_NAMES, vol_model.feature_importances_),
            key=lambda kv: kv[1], reverse=True,
        ),
    }

    if evaluate:
        result["evaluation"] = walk_forward_validate(
            X, y_return, y_vol, min_train_samples=min_train_samples, rf_params=params
        )
    else:
        result["evaluation"] = None

    return result


def predict_universe(returns_dict, window_length=21, fund_ages=None, rf_params=None, evaluate=False):
    """
    Run predict_fund_return_and_volatility for every fund.

    Parameters
    ----------
    returns_dict : dict {fund_id: list[float]}
    fund_ages : dict {fund_id: float}, optional (e.g. from
                load_fund_ages()); funds not present default to 0.0
                for backward compatibility

    Time complexity: O(n*T), n = funds.
    """
    fund_ages = fund_ages or {}
    predictions = {}
    for fund_id, returns in returns_dict.items():
        predictions[fund_id] = predict_fund_return_and_volatility(
            returns,
            window_length=window_length,
            fund_age=fund_ages.get(fund_id, 0.0),
            rf_params=rf_params,
            evaluate=evaluate,
        )
    return predictions


if __name__ == "__main__":
    print("=== PRISM Module 7: AI Prediction self-test (Random Forest) ===\n")

    # Synthetic fund with a clear upward volatility trend -> model
    # should predict rising volatility for the next window.
    returns = []
    for k in range(20):
        vol_level = 0.005 + k * 0.001  # volatility increases each window
        window = [0.001, -0.001, 0.001, -0.001, 0.002] * 4  # base pattern
        window = [r + vol_level * ((-1) ** i) for i, r in enumerate(window)]
        returns.extend(window)

    result = predict_fund_return_and_volatility(returns, window_length=20, fund_age=7.0, evaluate=True)
    print("Rising-volatility synthetic fund:")
    for k, v in result.items():
        if k == "evaluation":
            continue
        print(f"  {k}: {v}")

    print("\n[Sanity check] predicted volatility should reflect the rising trend "
          "seen in training windows, not just the historical average.")

    print()
    print_evaluation_report(result["evaluation"], fund_id="SYNTH")
