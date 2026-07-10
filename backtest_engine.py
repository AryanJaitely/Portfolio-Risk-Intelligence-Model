"""
PRISM - Module 12: Backtesting & Evaluation (FINAL MODULE)
=================================================================

Runs the ENTIRE PRISM pipeline (Modules 1-9, 11) on a FORMATION period
only, then holds the resulting weights fixed and measures REALIZED
performance on an unseen TEST period. This is the no-lookahead-bias
discipline every real backtest requires -- decisions must never see
the data they're evaluated on.

Synthetic data generation (numpy multivariate_normal) is used ONLY to
create a realistic historical scenario for demonstration -- this is a
TEST FIXTURE, not part of PRISM's own decision logic, same principle
as using numpy as a test oracle in earlier modules.

New metric introduced here: Maximum Drawdown (worst peak-to-trough
decline), which volatility alone does not capture.
"""

import numpy as np

from return_engine import annualize_return, annualize_volatility
from optimizer import simulated_annealing
from benchmark_engine import (
    equal_weight_portfolio, generate_random_portfolio,
    optimize_markowitz, compare_portfolios,
)
from prism_score import (
    compute_prism_score, standardize_holdings_to_fractions,
    build_predicted_covariance_matrix,
)
from covariance_engine import build_covariance_matrix, build_correlation_matrix
from overlap_engine import build_overlap_matrix
from consistency_engine import fund_consistency_score
from ai_prediction_engine import predict_fund_return_and_volatility


# ---------------------------------------------------------------------------
# 1. Synthetic historical data generator (TEST FIXTURE ONLY)
# ---------------------------------------------------------------------------
def generate_formation_and_test_returns(fund_ids, means, vols, corr_matrix,
                                         formation_days, test_days,
                                         crisis_shock=None, crisis_funds=None,
                                         seed=None):
    """
    Generate a realistic multi-fund daily return history using a
    multivariate normal distribution (numpy, used here ONLY to build a
    plausible synthetic dataset -- not part of PRISM's own scoring).

    Optionally injects a one-day adverse shock into specific funds,
    placed ONLY within the test period (never formation), simulating
    a "correlation breakdown" event -- funds that looked uncorrelated
    in calm formation data can still crash together in a real crisis
    if they share structural (sector/stock) exposure.

    Returns
    -------
    dict {fund_id: full_return_series}, plus formation_days (split index)
    """
    rng = np.random.default_rng(seed)
    n = len(fund_ids)

    cov = np.zeros((n, n))
    for a, fi in enumerate(fund_ids):
        for b, fj in enumerate(fund_ids):
            cov[a][b] = corr_matrix[fi][fj] * vols[fi] * vols[fj]

    total_days = formation_days + test_days
    mean_vector = [means[f] for f in fund_ids]
    samples = rng.multivariate_normal(mean_vector, cov, size=total_days)

    returns = {f: list(samples[:, idx]) for idx, f in enumerate(fund_ids)}

    if crisis_shock is not None and crisis_funds:
        # Place the shock day strictly inside the TEST period
        crisis_day = formation_days + (test_days // 2)
        for f in crisis_funds:
            returns[f][crisis_day] -= crisis_shock

    return returns, formation_days


# ---------------------------------------------------------------------------
# 2. Performance metrics (new: Maximum Drawdown)
# ---------------------------------------------------------------------------
def max_drawdown(returns):
    """
    Worst peak-to-trough decline in cumulative wealth.

    Wealth_t = exp(cumsum(log returns))   (log returns are additive, Module 2)
    Peak_t = running max of Wealth
    Drawdown_t = (Peak_t - Wealth_t) / Peak_t
    MaxDD = max_t(Drawdown_t)

    Time complexity: O(T)
    """
    wealth = 1.0
    peak = 1.0
    worst_dd = 0.0
    cumulative_log_return = 0.0

    for r in returns:
        cumulative_log_return += r
        wealth = np.exp(cumulative_log_return)
        peak = max(peak, wealth)
        drawdown = (peak - wealth) / peak
        worst_dd = max(worst_dd, drawdown)

    return worst_dd


def evaluate_strategy(returns):
    """
    Full set of realized-performance metrics for one return series.
    Time complexity: O(T)
    """
    T = len(returns)
    mean_daily = sum(returns) / T
    var_daily = sum((r - mean_daily) ** 2 for r in returns) / (T - 1)
    vol_daily = var_daily ** 0.5

    cumulative_return = np.exp(sum(returns)) - 1
    ann_return = annualize_return(mean_daily)
    ann_vol = annualize_volatility(vol_daily)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    mdd = max_drawdown(returns)

    return {
        "cumulative_return": cumulative_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
    }


# ---------------------------------------------------------------------------
# 3. Apply fixed weights to a test-period return matrix (no rebalancing)
# ---------------------------------------------------------------------------
def compute_portfolio_returns(weights, returns_dict, start_idx, end_idx):
    """
    Buy-and-hold: apply FIXED weights (found using formation data only)
    to each day's actual fund returns in [start_idx, end_idx).

    Time complexity: O(n * T_test)
    """
    T = end_idx - start_idx
    portfolio_returns = []
    for t in range(start_idx, end_idx):
        day_return = sum(w * returns_dict[f][t] for f, w in weights.items())
        portfolio_returns.append(day_return)
    return portfolio_returns


def run_single_backtest(seed, formation_days=500, test_days=250, crisis_shock=0.08):
    """
    One complete formation -> test backtest run. Returns
    {strategy_name: metrics_dict} for this single random draw.
    """
    fund_ids = ["A", "B", "C"]
    means = {"A": 0.0006, "B": 0.0005, "C": 0.0003}
    vols = {"A": 0.012, "B": 0.010, "C": 0.006}
    corr_matrix = {
        "A": {"A": 1.0, "B": 0.1, "C": -0.2},
        "B": {"A": 0.1, "B": 1.0, "C": 0.05},
        "C": {"A": -0.2, "B": 0.05, "C": 1.0},
    }

    full_returns, split_idx = generate_formation_and_test_returns(
        fund_ids, means, vols, corr_matrix, formation_days, test_days,
        crisis_shock=crisis_shock, crisis_funds=["A", "B"], seed=seed
    )
    formation_returns = {f: full_returns[f][:split_idx] for f in fund_ids}

    fund_means = {f: sum(r) / len(r) for f, r in formation_returns.items()}
    cov_hist = build_covariance_matrix(formation_returns, fund_means)
    corr_hist = build_correlation_matrix(cov_hist)

    predictions = {f: predict_fund_return_and_volatility(formation_returns[f], window_length=21)
                   for f in fund_ids}
    predicted_returns = {f: predictions[f]["predicted_daily_return"] for f in fund_ids}
    predicted_vols = {f: predictions[f]["predicted_daily_volatility"] for f in fund_ids}

    holdings_pct = {
        "A": {"HDFC Bank": 9.0, "Infosys": 6.0},
        "B": {"HDFC Bank": 8.0, "ICICI Bank": 6.0},
        "C": {"ITC": 5.0, "L&T": 4.0},
    }
    overlap_matrix = build_overlap_matrix(standardize_holdings_to_fractions(holdings_pct))

    all_sectors = ["Financials", "IT", "FMCG", "Industrials", "Unclassified/Other"]
    sectors_dict = {
        "A": {"Financials": 45.0, "IT": 20.0},
        "B": {"Financials": 48.0, "IT": 15.0},
        "C": {"FMCG": 30.0, "Industrials": 25.0},
    }
    consistency_dict = {f: fund_consistency_score(formation_returns[f], window_length=21) for f in fund_ids}

    cov_pred = build_predicted_covariance_matrix(corr_hist, predicted_vols)

    def prism_score_fn(weights):
        return compute_prism_score(
            weights, predicted_returns, corr_hist, predicted_vols,
            overlap_matrix, sectors_dict, all_sectors, consistency_dict, tau=0.5
        )["total_score"]

    strategies = {
        "Equal-Weight": equal_weight_portfolio(fund_ids),
        "Random": generate_random_portfolio(fund_ids, seed=seed + 1000),
        "Markowitz": optimize_markowitz(fund_ids, predicted_returns, cov_pred, num_iterations=1500, seed=seed),
        "PRISM": simulated_annealing(
            fund_ids, prism_score_fn, initial_temp=0.5, cooling_rate=0.995,
            num_iterations=1500, step_size=0.08, seed=seed
        )["best_weights"],
    }

    results = {}
    for name, w in strategies.items():
        port_returns = compute_portfolio_returns(w, full_returns, split_idx, split_idx + test_days)
        results[name] = evaluate_strategy(port_returns)
        results[name]["weights"] = w
    return results


def run_multi_seed_backtest(num_seeds=15):
    """
    Repeat the full backtest across many independent random draws, and
    report MEAN and STD per strategy per metric -- a single draw is not
    statistically meaningful on its own.
    """
    all_metrics = {name: {"annualized_return": [], "annualized_volatility": [],
                           "sharpe": [], "max_drawdown": []}
                   for name in ["Equal-Weight", "Random", "Markowitz", "PRISM"]}

    for seed in range(num_seeds):
        results = run_single_backtest(seed)
        for name, metrics in results.items():
            for key in all_metrics[name]:
                all_metrics[name][key].append(metrics[key])

    summary = {}
    for name, metric_lists in all_metrics.items():
        summary[name] = {}
        for key, values in metric_lists.items():
            mean = sum(values) / len(values)
            std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
            summary[name][key] = {"mean": mean, "std": std}
    return summary


# ---------------------------------------------------------------------------
# Manual self-test: single-draw demo, THEN the statistically honest
# multi-seed version
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== PRISM Module 12: Backtesting & Evaluation (FINAL MODULE) ===\n")

    fund_ids = ["A", "B", "C"]
    means = {"A": 0.0006, "B": 0.0005, "C": 0.0003}
    vols = {"A": 0.012, "B": 0.010, "C": 0.006}
    corr_matrix = {
        "A": {"A": 1.0, "B": 0.1, "C": -0.2},
        "B": {"A": 0.1, "B": 1.0, "C": 0.05},
        "C": {"A": -0.2, "B": 0.05, "C": 1.0},
    }

    formation_days, test_days = 500, 250
    print(f"Formation period: {formation_days} days (all decisions made here)")
    print(f"Test period: {test_days} days (UNSEEN during decision-making)")
    print("Injecting a one-day shared shock into A and B (both Financials-heavy,")
    print("both hold HDFC Bank) partway through the TEST period -- simulating a")
    print("sector crisis that formation-period correlation (0.1, looks low) did")
    print("not warn about.\n")

    full_returns, split_idx = generate_formation_and_test_returns(
        fund_ids, means, vols, corr_matrix, formation_days, test_days,
        crisis_shock=0.08, crisis_funds=["A", "B"], seed=7
    )
    formation_returns = {f: full_returns[f][:split_idx] for f in fund_ids}

    # --- Run the pipeline on FORMATION data only ---
    fund_means = {f: sum(r) / len(r) for f, r in formation_returns.items()}
    cov_hist = build_covariance_matrix(formation_returns, fund_means)
    corr_hist = build_correlation_matrix(cov_hist)

    predictions = {f: predict_fund_return_and_volatility(formation_returns[f], window_length=21)
                   for f in fund_ids}
    predicted_returns = {f: predictions[f]["predicted_daily_return"] for f in fund_ids}
    predicted_vols = {f: predictions[f]["predicted_daily_volatility"] for f in fund_ids}

    holdings_pct = {
        "A": {"HDFC Bank": 9.0, "Infosys": 6.0},
        "B": {"HDFC Bank": 8.0, "ICICI Bank": 6.0},
        "C": {"ITC": 5.0, "L&T": 4.0},
    }
    overlap_matrix = build_overlap_matrix(standardize_holdings_to_fractions(holdings_pct))

    all_sectors = ["Financials", "IT", "FMCG", "Industrials", "Unclassified/Other"]
    sectors_dict = {
        "A": {"Financials": 45.0, "IT": 20.0},
        "B": {"Financials": 48.0, "IT": 15.0},
        "C": {"FMCG": 30.0, "Industrials": 25.0},
    }
    consistency_dict = {f: fund_consistency_score(formation_returns[f], window_length=21) for f in fund_ids}

    cov_pred = build_predicted_covariance_matrix(corr_hist, predicted_vols)

    def prism_score_fn(weights):
        return compute_prism_score(
            weights, predicted_returns, corr_hist, predicted_vols,
            overlap_matrix, sectors_dict, all_sectors, consistency_dict, tau=0.5
        )["total_score"]

    # --- Build all 4 strategies using FORMATION data only ---
    equal_w = equal_weight_portfolio(fund_ids)
    random_w = generate_random_portfolio(fund_ids, seed=1)
    markowitz_w = optimize_markowitz(fund_ids, predicted_returns, cov_pred, num_iterations=2000, seed=42)
    prism_w = simulated_annealing(
        fund_ids, prism_score_fn, initial_temp=0.5, cooling_rate=0.995,
        num_iterations=2000, step_size=0.08, seed=42
    )["best_weights"]

    strategies = {
        "Equal-Weight": equal_w,
        "Random": random_w,
        "Markowitz": markowitz_w,
        "PRISM": prism_w,
    }

    print("Weights decided using FORMATION data only:")
    for name, w in strategies.items():
        print(f"  {name:<14}: {{k: round(v,3) for k,v in w.items()}}".replace("{k: round(v,3) for k,v in w.items()}", str({k: round(v, 3) for k, v in w.items()})))

    # --- Apply fixed weights to UNSEEN test period ---
    print(f"\n{'Strategy':<14}{'CumRet':>10}{'AnnRet':>10}{'AnnVol':>10}{'Sharpe':>10}{'MaxDD':>10}")
    print("-" * 64)
    results = {}
    for name, w in strategies.items():
        port_returns = compute_portfolio_returns(w, full_returns, split_idx, split_idx + test_days)
        metrics = evaluate_strategy(port_returns)
        results[name] = metrics
        print(f"{name:<14}{metrics['cumulative_return']:>10.4f}{metrics['annualized_return']:>10.4f}"
              f"{metrics['annualized_volatility']:>10.4f}{metrics['sharpe']:>10.3f}{metrics['max_drawdown']:>10.4f}")

    print(f"\n[Key result] Out-of-sample (unseen test period, includes the injected")
    print(f"A/B shock): PRISM's Max Drawdown = {results['PRISM']['max_drawdown']:.4f} vs "
          f"Markowitz's = {results['Markowitz']['max_drawdown']:.4f}")
    if results["PRISM"]["max_drawdown"] < results["Markowitz"]["max_drawdown"]:
        print("PRISM's structural risk awareness produced a SMALLER drawdown than")
        print("Markowitz's correlation-only view in THIS single draw.")
    print("\nA single draw is not statistically meaningful on its own -- Markowitz")
    print("optimization is also known to produce extreme/concentrated corner")
    print("solutions from estimation error (see its weights above), which can look")
    print("great OR terrible depending on luck in one draw. Running many independent")
    print("draws below for an honest comparison.\n")

    print("=" * 74)
    print(f"MULTI-SEED BACKTEST SUMMARY (15 independent formation/test draws)")
    print("=" * 74)
    summary = run_multi_seed_backtest(num_seeds=15)
    print(f"{'Strategy':<14}{'AnnRet':>14}{'AnnVol':>14}{'Sharpe':>14}{'MaxDD':>14}")
    print("-" * 74)
    for name, metrics in summary.items():
        print(f"{name:<14}"
              f"{metrics['annualized_return']['mean']:>7.4f}±{metrics['annualized_return']['std']:<6.4f}"
              f"{metrics['annualized_volatility']['mean']:>7.4f}±{metrics['annualized_volatility']['std']:<6.4f}"
              f"{metrics['sharpe']['mean']:>7.3f}±{metrics['sharpe']['std']:<6.3f}"
              f"{metrics['max_drawdown']['mean']:>7.4f}±{metrics['max_drawdown']['std']:<6.4f}")

    print(f"\n[Honest conclusion] Averaged over 15 independent draws:")
    print(f"  PRISM mean Max Drawdown : {summary['PRISM']['max_drawdown']['mean']:.4f}")
    print(f"  Markowitz mean Max Drawdown : {summary['Markowitz']['max_drawdown']['mean']:.4f}")
    print(f"  PRISM mean Ann. Volatility : {summary['PRISM']['annualized_volatility']['mean']:.4f}")
    print(f"  Markowitz mean Ann. Volatility : {summary['Markowitz']['annualized_volatility']['mean']:.4f}")
