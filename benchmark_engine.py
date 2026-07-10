"""
PRISM - Module 11: Benchmarking Suite
==========================================

Compares PRISM's optimized portfolio against three baselines:
    1. Equal-Weight
    2. Random Allocation (averaged over many samples, not one lucky draw)
    3. Markowitz (mean-variance / max-Sharpe) -- built here ONLY as a
       benchmark, using Module 9's SAME simulated annealing machinery
       with a DIFFERENT objective function. This isolates the
       comparison to "what are we optimizing for" rather than "which
       search algorithm searches better" -- an apples-to-apples test.

Design rule followed: no scipy.optimize/cvxpy anywhere; Markowitz here
reuses Module 9's own from-scratch optimizer.
"""

import random

from optimizer import simulated_annealing, generate_neighbor
from covariance_engine import portfolio_return, portfolio_variance


# ---------------------------------------------------------------------------
# 1. Baseline portfolio generators
# ---------------------------------------------------------------------------
def equal_weight_portfolio(fund_ids):
    """Time complexity: O(n)."""
    n = len(fund_ids)
    return {f: 1.0 / n for f in fund_ids}


def generate_random_portfolio(fund_ids, seed=None):
    """
    Uniformly random point on the simplex, via the standard
    Exponential-normalization method: draw i.i.d. Exponential(1)
    values and normalize by their sum. This gives a TRUE uniform
    distribution over the simplex (naive uniform-then-normalize does
    NOT -- it biases toward the center), which matters for a fair
    "random allocation" baseline.

    Time complexity: O(n)
    """
    rng = random.Random(seed)
    draws = [rng.expovariate(1.0) for _ in fund_ids]
    total = sum(draws)
    return {f: d / total for f, d in zip(fund_ids, draws)}


def average_random_portfolio_score(fund_ids, score_fn, num_samples=200, seed=None):
    """
    Random Allocation is a random VARIABLE, not a single number -- one
    lucky/unlucky draw isn't a fair baseline. We average score_fn over
    many random portfolios and report both mean and std.

    Time complexity: O(num_samples * cost(score_fn))
    """
    rng = random.Random(seed)
    scores = []
    for _ in range(num_samples):
        sample_seed = rng.randint(0, 10**9)
        weights = generate_random_portfolio(fund_ids, seed=sample_seed)
        scores.append(score_fn(weights))

    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / (len(scores) - 1)
    std_score = variance ** 0.5

    # Also return one representative sample portfolio for display purposes
    representative = generate_random_portfolio(fund_ids, seed=seed)

    return {"mean_score": mean_score, "std_score": std_score,
            "representative_weights": representative, "num_samples": num_samples}


# ---------------------------------------------------------------------------
# 2. Markowitz benchmark (max-Sharpe), built with Module 9's own optimizer
# ---------------------------------------------------------------------------
def make_sharpe_score_fn(predicted_returns, cov_matrix):
    """
    Classic Markowitz objective: maximize portfolio Sharpe ratio
    (return / volatility). This is intentionally the ONLY thing this
    score function considers -- no overlap, sector, or consistency
    terms -- because that omission is exactly what PRISM claims to fix.

    Time complexity of returned function: O(n^2) per call.
    """
    def sharpe_score_fn(weights):
        r = portfolio_return(weights, predicted_returns)
        var = portfolio_variance(weights, cov_matrix)
        vol = var ** 0.5
        if vol == 0:
            return 0.0
        return r / vol
    return sharpe_score_fn


def optimize_markowitz(fund_ids, predicted_returns, cov_matrix,
                        num_iterations=2000, seed=None):
    """
    Find the max-Sharpe portfolio using the SAME simulated annealing
    machinery as PRISM's own optimizer (Module 9), just with the
    Sharpe-only objective instead of the full PRISM score. Reusing the
    identical search algorithm ensures any performance difference we
    observe is due to the OBJECTIVE, not the search method.

    Time complexity: O(num_iterations * n^2)
    """
    score_fn = make_sharpe_score_fn(predicted_returns, cov_matrix)
    result = simulated_annealing(
        fund_ids, score_fn, initial_temp=0.5, cooling_rate=0.995,
        num_iterations=num_iterations, step_size=0.08, seed=seed
    )
    return result["best_weights"]


# ---------------------------------------------------------------------------
# 3. Comparison table
# ---------------------------------------------------------------------------
def compare_portfolios(portfolios, predicted_returns, predicted_vols, cov_matrix,
                        prism_score_fn):
    """
    Build a comparison table across named portfolios, showing BOTH the
    classic Sharpe ratio (what Markowitz optimizes for) and the full
    PRISM score (what PRISM optimizes for) -- this side-by-side view is
    exactly what exposes the "individually good funds, bad portfolio"
    blind spot in classic mean-variance approaches.

    Parameters
    ----------
    portfolios : dict {name: weights_dict}
    prism_score_fn : callable, weights -> PRISM total_score

    Time complexity: O(m * n^2), m = number of portfolios compared
    """
    rows = []
    for name, weights in portfolios.items():
        r = portfolio_return(weights, predicted_returns)
        var = portfolio_variance(weights, cov_matrix)
        vol = var ** 0.5
        sharpe = r / vol if vol > 0 else 0.0
        prism_total = prism_score_fn(weights)

        rows.append({
            "name": name,
            "weights": {k: round(v, 3) for k, v in weights.items()},
            "return": r,
            "volatility": vol,
            "sharpe": sharpe,
            "prism_score": prism_total,
        })
    return rows


def print_comparison_table(rows):
    print(f"{'Portfolio':<20}{'Return':>10}{'Vol':>10}{'Sharpe':>10}{'PRISM Score':>14}")
    print("-" * 64)
    for row in rows:
        print(f"{row['name']:<20}{row['return']:>10.5f}{row['volatility']:>10.5f}"
              f"{row['sharpe']:>10.3f}{row['prism_score']:>14.4f}")


# ---------------------------------------------------------------------------
# Manual self-test: the central demonstration of PRISM's value proposition
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from prism_score import compute_prism_score, standardize_holdings_to_fractions, build_predicted_covariance_matrix
    from overlap_engine import build_overlap_matrix

    print("=== PRISM Module 11: Benchmarking Suite ===\n")
    print("Scenario designed to expose Markowitz's blind spot:")
    print("Funds A and B have LOW historical return correlation (0.1),")
    print("so classical mean-variance sees them as good diversifiers.")
    print("But A and B secretly share HDFC Bank AND are both heavily")
    print("Financials-sector -- real structural risk that correlation")
    print("alone did not fully capture in this sample.\n")

    fund_ids = ["A", "B", "C"]
    predicted_returns = {"A": 0.0006, "B": 0.0005, "C": 0.0003}
    predicted_vols = {"A": 0.012, "B": 0.010, "C": 0.006}

    # KEY: low A-B correlation despite real structural overlap
    corr_matrix = {
        "A": {"A": 1.0, "B": 0.1, "C": -0.2},
        "B": {"A": 0.1, "B": 1.0, "C": 0.05},
        "C": {"A": -0.2, "B": 0.05, "C": 1.0},
    }

    holdings_pct = {
        "A": {"HDFC Bank": 9.0, "Infosys": 6.0},
        "B": {"HDFC Bank": 8.0, "ICICI Bank": 6.0},   # heavy overlap with A on HDFC Bank
        "C": {"ITC": 5.0, "L&T": 4.0},
    }
    overlap_matrix = build_overlap_matrix(standardize_holdings_to_fractions(holdings_pct))

    all_sectors = ["Financials", "IT", "FMCG", "Industrials", "Unclassified/Other"]
    sectors_dict = {
        "A": {"Financials": 45.0, "IT": 20.0},
        "B": {"Financials": 48.0, "IT": 15.0},   # both A and B heavy Financials
        "C": {"FMCG": 30.0, "Industrials": 25.0},
    }
    consistency_dict = {
        "A": {"consistency": 0.8, "cv": 0.25, "num_windows": 6, "win_rate": 0.83},
        "B": {"consistency": 0.75, "cv": 0.33, "num_windows": 6, "win_rate": 0.75},
        "C": {"consistency": 0.9, "cv": 0.11, "num_windows": 6, "win_rate": 0.9},
    }

    tau = 0.5
    cov_pred = build_predicted_covariance_matrix(corr_matrix, predicted_vols)

    def prism_score_fn(weights):
        return compute_prism_score(
            weights, predicted_returns, corr_matrix, predicted_vols,
            overlap_matrix, sectors_dict, all_sectors, consistency_dict, tau
        )["total_score"]

    # --- Build each baseline ---
    equal_w = equal_weight_portfolio(fund_ids)

    random_stats = average_random_portfolio_score(fund_ids, prism_score_fn, num_samples=200, seed=1)
    random_w = random_stats["representative_weights"]

    markowitz_w = optimize_markowitz(fund_ids, predicted_returns, cov_pred, num_iterations=2000, seed=42)

    prism_sa_result = simulated_annealing(
        fund_ids, prism_score_fn, initial_temp=0.5, cooling_rate=0.995,
        num_iterations=2000, step_size=0.08, seed=42
    )
    prism_w = prism_sa_result["best_weights"]

    portfolios = {
        "Equal-Weight": equal_w,
        "Random (sample)": random_w,
        "Markowitz (max-Sharpe)": markowitz_w,
        "PRISM (optimized)": prism_w,
    }

    rows = compare_portfolios(portfolios, predicted_returns, predicted_vols, cov_pred, prism_score_fn)
    print_comparison_table(rows)

    print(f"\nRandom Allocation over {random_stats['num_samples']} samples: "
          f"mean PRISM score = {random_stats['mean_score']:.4f}, "
          f"std = {random_stats['std_score']:.4f}")

    markowitz_row = next(r for r in rows if r["name"] == "Markowitz (max-Sharpe)")
    prism_row = next(r for r in rows if r["name"] == "PRISM (optimized)")
    print(f"\n[Key result] Markowitz Sharpe = {markowitz_row['sharpe']:.3f} "
          f"(highest by classic metric) but PRISM score = {markowitz_row['prism_score']:.4f}")
    print(f"PRISM-optimized Sharpe = {prism_row['sharpe']:.3f} but PRISM score = {prism_row['prism_score']:.4f}")
    print("If Markowitz's PRISM score is lower despite a competitive/higher Sharpe,")
    print("that demonstrates the core thesis: optimizing return/variance alone can")
    print("miss overlap and sector concentration risk that PRISM explicitly penalizes.")
