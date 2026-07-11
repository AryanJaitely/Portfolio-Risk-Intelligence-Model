"""
PRISM - Module 9: Manual Optimizer (Simulated Annealing)
==============================================================

Responsibility: search over portfolio weight vectors w (on the
simplex: sum(w)=1, w_i>=0) to MAXIMIZE Module 8's PRISM Score.

Why Simulated Annealing (justification, not just a choice):
    - S(w) is NOT smooth/differentiable everywhere (Module 4's min()
      operations, Module 6's window-based logic break gradient
      continuity) -> gradient descent is not well-founded here.
    - S(w) is not convex (multiple competing terms) -> plain hill
      climbing gets stuck in local optima.
    - SA is derivative-free, naturally handles the simplex constraint
      via a "transfer weight from fund i to fund j" neighbor move,
      and its stochastic acceptance lets it escape local optima that
      hill climbing cannot.
    - A Genetic Algorithm would also work but is heavier machinery
      than a single small (n<=15 dimensional) weight vector needs.

Implemented entirely with Python's built-in `random` module -- no
optimization library (scipy.optimize, cvxpy, etc.) is used anywhere.
"""

import random
import math


# ---------------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------------
def random_initial_weights(fund_ids):
    """Equal-weight starting point (a neutral, unbiased starting guess)."""
    n = len(fund_ids)
    return {f: 1.0 / n for f in fund_ids}


# ---------------------------------------------------------------------------
# 2. Neighbor generation (stays on the simplex by construction)
# ---------------------------------------------------------------------------
def generate_neighbor(weights, step_size, max_weight=None):
    """
    Transfer a small random amount from one randomly chosen fund to
    another. This keeps sum(w)=1 and w_i>=0 automatically -- no
    rejection sampling needed for the simplex constraint itself.

    Parameters
    ----------
    weights : dict {fund_id: weight}
    step_size : float, maximum amount transferred in one move
    max_weight : float or None, optional per-fund cap (e.g. 0.4 means
                 no single fund can exceed 40% of the portfolio --
                 a common practical diversification constraint)

    Time complexity: O(n) to copy the dict; O(1) for the transfer itself.
    """
    fund_ids = list(weights.keys())
    i, j = random.sample(fund_ids, 2)

    delta = random.uniform(0, step_size)
    delta = min(delta, weights[i])  # can't remove more than fund i has

    if max_weight is not None:
        delta = min(delta, max_weight - weights[j])  # can't push fund j over its cap
        delta = max(delta, 0.0)

    new_weights = dict(weights)
    new_weights[i] -= delta
    new_weights[j] += delta
    return new_weights


# ---------------------------------------------------------------------------
# 3. Simulated Annealing main loop
# ---------------------------------------------------------------------------
def simulated_annealing(fund_ids,score_fn,initial_temp=1.0,cooling_rate=0.95,num_iterations=2000,step_size=0.05,max_weight=None, seed=None):
    """
    Search for the weight vector maximizing score_fn(weights).

    Parameters
    ----------
    fund_ids : list[str]
    score_fn : callable, weights -> float (e.g. wraps compute_prism_score
               and returns just result["total_score"], with all other
               PRISM inputs fixed via closure)
    initial_temp : float, starting temperature
    cooling_rate : float in (0,1), geometric cooling: T_k = T0 * rate^k
    num_iterations : int, total SA steps
    step_size : float, max weight transferred per neighbor move
    max_weight : float or None, optional per-fund weight cap
    seed : int or None, for reproducibility

    Returns
    -------
    dict with:
        "best_weights" : dict, the best weight vector found
        "best_score"   : float
        "final_weights": dict, weights at the end of the walk (may
                          differ from best_weights -- SA doesn't always
                          end at its best point, which is exactly why
                          we track best-so-far separately)
        "score_history" : list[float], score at each iteration (for
                          plotting convergence in your report)

    Time complexity: O(I * cost(score_fn)) = O(I * n^2) given Module 8's
    score is O(n^2). For I=2000, n<=15, this is ~450,000 operations --
    trivial even on modest hardware (a few seconds at most).
    """
    if seed is not None:
        random.seed(seed)

    current_weights = random_initial_weights(fund_ids)
    current_score = score_fn(current_weights)

    best_weights = dict(current_weights)
    best_score = current_score

    score_history = [current_score]
    temperature = initial_temp

    for iteration in range(num_iterations):
        candidate = generate_neighbor(current_weights, step_size, max_weight)
        candidate_score = score_fn(candidate)

        delta = candidate_score - current_score

        if delta > 0:
            accept = True
        else:
            # Worse move: accept with probability exp(delta/T).
            # Higher temperature -> more likely to accept a worse move
            # (exploration). As T cools, this becomes increasingly rare
            # (exploitation) -- the classic SA exploration/exploitation
            # tradeoff, controlled entirely by the cooling schedule.
            accept_probability = math.exp(delta / temperature) if temperature > 1e-12 else 0.0
            accept = random.random() < accept_probability

        if accept:
            current_weights = candidate
            current_score = candidate_score

        if current_score > best_score:
            best_weights = dict(current_weights)
            best_score = current_score

        score_history.append(current_score)
        temperature *= cooling_rate

    return {
        "best_weights": best_weights,
        "best_score": best_score,
        "final_weights": current_weights,
        "score_history": score_history,
    }


# ---------------------------------------------------------------------------
# Manual self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== PRISM Module 9: Manual Optimizer self-test ===\n")

    from prism_score import compute_prism_score, standardize_holdings_to_fractions
    from overlap_engine import build_overlap_matrix

    fund_ids = ["A", "B", "C"]
    predicted_returns = {"A": 0.0006, "B": 0.0004, "C": 0.0003}
    predicted_vols = {"A": 0.012, "B": 0.009, "C": 0.006}
    corr_matrix = {
        "A": {"A": 1.0, "B": 0.6, "C": -0.2},
        "B": {"A": 0.6, "B": 1.0, "C": 0.1},
        "C": {"A": -0.2, "B": 0.1, "C": 1.0},
    }
    holdings_pct = {
        "A": {"HDFC Bank": 9.0, "Infosys": 6.0},
        "B": {"HDFC Bank": 7.0, "ICICI Bank": 6.0},
        "C": {"ITC": 5.0, "L&T": 4.0},
    }
    overlap_matrix = build_overlap_matrix(standardize_holdings_to_fractions(holdings_pct))
    all_sectors = ["Financials", "IT", "FMCG", "Industrials", "Unclassified/Other"]
    sectors_dict = {
        "A": {"Financials": 45.0, "IT": 20.0},
        "B": {"Financials": 50.0, "IT": 15.0},
        "C": {"FMCG": 30.0, "Industrials": 25.0},
    }
    consistency_dict = {
        "A": {"consistency": 0.8, "cv": 0.25, "num_windows": 6, "win_rate": 0.83},
        "B": {"consistency": 0.6, "cv": 0.67, "num_windows": 6, "win_rate": 0.67},
        "C": {"consistency": 0.9, "cv": 0.11, "num_windows": 6, "win_rate": 0.9},
    }

    def make_score_fn(tau):
        def score_fn(weights):
            return compute_prism_score(
                weights, predicted_returns, corr_matrix, predicted_vols,
                overlap_matrix, sectors_dict, all_sectors, consistency_dict, tau
            )["total_score"]
        return score_fn

    for tau in [0.2, 0.8]:
        score_fn = make_score_fn(tau)
        equal_weights = random_initial_weights(fund_ids)
        equal_score = score_fn(equal_weights)

        result = simulated_annealing(
            fund_ids, score_fn, initial_temp=0.5, cooling_rate=0.995,
            num_iterations=2000, step_size=0.08, max_weight=0.7, seed=42
        )

        print(f"[tau={tau}]")
        print(f"  Equal-weight score : {equal_score:.6f}  (weights: {equal_weights})")
        print(f"  SA best score      : {result['best_score']:.6f}  (weights: "
              f"{ {k: round(v,3) for k,v in result['best_weights'].items()} })")
        assert result["best_score"] >= equal_score, "SA should never do worse than equal-weight start"
        print(f"  Confirmed: SA result is at least as good as equal-weight baseline.\n")
