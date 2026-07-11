"""
PRISM - Module 10: Rule-Based Explanation Engine
======================================================

Responsibility: turn a portfolio's weights + Module 8's score
breakdown into a plain-English report -- why each fund got its
weight, and why the portfolio scored the way it did.

Almost no new math here. This module composes explain_* functions
already built in Modules 4, 5, and 6, plus simple rule-based
thresholds on weight size. Pure rule-based logic, no AI/ML -- per the
project's core design rule, explanations (like decisions) stay
deterministic and auditable.
"""

from overlap_engine import explain_fund_pair_overlap
from sector_engine import sector_concentration_penalty, explain_sector_concentration
from consistency_engine import explain_consistency


WEIGHT_THRESHOLDS = {
    "core": 0.30,       # >= 30% -> core holding
    "supporting": 0.10,  # >= 10% -> supporting holding
    "minor": 0.01,       # >= 1%  -> minor holding
    # below "minor" -> excluded / negligible
}


# ---------------------------------------------------------------------------
# 1. Weight classification (simple rule-based bands)
# ---------------------------------------------------------------------------
def classify_weight(weight):
    """Time complexity: O(1)."""
    if weight >= WEIGHT_THRESHOLDS["core"]:
        return "core holding"
    elif weight >= WEIGHT_THRESHOLDS["supporting"]:
        return "supporting holding"
    elif weight >= WEIGHT_THRESHOLDS["minor"]:
        return "minor holding"
    else:
        return "excluded / negligible"


# ---------------------------------------------------------------------------
# 2. Per-fund explanation
# ---------------------------------------------------------------------------
def explain_fund_allocation(fund_id, weights, predicted_returns, predicted_vols,
                             consistency_dict, overlap_matrix, holdings_dict):
    """
    Explain why one fund received its specific weight, referencing its
    predicted return/risk, consistency, and its largest overlap with
    any OTHER fund that is also actually in the portfolio (weight > 0)
    -- overlap with an excluded fund isn't relevant to explain.

    Time complexity: O(n) to scan for the largest overlap partner.
    """
    weight = weights[fund_id]
    role = classify_weight(weight)

    lines = [f"{fund_id} — {role} ({weight*100:.1f}% of portfolio)"]
    lines.append(
        f"    Predicted daily return: {predicted_returns[fund_id]:.5f} | "
        f"Predicted daily volatility: {predicted_vols[fund_id]:.5f}"
    )
    lines.append(f"    {explain_consistency(fund_id, consistency_dict[fund_id])}")

    partners = [
        (other, overlap_matrix[fund_id][other])
        for other in weights if other != fund_id and weights[other] > 0
    ]
    if partners:
        top_partner, top_overlap = max(partners, key=lambda x: x[1])
        if top_overlap > 0 and fund_id in holdings_dict and top_partner in holdings_dict:
            explanation = explain_fund_pair_overlap(
                fund_id, top_partner, holdings_dict[fund_id], holdings_dict[top_partner]
            )
            lines.append(f"    {explanation}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3. Portfolio-level summary
# ---------------------------------------------------------------------------
def explain_portfolio_summary(weights, score_result, sectors_dict, all_sector_names):
    """
    Rule-based commentary on Module 8's score breakdown. Every threshold
    used here matches conventions already established in earlier
    modules (e.g. Module 5's HHI bands), not new arbitrary cutoffs.

    Time complexity: O(n*S + N), dominated by re-running the sector
    breakdown for its exposure detail (cheap, see Module 5).
    """
    breakdown = score_result["breakdown"]
    lines = [f"Overall PRISM Score: {score_result['total_score']:.4f}"]

    lines.append(
        f"  Return: {breakdown['R_return']:.5f}  |  "
        f"Risk (volatility): {breakdown['sigma_risk']:.5f}"
    )

    if breakdown["D_diversification"] >= 0.3:
        lines.append(f"  Diversification reward is HIGH ({breakdown['D_diversification']:.3f}) "
                      f"— combined fund risk is meaningfully lower than the individual funds' average risk.")
    else:
        lines.append(f"  Diversification reward is LOW ({breakdown['D_diversification']:.3f}) "
                      f"— limited risk-reduction benefit from combining these funds.")

    if breakdown["O_overlap"] > 0.05:
        lines.append(f"  Overlap penalty is ELEVATED ({breakdown['O_overlap']:.4f}) "
                      f"— selected funds share meaningful common holdings.")
    else:
        lines.append(f"  Overlap penalty is LOW ({breakdown['O_overlap']:.4f}) "
                      f"— little redundant stock exposure across selected funds.")

    if breakdown["C_correlation"] < 0:
        lines.append(f"  Correlation term is NEGATIVE ({breakdown['C_correlation']:.4f}) "
                      f"— selected funds move against each other on average, actively reducing risk.")
    else:
        lines.append(f"  Correlation term is POSITIVE ({breakdown['C_correlation']:.4f}) "
                      f"— selected funds tend to move together.")

    sector_detail = sector_concentration_penalty(weights, sectors_dict, all_sector_names)
    lines.append(f"  {explain_sector_concentration(sector_detail)}")

    if breakdown["K_consistency"] >= 0.6:
        lines.append(f"  Portfolio-level consistency is HIGH ({breakdown['K_consistency']:.3f}) "
                      f"— dominated by historically steady performers.")
    else:
        lines.append(f"  Portfolio-level consistency is MODERATE-TO-LOW ({breakdown['K_consistency']:.3f}).")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Full report (the function other code / a UI would call)
# ---------------------------------------------------------------------------
def generate_full_report(weights, predicted_returns, predicted_vols, overlap_matrix,
                          holdings_dict, sectors_dict, all_sector_names,
                          consistency_dict, score_result):
    """
    Assemble the complete explanation report: portfolio summary first,
    then a per-fund explanation for every fund in the universe (not
    just selected ones -- excluded funds get explained too, since
    "why wasn't this fund chosen" is as important as "why was it").

    Time complexity: O(n^2) -- dominated by scanning overlap partners
    for each of the n funds.
    """
    sections = ["=" * 60, "PRISM PORTFOLIO EXPLANATION REPORT", "=" * 60, ""]
    sections.append(explain_portfolio_summary(weights, score_result, sectors_dict, all_sector_names))
    sections.append("")
    sections.append("-" * 60)
    sections.append("FUND-LEVEL BREAKDOWN")
    sections.append("-" * 60)

    # Selected funds first (descending weight), then excluded funds
    sorted_funds = sorted(weights.keys(), key=lambda f: weights[f], reverse=True)
    for fund_id in sorted_funds:
        sections.append(explain_fund_allocation(
            fund_id, weights, predicted_returns, predicted_vols,
            consistency_dict, overlap_matrix, holdings_dict
        ))
        sections.append("")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Manual self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from prism_score import compute_prism_score, standardize_holdings_to_fractions
    from overlap_engine import build_overlap_matrix
    from optimizer import simulated_annealing

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
    holdings_fractions = standardize_holdings_to_fractions(holdings_pct)
    overlap_matrix = build_overlap_matrix(holdings_fractions)
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

    def score_fn(weights):
        return compute_prism_score(
            weights, predicted_returns, corr_matrix, predicted_vols,
            overlap_matrix, sectors_dict, all_sectors, consistency_dict, tau=0.5
        )["total_score"]

    sa_result = simulated_annealing(
        ["A", "B", "C"], score_fn, initial_temp=0.5, cooling_rate=0.995,
        num_iterations=2000, step_size=0.08, max_weight=0.7, seed=42
    )
    best_weights = sa_result["best_weights"]

    score_result = compute_prism_score(
        best_weights, predicted_returns, corr_matrix, predicted_vols,
        overlap_matrix, sectors_dict, all_sectors, consistency_dict, tau=0.5
    )

    report = generate_full_report(
        best_weights, predicted_returns, predicted_vols, overlap_matrix,
        holdings_pct, sectors_dict, all_sectors, consistency_dict, score_result
    )
    print(report)
