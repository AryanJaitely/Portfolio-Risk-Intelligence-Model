"""
PRISM - Module 4: Overlap Penalty Engine
============================================

Responsibility of this file:
    Quantify how much redundant, duplicated stock exposure exists
    between mutual funds, using actual stock-level holdings data
    (not just statistical return correlation from Module 3).

Core original contribution (see README for full derivation):
    Standard "portfolio overlap" tools compute:
        R(i,j) = sum over common stocks of min(w_i,s, w_j,s)
    PRISM extends this with a concentration-amplification factor based
    on the Herfindahl-Hirschman Index (HHI) of the overlap itself:
        O(i,j) = R(i,j) * (1 + HHI_overlap(i,j))
    This distinguishes "overlap spread across many stocks" (safer) from
    "overlap concentrated in one or two stocks" (riskier), which
    standard overlap tools do not capture.

Design rule followed: every number is computed with explicit loops.
No pandas groupby tricks, no hidden similarity-metric libraries.
"""


# ---------------------------------------------------------------------------
# 1. Pairwise fund overlap score
# ---------------------------------------------------------------------------
def fund_pair_overlap(holdings_i, holdings_j):
    """
    Compute the PRISM overlap score between two funds' holdings.

    Parameters
    ----------
    holdings_i, holdings_j : dict {stock_name: weight_pct}
        weight_pct is expected on a 0-100 scale (matching holdings.csv),
        but the formula works identically on a 0-1 scale as long as
        both funds use the same scale.

    Returns
    -------
    float : overlap score. On the same scale as the input weights
            (e.g. if weights are 0-100, a score of 15 means roughly
            "15 percentage points of guaranteed duplicated exposure,
            amplified by concentration").

    Time complexity: O(H_i + H_j), where H = number of holdings per
    fund (small and roughly constant, e.g. ~10, since factsheets
    disclose only top holdings).
    """
    common_stocks = set(holdings_i.keys()) & set(holdings_j.keys())

    if not common_stocks:
        return 0.0

    contributions = []
    for stock in common_stocks:
        c = min(holdings_i[stock], holdings_j[stock])
        contributions.append(c)

    R = sum(contributions)
    if R == 0:
        return 0.0

    # Normalize contributions into shares that sum to 1, then compute
    # the Herfindahl-Hirschman Index of THIS overlap distribution.
    shares = [c / R for c in contributions]
    hhi_overlap = sum(p ** 2 for p in shares)

    return R * (1 + hhi_overlap)


def explain_fund_pair_overlap(fund_i_id, fund_j_id, holdings_i, holdings_j):
    """
    Human-readable, rule-based explanation of a fund pair's overlap.
    This is early groundwork for Module 10's full explanation engine --
    every score PRISM produces should be traceable back to plain
    English, not just a number.

    Time complexity: O(H_i + H_j)
    """
    common_stocks = set(holdings_i.keys()) & set(holdings_j.keys())

    if not common_stocks:
        return f"{fund_i_id} and {fund_j_id} share no common top holdings."

    # Sort common stocks by their overlap contribution, descending,
    # so the explanation highlights the biggest driver of overlap first.
    contributions = [(s, min(holdings_i[s], holdings_j[s])) for s in common_stocks]
    contributions.sort(key=lambda x: x[1], reverse=True)

    R = sum(c for _, c in contributions)
    top_stock, top_contribution = contributions[0]

    top_stock_share_of_overlap = top_contribution / R if R > 0 else 0

    concentration_note = ""
    if top_stock_share_of_overlap > 0.5:
        concentration_note = (
            f" Over half of this overlap comes from a single stock "
            f"({top_stock}), which increases concentrated single-stock risk."
        )

    stock_list = ", ".join(f"{s} ({holdings_i[s]:.1f}% vs {holdings_j[s]:.1f}%)"
                            for s, _ in contributions[:3])

    return (
        f"{fund_i_id} and {fund_j_id} share {len(common_stocks)} common holding(s), "
        f"totaling {R:.2f} percentage points of overlapping exposure. "
        f"Top contributors: {stock_list}."
        f"{concentration_note}"
    )


# ---------------------------------------------------------------------------
# 2. Full overlap matrix across the fund universe
# ---------------------------------------------------------------------------
def build_overlap_matrix(holdings_dict):
    """
    Build the full n x n fund-pair overlap matrix.

    Parameters
    ----------
    holdings_dict : dict {fund_id: {stock_name: weight_pct}}

    Returns
    -------
    dict of dict: overlap_matrix[fund_a][fund_b] -> overlap score
                  (diagonal entries are left as 0.0 -- a fund's
                  "overlap with itself" isn't a meaningful concept
                  in this context; self-concentration is handled
                  separately, it's not what this module measures)

    Time complexity: O(n^2 * H), H bounded/small -> effectively O(n^2)
    """
    fund_ids = list(holdings_dict.keys())
    overlap_matrix = {f: {} for f in fund_ids}

    for a in range(len(fund_ids)):
        for b in range(len(fund_ids)):
            fund_i = fund_ids[a]
            fund_j = fund_ids[b]
            if fund_i == fund_j:
                overlap_matrix[fund_i][fund_j] = 0.0
                continue
            if b < a:
                # already computed as (b, a) -- exploit symmetry
                overlap_matrix[fund_i][fund_j] = overlap_matrix[fund_j][fund_i]
                continue
            score = fund_pair_overlap(holdings_dict[fund_i], holdings_dict[fund_j])
            overlap_matrix[fund_i][fund_j] = score
            overlap_matrix[fund_j][fund_i] = score

    return overlap_matrix


# ---------------------------------------------------------------------------
# 3. Portfolio-level overlap penalty
# ---------------------------------------------------------------------------
def portfolio_overlap_penalty(weights, overlap_matrix):
    """
    Aggregate pairwise fund overlaps into a single portfolio-level
    penalty, using the same weighted double-sum pattern as Module 3's
    portfolio_variance() -- both measure "interaction risk" the same way.

    O(w) = sum_{i != j} w_i * w_j * O(i,j)

    Parameters
    ----------
    weights : dict {fund_id: weight}, should sum to 1
    overlap_matrix : dict of dict, from build_overlap_matrix()

    Time complexity: O(n^2)
    """
    total = 0.0
    for fund_i, w_i in weights.items():
        for fund_j, w_j in weights.items():
            if fund_i == fund_j:
                continue
            total += w_i * w_j * overlap_matrix[fund_i][fund_j]
    return total


# ---------------------------------------------------------------------------
# Manual self-test using hand-verifiable synthetic examples
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== PRISM Module 4: Overlap Penalty self-test ===\n")

    # Scenario A: overlap concentrated in ONE stock (should get amplified)
    fund_a1 = {"HDFC Bank": 10.0, "Infosys": 5.0, "TCS": 3.0}
    fund_a2 = {"HDFC Bank": 8.0, "Wipro": 4.0, "ITC": 2.0}
    # Common stock: HDFC Bank only -> min(10,8) = 8 -> R = 8, HHI = 1 (single stock)
    # Expected O = 8 * (1 + 1) = 16

    score_a = fund_pair_overlap(fund_a1, fund_a2)
    print(f"[Scenario A: single-stock overlap]")
    print(f"  Expected O(A1,A2) = 16.0")
    print(f"  Actual   O(A1,A2) = {score_a:.4f}")
    print(f"  {explain_fund_pair_overlap('FundA1', 'FundA2', fund_a1, fund_a2)}")

    # Scenario B: SAME total overlap (8), but spread across 4 stocks evenly
    fund_b1 = {"HDFC Bank": 2.0, "Infosys": 2.0, "TCS": 2.0, "ITC": 2.0, "Wipro": 5.0}
    fund_b2 = {"HDFC Bank": 2.0, "Infosys": 2.0, "TCS": 2.0, "ITC": 2.0, "L&T": 3.0}
    # Common: HDFC Bank, Infosys, TCS, ITC, each min = 2.0 -> R = 8, HHI = 4*(2/8)^2 = 0.25
    # Expected O = 8 * (1 + 0.25) = 10

    score_b = fund_pair_overlap(fund_b1, fund_b2)
    print(f"\n[Scenario B: SAME total overlap (8), spread across 4 stocks]")
    print(f"  Expected O(B1,B2) = 10.0")
    print(f"  Actual   O(B1,B2) = {score_b:.4f}")
    print(f"  {explain_fund_pair_overlap('FundB1', 'FundB2', fund_b1, fund_b2)}")

    print(f"\n[Key result] Both scenarios have IDENTICAL raw overlap (R=8),")
    print(f"but Scenario A (concentrated) scores {score_a:.1f} vs Scenario B")
    print(f"(spread out) scoring {score_b:.1f}. This is PRISM's original")
    print(f"contribution: concentration-aware overlap, not just raw overlap.")

    # Portfolio-level test with 3 funds
    print("\n\n[Portfolio-level overlap penalty test]")
    holdings = {
        "F1": {"HDFC Bank": 9.0, "Infosys": 6.0, "Reliance": 5.0},
        "F2": {"HDFC Bank": 7.0, "ICICI Bank": 6.0, "TCS": 4.0},
        "F3": {"ITC": 5.0, "L&T": 4.0, "Sun Pharma": 3.0},  # no overlap with F1/F2
    }
    overlap_matrix = build_overlap_matrix(holdings)

    print("Overlap matrix:")
    for i in holdings:
        row = "  ".join(f"{j}:{overlap_matrix[i][j]:.2f}" for j in holdings)
        print(f"  {i}: {row}")

    weights = {"F1": 0.4, "F2": 0.4, "F3": 0.2}
    penalty = portfolio_overlap_penalty(weights, overlap_matrix)
    print(f"\nPortfolio weights: {weights}")
    print(f"Portfolio overlap penalty O(w) = {penalty:.4f}")
    print("(F3 has zero overlap with F1/F2, so it only dilutes -- adding")
    print("more of F3 to the portfolio should LOWER this penalty. Try it.)")
