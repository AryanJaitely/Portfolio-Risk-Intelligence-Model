"""
PRISM - Module 2: Return & Volatility Engine
===============================================

Responsibility of this file:
    Convert raw NAV price series into the two fundamental fund-level
    statistics every later module depends on:
        mu (expected return)  and  sigma (volatility / risk)

Design rule followed throughout PRISM:
    Every calculation here is written as an explicit loop over the data,
    NOT as a call to numpy.mean() / numpy.std() / statistics.mean().
    This is deliberate: the entire point of PRISM is that every number
    in the final score must be traceable to a formula YOU derived and
    wrote, not to a library's internal (and possibly differently
    defined) implementation.

    numpy/statistics are only used in test_return_engine.py, and only
    as an independent "answer key" to verify our manual code is
    correct -- never inside the production computation path.

Depends on: data_layer.py (Module 1) for fetching NAV series.
"""

import math


# ---------------------------------------------------------------------------
# 1. Log returns
# ---------------------------------------------------------------------------
def compute_log_returns(nav_series):
    """
    Convert a NAV price series into daily log returns.

    r_t = ln(p_t / p_(t-1))

    Parameters
    ----------
    nav_series : list[float]
        Chronologically ordered (oldest first) NAV values.

    Returns
    -------
    list[float] of length (T - 1), where T = len(nav_series)

    Time complexity: O(T)
    """
    if len(nav_series) < 2:
        raise ValueError("Need at least 2 NAV points to compute a return.")

    returns = []
    for t in range(1, len(nav_series)):
        prev_price = nav_series[t - 1]
        curr_price = nav_series[t]
        if prev_price <= 0 or curr_price <= 0:
            raise ValueError("NAV values must be strictly positive.")
        r_t = math.log(curr_price / prev_price)
        returns.append(r_t)

    return returns


# ---------------------------------------------------------------------------
# 2. Mean return (first moment)
# ---------------------------------------------------------------------------
def mean_return(returns):
    """
    Arithmetic mean of a list of returns.

    mu = (1/T) * sum(r_t)

    Time complexity: O(T)
    """
    if len(returns) == 0:
        raise ValueError("Cannot compute mean of an empty return series.")

    total = 0.0
    for r in returns:
        total += r
    return total / len(returns)


# ---------------------------------------------------------------------------
# 3. Variance & Volatility (second moment)
# ---------------------------------------------------------------------------
def variance_return(returns, mean=None):
    """
    Sample variance of returns, using Bessel's correction (T-1 denominator)
    since the mean used here is itself estimated from the same sample.

    sigma^2 = (1 / (T-1)) * sum((r_t - mu)^2)

    Parameters
    ----------
    returns : list[float]
    mean : float, optional
        If not provided, it is computed internally (one extra O(T) pass).

    Time complexity: O(T)
    """
    T = len(returns)
    if T < 2:
        raise ValueError("Need at least 2 returns to compute variance.")

    if mean is None:
        mean = mean_return(returns)

    sum_sq_dev = 0.0
    for r in returns:
        sum_sq_dev += (r - mean) ** 2

    return sum_sq_dev / (T - 1)


def volatility(returns, mean=None):
    """
    Standard deviation of returns = sqrt(variance).
    This is our risk measure at the daily level.

    Time complexity: O(T) (dominated by variance_return)
    """
    var = variance_return(returns, mean)
    return math.sqrt(var)


# ---------------------------------------------------------------------------
# 4. Annualization
# ---------------------------------------------------------------------------
TRADING_DAYS_PER_YEAR = 252  # standard convention for equity markets


def annualize_return(daily_mean, periods_per_year=TRADING_DAYS_PER_YEAR):
    """
    Return scales LINEARLY with time because log returns are additive
    (see README section 1.4 for the full derivation).

    Time complexity: O(1)
    """
    return daily_mean * periods_per_year


def annualize_volatility(daily_vol, periods_per_year=TRADING_DAYS_PER_YEAR):
    """
    Volatility scales with the SQUARE ROOT of time, because variance
    (not std dev) is additive across independent periods.

    Time complexity: O(1)
    """
    return daily_vol * math.sqrt(periods_per_year)


# ---------------------------------------------------------------------------
# 5. CAGR - a second, independent return measure for sanity-checking
# ---------------------------------------------------------------------------
def cagr(start_price, end_price, num_years):
    """
    Compound Annual Growth Rate.

    CAGR = (end_price / start_price)^(1/num_years) - 1

    This answers: "what constant annual growth rate would have produced
    this exact price change?" Useful as an independent cross-check
    against annualize_return() -- if the two disagree wildly, it's a
    signal of either high volatility (compounding drag) or a data issue.

    Time complexity: O(1)
    """
    if start_price <= 0 or end_price <= 0:
        raise ValueError("Prices must be positive.")
    if num_years <= 0:
        raise ValueError("num_years must be positive.")

    return (end_price / start_price) ** (1.0 / num_years) - 1.0


# ---------------------------------------------------------------------------
# 6. Fund-level statistics bundle
# ---------------------------------------------------------------------------
class FundStats:
    """
    A simple, readable container for one fund's computed statistics.
    Deliberately a plain class, not a heavyweight dataclass/ORM model --
    this project favors clarity over enterprise patterns.
    """

    def __init__(self, fund_id, daily_returns, nav_series, dates):
        self.fund_id = fund_id
        self.daily_returns = daily_returns

        self.daily_mean = mean_return(daily_returns)
        self.daily_vol = volatility(daily_returns, mean=self.daily_mean)

        self.annual_return = annualize_return(self.daily_mean)
        self.annual_vol = annualize_volatility(self.daily_vol)

        num_years = len(nav_series) / TRADING_DAYS_PER_YEAR
        self.cagr = cagr(nav_series[0], nav_series[-1], num_years) if num_years > 0 else None

        self.start_date = dates[0]
        self.end_date = dates[-1]
        self.num_observations = len(daily_returns)

    def __repr__(self):
        return (
            f"FundStats({self.fund_id}: "
            f"annual_return={self.annual_return:.2%}, "
            f"annual_vol={self.annual_vol:.2%}, "
            f"cagr={self.cagr:.2%}, "
            f"n_obs={self.num_observations})"
        )


def build_fund_statistics(aligned_data, fund_ids=None):
    """
    Build FundStats for every fund in an aligned NAV dataset
    (the output shape of data_layer.fetch_nav_series_aligned).

    Parameters
    ----------
    aligned_data : dict
        {"dates": [...], "nav_matrix": {scheme_code: [prices...], ...}}
    fund_ids : optional list mapping scheme_code -> friendly fund_id
               (if not given, scheme codes are used as-is)

    Returns
    -------
    dict: {fund_id: FundStats}

    Time complexity: O(n * T) for n funds, T days each.
    """
    dates = aligned_data["dates"]
    nav_matrix = aligned_data["nav_matrix"]

    stats = {}
    for code, nav_series in nav_matrix.items():
        fund_id = code if fund_ids is None else fund_ids.get(code, code)
        returns = compute_log_returns(nav_series)
        stats[fund_id] = FundStats(fund_id, returns, nav_series, dates)

    return stats


# ---------------------------------------------------------------------------
# Manual self-test using synthetic data (works with NO internet access)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== PRISM Module 2: Return & Volatility Engine self-test ===\n")

    # A simple synthetic NAV series: steady 0.1% daily growth, no noise.
    # We use this because we can hand-verify the expected answer.
    synthetic_nav = [100.0]
    for _ in range(100):
        synthetic_nav.append(synthetic_nav[-1] * 1.001)  # +0.1% each day

    returns = compute_log_returns(synthetic_nav)
    mu = mean_return(returns)
    sigma = volatility(returns, mean=mu)

    print(f"[Synthetic constant-growth series]")
    print(f"  Daily mean return   : {mu:.6f}  (expected ~ ln(1.001) = {math.log(1.001):.6f})")
    print(f"  Daily volatility    : {sigma:.6f}  (expected ~ 0, since growth is noise-free)")
    print(f"  Annualized return   : {annualize_return(mu):.2%}")
    print(f"  Annualized vol      : {annualize_volatility(sigma):.2%}")
    ann_ret = annualize_return(mu)
    real_cagr = cagr(synthetic_nav[0], synthetic_nav[-1], 100 / 252)
    print(f"  CAGR                : {real_cagr:.2%}")

    print("\n[Important relationship, not a bug]")
    print("Annualized return above is an annualized LOG return (continuous")
    print("compounding scale). CAGR is a discrete compounding scale. They are")
    print("related by: CAGR = exp(annualized_log_return) - 1")
    print(f"  Check: exp({ann_ret:.4f}) - 1 = {math.exp(ann_ret) - 1:.2%}  "
          f"(should equal CAGR = {real_cagr:.2%})")
    print("For a NOISE-FREE series these match exactly (as shown above).")
    print("Once real, noisy data is used, a related but distinct effect appears:")
    print("E[exp(X)] >= exp(E[X]) (Jensen's inequality) -- higher volatility")
    print("causes CAGR to drift below what annualized log return would suggest.")
    print("This 'volatility drag' is a real phenomenon worth citing in your report.")
