"""
run_prism_real.py
==================
Bridges your REAL NAV data (loaded via your existing loader.py) with the
full PRISM pipeline (Modules 1-9, 11) to compute an actual PRISM-optimized
portfolio for your 10 real mutual funds.

Place this file in the SAME folder as: loader.py, data_layer.py,
return_engine.py, covariance_engine.py, overlap_engine.py, sector_engine.py,
consistency_engine.py, ai_prediction_engine.py, prism_score.py, optimizer.py,
fund_meta.csv, holdings.csv, sectors.csv

Run this ONLY after fund_meta.csv, holdings.csv, and sectors.csv all have
real rows for every fund (F01-F10). See fund_meta.csv for the fund_id <->
scheme_code mapping.
"""

from loader import load_hk7797_nav_folder
from data_layer import load_fund_meta, load_holdings, load_sectors
from return_engine import compute_log_returns, mean_return
from covariance_engine import build_covariance_matrix, build_correlation_matrix
from overlap_engine import build_overlap_matrix
from consistency_engine import fund_consistency_score
from ai_prediction_engine import predict_universe
from ai_prediction_engine import load_fund_ages
from prism_score import compute_prism_score, standardize_holdings_to_fractions
from optimizer import simulated_annealing

# ---------------------------------------------------------------------
# 1. CONFIG -- update NAV_FOLDER to match your machine
# ---------------------------------------------------------------------
NAV_FOLDER = r"C:\Users\Administrator\Desktop\MutualFundProject\data\archive (1)\DailyNAV"
TAU = 0.5  # risk tolerance: 0.0 = max safety, 1.0 = max return-seeking

# ---------------------------------------------------------------------
# 2. Load fund universe metadata (fund_id <-> scheme_code mapping)
# ---------------------------------------------------------------------
fund_meta = load_fund_meta("fund_meta.csv")
scheme_to_fund = {row["scheme_code"]: row["fund_id"] for row in fund_meta}
fund_codes = [row["scheme_code"] for row in fund_meta]
fund_names = {row["fund_id"]: row["fund_name"] for row in fund_meta}
fund_ages = load_fund_ages(fund_meta)
# ---------------------------------------------------------------------
# 3. Load real NAV history and align dates across all funds
# ---------------------------------------------------------------------
print("Loading real NAV data...")
data = load_hk7797_nav_folder(NAV_FOLDER, fund_codes)
nav_matrix_by_code = data["nav_matrix"]

# remap from scheme_code (loader's key) to fund_id (PRISM modules' key)
nav_matrix = {scheme_to_fund[code]: navs for code, navs in nav_matrix_by_code.items()}
fund_ids = list(nav_matrix.keys())
print(f"Loaded {len(fund_ids)} funds over {len(data['dates'])} trading days.\n")

# ---------------------------------------------------------------------
# 4. Returns, covariance, correlation (Modules 2-3)
# ---------------------------------------------------------------------
for fid, navs in nav_matrix.items():
    bad = [x for x in navs if x <= 0]
    if bad:
        print(fid, bad[:10])
returns_dict = {fid: compute_log_returns(navs) for fid, navs in nav_matrix.items()}
means_dict = {fid: mean_return(r) for fid, r in returns_dict.items()}

cov_matrix = build_covariance_matrix(returns_dict, means_dict)
corr_matrix = build_correlation_matrix(cov_matrix)

# ---------------------------------------------------------------------
# 5. AI-predicted next-period return/volatility (Module 7)
# ---------------------------------------------------------------------
predictions = predict_universe(returns_dict,window_length=21,fund_ages=fund_ages,evaluate=True)
predicted_returns = {fid: p["predicted_daily_return"] for fid, p in predictions.items()}
predicted_vols = {fid: p["predicted_daily_volatility"] for fid, p in predictions.items()}
print("\nPredicted returns:")
for fid, value in predicted_returns.items():
    print(fid, value)

print("\nPredicted volatilities:")
for fid, value in predicted_vols.items():
    print(fid, value)
print("\n================ MODEL EVALUATION ================\n")

for fid in fund_ids:
    print(f"\n{fid} ({fund_names[fid]})")

    evaluation = predictions[fid]["evaluation"]

    if evaluation is None:
        continue

    print(f"MAE Return : {evaluation['mae_return']:.6f}")
    print(f"RMSE Return: {evaluation['rmse_return']:.6f}")
    print(f"Direction Accuracy: {evaluation['directional_accuracy']:.2%}")

    print("Top Features:")

    for feature, importance in evaluation["feature_importance_return"][:5]:
        print(f"   {feature:<20} {importance:.3f}")
# ---------------------------------------------------------------------
# 6. Holdings overlap (Module 4) -- needs real holdings.csv
# ---------------------------------------------------------------------
holdings_pct = load_holdings("holdings.csv")
holdings_fractions = standardize_holdings_to_fractions(holdings_pct)
overlap_matrix = build_overlap_matrix(holdings_fractions)

# ---------------------------------------------------------------------
# 7. Sector concentration inputs (Module 5) -- needs real sectors.csv
# ---------------------------------------------------------------------
sectors_dict = load_sectors("sectors.csv")
all_sector_names = sorted({s for sd in sectors_dict.values() for s in sd}) + ["Unclassified/Other"]

# ---------------------------------------------------------------------
# 8. Consistency scores (Module 6)
# ---------------------------------------------------------------------
consistency_dict = {fid: fund_consistency_score(r) for fid, r in returns_dict.items()}

# ---------------------------------------------------------------------
# 9. Wrap Module 8's score as a function of weights only, then optimize (Module 9)
# ---------------------------------------------------------------------
def score_fn(weights):
    return compute_prism_score(
        weights, predicted_returns, corr_matrix, predicted_vols,
        overlap_matrix, sectors_dict, all_sector_names,
        consistency_dict, tau=TAU,
    )["total_score"]

print("Running simulated annealing optimizer...")
result = simulated_annealing(
    fund_ids,
    score_fn,
    initial_temp=1.0,
    cooling_rate=0.995,
    num_iterations=5000,
    step_size=0.02,
    max_weight=0.40,
    seed=42,
)

best_weights = result["best_weights"]
best_score = result["best_score"]

# ---------------------------------------------------------------------
# 10. Report
# ---------------------------------------------------------------------
print("\n========== PRISM OPTIMIZED PORTFOLIO (REAL DATA) ==========\n")
for fid, w in sorted(best_weights.items(), key=lambda x: -x[1]):
    print(f"{fid} ({fund_names.get(fid, '?')}): {w * 100:.2f}%")

print(f"\nPRISM Score: {best_score:.4f}")

breakdown = compute_prism_score(
    best_weights, predicted_returns, corr_matrix, predicted_vols,
    overlap_matrix, sectors_dict, all_sector_names,
    consistency_dict, tau=TAU,
)["breakdown"]

print("\nScore breakdown:")
for k, v in breakdown.items():
    print(f"  {k}: {v:.6f}")
