from loader import load_hk7797_nav_folder

folder_path = r"C:\Users\Administrator\Desktop\MutualFundProject\data\archive (1)\DailyNAV"

fund_codes = [
    "103174",  # Large Cap - Aditya Birla Sun Life Frontline Equity Fund - Growth
    "112277",  # Large Cap - Axis Bluechip Fund - Regular Plan - Growth
    "101592",  # Mid Cap - Aditya Birla Sun Life Midcap Fund - Growth
    "114564",  # Mid Cap - Axis Midcap Fund - Regular Plan - Growth
    "105804",  # Small Cap - Aditya Birla Sun Life Small Cap Fund - Growth
    "102001",  # HDFC Top 100 Fund
    "100047",  # Debt (Liquid) - Aditya Birla Sun Life Liquid Fund - Growth
    "103178",  # Debt (Corporate Bond) - Aditya Birla Sun Life Corporate Bond Fund - Growth - Regular Plan
    "102948",  # Hybrid - HDFC Hybrid Equity Fund - Growth
    "104685",  # Hybrid (Balanced Advantage) - ICICI Prudential Balanced Advantage Fund - Growth
]

data = load_hk7797_nav_folder(folder_path, fund_codes, drop_non_overlapping=True)

print("\n========== DATA LOADED SUCCESSFULLY ==========\n")
print("Funds Loaded:")
for code in data["nav_matrix"]:
    print(code)

print("\nTotal Dates:", len(data["dates"]))
print("\nFirst 10 Dates:")
print(data["dates"][:10])

print("\nSample NAV values:\n")
for code, navs in data["nav_matrix"].items():
    print(f"Scheme Code: {code}")
    print("First 10 NAVs:", navs[:10])
    print("-" * 50)