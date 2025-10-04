"""
HW1 - Innovation Diffusion Analysis (ASUS Zenbook Duo)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
BASE = Path(__file__).parent
DATA = BASE / "data"
IMG = BASE / "img"
REPORT = BASE / "report"

for d in (DATA, IMG, REPORT):
    d.mkdir(exist_ok=True)

# --------------------------------------------------------------------
# 1. Load Statista Excel data (robust loader for messy headers)
# --------------------------------------------------------------------
from pandas import ExcelFile

xlsx_path = DATA / "2_in_1_tablet_shipments_forecast.xlsx"
if not xlsx_path.exists():
    raise FileNotFoundError(f"Excel not found at {xlsx_path}")

def try_extract_years_ship(df_any):
    """Return tidy (year, adopters_millions) if we can detect columns, else None."""
    # Lower-case & strip
    cols = [str(c).strip().lower() for c in df_any.columns]
    df_any.columns = cols

    # 1) If there is an explicit 'year' column, use it
    year_col = next((c for c in cols if "year" in c), None)

    # 2) Candidate numeric columns for shipments
    cand_ship = [c for c in cols if any(k in c for k in ["shipment", "unit", "volume", "million", "shipments"])]
    ship_col = cand_ship[0] if cand_ship else None

    # If headers are useless ('unnamed'), treat all cols as potential
    if year_col is None:
        # Heuristic: find a column with 4+ integers between 1990 and 2100
        for c in df_any.columns:
            s = pd.to_numeric(df_any[c], errors="coerce")
            if s.notna().sum() >= 3:
                vals = s.dropna().astype(int)
                if len(vals) >= 3 and ((vals >= 1990) & (vals <= 2100)).mean() > 0.7:
                    year_col = c
                    break

    if ship_col is None:
        # Pick a numeric column (not the year col) with plausible magnitudes
        # Statista IDC numbers here are in "millions", e.g., 200–300 range.
        best = None
        for c in df_any.columns:
            if c == year_col:
                continue
            s = pd.to_numeric(df_any[c], errors="coerce")
            if s.notna().sum() >= 3:
                # Heuristic: values between ~1 and ~5000, with at least 3 non-nulls
                v = s.dropna()
                if (v.between(1, 5000).mean() > 0.7):
                    best = c
                    break
        ship_col = best

    if year_col is None or ship_col is None:
        return None

    out = pd.DataFrame({
        "year": pd.to_numeric(df_any[year_col], errors="coerce"),
        "adopters_millions": pd.to_numeric(df_any[ship_col], errors="coerce"),
    }).dropna()

    # Keep plausible years & positive shipments
    out = out[(out["year"].between(1990, 2100)) & (out["adopters_millions"] > 0)]
    if out.empty:
        return None

    # Coerce types and sort
    out = out.astype({"year": int, "adopters_millions": float}).sort_values("year").reset_index(drop=True)
    return out

# Try all sheets and multiple header strategies
tidy = None
xf = ExcelFile(xlsx_path)
print("Sheets found:", xf.sheet_names)

for sheet in xf.sheet_names:
    # Try with default header
    try:
        df0 = pd.read_excel(xlsx_path, sheet_name=sheet)
        if df0 is not None and not df0.empty:
            t = try_extract_years_ship(df0.copy())
            if t is not None:
                tidy = t; print(f"Parsed with default header on sheet '{sheet}'"); break
    except Exception:
        pass

    # Try header=None (no header) and then assign simple column names
    try:
        df1 = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
        # Make fake headers
        df1.columns = [f"col_{i}" for i in range(df1.shape[1])]
        t = try_extract_years_ship(df1.copy())
        if t is not None:
            tidy = t; print(f"Parsed with header=None on sheet '{sheet}'"); break
    except Exception:
        pass

if tidy is None:
    # Final debug output to help us identify columns
    print("Could not auto-parse. Here are the first 20 rows of the first sheet for inspection:")
    dbg = pd.read_excel(xlsx_path, sheet_name=0, header=None)
    print(dbg.head(20))
    raise RuntimeError("Could not detect Year and Shipments columns; please tell me which columns contain Year and the numbers.")

# Save and display the cleaned series
df = tidy.copy()
df.to_csv(DATA / "lookalike_timeseries.csv", index=False)
print("Cleaned series:")
print(df)

# --------------------------------------------------------------------
# 2. Bass model helper
# --------------------------------------------------------------------
def simulate_bass(p, q, M, periods):
    """Discrete Bass model simulation"""
    n = np.zeros(periods)
    cum = 0.0
    for t in range(periods):
        n[t] = (p + q * (cum / M)) * (M - cum)
        cum += n[t]
    return n

# --------------------------------------------------------------------
# 3. Fit Bass parameters (p, q, M)
# --------------------------------------------------------------------
y = df["adopters_millions"].values
years = df["year"].values
N = len(y)

def sse(p, q, M):
    if p <= 0 or q <= 0 or M <= max(y) * 1.05:
        return 1e18
    n = simulate_bass(p, q, M, N)
    return float(np.sum((n - y) ** 2))

rng = np.random.default_rng(42)
best = {"sse": 1e99, "p": None, "q": None, "M": None}

total = float(np.sum(y))
low_M = max(float(max(y)) * 1.2, total * 0.6)   # at least bigger than a single year and a big chunk of total
high_M = total * 5.0                             # generous upper bound

for _ in range(50000):
    p_try = float(rng.uniform(0.001, 0.08))
    q_try = float(rng.uniform(0.05, 0.95))
    # ensure low_M < high_M
    low, high = sorted([low_M, high_M])
    M_try = float(rng.uniform(low, high))
    s = sse(p_try, q_try, M_try)
    if s < best["sse"]:
        best.update({"p": p_try, "q": q_try, "M": M_try, "sse": s})


print("\nEstimated Bass parameters:")
print(best)

# --------------------------------------------------------------------
# 4. Plot fit
# --------------------------------------------------------------------
n_hat = simulate_bass(best["p"], best["q"], best["M"], N)
plt.plot(years, y, marker="o", label="Actual (M)")
plt.plot(years, n_hat, marker="s", label="Bass fit (M)")
plt.xlabel("Year")
plt.ylabel("Shipments (millions)")
plt.title("Look-alike diffusion fit (2-in-1 tablet shipments)")
plt.legend()
plt.tight_layout()
plt.savefig(IMG / "lookalike_fit.png", dpi=160)
plt.show()

# --------------------------------------------------------------------
# 5. Forecast Zenbook Duo adoption (transfer p, q)
# --------------------------------------------------------------------
forecast_years = np.arange(2025, 2036)
scenarios = {"low": 0.10, "base": 0.25, "high": 0.50}
rows = []
for label, M in scenarios.items():
    n = simulate_bass(best["p"], best["q"], M, len(forecast_years))
    cum = np.cumsum(n)
    for yr, nn, cc in zip(forecast_years, n, cum):
        rows.append({"scenario": label, "year": yr,
                     "new_adopters_millions": nn,
                     "cumulative_millions": cc})
forecast_df = pd.DataFrame(rows)
forecast_df.to_csv(DATA / "zenbook_duo_forecast.csv", index=False)
print(forecast_df.head())

# --------------------------------------------------------------------
# 6. Plot forecasts
# --------------------------------------------------------------------
plt.figure()
for sc in scenarios:
    d = forecast_df[forecast_df["scenario"] == sc]
    plt.plot(d["year"], d["new_adopters_millions"], marker="o", label=sc)
plt.xlabel("Year"); plt.ylabel("New adopters (M)")
plt.title("ASUS Zenbook Duo – annual adopters")
plt.legend(); plt.tight_layout()
plt.savefig(IMG / "zenbook_duo_annual.png", dpi=160)
plt.show()

plt.figure()
for sc in scenarios:
    d = forecast_df[forecast_df["scenario"] == sc]
    plt.plot(d["year"], d["cumulative_millions"], marker="s", label=sc)
plt.xlabel("Year"); plt.ylabel("Cumulative adopters (M)")
plt.title("ASUS Zenbook Duo – cumulative adopters")
plt.legend(); plt.tight_layout()
plt.savefig(IMG / "zenbook_duo_cumulative.png", dpi=160)
plt.show()