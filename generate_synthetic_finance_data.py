import numpy as np
import pandas as pd

np.random.seed(42)

# ==========================
# CONFIG
# ==========================
NUM_COMPANIES = 120
NUM_DAYS = 250
SECTORS = ["Energy", "IT", "Banking", "Healthcare", "Manufacturing", "Finance"]
START_DATE = "2023-01-01"

# ==========================
# COMPANY LIST
# ==========================
companies = [f"COMP_{i:03d}" for i in range(NUM_COMPANIES)]

# ==========================
# MULTI-SECTOR ASSIGNMENT
# ==========================
company_sector_rows = []

for company in companies:
    primary_sector = np.random.choice(SECTORS)
    
    # 30% companies have secondary exposure
    if np.random.rand() < 0.3:
        secondary_sector = np.random.choice(
            [s for s in SECTORS if s != primary_sector]
        )
        company_sector_rows.append((company, primary_sector))
        company_sector_rows.append((company, secondary_sector))
    else:
        company_sector_rows.append((company, primary_sector))

company_sector_df = pd.DataFrame(
    company_sector_rows,
    columns=["company", "sector"]
)

company_sector_df.to_csv("data/company_sector.csv", index=False)

# ==========================
# SECTOR VOLATILITY PROFILES
# ==========================
sector_vol = {
    "Energy": 0.020,
    "IT": 0.015,
    "Banking": 0.012,
    "Healthcare": 0.010,
    "Manufacturing": 0.013,
    "Finance": 0.016
}

# ==========================
# GENERATE PRICE DATA
# ==========================
dates = pd.date_range(start=START_DATE, periods=NUM_DAYS)

price_rows = []

for company in companies:
    sectors_of_company = company_sector_df[
        company_sector_df["company"] == company
    ]["sector"].tolist()

    # Base volatility is average of its sectors
    base_vol = np.mean([sector_vol[s] for s in sectors_of_company])

    price = np.random.uniform(100, 2000)

    for date in dates:
        # stochastic volatility
        vol = abs(np.random.normal(base_vol, base_vol * 0.3))

        # drift + noise
        daily_return = np.random.normal(0.0005, vol)
        price *= (1 + daily_return)

        price_rows.append((
            date.strftime("%Y-%m-%d"),
            company,
            round(price, 2)
        ))

prices_df = pd.DataFrame(
    price_rows,
    columns=["date", "company", "close"]
)

prices_df.to_csv("data/prices.csv", index=False)

print("✅ Synthetic finance data generated")
print(f"Companies: {NUM_COMPANIES}")
print(f"Days: {NUM_DAYS}")
print(f"Sectors: {len(SECTORS)}")
