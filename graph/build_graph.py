import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1. LOAD DATA
# -----------------------------
prices = pd.read_csv(
    "C:/Users/ak500/OneDrive/Desktop/IT/graph-finance/data/prices.csv"
)
sectors_df = pd.read_csv(
    "C:/Users/ak500/OneDrive/Desktop/IT/graph-finance/data/company_sector.csv"
)

# Unique companies
companies = sorted(prices["company"].unique())
num_companies = len(companies)

# Encode companies
le = LabelEncoder()
prices["company_id"] = le.fit_transform(prices["company"])

# =============================
# BUILD MULTI-SECTOR MAP
# =============================
sector_list = sorted(sectors_df["sector"].unique())
sector_to_idx = {s: i for i, s in enumerate(sector_list)}

company_sector_map = (
    sectors_df
    .groupby("company")["sector"]
    .apply(list)
    .to_dict()
)

# =============================
# FEATURE ENGINEERING
# =============================
features = []

for company in companies:
    cid = le.transform([company])[0]
    subset = prices[prices["company"] == company]

    returns = subset["close"].pct_change().dropna()

    mean_return = returns.mean()
    volatility = returns.std()

    # firm identity embedding (break symmetry)
    firm_id_feat = cid / num_companies

    # multi-sector one-hot exposure
    sector_vec = np.zeros(len(sector_list))
    for s in company_sector_map[company]:
        sector_vec[sector_to_idx[s]] = 1.0

    # firm-specific volatility sensitivity
    # >1  = volatility amplifies risk
    # <1  = volatility dampens risk
    vol_sensitivity = np.random.uniform(0.6, 1.8)

    features.append(
        [mean_return, volatility, firm_id_feat, vol_sensitivity, *sector_vec]
    )


x = torch.tensor(features, dtype=torch.float)

# =============================
# BUILD EDGES (CORRELATION)
# =============================
pivot = prices.pivot_table(
    index="date",
    columns="company_id",
    values="close",
    aggfunc="mean"
)

returns = pivot.pct_change().dropna()
corr = returns.corr()

edges = []
CORR_THRESHOLD = 0.15

for i in range(num_companies):
    for j in range(num_companies):
        if i != j and abs(corr.iloc[i, j]) >= CORR_THRESHOLD:
            edges.append([i, j])

# self loops
for i in range(num_companies):
    edges.append([i, i])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# =============================
# SAVE GRAPH
# =============================
data = Data(x=x, edge_index=edge_index)
torch.save(data, "data/graph_data.pt")

print("✅ Graph built with heterogeneous node features")
print("Nodes:", data.num_nodes)
print("Features per node:", data.num_node_features)