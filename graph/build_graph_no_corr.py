import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1. LOAD DATA
# -----------------------------
prices = pd.read_csv(
    "C:/Users/ak500/OneDrive/Desktop/IT/graph-finance/data/prices.csv"
)
sectors = pd.read_csv(
    "C:/Users/ak500/OneDrive/Desktop/IT/graph-finance/data/company_sector.csv"
)

# -----------------------------
# 2. ENCODE COMPANIES
# -----------------------------
le = LabelEncoder()
prices["company_id"] = le.fit_transform(prices["company"])
num_nodes = len(le.classes_)

# -----------------------------
# 3. FEATURE ENGINEERING
# -----------------------------
features = []
risk_labels = []

for company_id in range(num_nodes):
    subset = prices[prices["company_id"] == company_id].sort_values("date")
    returns = subset["close"].pct_change().dropna()

    mean_return = returns.mean()
    volatility = returns.std()

    features.append([mean_return, volatility])
    risk_labels.append(volatility)   # volatility = risk proxy

x = torch.tensor(features, dtype=torch.float)
y = torch.tensor(risk_labels, dtype=torch.float).view(-1, 1)

# -----------------------------
# 4. BUILD EDGES
# -----------------------------
edges = []

# (A) SELF-LOOPS (MANDATORY FOR GCN)
for i in range(num_nodes):
    edges.append([i, i])

# (B) SECTOR-BASED EDGES
sector_map = dict(zip(sectors["company"], sectors["sector"]))

for i, ci in enumerate(le.classes_):
    for j, cj in enumerate(le.classes_):
        if i != j and sector_map.get(ci) == sector_map.get(cj):
            edges.append([i, j])

# -----------------------------
# 5. FINAL GRAPH OBJECT
# -----------------------------
if len(edges) == 0:
    raise ValueError("No edges created — graph is invalid.")

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

data = Data(
    x=x,
    edge_index=edge_index,
    y=y
)

torch.save(
    data,
    "C:/Users/ak500/OneDrive/Desktop/IT/graph-finance/data/graph_data_no_corr.pt"
)

print("Graph built successfully")
print(f"Nodes: {num_nodes}")
print(f"Edges: {edge_index.shape[1]}")
