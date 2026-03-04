import torch
import pandas as pd
import numpy as np
from model.gnn_model import FinancialGNN

# Load data
data = torch.load("data/graph_data.pt", weights_only=False)
sectors = pd.read_csv("data/company_sector.csv")

# Load model
model = FinancialGNN(input_dim=data.x.shape[1])
model.load_state_dict(torch.load("model_weights.pt", weights_only=False))
model.eval()

# Baseline prediction
with torch.no_grad():
    baseline = model(data).squeeze().numpy()

# Apply shock
shock_node = 0
shock_strength = 3.0

shocked = data.clone()
shocked.x[shock_node, 1] += shock_strength

with torch.no_grad():
    shocked_pred = model(shocked).squeeze().numpy()

# Delta risk
delta_risk = shocked_pred - baseline

# Map node → sector
sector_map = dict(zip(sectors["company"], sectors["sector"]))

# LabelEncoder order
companies = list(sector_map.keys())

sector_changes = {}

for idx, delta in enumerate(delta_risk):
    company = companies[idx]
    sector = sector_map[company]

    sector_changes.setdefault(sector, []).append(delta)

# Compute percentage impact
print("\nSector-wise Risk Impact (%)")
for sector, values in sector_changes.items():
    avg_change = np.mean(values)
    impact_pct = (avg_change / shock_strength) * 100
    print(f"{sector}: {impact_pct:.2f}%")

print("\nSUMMARY:")
for sector, values in sector_changes.items():
    avg = np.mean(values)
    if abs(avg) < 0.001:
        print(f"- {sector}: No significant impact")
    else:
        print(f"- {sector}: Affected (avg ΔRisk = {avg:.4f})")
