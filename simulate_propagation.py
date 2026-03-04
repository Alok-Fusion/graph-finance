import torch
import numpy as np
from model.gnn_model import FinancialGNN

# -----------------------------
# LOAD GRAPH + MODEL
# -----------------------------
data = torch.load(
    "data/graph_data.pt",
    weights_only=False
)

assert not torch.isnan(data.x).any(), "NaNs found in node features"

model = FinancialGNN(input_dim=data.x.shape[1])
model.load_state_dict(torch.load("model_weights.pt", weights_only=False))
model.eval()

# -----------------------------
# BASELINE PREDICTION
# -----------------------------
with torch.no_grad():
    baseline_risk = model(data).squeeze().numpy()

# -----------------------------
# APPLY SHOCK TO ONE COMPANY
# -----------------------------
shock_node = 0          # company index to shock
shock_strength = 3.0    # magnitude of shock

shocked_data = data.clone()
shocked_data.x[shock_node, 1] += shock_strength  # increase volatility feature

with torch.no_grad():
    shocked_risk = model(shocked_data).squeeze().numpy()

# -----------------------------
# MEASURE PROPAGATION
# -----------------------------
risk_diff = shocked_risk - baseline_risk

print("Risk change after shock:")
for i, delta in enumerate(risk_diff):
    print(f"Company {i}: ΔRisk = {delta:.4f}")
