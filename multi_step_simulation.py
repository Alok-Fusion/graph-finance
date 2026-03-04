import torch
import numpy as np
from model.gnn_model import FinancialGNN

# -----------------------------
# LOAD DATA + MODEL
# -----------------------------
data = torch.load("data/graph_data.pt", weights_only=False)

model = FinancialGNN(input_dim=data.x.shape[1])
model.load_state_dict(torch.load("model_weights.pt", weights_only=False))
model.eval()

# -----------------------------
# PARAMETERS
# -----------------------------
shock_nodes = [0, 2]       # multiple shocks
shock_strength = 3.0
time_steps = 5
decay = 0.6               # how shock weakens over time

# -----------------------------
# BASELINE
# -----------------------------
with torch.no_grad():
    baseline = model(data).squeeze().numpy()

# -----------------------------
# TIME-STEP SIMULATION
# -----------------------------
history = []

current_data = data.clone()

for t in range(time_steps):
    # Apply shock (decaying over time)
    for node in shock_nodes:
        current_data.x[node, 1] += shock_strength * (decay ** t)

    with torch.no_grad():
        pred = model(current_data).squeeze().numpy()

    delta = pred - baseline
    history.append(delta)

# Save result
np.save("risk_evolution.npy", np.array(history))
print("Multi-step simulation completed.")
print("Risk evolution saved to risk_evolution.npy")