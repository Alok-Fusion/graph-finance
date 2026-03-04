import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from model.gnn_model import FinancialGNN

# -----------------------------
# LOAD DATA & MODEL
# -----------------------------
data = torch.load("data/graph_data.pt", weights_only=False)
sectors_df = pd.read_csv("data/company_sector.csv")

companies = sectors_df["company"].tolist()
sector_map = dict(zip(sectors_df["company"], sectors_df["sector"]))

model = FinancialGNN(input_dim=data.x.shape[1])
model.load_state_dict(torch.load("model_weights.pt", weights_only=False))
model.eval()

# -----------------------------
# BASELINE
# -----------------------------
with torch.no_grad():
    baseline = model(data).squeeze().numpy()

# -----------------------------
# RANDOM SHOCK EXPERIMENT
# -----------------------------
N_RUNS = 50
shock_strength = 3.0
results = []

vol_idx = 1
sens_idx = 3

num_nodes = data.x.shape[0]

for _ in range(N_RUNS):
    shock_node = np.random.randint(0, num_nodes)

    shocked = data.clone()
    shocked.x[shock_node, vol_idx] += (
        shock_strength * shocked.x[shock_node, sens_idx]
    )

    with torch.no_grad():
        pred = model(shocked).squeeze().numpy()

    delta = (pred - baseline) / shock_strength * 100

    results.append({
        "Shocked Company": companies[shock_node],
        "Sector": sector_map[companies[shock_node]],
        "Avg Impact (%)": float(np.mean(delta)),
        "Std Impact (%)": float(np.std(delta))
    })

df = pd.DataFrame(results)

print("\nRANDOM SHOCK VALIDATION SUMMARY")
print(df.groupby("Sector")[["Avg Impact (%)", "Std Impact (%)"]].mean())

# -----------------------------
# AGGREGATE FOR PLOTTING
# -----------------------------
summary = (
    df.groupby("Sector")
    .agg({
        "Avg Impact (%)": "mean",
        "Std Impact (%)": "mean"
    })
    .reset_index()
)

# -----------------------------
# PLOT
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    summary["Sector"],
    summary["Avg Impact (%)"],
    yerr=summary["Std Impact (%)"],
    capsize=6,
    color="#3b82f6",
    alpha=0.8
)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Average Impact (%)")
ax.set_title("Random Shock Robustness Across Sectors")

plt.tight_layout()
plt.show()

# =============================
# ABLATION STUDY
# =============================
ablated_results = []

for _ in range(N_RUNS):
    shock_node = np.random.randint(0, num_nodes)

    shocked = data.clone()

    # ABLATION: remove sensitivity scaling
    shocked.x[shock_node, vol_idx] += shock_strength

    with torch.no_grad():
        pred = model(shocked).squeeze().numpy()

    delta = (pred - baseline) / shock_strength * 100

    ablated_results.append({
        "Shocked Company": companies[shock_node],
        "Sector": sector_map[companies[shock_node]],
        "Avg Impact (%)": np.mean(delta)
    })

ablated_df = pd.DataFrame(ablated_results)

# Aggregate by sector
ablated_sector = (
    ablated_df
    .groupby("Sector")["Avg Impact (%)"]
    .mean()
    .reset_index()
    .rename(columns={"Avg Impact (%)": "Ablated Impact (%)"})
)

comparison = summary.merge(
    ablated_sector,
    on="Sector",
    how="inner"
)

# -----------------------------
# ABLATION COMPARISON PLOT
# -----------------------------
fig, ax = plt.subplots(figsize=(9, 5))

x = np.arange(len(comparison))
width = 0.35

ax.bar(
    x - width/2,
    comparison["Avg Impact (%)"],
    width,
    label="Full Model",
    color="#3b82f6"
)

ax.bar(
    x + width/2,
    comparison["Ablated Impact (%)"],
    width,
    label="Ablated (No Sensitivity)",
    color="#9ca3af"
)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(comparison["Sector"], rotation=30)
ax.set_ylabel("Net Impact (%)")
ax.set_title("Ablation Study: Effect of Volatility Sensitivity")
ax.legend()

plt.tight_layout()
plt.show()
