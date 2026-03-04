import numpy as np
import pandas as pd
import torch
from pyvis.network import Network
from model.gnn_model import FinancialGNN

# =============================
# LOAD DATA
# =============================
data = torch.load("data/graph_data.pt", weights_only=False)
prices = pd.read_csv("data/prices.csv")
sectors_df = pd.read_csv("data/company_sector.csv")

companies = sectors_df["company"].tolist()
sector_map = dict(zip(sectors_df["company"], sectors_df["sector"]))
sectors = sorted(set(sector_map.values()))

# =============================
# LOAD MODEL
# =============================
model = FinancialGNN(input_dim=data.x.shape[1])
model.load_state_dict(torch.load("model_weights.pt", weights_only=False))
model.eval()

# =============================
# BASELINE
# =============================
with torch.no_grad():
    baseline = model(data).squeeze().numpy()

shock_node = 0        # index of shocked company
shock_strength = 3.0  # magnitude

# =============================
# APPLY DUAL SHOCKS
# =============================
results = {}

for label, shock in {
    "Positive Shock": +shock_strength,
    "Negative Shock": -shock_strength
}.items():

    shocked = data.clone()
    shocked.x[shock_node, 1] += shock

    with torch.no_grad():
        pred = model(shocked).squeeze().numpy()

    # shock-relative % (CORRECT)
    delta = (pred - baseline) / shock_strength * 100
    delta = np.clip(delta, -20, 20)

    results[label] = delta.astype(float)

# =============================
# NET EFFECT (ASYMMETRY)
# =============================
net_effect = (results["Positive Shock"] - results["Negative Shock"]).astype(float)

# =============================
# CORRELATION MATRIX
# =============================
prices["company_id"] = prices["company"].apply(lambda x: companies.index(x))

pivot = prices.pivot_table(
    index="date",
    columns="company_id",
    values="close",
    aggfunc="mean"
)

returns = pivot.pct_change().dropna()
corr = returns.corr()

# =============================
# SECTOR AGGREGATION (SAFE)
# =============================
CORR_THRESHOLD = 0.25
sector_risk = {s: 0.0 for s in sectors}

for i, company in enumerate(companies):
    for sector in sectors:
        sector_companies = [
            companies.index(c)
            for c, s in sector_map.items()
            if s == sector
        ]

        # SAFE correlation extraction
        vals = [
            float(corr.iloc[i, j])
            for j in sector_companies
            if not np.isnan(corr.iloc[i, j])
        ]

        if len(vals) == 0:
            continue

        avg_corr = float(np.mean(vals))

        if abs(avg_corr) >= CORR_THRESHOLD:
            sector_risk[sector] += abs(avg_corr * net_effect[i])

# =============================
# BUILD GRAPH
# =============================
net = Network(
    height="800px",
    width="100%",
    bgcolor="#020617",
    font_color="white"
)

net.force_atlas_2based(
    gravity=-12,
    central_gravity=0.015,
    spring_length=200,
    spring_strength=0.03
)

# =============================
# SECTOR NODES
# =============================
for sector in sectors:
    raw = float(sector_risk[sector])
    size = float(min(90 + np.log1p(abs(raw)) * 150, 230))
    color = "#dc2626" if raw > 0 else "#16a34a"

    net.add_node(
        sector,
        label=sector,
        group="sector",
        size=size,
        color=color,
        title=f"Sector: {sector}<br>Net Impact: {raw:.2f}%"
    )

# =============================
# COMPANY NODES
# =============================
for i, company in enumerate(companies):
    r = float(net_effect[i])
    size = float(min(35 + np.log1p(abs(r)) * 110, 130))
    color = "#ef4444" if r > 0 else "#22c55e"

    net.add_node(
        company,
        label=company,
        group="company",
        size=size,
        color=color,
        title=(
            f"Company: {company}<br>"
            f"+Shock: {results['Positive Shock'][i]:.2f}%<br>"
            f"-Shock: {results['Negative Shock'][i]:.2f}%<br>"
            f"Net Effect: {r:.2f}%"
        )
    )

# =============================
# EDGES (MEANINGFUL ONLY)
# =============================
EDGE_THRESHOLD = 0.3

for i, company in enumerate(companies):
    for sector in sectors:
        if sector_map[company] != sector:
            continue

        net.add_edge(
            company,
            sector,
            color="#64748b",
            width=2
        )

# =============================
# LEGEND
# =============================
net.html += """
<div style="
position:fixed;
bottom:20px;
left:20px;
background:#020617;
padding:12px;
color:white;
font-size:14px;
border-radius:10px;
box-shadow:0 0 10px #000;">
<b>Legend</b><br>
🔴 Red → Net Risk Increase<br>
🟢 Green → Net Risk Decrease<br>
⭕ Node Size → Asymmetry Magnitude<br>
⚪ Edge → Sector Membership
</div>
"""

# =============================
# SAVE
# =============================
net.write_html("graphdb_dual_shock_risk.html", open_browser=True)

print("\nDUAL SHOCK RESULTS (Net Effect %)")
for c, r in zip(companies, net_effect):
    print(f"{c}: {r:.2f}%")

print("\nGraph saved as graphdb_dual_shock_risk.html")
