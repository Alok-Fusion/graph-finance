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
# BASELINE + SHOCK
# =============================
with torch.no_grad():
    baseline = model(data).squeeze().numpy()

shock_node = 0
shock_strength = 3.0

shocked = data.clone()
shocked.x[shock_node, 1] += shock_strength

with torch.no_grad():
    shocked_pred = model(shocked).squeeze().numpy()

# =============================
# CORRECT RISK % (SHOCK-RELATIVE)
# =============================
delta_risk = (shocked_pred - baseline).astype(float)
delta_risk_pct = (delta_risk / shock_strength) * 100

# clip extreme values (visual safety)
delta_risk_pct = np.clip(delta_risk_pct, -20, 20)

# log-scale for visualization clarity
vis_risk = np.sign(delta_risk_pct) * np.log1p(np.abs(delta_risk_pct))

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
# SECTOR AGGREGATED RISK (%)
# =============================
CORR_THRESHOLD = 0.2
sector_risk = {s: 0.0 for s in sectors}

for i, company in enumerate(companies):
    for sector in sectors:
        sector_companies = [
            companies.index(c)
            for c, s in sector_map.items()
            if s == sector
        ]

        avg_corr = np.mean([
            corr.iloc[i, j]
            for j in sector_companies
            if not np.isnan(corr.iloc[i, j])
        ])

        if abs(avg_corr) >= CORR_THRESHOLD:
            sector_risk[sector] += abs(avg_corr * delta_risk_pct[i])

# =============================
# GRAPHDB VISUALIZATION
# =============================
net = Network(
    height="800px",
    width="100%",
    bgcolor="#020617",
    font_color="white"
)
net.force_atlas_2based(
    gravity=-20,
    central_gravity=0.02,
    spring_length=140
)

# -----------------------------
# SECTOR NODES (CONTROLLED SIZE)
# -----------------------------
for sector in sectors:
    raw = sector_risk[sector]
    risk = np.log1p(abs(raw))

    size = min(80 + risk * 160, 220)
    color = "#dc2626" if raw > 0 else "#16a34a"

    net.add_node(
        sector,
        label=sector,
        shape="ellipse",
        size=size,
        color=color,
        title=f"Sector: {sector}<br>Total Impact: {raw:.2f}%"
    )

# -----------------------------
# COMPANY NODES
# -----------------------------
for i, company in enumerate(companies):
    r = vis_risk[i]

    size = min(25 + abs(r) * 100, 120)
    color = "#ef4444" if r > 0 else "#22c55e"

    net.add_node(
        company,
        label=company,
        size=size,
        color=color,
        title=f"Company: {company}<br>ΔRisk: {delta_risk_pct[i]:.2f}%"
    )

# -----------------------------
# EDGES (MULTI-SECTOR SPILLOVER)
# -----------------------------
for i, company in enumerate(companies):
    for sector in sectors:
        sector_companies = [
            companies.index(c)
            for c, s in sector_map.items()
            if s == sector
        ]

        avg_corr = np.mean([
            corr.iloc[i, j]
            for j in sector_companies
            if not np.isnan(corr.iloc[i, j])
        ])

        if abs(avg_corr) >= CORR_THRESHOLD:
            edge_color = (
                "#facc15" if sector_map[company] != sector else "#64748b"
            )

            net.add_edge(
                company,
                sector,
                value=float(abs(avg_corr) * 10),
                color=edge_color,
                title=f"Avg Corr: {avg_corr:.2f}"
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
🔴 Red Node → Risk Increase (%)<br>
🟢 Green Node → Risk Decrease (%)<br>
⭕ Node Size → Impact Magnitude<br>
🟡 Yellow Edge → Cross-Sector Spillover<br>
⚪ Grey Edge → Same-Sector Exposure
</div>
"""

# =============================
# SAVE
# =============================
net.write_html("graphdb_multisector_risk_percent_clean.html", open_browser=True)

print("\nCOMPANY RISK IMPACT (%)")
for c, r in zip(companies, delta_risk_pct):
    print(f"{c}: {r:.2f}%")

print("\nSECTOR RISK IMPACT (%)")
for sector, risk in sector_risk.items():
    print(f"{sector}: {risk:.2f}%")

print("\nGraph saved as graphdb_multisector_risk_percent_clean.html")
