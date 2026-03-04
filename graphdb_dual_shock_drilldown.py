import json
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

shock_node = 0
shock_strength = 3.0

results = {}

for label, shock in {
    "Positive Shock": +shock_strength,
    "Negative Shock": -shock_strength
}.items():

    shocked = data.clone()
    shocked.x[shock_node, 1] += shock

    with torch.no_grad():
        pred = model(shocked).squeeze().numpy()

    delta = (pred - baseline) / shock_strength * 100
    delta = np.clip(delta, -20, 20)
    results[label] = delta.astype(float)

net_effect = results["Positive Shock"] - results["Negative Shock"]

# =============================
# PREPARE DRILL-DOWN DATA
# =============================
company_details = {}

for i, c in enumerate(companies):
    company_details[c] = {
        "sector": sector_map[c],
        "positive_shock": float(results["Positive Shock"][i]),
        "negative_shock": float(results["Negative Shock"][i]),
        "net_effect": float(net_effect[i])
    }

# =============================
# GRAPH
# =============================
net = Network(
    height="700px",
    width="70%",
    bgcolor="#020617",
    font_color="white"
)

for i, c in enumerate(companies):
    r = float(net_effect[i])
    size = float(min(35 + np.log1p(abs(r)) * 110, 120))
    color = "#ef4444" if r > 0 else "#22c55e"

    net.add_node(
        c,
        label=c,
        size=size,
        color=color
    )

# Simple edges (sector membership)
for c in companies:
    net.add_node(sector_map[c], label=sector_map[c], color="#64748b", size=60)
    net.add_edge(c, sector_map[c], color="#64748b")

# =============================
# CUSTOM HTML (INFO PANEL)
# =============================
details_json = json.dumps(company_details)

net.html += f"""
<div style="
position:fixed;
top:0;
right:0;
width:30%;
height:100%;
background:#020617;
color:white;
padding:20px;
overflow:auto;
border-left:1px solid #334155;">
<h2>Company Details</h2>
<div id="info">Click a company node</div>
</div>

<script>
const companyData = {details_json};

network.on("click", function(params) {{
    if (params.nodes.length === 0) return;
    const node = params.nodes[0];
    if (!(node in companyData)) return;

    const d = companyData[node];
    document.getElementById("info").innerHTML = `
        <b>Company:</b> ${{node}}<br><br>
        <b>Sector:</b> ${{d.sector}}<br><br>
        <b>Positive Shock:</b> ${{d.positive_shock.toFixed(2)}}%<br>
        <b>Negative Shock:</b> ${{d.negative_shock.toFixed(2)}}%<br><br>
        <b>Net Effect:</b> ${{d.net_effect.toFixed(2)}}%
    `;
}});
</script>
"""

# =============================
# SAVE
# =============================
net.write_html("graphdb_dual_shock_drilldown.html", open_browser=True)

print("Saved: graphdb_dual_shock_drilldown.html")
