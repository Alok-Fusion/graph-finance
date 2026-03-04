import numpy as np
import pandas as pd
from pyvis.network import Network
import json

# -----------------------------
# LOAD DATA
# -----------------------------
risk_history = np.load("risk_evolution.npy")
sectors_df = pd.read_csv("data/company_sector.csv")

companies = sectors_df["company"].tolist()
sector_map = dict(zip(sectors_df["company"], sectors_df["sector"]))

# -----------------------------
# CREATE NETWORK
# -----------------------------
net = Network(height="750px", width="100%", bgcolor="#0f172a", font_color="white")
net.force_atlas_2based()

# Add sector nodes
for sector in set(sector_map.values()):
    net.add_node(sector, label=sector, color="#64748b", size=35)

# Add company nodes (initial)
for i, company in enumerate(companies):
    net.add_node(company, label=company, size=25)

# Add edges company → sector
for company, sector in sector_map.items():
    net.add_edge(company, sector, color="#334155")

# -----------------------------
# ADD TIME-STEP DATA (CUSTOM JS)
# -----------------------------
frames = []
for t, risks in enumerate(risk_history):
    frame = {}
    for i, risk in enumerate(risks):
        frame[companies[i]] = float(risk)
    frames.append(frame)

net.set_options(json.dumps({
    "physics": {"enabled": True},
    "interaction": {"hover": True}
}))

net.write_html("graphdb_time_evolution.html")

# Inject animation data
with open("graphdb_time_evolution.html", "a", encoding="utf-8") as f:
    f.write("\n<!-- Risk evolution data -->\n")
    f.write("<script>\n")
    f.write(f"var riskFrames = {json.dumps(frames)};\n")
    f.write("</script>\n")

print("Animated GraphDB saved as graphdb_time_evolution.html")
print("Use a local server to view the animation correctly.")