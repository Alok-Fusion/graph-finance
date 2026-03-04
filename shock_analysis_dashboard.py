import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
from model.gnn_model import FinancialGNN

st.set_page_config(layout="wide", page_title="Dual Shock Analysis")

# =============================
# LOAD DATA
# =============================
data = torch.load("data/graph_data.pt", weights_only=False)
sectors_df = pd.read_csv("data/company_sector.csv")

companies = sectors_df["company"].unique().tolist()
sector_map = (
    sectors_df
    .groupby("company")["sector"]
    .apply(list)
    .to_dict()
)


# =============================
# USER CONTROLS
# =============================
st.sidebar.header("⚙️ Shock Configuration")

shock_company = st.sidebar.selectbox(
    "Select company to shock:",
    companies
)

shock_node = companies.index(shock_company)
shock_strength = st.sidebar.slider(
    "Shock magnitude",
    min_value=1.0,
    max_value=5.0,
    value=3.0,
    step=0.5
)

st.sidebar.info(
    "Positive shock = volatility increase\n"
    "Negative shock = volatility decrease"
)

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

results = {}

for label, shock in {
    "Positive Shock": +shock_strength,
    "Negative Shock": -shock_strength
}.items():

    shocked = data.clone()
    # shock primary company
    # volatility index = 1
    # sensitivity index = 3 (based on feature order)
    vol_idx = 1
    sens_idx = 3

    # primary shock
    shocked.x[shock_node, vol_idx] += shock
    # sector spillover
    for i, c in enumerate(companies):
        if i != shock_node:
            if any(s in sector_map[c] for s in sector_map[shock_company]):
                shocked.x[i, vol_idx] += 0.3 * shock * shocked.x[i, sens_idx]


    # shock sector peers (weaker)
    for i, c in enumerate(companies):
        if c != shock_company:
            if any(s in sector_map[c] for s in sector_map[shock_company]):
                shocked.x[i, 1] += 0.3 * shock

    with torch.no_grad():
        pred = model(shocked).squeeze().numpy()

    delta = (pred - baseline) / shock_strength * 100
    delta = np.tanh(delta / 5) * 10
    results[label] = delta.astype(float)

net_effect = results["Positive Shock"] - results["Negative Shock"]

# =============================
# COMPANY TABLE
# =============================
company_df = pd.DataFrame({
    "Company": companies,
    "Sectors": [", ".join(sector_map[c]) for c in companies],
    "Positive Shock (%)": results["Positive Shock"],
    "Negative Shock (%)": results["Negative Shock"],
    "Net Effect (%)": net_effect
})

company_df["Interpretation"] = company_df["Net Effect (%)"].apply(
    lambda x: "Downside-stabilizing" if x > 0 else
              "Upside-stabilizing" if x < 0 else
              "Neutral"
)

company_df["Abs Impact"] = company_df["Net Effect (%)"].abs()
company_df = company_df.sort_values("Abs Impact", ascending=False)

company_df = company_df.round(3)

# =============================
# ECONOMIC INTUITION TABLE
# =============================

# sensitivity feature index (as defined earlier)
SENS_IDX = 3

sensitivities = data.x[:, SENS_IDX].detach().cpu().numpy()

company_df["Volatility Sensitivity"] = sensitivities
company_df["Volatility Sensitivity"] = company_df["Volatility Sensitivity"].round(2)

def economic_label(row):
    if row["Volatility Sensitivity"] > 1.3 and row["Net Effect (%)"] > 5:
        return "Defensive / Leverage-sensitive"
    elif row["Volatility Sensitivity"] > 1.3 and row["Net Effect (%)"] < -5:
        return "Volatility-benefiting (Convex payoff)"
    elif abs(row["Net Effect (%)"]) < 3:
        return "Risk-insulated / Stable"
    else:
        return "Moderate asymmetric exposure"

company_df["Economic Interpretation"] = company_df.apply(economic_label, axis=1)

# =============================
# SECTOR AGGREGATION
# =============================
sector_df = (
    company_df
    .assign(Sector=company_df["Sectors"].str.split(", "))
    .explode("Sector")
    .groupby("Sector")
    .agg({
        "Positive Shock (%)": "mean",
        "Negative Shock (%)": "mean",
        "Net Effect (%)": "mean"
    })
    .reset_index()
)

sector_df["Interpretation"] = sector_df["Net Effect (%)"].apply(
    lambda x: "Defensive Sector" if x > 0 else
              "Pro-cyclical Sector" if x < 0 else
              "Neutral"
)

sector_df["Abs Impact"] = sector_df["Net Effect (%)"].abs()
sector_df = sector_df.sort_values("Abs Impact", ascending=False)
sector_df = sector_df.round(3)

# =============================
# UI TABS
# =============================
tab1, tab2 = st.tabs(["🏢 Company Analysis", "🏭 Sector Analysis"])

# ==================================================
# TAB 1 — COMPANY ANALYSIS
# ==================================================
with tab1:
    st.header("🏢 Company-Level Shock Impact")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Companies Ranked by Shock Sensitivity")
        st.dataframe(
            company_df[[
                "Company",
                "Sectors",
                "Positive Shock (%)",
                "Negative Shock (%)",
                "Net Effect (%)",
                "Interpretation"
            ]],
            use_container_width=True
        )

    company = st.selectbox(
        "Select a company for detailed view:",
        company_df["Company"].tolist()
    )

    row = company_df[company_df["Company"] == company].iloc[0]

    with col2:
        st.subheader(f"📌 {company} — Detailed Impact")

        st.markdown(
            f"**Interpretation:** `{row['Interpretation']}`  \n"
            f"**Sector:** `{row['Sectors']}`"
        )

        metrics = pd.DataFrame({
            "Metric": ["Positive Shock", "Negative Shock", "Net Effect"],
            "Value (%)": [
                row["Positive Shock (%)"],
                row["Negative Shock (%)"],
                row["Net Effect (%)"]
            ]
        })

        st.table(metrics)

        fig, ax = plt.subplots()
        ax.bar(
            metrics["Metric"],
            metrics["Value (%)"],
            color=["#22c55e", "#ef4444", "#3b82f6"]
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Impact (%)")
        ax.set_title("Shock Response Profile")

        st.pyplot(fig)

        st.subheader("📌 Economic Intuition Summary (Top Impact Firms)")

        top_econ = company_df.sort_values(
            "Abs Impact", ascending=False
        ).head(10)

        st.dataframe(
            top_econ[[
                "Company",
                "Sectors",
                "Volatility Sensitivity",
                "Positive Shock (%)",
                "Negative Shock (%)",
                "Net Effect (%)",
                "Economic Interpretation"
            ]],
            use_container_width=True
        )

# ==================================================
# TAB 2 — SECTOR ANALYSIS
# ==================================================
with tab2:
    st.header("🏭 Sector-Level Aggregated Impact")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Sectors Ranked by Systemic Impact")
        st.dataframe(
            sector_df[[
                "Sector",
                "Positive Shock (%)",
                "Negative Shock (%)",
                "Net Effect (%)",
                "Interpretation"
            ]],
            use_container_width=True
        )

    sector = st.selectbox(
        "Select a sector for detailed view:",
        sector_df["Sector"].tolist()
    )

    srow = sector_df[sector_df["Sector"] == sector].iloc[0]

    with col2:
        st.subheader(f"📌 {sector} Sector — Aggregated Impact")

        st.markdown(
            f"**Interpretation:** `{srow['Interpretation']}`"
        )

        metrics = pd.DataFrame({
            "Metric": ["Avg Positive Shock", "Avg Negative Shock", "Avg Net Effect"],
            "Value (%)": [
                srow["Positive Shock (%)"],
                srow["Negative Shock (%)"],
                srow["Net Effect (%)"]
            ]
        })

        st.table(metrics)

        fig, ax = plt.subplots()
        ax.bar(
            metrics["Metric"],
            metrics["Value (%)"],
            color=["#22c55e", "#ef4444", "#3b82f6"]
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("Impact (%)")
        ax.set_title("Sector Shock Profile")

        st.pyplot(fig)