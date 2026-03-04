<div align="center">

# 📊 Graph-Finance 📉

**A state-of-the-art Graph Neural Network (GNN) system for analyzing systemic financial risk, dual-shock scenarios, and market contagion.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

</div>

---

## 🌟 Overview

**Graph-Finance** leverages Graph Neural Networks to model and simulate how economic shocks (e.g., volatility spikes, sudden credit crunches) propagate through inter-company networks and entire industrial sectors. 

By representing companies as nodes and their financial relationships/correlations as edges, the system uncovers non-linear dependencies, identifying **Risk-insulated** fortresses and **Volatility-benefiting** assets within a complex financial ecosystem.

---

## 🔥 Key Features

- **🧠 Advanced Graph Neural Networks (GNN):** 
  Built precisely with PyTorch and PyTorch Geometric, extracting robust structural embeddings to predict company sensitivities dynamically.
- **🎯 Bounded Unsupervised Learning:** 
  The model is trained entirely unsupervised using unique constraints (variance targets, graph smoothness, and scale regularization) to ensure realistic, grounded risk outputs.
- **⚡ Systemic Risk & Contagion Simulation:** 
  Includes robust multi-step propagation scripts to observe how isolated events escalate into multi-sector systemic threats.
- **🏢 Dual-Shock Scenario Dashboard:** 
  A highly interactive **Streamlit Dashboard** that lets you simulate both positive and negative shocks. It instantly computes **net-effects** and translates complex model outputs into distinct economic categories (*Defensive / Leverage-sensitive*, *Risk-insulated*, *Pro-cyclical Sector*).

---

## 🏗️ Project Architecture & Structure

```text
graph-finance/
├── 📂 data/                           # Generated synthetic graph objects & sector mappings
├── 📂 lib/                            # Helper utilities and project libraries
├── 📂 model/                          # Financial GNN architectures (e.g., FinancialGNN)
├── 📂 graph/                          # Drilldown views, animations, and risk evolution outputs
│
├── 🧠 train.py & variants             # Unsupervised GNN training pipelines
├── ⚡ simulate_propagation.py         # Simulates multi-step market shock contagion
├── 📊 shock_analysis_dashboard.py     # Main interactive Dual-Shock Streamlit Dashboard
├── 🧬 generate_synthetic_finance_data # Generator for deterministic synthetic finance ecosystems
└── 📄 requirements.txt                # Core project dependencies
```

---

## 🚀 Getting Started

### 1️⃣ Installation

Make sure your virtual environment is active, then install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2️⃣ Generating Data & Training the Model

If you are running the project natively for the first time, build the synthetic data and execute the unsupervised training sequence:

```bash
python generate_synthetic_finance_data.py
python train.py
```

### 3️⃣ Launch the Interactive Dashboard

To boot the interactive dual-shock Streamlit interface:

```bash
streamlit run shock_analysis_dashboard.py
```

> **Tip:** You can adjust the **Shock Magnitude** via the sidebar controls and select individual target companies to measure their isolated impact and the ensuing sector spillover effects!

---

<div align="center">
  <i>Built with precision to push the boundaries of Machine Learning in Quantitative Finance.</i>
</div>
