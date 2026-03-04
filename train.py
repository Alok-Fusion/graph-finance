import torch
from model.gnn_model import FinancialGNN

# =============================
# LOAD GRAPH
# =============================
data = torch.load("data/graph_data.pt", weights_only=False)

model = FinancialGNN(input_dim=data.x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

model.train()

edge_index = data.edge_index

def graph_smoothness(out, edge_index):
    src, dst = edge_index
    return ((out[src] - out[dst]) ** 2).mean()

TARGET_VAR = 1.0  # desired output variance

# =============================
# TRAINING
# =============================
for epoch in range(300):
    optimizer.zero_grad()

    out = model(data).squeeze()

    # 1. Variance constraint (bounded)
    var_loss = (out.var() - TARGET_VAR) ** 2

    # 2. Graph smoothness
    smooth_loss = graph_smoothness(out, edge_index)

    # 3. Scale regularization
    reg_loss = (out ** 2).mean()

    loss = var_loss + 0.5 * smooth_loss + 0.01 * reg_loss

    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(
            f"Epoch {epoch} | "
            f"Loss: {loss.item():.4f} | "
            f"Var: {out.var().item():.4f}"
        )

torch.save(model.state_dict(), "model_weights.pt")
print("✅ Model trained (bounded unsupervised)")
