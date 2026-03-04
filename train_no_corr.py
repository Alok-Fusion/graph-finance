import torch
from model.gnn_model import FinancialGNN

data = torch.load(
    "data/graph_data_no_corr.pt",
    weights_only=False
)

model = FinancialGNN(input_dim=data.x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = ((out - data.y) ** 2).mean()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"[NO-CORR] Epoch {epoch} | Loss: {loss.item():.4f}")
