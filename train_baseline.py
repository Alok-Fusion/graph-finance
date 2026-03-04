import torch
from model.baseline_mlp import BaselineMLP

data = torch.load(
    "data/graph_data.pt",
    weights_only=False
)

x = data.x
y = data.y

model = BaselineMLP(input_dim=x.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    optimizer.zero_grad()
    out = model(x)
    loss = ((out - y) ** 2).mean()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"[BASELINE] Epoch {epoch} | Loss: {loss.item():.4f}")
