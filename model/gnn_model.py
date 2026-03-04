import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class FinancialGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=0.3, training=self.training)

        h2 = self.conv2(h, edge_index)
        h = h + h2  # residual connection

        # Nonlinear risk mapping
        out = self.out(h)
        out = torch.tanh(out) * 5   # bound outputs, preserve asymmetry
        return out

