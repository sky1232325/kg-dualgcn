"""
Relational Graph Convolutional Network (R-GCN) Encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class RGCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_relations, num_layers=2, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.layers.append(RGCNConv(in_dim, hidden_dim, num_relations))
        self.norms.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(RGCNConv(hidden_dim, hidden_dim, num_relations))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.layers.append(RGCNConv(hidden_dim, out_dim, num_relations))
        self.norms.append(nn.LayerNorm(out_dim))

    def forward(self, x, edge_index, edge_type):
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x, edge_index, edge_type)
            x = norm(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
