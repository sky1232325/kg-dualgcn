
"""
Bidirectional Cross-Attention Fusion Module
"""
import torch
import torch.nn as nn


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, dim_text, dim_graph, fusion_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.text_proj = nn.Linear(dim_text, fusion_dim)
        self.graph_proj = nn.Linear(dim_graph, fusion_dim)
        self.text2graph_attn = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.text2graph_norm = nn.LayerNorm(fusion_dim)
        self.graph2text_attn = nn.MultiheadAttention(
            fusion_dim, num_heads, dropout=dropout, batch_first=True)
        self.graph2text_norm = nn.LayerNorm(fusion_dim)
        self.fusion_proj = nn.Linear(fusion_dim * 2, fusion_dim)
        self.fusion_norm = nn.LayerNorm(fusion_dim)

    def forward(self, text_features, graph_features, text_mask=None):
        T = self.text_proj(text_features)
        G = self.graph_proj(graph_features)
        text_kpm = (text_mask == 0) if text_mask is not None else None
        T_enh, _ = self.text2graph_attn(query=T, key=G, value=G)
        T_out = self.text2graph_norm(T + T_enh)
        G_enh, _ = self.graph2text_attn(query=G, key=T, value=T, key_padding_mask=text_kpm)
        G_out = self.graph2text_norm(G + G_enh)
        T_g = T_out.mean(dim=1)
        G_g = G_out.mean(dim=1)
        fused = self.fusion_norm(self.fusion_proj(torch.cat([T_g, G_g], -1)))
        return fused
