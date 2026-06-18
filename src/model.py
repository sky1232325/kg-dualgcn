
"""
Knowledge-Guided Dual-Channel GCN for Explainable AD Detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rgcn_layer import RGCNEncoder
from .cross_attention import BidirectionalCrossAttention
from .explainer import TextExplainer, GraphExplainer


class SemanticChannel(nn.Module):
    def __init__(self, bert_model_name, hidden_dim, freeze_layers=0):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(bert_model_name)
        if freeze_layers > 0:
            for layer in self.bert.encoder.layer[:freeze_layers]:
                for p in layer.parameters():
                    p.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                        token_type_ids=token_type_ids, output_hidden_states=True)
        return out.last_hidden_state, out.last_hidden_state[:, 0, :]


class KnowledgeChannel(nn.Module):
    def __init__(self, node_embed_dim, hidden_dim, num_relations, num_layers, dropout):
        super().__init__()
        self.rgcn = RGCNEncoder(node_embed_dim, hidden_dim, hidden_dim,
                                 num_relations, num_layers, dropout)

    def forward(self, node_features, edge_index, edge_type, batch=None):
        node_emb = self.rgcn(node_features, edge_index, edge_type)
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            graph_emb = global_mean_pool(node_emb, batch)
        else:
            graph_emb = node_emb.mean(dim=0, keepdim=True)
        return node_emb, graph_emb


class DualChannelADModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.semantic_channel = SemanticChannel(
            config.bert_model_name, config.bert_hidden_dim, config.bert_freeze_layers)
        self.knowledge_channel = KnowledgeChannel(
            config.node_embed_dim, config.gcn_hidden_dim,
            config.gcn_num_relations, config.gcn_num_layers, config.gcn_dropout)
        self.fusion = BidirectionalCrossAttention(
            config.bert_hidden_dim, config.gcn_hidden_dim,
            config.fusion_dim, config.num_attention_heads, config.attention_dropout)
        self.classifier = nn.Sequential(
            nn.Linear(config.fusion_dim, config.classifier_hidden_dim),
            nn.ReLU(), nn.Dropout(config.classifier_dropout),
            nn.Linear(config.classifier_hidden_dim, config.num_classes))
        if config.use_explanation:
            self.text_explainer = TextExplainer()
            self.graph_explainer = GraphExplainer()

    def forward(self, batch):
        text_seq, text_cls = self.semantic_channel(
            batch["input_ids"], batch["attention_mask"])
        node_emb, graph_emb = self.knowledge_channel(
            batch["node_features"], batch["edge_index"],
            batch["edge_type"], batch.get("node_batch"))
        # Reshape node_emb from (total_nodes, d) to (B, max_nodes, d)
        B = batch["input_ids"].shape[0]
        if node_emb.dim() == 2 and "node_batch" in batch:
            nb = batch["node_batch"]
            max_nodes = (nb == 0).sum().item() if (nb == 0).any() else node_emb.size(0) // B
            d = node_emb.size(-1)
            node_emb_3d = torch.zeros(B, max_nodes, d, device=node_emb.device)
            for i in range(B):
                mask = (nb == i)
                n_i = mask.sum().item()
                node_emb_3d[i, :n_i] = node_emb[mask]
            node_emb = node_emb_3d
        elif node_emb.dim() == 2:
            node_emb = node_emb.unsqueeze(0).expand(B, -1, -1)
        fused = self.fusion(text_seq, node_emb, batch["attention_mask"])
        logits = self.classifier(fused)
        return {"logits": logits, "probs": F.softmax(logits, -1),
                "text_seq": text_seq, "text_cls": text_cls,
                "node_emb": node_emb, "graph_emb": graph_emb, "fused": fused}

    def explain(self, batch):
        with torch.enable_grad():
            return {
                "text_attributions": self.text_explainer.explain(self, batch),
                "graph_attributions": self.graph_explainer.explain(self, batch),
            }
