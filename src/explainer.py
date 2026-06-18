
"""
Explainability: Integrated Gradients (text) + GNNExplainer (graph)
"""
import torch
import torch.nn.functional as F


class TextExplainer:
    def explain(self, model, batch, target_class=None, n_steps=20):
        model.eval()
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        embed_fn = model.semantic_channel.bert.embeddings.word_embeddings
        orig_embeds = embed_fn(input_ids)
        baseline = torch.zeros_like(orig_embeds)
        attributions = torch.zeros_like(orig_embeds)
        for alpha in torch.linspace(0, 1, n_steps):
            interp = (baseline + alpha * (orig_embeds - baseline)).requires_grad_(True)
            out = model.semantic_channel.bert(inputs_embeds=interp, attention_mask=attention_mask)
            cls_out = out.last_hidden_state[:, 0, :]
            score = cls_out.sum()
            score.backward(retain_graph=True)
            attributions += interp.grad / n_steps
        token_scores = attributions.norm(dim=-1) * attention_mask.float()
        return token_scores.detach().cpu()


class GraphExplainer:
    def explain(self, model, batch, num_epochs=50):
        model.eval()
        nf = batch["node_features"].detach().requires_grad_(True)
        ei = batch["edge_index"]
        et = batch["edge_type"]
        node_mask = torch.ones(nf.size(0), device=nf.device, requires_grad=True)
        edge_mask = torch.ones(ei.size(1), device=ei.device, requires_grad=True)
        opt = torch.optim.Adam([node_mask, edge_mask], lr=0.01)
        for _ in range(num_epochs):
            opt.zero_grad()
            masked = nf * node_mask.unsqueeze(-1).sigmoid()
            emb = model.knowledge_channel.rgcn(masked, ei, et)
            g = emb.mean(dim=0, keepdim=True)
            logits = model.classifier(g)
            loss = -logits.max()
            loss.backward()
            opt.step()
        return {
            "node_importance": node_mask.detach().sigmoid().cpu(),
            "edge_importance": edge_mask.detach().sigmoid().cpu(),
            "edge_index": ei.cpu(), "edge_type": et.cpu(),
        }
