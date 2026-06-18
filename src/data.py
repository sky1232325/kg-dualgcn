"""
Data processing pipeline:
Text preprocessing -> UMLS entity extraction -> Subgraph construction -> Dataset
"""
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional


NEGATION_TRIGGERS = [
    "no", "not", "without", "denies", "denied", "absent",
    "never", "none", "negative", "cannot", "unable",
    "\u65e0", "\u5426\u5b9a", "\u672a\u89c1", "\u672a\u53ca"
]


def preprocess_text(text, max_length=512):
    """Clinical text preprocessing: de-identify, clean, normalize."""
    text = re.sub(r'\b1[3-9]\d{9}\b', '[PHONE]', text)
    text = re.sub(r'\b\d{6,10}\b', '[ID]', text)
    text = text.replace('\u3000', ' ').replace('\xa0', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def detect_negation(text, entity_span, window=5):
    """Check if an entity mention is in a negation context."""
    start, end = entity_span
    prefix = text[max(0, start - window * 3):start].lower()
    for trigger in NEGATION_TRIGGERS:
        if trigger in prefix:
            return True
    return False


class UMLSEntityExtractor:
    """Extract UMLS entities from clinical text using QuickUMLS or regex fallback."""

    def __init__(self, quickumls_path=None, threshold=0.7):
        self.quickumls = None
        if quickumls_path:
            try:
                from quickumls import QuickUMLS
                self.quickumls = QuickUMLS(quickumls_path, threshold=threshold)
            except ImportError:
                pass

    def extract(self, text):
        """Extract entities with CUI codes."""
        if self.quickumls:
            return self._extract_quickumls(text)
        return self._extract_regex(text)

    def _extract_quickumls(self, text):
        matches = self.quickumls.match(text, best_match=True)
        entities = []
        for match in matches:
            for m in match:
                if not detect_negation(text, (m["start"], m["end"])):
                    entities.append({
                        "cui": m["cui"], "name": m["term"],
                        "start": m["start"], "end": m["end"],
                        "similarity": m["similarity"],
                    })
        return entities

    def _extract_regex(self, text):
        """Regex-based fallback for demo/testing."""
        AD_TERMS = {
            "C0002395": ["\u963f\u5c14\u8328\u6d77\u9ed8", "Alzheimer", "AD", "\u8001\u5e74\u75f4\u5446"],
            "C0338656": ["\u8bb0\u5fc6\u529b\u4e0b\u964d", "\u8bb0\u5fc6\u51cf\u9000"],
            "C0006118": ["MCI", "\u8f7b\u5ea6\u8ba4\u77e5\u969c\u788d"],
            "C0011206": ["\u5b9a\u5411\u529b\u969c\u788d"],
            "C0085584": ["\u8bed\u8a00\u969c\u788d", "\u5931\u8bed"],
            "C0020538": ["\u9ad8\u8840\u538b"],
            "C0011849": ["\u7cd6\u5c3f\u75c5"],
            "C0038454": ["\u8111\u840e\u7f29"],
            "C0242422": ["\u8ba4\u77e5\u529f\u80fd\u4e0b\u964d"],
            "C0030319": ["\u8111\u8840\u7ba1\u75c5"],
            "C0026769": ["\u5e15\u91d1\u68ee"],
            "C0007222": ["\u5fc3\u8840\u7ba1\u75c5"],
        }
        entities = []
        for cui, terms in AD_TERMS.items():
            for term in terms:
                for m in re.finditer(re.escape(term), text):
                    if not detect_negation(text, (m.start(), m.end())):
                        entities.append({
                            "cui": cui, "name": term,
                            "start": m.start(), "end": m.end(),
                            "similarity": 1.0,
                        })
        seen = set()
        unique = []
        for e in entities:
            key = (e["cui"], e["start"])
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique


class SubgraphBuilder:
    """Build patient-specific UMLS subgraph from extracted entities."""

    def __init__(self, umls_relations=None):
        self.native_relations = umls_relations or {
            ("C0002395", "C0338656"): "manifestation_of",
            ("C0002395", "C0006118"): "progresses_to",
            ("C0002395", "C0038454"): "associated_with",
            ("C0006118", "C0338656"): "manifestation_of",
            ("C0006118", "C0242422"): "has_finding",
            ("C0011206", "C0002395"): "manifestation_of",
            ("C0085584", "C0002395"): "manifestation_of",
            ("C0020538", "C0002395"): "risk_factor_of",
            ("C0011849", "C0002395"): "risk_factor_of",
            ("C0242422", "C0006118"): "associated_with",
            ("C0038454", "C0002395"): "associated_with",
        }
        self.rel_types = {
            "manifestation_of": 0, "progresses_to": 1,
            "associated_with": 2, "risk_factor_of": 3,
            "has_finding": 3, "co_occurrence": 2,
        }

    def build(self, entities):
        """Build graph: returns (node_features, edge_index, edge_type)."""
        if not entities:
            return (torch.zeros(1, 768), torch.zeros(2, 0, dtype=torch.long),
                    torch.zeros(0, dtype=torch.long))

        cui_set = list(dict.fromkeys(e["cui"] for e in entities))
        cui2idx = {c: i for i, c in enumerate(cui_set)}
        num_nodes = len(cui_set)
        node_features = torch.randn(num_nodes, 768) * 0.1

        edges_src, edges_dst, edge_types = [], [], []

        for (c1, c2), rel_name in self.native_relations.items():
            if c1 in cui2idx and c2 in cui2idx:
                rtype = self.rel_types.get(rel_name, 2)
                edges_src.extend([cui2idx[c1], cui2idx[c2]])
                edges_dst.extend([cui2idx[c2], cui2idx[c1]])
                edge_types.extend([rtype, rtype])

        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                c1, c2 = entities[i]["cui"], entities[j]["cui"]
                if c1 != c2 and c1 in cui2idx and c2 in cui2idx:
                    edges_src.extend([cui2idx[c1], cui2idx[c2]])
                    edges_dst.extend([cui2idx[c2], cui2idx[c1]])
                    edge_types.extend([self.rel_types["co_occurrence"]] * 2)

        if not edges_src:
            edge_index = torch.arange(num_nodes).unsqueeze(0).repeat(2, 1)
            edge_type = torch.zeros(num_nodes, dtype=torch.long)
        else:
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
            edge_type = torch.tensor(edge_types, dtype=torch.long)

        return node_features, edge_index, edge_type


class ClinicalNoteDataset(Dataset):
    """Dataset for clinical notes with dual-channel features."""

    def __init__(self, texts, labels, tokenizer, max_length=512,
                 extractor=None, graph_builder=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.extractor = extractor or UMLSEntityExtractor()
        self.graph_builder = graph_builder or SubgraphBuilder()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        clean_text = preprocess_text(text, self.max_length)
        encoding = self.tokenizer(
            clean_text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        entities = self.extractor.extract(clean_text)
        node_features, edge_index, edge_type = self.graph_builder.build(entities)
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "node_features": node_features, "edge_index": edge_index,
            "edge_type": edge_type,
            "label": torch.tensor(label, dtype=torch.long),
            "text": clean_text, "entities": entities,
        }


def collate_fn(batch):
    """Custom collate for variable-size graphs."""
    B = len(batch)
    max_nodes = max(item["node_features"].size(0) for item in batch)
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    d = batch[0]["node_features"].size(1)
    padded_nf = torch.zeros(B * max_nodes, d)
    node_batch = torch.zeros(B * max_nodes, dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["node_features"].size(0)
        padded_nf[i * max_nodes:i * max_nodes + n] = item["node_features"]
        node_batch[i * max_nodes:i * max_nodes + n] = i
    edge_offsets = torch.cumsum(
        torch.tensor([0] + [item["node_features"].size(0) for item in batch[:-1]]), 0)
    all_src, all_dst, all_types = [], [], []
    for i, item in enumerate(batch):
        if item["edge_index"].size(1) > 0:
            all_src.append(item["edge_index"][0] + edge_offsets[i])
            all_dst.append(item["edge_index"][1] + edge_offsets[i])
            all_types.append(item["edge_type"])
    if all_src:
        edge_index = torch.stack([torch.cat(all_src), torch.cat(all_dst)])
        edge_type = torch.cat(all_types)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_type = torch.zeros(0, dtype=torch.long)
    return {
        "input_ids": input_ids, "attention_mask": attention_mask,
        "node_features": padded_nf, "edge_index": edge_index,
        "edge_type": edge_type, "node_batch": node_batch,
        "labels": labels,
        "texts": [item["text"] for item in batch],
        "entities_list": [item["entities"] for item in batch],
    }
