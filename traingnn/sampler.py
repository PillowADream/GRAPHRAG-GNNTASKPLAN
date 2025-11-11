# sampler.py (modified for M3 anti-frequency & pair_weight support)
import random
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any, Callable

import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Optional: TF-IDF similarity for text negatives
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _HAS_SK = True
except Exception:
    _HAS_SK = False


# -----------------------------
# Utilities
# -----------------------------

def _bfs_shortest_paths(start: str, adj: Dict[str, List[str]]) -> Dict[str, int]:
    """Unweighted shortest path lengths from start to all nodes in adj (BFS)."""
    dist = {start: 0}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj.get(u, []):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


class _DictDataset(Dataset):
    def __init__(self, payload: Dict[str, torch.Tensor]) -> None:
        self.payload = payload
        # validate equal length
        n = None
        for v in payload.values():
            if n is None:
                n = v.shape[0]
            else:
                assert v.shape[0] == n, "All payload tensors must have the same length"
        self._n = int(n if n is not None else 0)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.payload.items()}


# -----------------------------
# Train Sampler
# -----------------------------

class TrainSampler:
    """
    Enhanced sampler for LM+GNN training.

    Features:
      - Text-hard negatives (TF-IDF/BOW similarity to step text).
      - Structural hard negatives (prefer cross-community; prefer large shortest path).
      - Anti-frequency reweighting: pair_weight = freq(pos)^(-af_pow) * edge_weight_fn(pos,neg).
      - Anti-frequency in negative ranking: score *= freq(neg)^(-neg_af_pow) and *= edge_weight_fn(pos,neg)^neg_edge_pow.

    Args:
      raw_contents: List of (step_text, pos_tool)
      num_negatives: negatives per positive sample
      sample_graph: adjacency dict on tool space (tool -> List[tool]); can be None
      tool2index: mapping tool string -> tool id (int)
      hard_negative: enable text-hard negative
      batch_size: batch size for DataLoader
      structure_hard_negative: enable structural constraints (cross-community / min shortest path)
      communities: mapping tool -> community id (int), optional
      min_sp: minimal shortest path length when picking negatives (only used if structure_hard_negative)
      tool_texts: optional mapping tool -> textual description for text similarity; fallback to tool name
      anti_freq: enable anti-frequency
      af_pow: exponent for positive frequency weighting (pos_freq ** -af_pow)
      neg_af_pow: exponent for negative frequency in ranking (neg_freq ** -neg_af_pow)
      edge_weight_fn: callable (pos_tool, neg_tool) -> edge weight (already mapped, e.g., (freq_edge + eps)^(-beta)); if None, treated as 1.0
      neg_edge_pow: exponent applied in ranking for edge_weight_fn output
      seed: random seed
    """
    def __init__(
        self,
        raw_contents: List[Tuple[str, str]],
        num_negatives: int,
        sample_graph: Optional[Dict[str, List[str]]],
        tool2index: Dict[str, int],
        hard_negative: bool = True,
        batch_size: int = 128,
        structure_hard_negative: bool = True,
        communities: Optional[Dict[str, int]] = None,
        min_sp: int = 2,
        tool_texts: Optional[Dict[str, str]] = None,
        # Anti-frequency
        anti_freq: bool = False,
        af_pow: float = 0.5,
        neg_af_pow: float = 0.0,
        edge_weight_fn: Optional[Callable[[str, str], float]] = None,
        neg_edge_pow: float = 1.0,
        seed: int = 1,
    ) -> None:
        self.raw_contents = list(raw_contents)
        self.num_negatives = int(num_negatives)
        self.sample_graph = sample_graph or {}
        self.tool2index = tool2index
        self.index2tool = {v: k for k, v in tool2index.items()}
        self.all_tools: List[str] = list(tool2index.keys())
        self.hard_negative = bool(hard_negative)
        self.batch_size = int(batch_size)
        self.structure_hard_negative = bool(structure_hard_negative)
        self.communities = communities
        self.min_sp = int(min_sp)
        self.seed = int(seed)
        random.seed(self.seed)

        # Texts for similarity
        self.tool_texts = tool_texts or {t: t for t in self.all_tools}
        self._tfidf = None
        self._tool_mat = None
        if _HAS_SK:
            try:
                self._tfidf = TfidfVectorizer(max_features=4096, ngram_range=(1,2))
                corpus = [self.tool_texts[t] for t in self.all_tools]
                self._tool_mat = self._tfidf.fit_transform(corpus)  # shape (T, V)
            except Exception:
                self._tfidf = None
                self._tool_mat = None

        # Frequencies (for anti-frequency)
        self.tool_freq: Dict[str, int] = defaultdict(int)
        for _, pos in self.raw_contents:
            self.tool_freq[pos] += 1

        self.anti_freq = bool(anti_freq)
        self.af_pow = float(af_pow)
        self.neg_af_pow = float(neg_af_pow)
        self.edge_weight_fn = edge_weight_fn
        self.neg_edge_pow = float(neg_edge_pow)

        # Cache for shortest paths per source tool
        self._sp_cache: Dict[str, Dict[str, int]] = {}

    # -----------------------------
    # Text similarity helpers
    # -----------------------------
    def _step_to_vec(self, step_text: str):
        if self._tfidf is None:
            return None
        try:
            return self._tfidf.transform([step_text])  # (1, V)
        except Exception:
            return None

    def _text_sim_scores(self, step_text: str) -> Dict[str, float]:
        """Return similarity scores between step text and each tool."""
        if self._tfidf is not None and self._tool_mat is not None:
            q = self._step_to_vec(step_text)  # (1, V)
            if q is not None:
                # cosine sim: (q * T)^ / (||q|| * ||T||) ; sklearn rows are normalized -> dot is cosine
                sims = (q @ self._tool_mat.T).A.flatten()
                return {tool: float(sims[i]) for i, tool in enumerate(self.all_tools)}
        # Fallback: simple word overlap
        words = set(step_text.lower().split())
        scores = {}
        for t in self.all_tools:
            tw = set(self.tool_texts[t].lower().split())
            inter = len(words & tw)
            scores[t] = float(inter) / (len(words) + 1e-6)
        return scores

    # -----------------------------
    # Structural constraints / distances
    # -----------------------------
    def _get_sp(self, src: str) -> Dict[str, int]:
        if src not in self._sp_cache:
            self._sp_cache[src] = _bfs_shortest_paths(src, self.sample_graph)
        return self._sp_cache[src]

    def _prefer_cross_community(self, pos_tool: str, cand_tools: List[str]) -> List[str]:
        if self.communities is None:
            return cand_tools
        c_pos = self.communities.get(pos_tool, -1)
        cross = [t for t in cand_tools if self.communities.get(t, -1) != c_pos]
        return cross if len(cross) > 0 else cand_tools

    # -----------------------------
    # Negative picking
    # -----------------------------
    def _pick_hard_negatives(self, step_text: str, pos_tool: str) -> List[str]:
        # base candidates: all tools except pos
        cands = [t for t in self.all_tools if t != pos_tool]
        if len(cands) == 0:
            return []

        # text similarity scores
        tool2sim = self._text_sim_scores(step_text) if self.hard_negative else {t: 1.0 for t in cands}

        # ranking with anti-frequency on negatives and edge weights
        def _rank_score(neg_t: str) -> float:
            base = float(tool2sim.get(neg_t, 0.0))
            if self.neg_af_pow > 0.0:
                fneg = float(self.tool_freq.get(neg_t, 1))
                base *= fneg ** (-self.neg_af_pow)
            if self.edge_weight_fn is not None:
                ew = float(self.edge_weight_fn(pos_tool, neg_t))
                try:
                    base *= max(1e-12, ew) ** self.neg_edge_pow
                except Exception:
                    base *= 1.0
            return base

        # structural constraints
        if self.structure_hard_negative:
            cands = self._prefer_cross_community(pos_tool, cands)
            if self.min_sp > 1 and self.sample_graph:
                sp = self._get_sp(pos_tool)
                far = [t for t in cands if sp.get(t, 1 << 30) >= self.min_sp]
                if len(far) > 0:
                    cands = far

        # sort by score high->low, then take top-K, then sample uniformly from top band
        cands.sort(key=_rank_score, reverse=True)
        k = min(len(cands), max(8, self.num_negatives * 4))
        top = cands[:k]
        # random sample num_negatives without replacement
        if len(top) <= self.num_negatives:
            return top
        return random.sample(top, self.num_negatives)

    # -----------------------------
    # Public API
    # -----------------------------
    def sample(
        self,
        shuffle: bool = True,
        maximum: Optional[int] = None,
        tmp_print: bool = True,
    ) -> Tuple[DataLoader, List[str]]:
        entries: List[Tuple[str, str, str]] = []  # (step_text, pos_tool, neg_tool)
        step_indices: List[int] = []              # ★ 每条 entry 对应其“原始样本”的下标
        weights_raw: List[float] = []
        all_step_texts: List[str] = []

        for raw_i, (step_text, pos_tool) in enumerate(self.raw_contents):
            all_step_texts.append(step_text)
            neg_list = self._pick_hard_negatives(step_text, pos_tool)

            # positive component of pair weight
            if self.anti_freq:
                fpos = float(self.tool_freq.get(pos_tool, 1))
                w_pos = fpos ** (-self.af_pow)
            else:
                w_pos = 1.0

            for neg_tool in neg_list:
                entries.append((step_text, pos_tool, neg_tool))
                step_indices.append(raw_i)  # ★ 关键修复：保存原始样本索引
                if self.anti_freq:
                    if self.edge_weight_fn is not None:
                        try:
                            w_edge = float(self.edge_weight_fn(pos_tool, neg_tool))
                        except Exception:
                            w_edge = 1.0
                    else:
                        w_edge = 1.0
                    weights_raw.append(w_pos * w_edge)

        if len(entries) == 0:
            # Fallback to a dummy sample to avoid崩溃
            step_text, pos_tool = self.raw_contents[0]
            neg_tool = (self.all_tools[0] if self.all_tools[0] != pos_tool else self.all_tools[1])
            entries.append((step_text, pos_tool, neg_tool))
            step_indices.append(0)  # ★ dummy 也对应第 0 条样本
            if self.anti_freq:
                fpos = float(self.tool_freq.get(pos_tool, 1))
                w_pos = fpos ** (-self.af_pow)
                w_edge = float(self.edge_weight_fn(pos_tool, neg_tool)) if self.edge_weight_fn is not None else 1.0
                weights_raw.append(w_pos * w_edge)

        # --- 构造张量 ---
        pos_ids = torch.LongTensor([self.tool2index[e[1]] for e in entries])
        neg_ids = torch.LongTensor([self.tool2index[e[2]] for e in entries])
        step_idx = torch.LongTensor(step_indices)   # ★ 用原始样本下标构造
        pair_weight = None
        if self.anti_freq:
            pair_weight = torch.tensor(weights_raw, dtype=torch.float32)

        # --- maximum 截断（评估/快速跑会用）---
        if maximum is not None and len(entries) > int(maximum):
            n = int(maximum)
            pos_ids = pos_ids[:n]
            neg_ids = neg_ids[:n]
            step_idx = step_idx[:n]
            if pair_weight is not None:
                pair_weight = pair_weight[:n]

        # --- 归一化 pair_weight 到均值 1.0（截断之后做）---
        if pair_weight is not None and pair_weight.numel() > 0:
            s = float(pair_weight.sum().item())
            scale = (pair_weight.numel() / s) if s > 0 else 1.0
            pair_weight = pair_weight * scale
            if tmp_print:
                m = pair_weight.mean().item()
                mi = pair_weight.min().item()
                ma = pair_weight.max().item()
                print(f"[Anti-Freq|batch0-like] mean={m:.4f} min={mi:.4f} max={ma:.4f} (after norm)")

        payload: Dict[str, torch.Tensor] = {
            "step_idx": step_idx,
            "pos_ids": pos_ids,
            "neg_ids": neg_ids,
        }
        if pair_weight is not None:
            payload["pair_weight"] = pair_weight

        dataset = _DictDataset(payload)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

        return train_loader, all_step_texts