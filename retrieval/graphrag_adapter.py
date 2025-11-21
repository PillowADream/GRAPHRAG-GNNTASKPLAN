# retrieval/graphrag_adapter.py
# Minimal, dependency-light GraphRAG adapter that always returns tool ids.
from pathlib import Path
import json
from difflib import SequenceMatcher

class GraphRAGAdapter:
    """
    A tiny local retriever:
      - Loads tool_desc.json & graph_desc.json (prefer index_dir, else its parent dataset dir)
      - Ranks tools by simple textual similarity between query and concatenated tool text
      - Returns dict{'nodes': [{'id','name','score'}], 'communities': [{'seed','members'}], 'evidence': []}
    Optional: if rapidfuzz is installed, uses it for better matching.
    """
    def __init__(self, index_dir):
        self.index_dir = Path(index_dir)
        # locate files (allow both .../data/<ds>/graphrag and .../data/<ds>)
        self.tool_desc_path = self._find_one("tool_desc.json")
        self.graph_desc_path = self._find_one("graph_desc.json")

        tdesc = json.load(open(self.tool_desc_path, "r", encoding="utf-8"))
        gdesc = json.load(open(self.graph_desc_path, "r", encoding="utf-8"))

        # tolerate two shapes: {"nodes":[...]} or a plain list
        self.tools = tdesc["nodes"] if isinstance(tdesc, dict) and "nodes" in tdesc else tdesc
        self.links = gdesc["links"] if isinstance(gdesc, dict) and "links" in gdesc else gdesc

        # build text per tool
        self.tool_map = {}   # id -> meta
        self.all_items = []  # [(id, text, name)]
        for t in self.tools:
            tid = t["id"]
            name = t.get("name") or tid
            desc = t.get("description") or ""
            aliases = " ".join(a for a in t.get("aliases", []) if isinstance(a, str))
            extra = []
            for k in ("input-type", "output-type", "category", "tags"):
                v = t.get(k)
                if isinstance(v, str): extra.append(v)
                elif isinstance(v, list): extra.extend([x for x in v if isinstance(x, str)])
            text = " ".join([tid, name, aliases, desc] + extra).strip()
            self.tool_map[tid] = {"id": tid, "name": name, "text": text}
            self.all_items.append((tid, text, name))

        # try rapidfuzz
        self._rf = None
        try:
            from rapidfuzz import fuzz
            self._rf = fuzz
        except Exception:
            self._rf = None

        # adjacency (for lightweight communities)
        self.adj = {}
        for e in self.links:
            s, t = e["source"], e["target"]
            self.adj.setdefault(s, set()).add(t)
            self.adj.setdefault(t, set()).add(s)

    def _find_one(self, fname: str) -> Path:
        p1 = self.index_dir / fname
        if p1.exists(): return p1
        p2 = self.index_dir.parent / fname
        if p2.exists(): return p2
        # also try two levels up (for very custom layouts)
        p3 = self.index_dir.parent.parent / fname
        if p3.exists(): return p3
        raise FileNotFoundError(f"Cannot locate {fname} under {self.index_dir} or its parents")

    def _score(self, q: str, t: str) -> int:
        if not q or not t: return 0
        ql, tl = q.lower(), t.lower()
        if self._rf is not None:
            # token_set_ratio is robust for short names + longer descriptions
            return int(self._rf.token_set_ratio(ql, tl))
        # fallback to difflib ratio in [0,100]
        return int(100 * SequenceMatcher(None, ql, tl).ratio())

    def search(self, query: str, mode: str = "auto", topk_nodes: int = 8, topk_comms: int = 3):
        # rank tools
        scored = []
        for tid, text, name in self.all_items:
            s = self._score(query, text)
            scored.append((s, tid, name))
        scored.sort(reverse=True)
        top = [x for x in scored[:max(1, topk_nodes)] if x[0] > 0]

        nodes = [{"id": tid, "name": name, "score": int(s)} for (s, tid, name) in top]

        # simple “communities”: neighbors of top seeds
        comms = []
        for _, tid, _ in top[:max(0, topk_comms)]:
            mems = list(self.adj.get(tid, []))
            if mems:
                comms.append({"seed": tid, "members": mems[: min(12, len(mems))]})

        return {"nodes": nodes, "communities": comms, "evidence": []}
