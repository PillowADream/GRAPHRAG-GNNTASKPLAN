# scripts/build_rag_feat.py
import json
import pickle
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if (p / "data").exists() and (p / "prediction").exists():
            return p
    return Path.cwd()


def _load_graph(ds_dir: Path):
    g = json.load(open(ds_dir / "graph_desc.json", "r", encoding="utf-8"))
    nodes, links = g["nodes"], g["links"]
    id2idx = {n["id"]: i for i, n in enumerate(nodes)}
    N = len(nodes)

    # degree & adjacency
    deg = np.zeros(N, dtype=np.int32)
    adj = defaultdict(set)
    for e in links:
        s, t = e["source"], e["target"]
        if s in id2idx and t in id2idx:
            i, j = id2idx[s], id2idx[t]
            deg[i] += 1
            deg[j] += 1
            adj[i].add(j)
            adj[j].add(i)

    return nodes, id2idx, deg, adj


def _build_communities(ds_dir: Path, id2idx, adj) -> np.ndarray:
    """
    优先从 GraphRAG 的 communities.json 读取；否则退化为连通分量。
    社区 ID 从 1 开始，未命中的为 0。
    """
    N = len(id2idx)
    comm_id = np.zeros(N, dtype=np.int32)

    comm_json = ds_dir / "graphrag" / "communities.json"
    if comm_json.exists():
        try:
            cs = json.load(open(comm_json, "r", encoding="utf-8"))
            cid = 1
            for c in cs:
                mems = c.get("members", [])
                for m in mems:
                    if m in id2idx:
                        comm_id[id2idx[m]] = cid
                cid += 1
            if comm_id.max() > 0:
                return comm_id
        except Exception:
            # 回退到连通分量
            pass

    # fallback: 简单 BFS 连通分量
    seen = set()
    cid = 1
    for i in range(N):
        if i in seen:
            continue
        q = [i]
        seen.add(i)
        has_edge = False
        while q:
            u = q.pop()
            comm_id[u] = cid
            for v in adj[u]:
                has_edge = True
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        # 如果该连通块实际上是孤点，也仍给一个 cid
        cid += 1
    return comm_id


def main(ds: str, tfidf_dim: int = 64, comm_dim: int = 64):
    ROOT = _repo_root()
    ds_dir = ROOT / "data" / ds

    nodes, id2idx, deg, adj = _load_graph(ds_dir)
    N = len(nodes)
    print(f"[INFO] dataset={ds}, |nodes|={N}")

    # ========= scalar features =========
    deg_log = np.log1p(deg).astype(np.float32)

    comm_id = _build_communities(ds_dir, id2idx, adj)
    # 社区大小（对每个节点取其所在社区大小的 log）
    comm2size = defaultdict(int)
    for c in comm_id:
        comm2size[int(c)] += 1
    comm_size_log = np.array([np.log1p(comm2size[int(c)]) for c in comm_id], dtype=np.float32)

    # ========= TF-IDF for tool descriptions =========
    texts = []
    for n in nodes:
        txt = (n.get("name", "") + "\n" + n.get("description", "") + "\n" + n.get("desc", "")).strip()
        texts.append(txt)

    # 词表上限保持较大，但 SVD 维度需合法（< n_features）
    tfidf = TfidfVectorizer(max_features=4096)
    X = tfidf.fit_transform(texts)  # (N, V)
    V = X.shape[1]

    # 保障 n_components 合法：TruncatedSVD 需要 1 <= n_components < n_features
    tfidf_dim_eff = max(1, min(tfidf_dim, max(1, V - 1)))
    if tfidf_dim_eff != tfidf_dim:
        print(f"[WARN] adjust tfidf_dim from {tfidf_dim} -> {tfidf_dim_eff} (V={V})")

    svd = TruncatedSVD(n_components=tfidf_dim_eff, random_state=0)
    X_tfidf = svd.fit_transform(X).astype(np.float32)  # (N, tfidf_dim_eff)

    # ========= community text vectors =========
    # 关键改动：先在与 X_tfidf 相同的空间做“成员均值”，避免维度不一致
    C = int(comm_id.max()) + 1  # 包含 0 类（未分配）的话也保留一个桶
    d0 = X_tfidf.shape[1]

    Xc = np.zeros((C, d0), dtype=np.float32)
    cnt = np.zeros(C, dtype=np.int32)
    for i, c in enumerate(comm_id):
        c = int(c)
        if c < 0 or c >= C:
            continue
        Xc[c] += X_tfidf[i]
        cnt[c] += 1
    nz = cnt > 0
    Xc[nz] = Xc[nz] / cnt[nz, None]

    # 如需把社区向量压到 comm_dim（<= d0），再做一次 SVD；否则直接沿用 d0 维
    if comm_dim < d0:
        svd_c = TruncatedSVD(n_components=comm_dim, random_state=0)
        Xc_proj = svd_c.fit_transform(Xc).astype(np.float32)  # (C, comm_dim)
        comm_text_vec = np.stack([Xc_proj[int(c)] for c in comm_id], axis=0)
        comm_text_dim = comm_dim
    elif comm_dim == d0:
        comm_text_vec = np.stack([Xc[int(c)] for c in comm_id], axis=0)
        comm_text_dim = d0
    else:
        # 想要的维度大于可用维度：直接保持 d0，并给出提示
        print(f"[WARN] comm_dim({comm_dim}) > TF-IDF dim({d0}); keep {d0} dims for community vectors.")
        comm_text_vec = np.stack([Xc[int(c)] for c in comm_id], axis=0)
        comm_text_dim = d0

    out = {
        "id_list": [n["id"] for n in nodes],     # (N,)
        "deg_log": deg_log,                      # (N,)
        "comm_id": comm_id,                      # (N,)
        "comm_size_log": comm_size_log,          # (N,)
        "tool_text_vec": X_tfidf,                # (N, tfidf_dim_eff)
        "comm_text_vec": comm_text_vec,          # (N, comm_text_dim)
    }

    (ds_dir / "cache").mkdir(parents=True, exist_ok=True)
    save_path = ds_dir / "cache" / "rag_feat_static.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(out, f)

    print("[OK] saved:", save_path)
    print(
        f"[SHAPE] deg_log={deg_log.shape}, "
        f"comm_id={comm_id.shape}, "
        f"comm_size_log={comm_size_log.shape}, "
        f"tool_text_vec={X_tfidf.shape}, "
        f"comm_text_vec={comm_text_vec.shape}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tfidf_dim", type=int, default=64)
    ap.add_argument("--comm_dim", type=int, default=64)
    args = ap.parse_args()
    main(args.dataset, args.tfidf_dim, args.comm_dim)