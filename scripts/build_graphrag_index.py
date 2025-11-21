# scripts/build_graphrag_index.py
import json, argparse, os, pathlib, math
import numpy as np
from collections import defaultdict, deque
from joblib import dump
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.neighbors import NearestNeighbors

def load_graph(ds_root: pathlib.Path):
    tools = json.load(open(ds_root/"tool_desc.json", "r", encoding="utf-8"))["nodes"]
    links = json.load(open(ds_root/"graph_desc.json","r",encoding="utf-8"))["links"]
    tool_by_id = {t["id"]: t for t in tools}
    # 无向邻接（社区/2-hop）
    nbr = defaultdict(set)
    for e in links:
        a,b = e["source"], e["target"]
        nbr[a].add(b); nbr[b].add(a)
    return tool_by_id, links, nbr

def mk_aliases(t):
    aliases = set()
    fields = [t.get("id",""), t.get("name",""), t.get("display_name","")]
    # 可选：仓库若定义了 'alias'/'synonyms' 字段也收集
    for k in ("alias","aliases","synonyms","short_name"):
        v = t.get(k, [])
        if isinstance(v, str): fields.append(v)
        elif isinstance(v, list): fields.extend(v)
    for s in fields:
        if not s: continue
        s0 = s.strip()
        aliases.add(s0)
        aliases.add(s0.replace("_"," ").replace("-"," "))
        aliases.add(s0.lower())
    return sorted(a for a in aliases if a)

def node_doc(tool):
    # 将工具描述打平为检索文本
    parts = [tool.get("id",""), tool.get("name",""), tool.get("description","")]
    for k in ("input-type","output-type","inputs","outputs","examples"):
        v = tool.get(k)
        if isinstance(v, str): parts.append(v)
        elif isinstance(v, list): parts.extend([str(x) for x in v])
        elif isinstance(v, dict): parts.append(json.dumps(v, ensure_ascii=False))
    return " \n".join([p for p in parts if p])

def ego2_doc(seed, tool_by_id, nbr):
    # 以 seed 为中心的 2-hop 社区文档
    visited={seed}; q=deque([(seed,0)]); order=[]
    while q:
        u,d=q.popleft()
        order.append(u)
        if d==2: continue
        for v in nbr.get(u, []):
            if v not in visited:
                visited.add(v); q.append((v,d+1))
    # 聚合文本
    lines=[f"[CENTER] {seed}"]
    for u in order:
        t=tool_by_id[u]
        lines.append(f"[NODE] {u} :: {t.get('name','')}")
        if u!=seed:
            lines.append(node_doc(t))
    return {"center": seed, "members": order, "text": "\n".join(lines)}

def embed_texts(texts, model_id, device):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    lm  = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    lm.eval()
    embs=[]
    bs=32
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch=texts[i:i+bs]
            x = tok(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            out = lm(**x, output_hidden_states=True)
            # E5: 取最后层 CLS；你的 M0 中用过同策略
            h = out.hidden_states[-1][:,0,:]  # [B, H]
            h = torch.nn.functional.normalize(h, p=2, dim=-1)
            embs.append(h.detach().cpu().numpy())
    return np.vstack(embs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["huggingface","multimedia","dailylife","tmdb","ultratool"])
    ap.add_argument("--lm_name", default="intfloat/e5-large")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--build_ego2", type=int, default=1)
    args = ap.parse_args()

    root = pathlib.Path("data")/args.dataset
    out_dir = root/"graphrag"
    out_dir.mkdir(parents=True, exist_ok=True)

    tool_by_id, links, nbr = load_graph(root)
    tools = list(tool_by_id.values())
    # 1) 节点文档
    node_docs=[]
    for t in tools:
        node_docs.append({
            "id": t["id"],
            "aliases": mk_aliases(t),
            "text": node_doc(t)
        })
    json.dump(node_docs, open(out_dir/"nodes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 2) 社区文档（2-hop ego）
    comm_docs=[]
    if args.build_ego2:
        for tid in tool_by_id:
            comm_docs.append(ego2_doc(tid, tool_by_id, nbr))
        json.dump(comm_docs, open(out_dir/"communities_ego2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 3) 向量化 + 索引
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    node_emb = embed_texts([d["text"] for d in node_docs], args.lm_name, device)
    np.save(out_dir/"nodes_emb.npy", node_emb)
    nn_nodes = NearestNeighbors(n_neighbors=50, metric="cosine").fit(node_emb)
    dump(nn_nodes, out_dir/"nodes_nn.joblib")

    if args.build_ego2:
        comm_emb = embed_texts([d["text"] for d in comm_docs], args.lm_name, device)
        np.save(out_dir/"communities_ego2_emb.npy", comm_emb)
        nn_comm = NearestNeighbors(n_neighbors=20, metric="cosine").fit(comm_emb)
        dump(nn_comm, out_dir/"communities_ego2_nn.joblib")

    # 4) 软别名缓存
    ent_map={}
    for d in node_docs:
        for a in d["aliases"]:
            ent_map.setdefault(a, set()).add(d["id"])
    ent_map = {k: sorted(list(v)) for k,v in ent_map.items()}
    json.dump(ent_map, open(out_dir/"entity2tool.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)

    print(f"[GraphRAG] built at {out_dir}")

if __name__ == "__main__":
    main()