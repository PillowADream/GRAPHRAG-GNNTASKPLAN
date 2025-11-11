# scripts/rag_coverage_summary.py
import argparse, json, sys
from pathlib import Path
import statistics as st
from typing import List, Dict

def load_records(path: Path) -> List[Dict]:
    recs = []
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    if not txt:
        return recs
    # 兼容 JSONL 与 单一 JSON 数组 两种格式
    if txt[0] == "[":
        arr = json.loads(txt)
        for r in arr:
            if isinstance(r, dict):
                recs.append(r)
    else:
        for line in txt.splitlines():
            line = line.strip()
            if not line: continue
            try:
                r = json.loads(line)
                if isinstance(r, dict):
                    recs.append(r)
            except Exception:
                pass
    return recs

def find_latest_pred(root: Path) -> Path:
    # 在 prediction/** 下找包含 graphsearch & auto 的文件，取 mtime 最新
    cand = list(root.glob("prediction/**/graphsearch_*_auto*.json"))
    if not cand:
        # 退而求其次找任意 graphsearch_*.json
        cand = list(root.glob("prediction/**/graphsearch_*.json"))
    if not cand:
        raise FileNotFoundError("No prediction files found under prediction/**")
    cand.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cand[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=str, default=None,
                    help="prediction json/jsonl path. If omitted, auto-pick the latest under prediction/**")
    ap.add_argument("--show_examples", type=int, default=5,
                    help="print up to N examples where cand_src_rag_avg>0 (and success)")
    args = ap.parse_args()

    repo = Path.cwd()
    pred_path = Path(args.pred) if args.pred else find_latest_pred(repo)

    recs = load_records(pred_path)
    if not recs:
        print(f"No records loaded from {pred_path}")
        return

    vals = []
    succ = 0
    for r in recs:
        if r.get("rag_used"):
            v = r.get("cand_src_rag_avg")
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if r.get("status") == "succ":
            succ += 1

    print(f"[file] {pred_path}")
    print(f"[count] total={len(recs)}, succ={succ}")

    if not vals:
        print("No RAG stats found (cand_src_rag_avg missing or zero everywhere).")
        return

    gt0 = sum(v > 0 for v in vals)
    print(f"RAG-cand per step: mean={st.mean(vals):.3f}, median={st.median(vals):.3f}, >0 ratio={gt0/len(vals):.2%}")

    # 选出若干“RAG 有候选”的样例（success 优先，且 node_f1 高）
    examples = []
    for r in recs:
        if not r.get("rag_used"): continue
        v = r.get("cand_src_rag_avg", 0.0) or 0.0
        if isinstance(v, str):
            try: v = float(v)
            except: v = 0.0
        if v <= 0: continue
        score = (1.0 if r.get("status") == "succ" else 0.0) + float(r.get("node_f1", 0.0))
        examples.append((score, r))

    examples.sort(key=lambda x: x[0], reverse=True)
    n = min(args.show_examples, len(examples))
    if n > 0:
        print(f"\nTop-{n} examples with RAG candidates (>0):")
        for i in range(n):
            r = examples[i][1]
            print(json.dumps({
                "id": r.get("id"),
                "status": r.get("status"),
                "node_f1": r.get("node_f1"),
                "link_f1": r.get("link_f1"),
                "cand_src_rag_avg": r.get("cand_src_rag_avg"),
                "cand_src_graph_avg": r.get("cand_src_graph_avg"),
                "rag_mode": r.get("rag_mode")
            }, ensure_ascii=False))

if __name__ == "__main__":
    main()