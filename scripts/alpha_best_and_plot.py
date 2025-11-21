# scripts/alpha_best_and_plot.py
import argparse, json, math
from pathlib import Path
from statistics import mean
import matplotlib.pyplot as plt

def load_jsonl(p: Path):
    rows = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def safe_mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not math.isnan(x)]
    return mean(xs) if xs else None

def agg_metrics(rows):
    """返回 (PSR, Node-F1, Link-F1, Latency) 的均值"""
    psr = safe_mean([1.0 if r.get("success") else 0.0 for r in rows])
    nf  = safe_mean([r.get("node_f1") for r in rows])
    lf  = safe_mean([r.get("link_f1") for r in rows])
    lat = safe_mean([r.get("latency_sec") for r in rows])
    return psr, nf, lf, lat

def select_best_alpha(sgc_rows):
    """按 Node-F1 -> PSR -> -Latency 选择最优 α；返回 best_alpha, 按 α 分组的汇总字典"""
    by_alpha = {}
    for r in sgc_rows:
        a = r.get("alpha")
        if a is None:
            continue
        by_alpha.setdefault(a, []).append(r)

    scored = []
    for a, rows in by_alpha.items():
        psr, nf, lf, lat = agg_metrics(rows)
        scored.append((a, nf if nf is not None else -1.0, psr if psr is not None else -1.0, lat if lat is not None else float("inf")))
    if not scored:
        return None, {}

    # 排序：Node-F1 ↓，PSR ↓，Latency ↑（更小更好，所以取负作为次序或显式比较）
    scored.sort(key=lambda t: (t[1], t[2], - (1e9 if math.isinf(t[3]) else -t[3])), reverse=True)
    best_alpha = scored[0][0]
    summary = {a: {"node_f1": nf, "psr": psr, "latency": lat} for (a, nf, psr, lat) in scored}
    return best_alpha, summary

def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset name, e.g., dailylife / huggingface / multimedia")
    ap.add_argument("--runs_dir", default="runs", help="runs directory (default: runs)")
    ap.add_argument("--subdir", default="base", help="sub directory under runs/<ds>/ (default: base)")
    ap.add_argument("--outfile", default="psr_vs_latency.png", help="output figure filename")
    args = ap.parse_args()

    ds = args.dataset
    base = Path(args.runs_dir) / ds / args.subdir

    # 输入文件
    p_direct = base / "direct.jsonl"
    p_graph  = base / "graphsearch_beam_2.jsonl"
    p_sgc    = base / "sgc.jsonl"

    # 输出（best α 的 SGC）
    p_sgc_best = base / "sgc_alpha_best.jsonl"

    # 1) 读取数据
    direct_rows = load_jsonl(p_direct)
    graph_rows  = load_jsonl(p_graph)
    sgc_rows    = load_jsonl(p_sgc)

    if not sgc_rows:
        print(f"[WARN] Not found or empty: {p_sgc}")
        return

    # 2) 选出 best α 并写出过滤后的文件
    best_alpha, alpha_summary = select_best_alpha(sgc_rows)
    if best_alpha is None:
        print("[ERROR] No alpha groups found in SGC jsonl.")
        return

    sgc_best_rows = [r for r in sgc_rows if r.get("alpha") == best_alpha]
    ensure_dir(p_sgc_best)
    with p_sgc_best.open("w", encoding="utf-8") as w:
        for r in sgc_best_rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 汇总三条曲线的代表点：Direct、GraphSearch、SGC(best α)
    methods = []
    if direct_rows:
        psr, nf, lf, lat = agg_metrics(direct_rows)
        methods.append(("Direct", psr, lat))
    if graph_rows:
        psr, nf, lf, lat = agg_metrics(graph_rows)
        methods.append(("GraphSearch (beam=2)", psr, lat))
    # SGC-best
    psr, nf, lf, lat = agg_metrics(sgc_best_rows)
    methods.append((f"SGC (α={best_alpha})", psr, lat))

    # 打印摘要
    print("== α summary (averages) ==")
    for a, m in sorted(alpha_summary.items(), key=lambda kv: kv[0]):
        print(f"alpha={a}: Node-F1={m['node_f1']}, PSR={m['psr']}, Latency={m['latency']}s")
    print(f"\n[Best α] {best_alpha}")
    print(f"[Wrote]  {p_sgc_best}")

    # 3) 画 PSR vs Latency
    xs, ys, labels = [], [], []
    for name, psr_v, lat_v in methods:
        if psr_v is None or lat_v is None:
            continue
        xs.append(lat_v); ys.append(psr_v); labels.append(name)

    if not xs:
        print("[WARN] Nothing to plot (no PSR/Latency found).")
        return

    plt.figure(figsize=(6, 4.5))
    plt.scatter(xs, ys, s=60)
    for x, y, lab in zip(xs, ys, labels):
        plt.annotate(lab, (x, y), xytext=(5, 5), textcoords="offset points")
    plt.xlabel("Average Latency (sec)")
    plt.ylabel("PSR (success ratio)")
    plt.title(f"Cost–Effect Baseline (dataset={ds})")
    plt.grid(True, linestyle="--", alpha=0.4)

    out_fig = base / args.outfile
    plt.tight_layout()
    plt.savefig(out_fig, dpi=200)
    print(f"[Saved] {out_fig}")

if __name__ == "__main__":
    main()