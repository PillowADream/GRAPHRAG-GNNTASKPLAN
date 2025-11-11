# eval_aggregate.py
import json, argparse, glob, os, csv
from statistics import mean
from pathlib import Path

# ----------------------------- Common helpers -----------------------------
def parse_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return round(mean(xs), 4) if xs else None

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def try_load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# ----------------------------- Eval (per-sample) -----------------------------
def load_gt(ds: str):
    """Load valid node/link universe and GT per-sample sets."""
    base = Path("data") / ds
    tool = json.load(open(base / "tool_desc.json", "r", encoding="utf-8"))["nodes"]
    graph = json.load(open(base / "graph_desc.json", "r", encoding="utf-8"))["links"]
    valid_nodes = {n["id"] for n in tool}
    valid_links = {f'{e["source"]}, {e["target"]}' for e in graph}

    gt = {}
    with open(base / "data.json", "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            gid = d["id"]
            gnodes = [
                (n["task"] if isinstance(n, dict) else n)
                for n in d.get("task_nodes", [])
            ]
            glinks = [
                f'{e["source"]}, {e["target"]}'
                for e in d.get("task_links", [])
            ]
            gt[gid] = (set(gnodes), set(glinks))
    return valid_nodes, valid_links, gt

def norm_links(L):
    """Normalize links into 'src, tgt' strings."""
    out = []
    for e in (L or []):
        if isinstance(e, dict) and "source" in e and "target" in e:
            out.append(f'{e["source"]}, {e["target"]}')
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            out.append(f"{e[0]}, {e[1]}")
        elif isinstance(e, str) and "," in e:
            a, b = e.split(",", 1)
            out.append(f"{a.strip()}, {b.strip()}")
    return out

def norm_nodes(L):
    """Normalize nodes to list of string tool ids."""
    out = []
    for n in (L or []):
        if isinstance(n, dict) and "task" in n:
            out.append(n["task"])
        elif isinstance(n, str):
            out.append(n)
    return out

def f1_score(pred_list, gt_set):
    ps, gs = set(pred_list), set(gt_set)
    if not ps and not gs:
        return 1.0
    tp = len(ps & gs)
    if tp == 0:
        return 0.0
    p = tp / max(1, len(ps))
    r = tp / max(1, len(gs))
    return 2 * p * r / (p + r + 1e-12)

def enrich_row_metrics(row, caches):
    """Fill missing node_f1/link_f1 and hallucination metrics."""
    valid_nodes, valid_links, gt = caches
    sid = row.get("sid", row.get("id"))
    pred_nodes = norm_nodes(row.get("task_nodes", []))
    pred_links = norm_links(row.get("task_links", []))

    gnodes, glinks = gt.get(sid, (set(), set()))

    # F1 fallback
    if row.get("node_f1") is None:
        row["node_f1"] = round(f1_score(pred_nodes, gnodes), 4)
    if row.get("link_f1") is None:
        row["link_f1"] = round(f1_score(pred_links, glinks), 4)

    # Hallucination fallback (NH-1/NH-2/LH-1/LH-2)
    need_hall = any(k not in row or row.get(k) is None for k in ("nh1", "nh2", "lh1", "lh2"))
    if need_hall:
        n = len(pred_nodes)
        m = len(pred_links)
        nh1 = (sum(1 for x in pred_nodes if x not in valid_nodes) / max(1, n)) if n else 0.0
        nh2 = (sum(1 for x in pred_nodes if x not in gnodes) / max(1, n)) if n else 0.0
        lh1 = (sum(1 for x in pred_links if x not in valid_links) / max(1, m)) if m else 0.0
        lh2 = (sum(1 for x in pred_links if x not in glinks) / max(1, m)) if m else 0.0
        row.setdefault("nh1", round(nh1, 4))
        row.setdefault("nh2", round(nh2, 4))
        row.setdefault("lh1", round(lh1, 4))
        row.setdefault("lh2", round(lh2, 4))

    # success fallback from node_f1
    if "success" not in row or row.get("success") is None:
        row["success"] = bool(row["node_f1"] is not None and row["node_f1"] >= 0.99)

    # unify latency field
    if row.get("latency_sec") is None and row.get("cost_time") is not None:
        row["latency_sec"] = row.get("cost_time")

    return row

def agg_one(file, caches):
    rows = [enrich_row_metrics(r, caches) for r in parse_jsonl(file)]
    n = len(rows)

    psr = round(sum(1 for r in rows if r.get("success") is True) / n, 4) if n else 0.0
    lat = safe_mean([r.get("latency_sec", r.get("cost_time")) for r in rows])
    n_f1 = safe_mean([r.get("node_f1") for r in rows])
    l_f1 = safe_mean([r.get("link_f1") for r in rows])

    # Hallucination aggregates
    nh1 = safe_mean([r.get("nh1") for r in rows])
    nh2 = safe_mean([r.get("nh2") for r in rows])
    lh1 = safe_mean([r.get("lh1") for r in rows])
    lh2 = safe_mean([r.get("lh2") for r in rows])

    tokp = safe_mean([r.get("tokens_prompt") for r in rows])
    tokr = safe_mean([r.get("tokens_completion") for r in rows])

    return {
        "file": file,
        "samples": n,
        "PSR": psr,
        "Node-F1": n_f1,
        "Link-F1": l_f1,
        "Latency": lat,
        "NH-1": nh1,
        "NH-2": nh2,
        "LH-1": lh1,
        "LH-2": lh2,
        "TokPrompt": tokp,
        "TokResp": tokr,
    }

def eval_mode(ds):
    base_dir = f"runs/{ds}"
    files = glob.glob(os.path.join(base_dir, "**", "*.jsonl"), recursive=True)
    # 排除训练曲线日志
    files = [f for f in files if os.path.basename(f) != "metrics_epoch.jsonl"]
    files = sorted(files)

    caches = load_gt(ds)  # (valid_nodes, valid_links, gt)

    print(f"# Aggregate (eval) for dataset={ds}")
    for f in files:
        r = agg_one(f, caches)
        rel = os.path.relpath(r["file"])
        print(
            f"{rel}\n  samples={r['samples']} PSR={r['PSR']} "
            f"Node-F1={r['Node-F1']} Link-F1={r['Link-F1']} Latency={r['Latency']} "
            f"NH-1={r['NH-1']} NH-2={r['NH-2']} LH-1={r['LH-1']} LH-2={r['LH-2']} "
            f"Tok(P/R)=({r['TokPrompt']},{r['TokResp']})"
        )

# ----------------------------- Train (epoch metrics) -----------------------------
def read_metrics_epoch(jsonl_path):
    rows = list(parse_jsonl(jsonl_path))
    # 排序并只保留关心字段
    rows = sorted(rows, key=lambda r: r.get("epoch", 0))
    out = []
    for r in rows:
        out.append({
            "epoch": int(r.get("epoch", 0)),
            "train_loss": r.get("train_loss"),
            "eval_acc": r.get("eval_acc"),
            "alpha": r.get("alpha"),
            "gate_mean": r.get("gate_mean"),
            "rag_lr": r.get("rag_lr"),
            "pw_q10": r.get("pw_q10"),
            "pw_q50": r.get("pw_q50"),
            "pw_q90": r.get("pw_q90"),
            "time": r.get("time"),
        })
    return out

def write_csv(path, rows, header):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(k, "") for k in header])

def plot_series(fig_path, series_list, labels, xlab, ylab, title=None):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib not available, skip plot: {e}")
        return
    if not series_list:
        return
    plt.figure(figsize=(7.2, 4.2))
    for (xs, ys), lb in zip(series_list, labels):
        if xs and ys:
            plt.plot(xs, ys, label=lb)
    plt.xlabel(xlab); plt.ylabel(ylab)
    if title: plt.title(title)
    if len(series_list) > 1:
        plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return round(mean(xs), 4) if xs else None

def enrich_row_metrics(row, caches):
    # ... 你已有的 enrich 逻辑 ...
    # [M4] 兼容 ToolLLM 行为字段
    if "param_f1" not in row or row.get("param_f1") is None:
        # 若没有显式 param_f1，但有 param_pred/param_gt，可以按键+值精确匹配计算一次
        pred = row.get("param_pred"); gt = row.get("param_gt")
        if isinstance(pred, dict) and isinstance(gt, dict):
            tp = 0
            P, G = set(pred.keys()), set(gt.keys())
            for k in P & G:
                a, b = pred[k], gt[k]
                eq = False
                try:
                    if isinstance(a,(int,float)) or isinstance(b,(int,float)):
                        eq = float(a) == float(b)
                    else:
                        eq = str(a).strip().lower() == str(b).strip().lower()
                except Exception:
                    eq = False
                if eq: tp += 1
            if (P or G):
                prec = tp / max(1, len(P))
                rec  = tp / max(1, len(G))
                row["param_f1"] = round(2*prec*rec/(prec+rec+1e-12), 4)
            else:
                row["param_f1"] = 1.0

    if "exec_success" in row and row.get("tool_exec_consistency") is None:
        # 把 exec_success 当作一致性二值（成功=1，失败=0）
        v = row.get("exec_success")
        row["tool_exec_consistency"] = (1.0 if v is True else 0.0 if v is False else None)

    return row

def agg_one(file, caches):
    rows = [enrich_row_metrics(r, caches) for r in parse_jsonl(file)]
    n = len(rows)
    # ... 你已有的聚合 ...
    param_f1 = safe_mean([r.get("param_f1") for r in rows])
    tec = safe_mean([r.get("tool_exec_consistency") for r in rows])

    out = {
        "file": file, "samples": n,
        "PSR": psr, "Node-F1": n_f1, "Link-F1": l_f1, "Latency": lat,
        "NH-1": nh1, "NH-2": nh2, "LH-1": lh1, "LH-2": lh2,
        "TokPrompt": tokp, "TokResp": tokr,
        # [M4] 新增两项
        "Param-F1": param_f1,
        "ToolExec-Consistency": tec,
    }
    return out

def train_mode(ds, do_plot=False):
    base_dir = f"runs/{ds}"
    metrics_files = glob.glob(os.path.join(base_dir, "**", "metrics_epoch.jsonl"), recursive=True)
    metrics_files = sorted(metrics_files)
    if not metrics_files:
        print(f"[INFO] No metrics_epoch.jsonl found under {base_dir}")
        return

    agg_dir = ensure_dir(os.path.join(base_dir, "_aggregate"))
    overlay_series_acc, overlay_series_loss, overlay_series_alpha, overlay_series_gate, overlay_series_pw50 = [], [], [], [], []
    overlay_labels = []

    summary_rows = []

    print(f"# Aggregate (train) for dataset={ds}")
    for mf in metrics_files:
        run_dir = os.path.dirname(mf)
        run_name = os.path.basename(run_dir)
        metrics = read_metrics_epoch(mf)

        # 1) 导出每 run 的 CSV
        per_run_csv = os.path.join(run_dir, "epoch_metrics.csv")
        write_csv(
            per_run_csv,
            metrics,
            header=["epoch", "train_loss", "eval_acc", "alpha", "gate_mean", "rag_lr", "pw_q10", "pw_q50", "pw_q90", "time"]
        )
        print(f"[CSV] {os.path.relpath(per_run_csv)}")

        # 2) 可选：画单 run 曲线
        if do_plot:
            xs = [r["epoch"] for r in metrics]
            acc = [r["eval_acc"] for r in metrics]
            loss = [r["train_loss"] for r in metrics]
            alpha = [r["alpha"] for r in metrics]
            gate = [r["gate_mean"] for r in metrics]
            q10 = [r["pw_q10"] for r in metrics]
            q50 = [r["pw_q50"] for r in metrics]
            q90 = [r["pw_q90"] for r in metrics]

            plot_series(os.path.join(run_dir, "acc_curve.png"),   [ (xs, acc) ],   [run_name], "epoch", "eval_acc",   title=run_name)
            plot_series(os.path.join(run_dir, "loss_curve.png"),  [ (xs, loss) ],  [run_name], "epoch", "train_loss", title=run_name)
            plot_series(os.path.join(run_dir, "alpha_curve.png"), [ (xs, alpha) ], [run_name], "epoch", "alpha",      title=run_name)
            if any(g is not None for g in gate):
                plot_series(os.path.join(run_dir, "gate_curve.png"),  [ (xs, gate) ],  [run_name], "epoch", "gate_mean",  title=run_name)
            if any(q is not None for q in q50):
                # 在单图里画 q10/q50/q90
                series = []
                labs = []
                if any(q is not None for q in q10): series.append((xs, q10)); labs.append("pw_q10")
                if any(q is not None for q in q50): series.append((xs, q50)); labs.append("pw_q50")
                if any(q is not None for q in q90): series.append((xs, q90)); labs.append("pw_q90")
                plot_series(os.path.join(run_dir, "pw_quantiles.png"), series, labs, "epoch", "pair_weight", title=run_name)

        # 3) 汇总 overlay 曲线
        xs = [r["epoch"] for r in metrics]
        overlay_series_acc.append( (xs, [r["eval_acc"] for r in metrics]) )
        overlay_series_loss.append( (xs, [r["train_loss"] for r in metrics]) )
        overlay_series_alpha.append( (xs, [r["alpha"] for r in metrics]) )
        overlay_series_gate.append( (xs, [r["gate_mean"] for r in metrics]) )
        overlay_series_pw50.append( (xs, [r["pw_q50"] for r in metrics]) )
        # label 优先 seed，其次 run 目录名
        summary = try_load_json(os.path.join(run_dir, "train_summary.json")) or {}
        seed = None
        if summary and isinstance(summary.get("args"), dict):
            seed = summary["args"].get("seed")
        label = f"{run_name}" if seed is None else f"{run_name}-seed{seed}"
        overlay_labels.append(label)

        # 4) 汇总表格行
        last = metrics[-1] if metrics else {}
        best_acc = None
        total_time = None
        if summary:
            best_acc = summary.get("best_eval_acc")
            total_time = summary.get("total_time_sec")
        summary_rows.append({
            "run_dir": run_dir,
            "seed": seed,
            "final_epoch": last.get("epoch"),
            "final_eval_acc": last.get("eval_acc"),
            "final_train_loss": last.get("train_loss"),
            "final_alpha": last.get("alpha"),
            "final_gate_mean": last.get("gate_mean"),
            "best_eval_acc": best_acc,
            "total_time_sec": total_time,
        })

    # 5) 输出聚合总表
    summary_csv = os.path.join(agg_dir, "runs_summary.csv")
    write_csv(summary_csv, summary_rows, header=[
        "run_dir","seed","final_epoch","final_eval_acc","final_train_loss",
        "final_alpha","final_gate_mean","best_eval_acc","total_time_sec"
    ])
    print(f"[CSV] {os.path.relpath(summary_csv)}")

    # 6) 可选：叠加曲线
    if do_plot:
        plot_series(os.path.join(agg_dir, "acc_overlay.png"),   overlay_series_acc,   overlay_labels, "epoch", "eval_acc",   title=f"{ds} / eval_acc")
        plot_series(os.path.join(agg_dir, "loss_overlay.png"),  overlay_series_loss,  overlay_labels, "epoch", "train_loss", title=f"{ds} / train_loss")
        plot_series(os.path.join(agg_dir, "alpha_overlay.png"), overlay_series_alpha, overlay_labels, "epoch", "alpha",      title=f"{ds} / alpha")
        # gate 可能为空
        if any(any(y is not None for y in ys) for (_, ys) in overlay_series_gate):
            plot_series(os.path.join(agg_dir, "gate_overlay.png"), overlay_series_gate, overlay_labels, "epoch", "gate_mean", title=f"{ds} / gate_mean")
        if any(any(y is not None for y in ys) for (_, ys) in overlay_series_pw50):
            plot_series(os.path.join(agg_dir, "pw50_overlay.png"), overlay_series_pw50, overlay_labels, "epoch", "pw_q50", title=f"{ds} / pair_weight@q50")

# ----------------------------- Entrypoint -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="dataset name under runs/<dataset>/")
    ap.add_argument("--mode", choices=["train", "eval"], default="train",
                    help="train: 读取 metrics_epoch.jsonl; eval: 读取按样本的预测 JSONL")
    ap.add_argument("--plot", action="store_true", help="生成 PNG 曲线")
    args = ap.parse_args()

    if args.mode == "train":
        train_mode(args.dataset, do_plot=args.plot)
    else:
        eval_mode(args.dataset)