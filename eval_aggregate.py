# eval_aggregate.py
import json, argparse, glob, os, csv
from statistics import mean
from pathlib import Path

# ---------------- Common helpers ----------------
def parse_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)

def safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return round(mean(xs), 4) if xs else None

def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def try_load_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def norm_nodes(L):
    out = []
    for n in (L or []):
        if isinstance(n, dict) and "task" in n:
            out.append(n["task"])
        elif isinstance(n, str):
            out.append(n)
    return out

def norm_links(L):
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

def f1_score(pred_list, gt_set):
    ps, gs = set(pred_list), set(gt_set)
    if not ps and not gs: return 1.0
    tp = len(ps & gs)
    if tp == 0: return 0.0
    p = tp / max(1, len(ps)); r = tp / max(1, len(gs))
    return 2*p*r/(p+r+1e-12)

# ---------------- Ground truth cache ----------------
def load_gt(ds: str):
    base = Path("data")/ds
    tool = json.load(open(base/"tool_desc.json","r",encoding="utf-8"))["nodes"]
    graph = json.load(open(base/"graph_desc.json","r",encoding="utf-8"))["links"]
    valid_nodes = {n["id"] for n in tool}
    valid_links = {f'{e["source"]}, {e["target"]}' for e in graph}

    gt = {}
    with open(base/"data.json","r",encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            gid = d["id"]
            gnodes = [(n["task"] if isinstance(n,dict) else n) for n in d.get("task_nodes",[])]
            glinks = [f'{e["source"]}, {e["target"]}' for e in d.get("task_links",[])]
            gt[gid] = (set(gnodes), set(glinks))
    return valid_nodes, valid_links, gt

# ---------------- Enrich per-row ----------------
def enrich_row_metrics(row, caches):
    valid_nodes, valid_links, gt = caches
    sid = row.get("sid", row.get("id"))
    pred_nodes = norm_nodes(row.get("task_nodes", []))
    pred_links = norm_links(row.get("task_links", []))
    gnodes, glinks = gt.get(sid, (set(), set()))

    # F1
    if row.get("node_f1") is None:
        row["node_f1"] = round(f1_score(pred_nodes, gnodes), 4)
    if row.get("link_f1") is None:
        row["link_f1"] = round(f1_score(pred_links, glinks), 4)

    # Hallucination
    need_hall = any(k not in row or row.get(k) is None for k in ("nh1","nh2","lh1","lh2"))
    if need_hall:
        n, m = len(pred_nodes), len(pred_links)
        nh1 = (sum(1 for x in pred_nodes if x not in valid_nodes) / max(1,n)) if n else 0.0
        nh2 = (sum(1 for x in pred_nodes if x not in gnodes) / max(1,n)) if n else 0.0
        lh1 = (sum(1 for x in pred_links if x not in valid_links) / max(1,m)) if m else 0.0
        lh2 = (sum(1 for x in pred_links if x not in glinks) / max(1,m)) if m else 0.0
        row.setdefault("nh1", round(nh1,4))
        row.setdefault("nh2", round(nh2,4))
        row.setdefault("lh1", round(lh1,4))
        row.setdefault("lh2", round(lh2,4))

    # Success & latency
    if "success" not in row or row.get("success") is None:
        row["success"] = bool(row["node_f1"] is not None and row["node_f1"] >= 0.99)
    if row.get("latency_sec") is None and row.get("cost_time") is not None:
        row["latency_sec"] = row.get("cost_time")

    # [M4] ToolExec-Consistency：优先用顶层；无则从 toollm[].exec_success 汇总
    if row.get("tool_exec_consistency") is None:
        tl = row.get("toollm")
        if isinstance(tl, list) and tl:
            vals = []
            for r in tl:
                v = r.get("exec_success")
                if v is True: vals.append(1.0)
                elif v is False: vals.append(0.0)
            if vals:
                row["tool_exec_consistency"] = round(mean(vals), 4)

    # [M4] Param-F1: 顶层 > 平均 toollm[*].param_f1 > (param_pred,param_gt) 精确匹配
    if row.get("param_f1") is None and row.get("param_f1_mean") is not None:
        row["param_f1"] = row["param_f1_mean"]
    if row.get("param_f1") is None:
        tl = row.get("toollm")
        if isinstance(tl, list) and tl:
            vals = [r.get("param_f1") for r in tl if isinstance(r.get("param_f1"), (int,float))]
            if vals:
                row["param_f1"] = round(mean(vals), 4)
    if row.get("param_f1") is None:
        pred, gtargs = row.get("param_pred"), row.get("param_gt")
        if isinstance(pred, dict) and isinstance(gtargs, dict):
            P, G = set(pred.keys()), set(gtargs.keys())
            tp = 0
            for k in P & G:
                a, b = pred[k], gtargs[k]; eq = False
                try:
                    if isinstance(a,(int,float)) or isinstance(b,(int,float)):
                        eq = float(a) == float(b)
                    else:
                        eq = str(a).strip().lower() == str(b).strip().lower()
                except Exception: eq = False
                if eq: tp += 1
            if (P or G):
                prec = tp / max(1,len(P)); rec = tp / max(1,len(G))
                row["param_f1"] = round(2*prec*rec/(prec+rec+1e-12),4)
            else:
                row["param_f1"] = 1.0
    return row

# ---------------- Aggregation ----------------
def agg_rows(rows, caches):
    rows = [enrich_row_metrics(r, caches) for r in rows]
    n = len(rows)
    psr = round(sum(1 for r in rows if r.get("success") is True)/n, 4) if n else 0.0
    lat = safe_mean([r.get("latency_sec", r.get("cost_time")) for r in rows])
    n_f1 = safe_mean([r.get("node_f1") for r in rows])
    l_f1 = safe_mean([r.get("link_f1") for r in rows])
    nh1 = safe_mean([r.get("nh1") for r in rows]); nh2 = safe_mean([r.get("nh2") for r in rows])
    lh1 = safe_mean([r.get("lh1") for r in rows]); lh2 = safe_mean([r.get("lh2") for r in rows])
    
    # [M5] Cost & RAG Trigger Stats
    tokp = safe_mean([r.get("tokens_prompt") for r in rows])
    tokr = safe_mean([r.get("tokens_completion") for r in rows])
    # 新增：聚合 RAG 触发率 (从 prediction json 的字段读取)
    rag_trig = safe_mean([r.get("rag_triggered_ratio") for r in rows])

    # [M4] Exec
    tec  = safe_mean([r.get("tool_exec_consistency") for r in rows])
    pf1  = safe_mean([r.get("param_f1") for r in rows])

    return {
        "samples": n, "PSR": psr, "Node-F1": n_f1, "Link-F1": l_f1, "Latency": lat,
        "NH-1": nh1, "NH-2": nh2, "LH-1": lh1, "LH-2": lh2,
        "TokPrompt": tokp, "TokResp": tokr, "RAG-Trig": rag_trig, # <--- Added
        "ToolExec-Consistency": tec, "Param-F1": pf1
    }

def dedup_merge(primary, secondary):
    """
    primary: 以 prediction 为主（包含 toollm），secondary: 日志/其他来源。
    同一个 id/sid 只保留一条；字段为空时用另一侧补齐。
    """
    by_id = {}
    def _key(d): return d.get("sid", d.get("id"))
    for d in primary:
        k = _key(d)
        if k is not None: by_id[k] = d
    for d in secondary:
        k = _key(d)
        if k is None: continue
        if k not in by_id:
            by_id[k] = d
        else:
            # 以 primary 为底，secondary 只用来补缺字段
            base = by_id[k]
            for kk, vv in d.items():
                if kk not in base or base.get(kk) is None:
                    base[kk] = vv
    return list(by_id.values())

# ---------------- Modes ----------------
def eval_mode(ds, logs_paths=None, preds_paths=None):
    caches = load_gt(ds)

    # 收集日志（可选）
    log_files = []
    if logs_paths:
        for p in logs_paths:
            if os.path.isdir(p):
                log_files.extend(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True))
            else:
                log_files.append(p)
    else:
        # 默认：runs/<ds>/**/*.jsonl（排除训练曲线）
        base_dir = f"runs/{ds}"
        log_files = glob.glob(os.path.join(base_dir, "**", "*.jsonl"), recursive=True)
        log_files = [f for f in log_files if os.path.basename(f) != "metrics_epoch.jsonl"]
    log_rows = []
    for lf in sorted(set(log_files)):
        try:
            log_rows.extend(list(parse_jsonl(lf)))
        except Exception:
            pass

    # 收集预测（可选）
    pred_rows = []
    if preds_paths:
        for p in preds_paths:
            pred_rows.extend(list(parse_jsonl(p)))

    # 合并去重（prediction 优先）
    rows = dedup_merge(pred_rows, log_rows)

    # 聚合并落表
    print(f"# Aggregate (eval) for dataset={ds}")
    result = agg_rows(rows, caches)
    
    # 格式化输出（增加 RAG-Trig）
    print(f"  samples={result['samples']} PSR={result['PSR']} "
          f"Node-F1={result['Node-F1']} Link-F1={result['Link-F1']} Latency={result['Latency']} "
          f"NH-1={result['NH-1']} NH-2={result['NH-2']} LH-1={result['LH-1']} LH-2={result['LH-2']} "
          f"Tok(P/R)=({result['TokPrompt']},{result['TokResp']}) RAG-Trig={result['RAG-Trig']} "
          f"ToolExec-Consistency={result['ToolExec-Consistency']} Param-F1={result['Param-F1']}")

    out_dir = ensure_dir("tables")
    out_csv = os.path.join(out_dir, f"{ds}_eval_agg.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file","samples","PSR","Node-F1","Link-F1","Latency","NH-1","NH-2","LH-1","LH-2",
                    "TokPrompt","TokResp","RAG-Trig","ToolExec-Consistency","Param-F1"])
        # 写一行汇总（file 留空）
        w.writerow(["", result["samples"], result["PSR"], result["Node-F1"], result["Link-F1"], result["Latency"],
                    result["NH-1"], result["NH-2"], result["LH-1"], result["LH-2"],
                    result["TokPrompt"], result["TokResp"], result["RAG-Trig"],
                    result["ToolExec-Consistency"], result["Param-F1"]])
    print(f"[CSV] {out_csv}")

def train_mode(ds, do_plot=False):
    base_dir = f"runs/{ds}"
    metrics_files = glob.glob(os.path.join(base_dir, "**", "metrics_epoch.jsonl"), recursive=True)
    metrics_files = sorted(metrics_files)
    if not metrics_files:
        print(f"[INFO] No metrics_epoch.jsonl found under {base_dir}")
        return

    agg_dir = ensure_dir(os.path.join(base_dir, "_aggregate"))
    overlay_series_acc, overlay_series_loss, overlay_series_alpha, overlay_series_gate, overlay_series_pw50 = [], [], [], [], []
    overlay_labels, summary_rows = [], []

    def read_metrics_epoch(jsonl_path):
        rows = list(parse_jsonl(jsonl_path))
        rows = sorted(rows, key=lambda r: r.get("epoch", 0))
        out = []
        for r in rows:
            out.append({
                "epoch": int(r.get("epoch", 0)), "train_loss": r.get("train_loss"),
                "eval_acc": r.get("eval_acc"), "alpha": r.get("alpha"),
                "gate_mean": r.get("gate_mean"), "rag_lr": r.get("rag_lr"),
                "pw_q10": r.get("pw_q10"), "pw_q50": r.get("pw_q50"), "pw_q90": r.get("pw_q90"),
                "time": r.get("time"),
            })
        return out

    def write_csv(path, rows, header):
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(header)
            for r in rows: w.writerow([r.get(k,"") for k in header])

    def plot_series(fig_path, series_list, labels, xlab, ylab, title=None):
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[WARN] matplotlib not available, skip plot: {e}"); return
        if not series_list: return
        plt.figure(figsize=(7.2, 4.2))
        for (xs, ys), lb in zip(series_list, labels):
            if xs and ys: plt.plot(xs, ys, label=lb)
        plt.xlabel(xlab); plt.ylabel(ylab)
        if title: plt.title(title)
        if len(series_list) > 1: plt.legend()
        plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(fig_path, dpi=160); plt.close()

    print(f"# Aggregate (train) for dataset={ds}")
    for mf in metrics_files:
        run_dir = os.path.dirname(mf); run_name = os.path.basename(run_dir)
        metrics = read_metrics_epoch(mf)
        per_run_csv = os.path.join(run_dir, "epoch_metrics.csv")
        write_csv(per_run_csv, metrics, header=[
            "epoch","train_loss","eval_acc","alpha","gate_mean","rag_lr","pw_q10","pw_q50","pw_q90","time"
        ])
        print(f"[CSV] {os.path.relpath(per_run_csv)}")

        if do_plot:
            xs = [r["epoch"] for r in metrics]
            def pick(key): return [r[key] for r in metrics]
            plot_series(os.path.join(run_dir, "acc_curve.png"),   [ (xs, pick("eval_acc")) ], [run_name], "epoch","eval_acc",   run_name)
            plot_series(os.path.join(run_dir, "loss_curve.png"),  [ (xs, pick("train_loss")) ], [run_name], "epoch","train_loss", run_name)
            plot_series(os.path.join(run_dir, "alpha_curve.png"), [ (xs, pick("alpha")) ],    [run_name], "epoch","alpha",      run_name)
            g = pick("gate_mean")
            if any(v is not None for v in g):
                plot_series(os.path.join(run_dir, "gate_curve.png"), [ (xs, g) ], [run_name], "epoch","gate_mean", run_name)
            q10, q50, q90 = pick("pw_q10"), pick("pw_q50"), pick("pw_q90")
            series, labs = [], []
            if any(v is not None for v in q10): series.append((xs, q10)); labs.append("pw_q10")
            if any(v is not None for v in q50): series.append((xs, q50)); labs.append("pw_q50")
            if any(v is not None for v in q90): series.append((xs, q90)); labs.append("pw_q90")
            if series:
                plot_series(os.path.join(run_dir, "pw_quantiles.png"), series, labs, "epoch","pair_weight", run_name)

        xs = [r["epoch"] for r in metrics]
        overlay_series_acc.append((xs, [r["eval_acc"] for r in metrics]))
        overlay_series_loss.append((xs, [r["train_loss"] for r in metrics]))
        overlay_series_alpha.append((xs, [r["alpha"] for r in metrics]))
        overlay_series_gate.append((xs, [r["gate_mean"] for r in metrics]))
        overlay_series_pw50.append((xs, [r["pw_q50"] for r in metrics]))

        summary = try_load_json(os.path.join(run_dir, "train_summary.json")) or {}
        seed = summary.get("args",{}).get("seed") if isinstance(summary.get("args"), dict) else None
        label = f"{run_name}" if seed is None else f"{run_name}-seed{seed}"
        overlay_labels.append(label)

        last = metrics[-1] if metrics else {}
        best_acc = summary.get("best_eval_acc") if summary else None
        total_time = summary.get("total_time_sec") if summary else None
        summary_rows.append({
            "run_dir": run_dir, "seed": seed,
            "final_epoch": last.get("epoch"), "final_eval_acc": last.get("eval_acc"),
            "final_train_loss": last.get("train_loss"), "final_alpha": last.get("alpha"),
            "final_gate_mean": last.get("gate_mean"), "best_eval_acc": best_acc,
            "total_time_sec": total_time,
        })

    summary_csv = os.path.join(agg_dir, "runs_summary.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "run_dir","seed","final_epoch","final_eval_acc","final_train_loss",
            "final_alpha","final_gate_mean","best_eval_acc","total_time_sec"
        ])
        for r in summary_rows:
            w.writerow([r.get(k,"") for k in [
                "run_dir","seed","final_epoch","final_eval_acc","final_train_loss",
                "final_alpha","final_gate_mean","best_eval_acc","total_time_sec"
            ]])
    print(f"[CSV] {os.path.relpath(summary_csv)}")

    if do_plot:
        plot_series(os.path.join(agg_dir, "acc_overlay.png"),   overlay_series_acc,   overlay_labels, "epoch","eval_acc",   f"{ds} / eval_acc")
        plot_series(os.path.join(agg_dir, "loss_overlay.png"),  overlay_series_loss,  overlay_labels, "epoch","train_loss", f"{ds} / train_loss")
        plot_series(os.path.join(agg_dir, "alpha_overlay.png"), overlay_series_alpha, overlay_labels, "epoch","alpha",      f"{ds} / alpha")
        if any(any(y is not None for y in ys) for _, ys in overlay_series_gate):
            plot_series(os.path.join(agg_dir, "gate_overlay.png"), overlay_series_gate, overlay_labels, "epoch","gate_mean", f"{ds} / gate_mean")
        if any(any(y is not None for y in ys) for _, ys in overlay_series_pw50):
            plot_series(os.path.join(agg_dir, "pw50_overlay.png"), overlay_series_pw50, overlay_labels, "epoch","pw_q50",    f"{ds} / pair_weight@q50")

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--mode", choices=["train","eval"], default="train")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--logs", nargs="*", help="one or more log files or directories")
    ap.add_argument("--preds", nargs="*", help="one or more prediction jsonl files")
    args = ap.parse_args()

    if args.mode == "train":
        train_mode(args.dataset, do_plot=args.plot)
    else:
        eval_mode(args.dataset, logs_paths=args.logs, preds_paths=args.preds)