# scripts/plot_m6.py
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
from pathlib import Path

# 引入 eval_aggregate 的逻辑 (假设在根目录运行)
sys.path.append(os.getcwd())
# 注意：请确保项目根目录下有 eval_aggregate.py
from eval_aggregate import load_gt, agg_rows, dedup_merge, parse_jsonl

def get_metrics(ds, run_tag, llm="CodeLlama-13b-Instruct-hf"):
    # 适配 Windows 路径分隔符
    llm_short = "CodeLlama-13b" if "CodeLlama-13b" in llm else llm
    
    # 预测文件路径
    pred_filename = f"graphsearch_beam_2_{run_tag}_auto.json"
    pred_path = os.path.join("prediction", ds, llm_short, pred_filename)
    
    # 日志文件路径模式 (匹配追加的日志)
    log_pattern = os.path.join("runs", ds, "**", f"graphsearch_beam_2_{run_tag}_*.jsonl")
    
    log_files = glob.glob(log_pattern, recursive=True)
    
    # 如果没找到文件，返回空
    if not os.path.exists(pred_path) and not log_files:
        print(f"[WARN] No data for {ds} {run_tag} (Checked: {pred_path})")
        return None

    print(f"[INFO] Loading {ds} - {run_tag}...")
    caches = load_gt(ds)
    pred_rows = list(parse_jsonl(pred_path)) if os.path.exists(pred_path) else []
    log_rows = []
    for lf in log_files:
        try: log_rows.extend(list(parse_jsonl(lf)))
        except: pass
        
    rows = dedup_merge(pred_rows, log_rows)
    return agg_rows(rows, caches)

def plot_m6(datasets=["tmdb", "dailylife", "huggingface"]):
    # 配置列表
    configs = [
        {"tag": "base", "label": "Graph Only", "color": "gray", "marker": "o"},
        {"tag": "full_rag", "label": "Full RAG", "color": "blue", "marker": "s"},
        {"tag": "trig_rag", "label": "Triggered RAG", "color": "green", "marker": "^"},
        {"tag": "self_correct", "label": "Self-Correct (Ours)", "color": "red", "marker": "*"},
    ]
    
    # 设置绘图风格
    sns.set_style("whitegrid")
    
    # 1. Pareto Curve: Cost vs Performance (Node-F1)
    for ds in datasets:
        plt.figure(figsize=(6, 5))
        data_points = []
        for cfg in configs:
            m = get_metrics(ds, cfg["tag"])
            if m:
                # 计算总 Token (Prompt + Resp)
                tokens = (m.get("TokPrompt") or 0) + (m.get("TokResp") or 0)
                # 如果 Baseline 没有 Token 记录（可能因为没跑 LLM 统计），手动设一个估计值或跳过
                if tokens == 0 and cfg["tag"] == "base": continue 
                
                data_points.append({
                    "Tokens": tokens,
                    "Node-F1": m["Node-F1"],
                    "Label": cfg["label"],
                    "Color": cfg["color"],
                    "Marker": cfg["marker"]
                })

        if not data_points: 
            plt.close()
            continue
        
        # 绘图
        for dp in data_points:
            plt.scatter(dp["Tokens"], dp["Node-F1"], color=dp["Color"], marker=dp["Marker"], s=100, label=dp["Label"])
            # 防止文字重叠，稍微偏移
            plt.text(dp["Tokens"], dp["Node-F1"]+0.002, dp["Label"], fontsize=9)
            
        plt.xlabel("Avg. Token Consumption")
        plt.ylabel("Node-F1 Score")
        plt.title(f"Cost-Effectiveness Frontier ({ds})")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        out_fig = os.path.join("figs", f"m6_pareto_{ds}.pdf")
        plt.savefig(out_fig)
        print(f"[PLOT] Generated {out_fig}")
        plt.close()

    # 2. Execution Gap Bar Chart: PSR vs Param-F1 (Focus on Self-Correct)
    rows = []
    for ds in datasets:
        # Base vs Self-Correct
        m_base = get_metrics(ds, "base")
        m_ours = get_metrics(ds, "self_correct")
        
        if m_base:
            rows.append({"Dataset": ds, "Method": "Baseline", "Metric": "Planning (PSR)", "Value": m_base["PSR"]})
            # Baseline 默认 Param-F1 极低（或设为 0 以示对比）
            base_pf1 = m_base.get("Param-F1") or 0.01
            rows.append({"Dataset": ds, "Method": "Baseline", "Metric": "Grounding (Param-F1)", "Value": base_pf1})
            
        if m_ours:
            rows.append({"Dataset": ds, "Method": "Ours (M4)", "Metric": "Planning (PSR)", "Value": m_ours["PSR"]})
            rows.append({"Dataset": ds, "Method": "Ours (M4)", "Metric": "Grounding (Param-F1)", "Value": m_ours.get("Param-F1", 0)})

    if rows:
        df = pd.DataFrame(rows)
        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x="Dataset", y="Value", hue="Metric", palette="viridis")
        plt.title("The Planning-Grounding Gap & Improvement")
        plt.ylabel("Score")
        plt.ylim(0, 1.1)
        plt.tight_layout()
        out_fig = os.path.join("figs", "m6_execution_gap.pdf")
        plt.savefig(out_fig)
        print(f"[PLOT] Generated {out_fig}")

if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    plot_m6()