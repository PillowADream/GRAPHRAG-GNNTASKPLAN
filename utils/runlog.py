# utils/runlog.py
import os, json, datetime, threading

_LOCK = threading.Lock()

def _ts():
    return datetime.datetime.now().isoformat(timespec="milliseconds")

def _ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def open_jsonl(path: str, mode: str = "a", encoding: str = "utf-8"):
    return open(path, mode, encoding=encoding)

def log_sample(
    fp,
    ds: str,
    method: str,
    llm: str,
    seed: int,
    sid: str,
    success: bool,
    latency_sec: float = None,
    tokens_prompt: int = None,
    tokens_completion: int = None,
    temperature: float = None,
    top_p: float = None,
    alpha: float = None,
    graphsearch: bool = None,
    task_steps=None,
    task_nodes=None,
    task_links=None,
    err=None,
    **extra,  # <- 新增
):
    row = {
        "dataset": ds,
        "method": method,
        "llm": llm,
        "seed": seed,
        "sid": sid,
        "success": bool(success),
        "latency_sec": latency_sec,
        "tokens_prompt": tokens_prompt,
        "tokens_completion": tokens_completion,
        "temperature": temperature,
        "top_p": top_p,
        "alpha": alpha,
        "graphsearch": graphsearch,
        "err": err,
    }

    if task_steps is not None:
        row["task_steps"] = task_steps
    if task_nodes is not None:
        row["task_nodes"] = task_nodes
    if task_links is not None:
        row["task_links"] = task_links

    # M4: 允许透传 ToolLLM 相关字段
    for k, v in extra.items():
        row[k] = v

    fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    fp.flush()