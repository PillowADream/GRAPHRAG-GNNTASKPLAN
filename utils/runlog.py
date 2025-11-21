# utils/runlog.py
import os, json, datetime, threading

_LOCK = threading.Lock()

def _ts():
    return datetime.datetime.now().isoformat(timespec="milliseconds")

def _ensure_dir(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)

def open_jsonl(log_path:str):
    _ensure_dir(log_path)
    return open(log_path, "a", encoding="utf-8")

def log_sample(fp, *,
               ds:str, method:str, llm:str, seed:int, sid:str,
               success:bool, latency_sec:float=None,
               tokens_prompt:int=None, tokens_completion:int=None,
               temperature:float=None, top_p:float=None,
               alpha:float=None, graphsearch:str=None,
               task_steps=None, task_nodes=None, task_links=None, err:str=None):
    rec = {
        "ts": _ts(),
        "ds": ds, "method": method, "llm": llm, "seed": seed, "id": str(sid),
        "success": bool(success),
        "latency_sec": latency_sec,
        "tokens_prompt": tokens_prompt, "tokens_completion": tokens_completion,
        "temperature": temperature, "top_p": top_p,
        "alpha": alpha, "graphsearch": graphsearch,
        "task_steps": task_steps, "task_nodes": task_nodes, "task_links": task_links,
        "err": err
    }
    line = json.dumps(rec, ensure_ascii=False)
    with _LOCK:
        fp.write(line + "\n")
        fp.flush()