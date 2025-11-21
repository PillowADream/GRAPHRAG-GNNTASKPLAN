# trainfree/graphsearch.py
import click
import os
import json
import copy
import requests
import time
import sys
import re
from pathlib import Path

# python -m trainfree.graphsearch --dataset dailylife --llm CodeLlama-13b-Instruct-hf --strategy beam --width 2 --threshold 3 --use_graphrag 1 --rag_mode auto --rag_topk_nodes 6 --rag_topk_comms 3 --rag_use_communities 0 --rag_min_score_nodes 50 --mode full --overwrite 1 --run_tag tuned50

def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if (p / "data").exists() and (p / "prediction").exists():
            return p
    return Path.cwd()

def _extract_tool_id_from_step_text(text: str):
    """
    从 'Step X: Call <tool_id> API ...' 中抽取 tool_id。
    抽取失败返回 None。
    """
    if not isinstance(text, str):
        try:
            text = _as_step_text(text)
        except Exception:
            return None
    m = re.search(r"Call\s+([A-Za-z0-9_]+)\s+API", text)
    return m.group(1) if m else None

def _build_rag_query_from_step(step_text: str) -> str:
    """
    把 'Step k: Call deliver_package API with ...' 这类句子，压缩成更适合检索的关键词：
    1) 若能提取到 tool_id，就直接用 tool_id 的“空格版”（deliver package）
    2) 否则：去掉 'Step/Call/API' 等词与标点，只保留字母数字空格
    """
    tid = _extract_tool_id_from_step_text(step_text)
    if isinstance(tid, str) and tid:
        return tid.replace("_", " ")

    s = re.sub(r"Step\s*\d+:\s*", "", step_text, flags=re.I)
    s = re.sub(r"\b(Call|API)\b.*", "", s, flags=re.I)  # 截断到 API 前
    s = re.sub(r"[^A-Za-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or step_text

ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import reformat_steps, get_cur_time
from utils.runlog import open_jsonl, log_sample
from retrieval.graphrag_adapter import GraphRAGAdapter
from retrieval.entity_linking import EntityLinker
# ====== M4: ToolLLM 适配器 ======
try:
    from retrieval.toollm_adapter import ToolLLMAdapter
except Exception:
    ToolLLMAdapter = None  # 若未放置该文件，后续根据 --use_toolllm 做检查

# ============== HTTP ==============
def _post_chat(url, payload, api_key=None, timeout=300):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    txt = resp.text
    try:
        data = resp.json()
    except Exception:
        raise Exception(f"HTTP {resp.status_code}: {txt}")
    if resp.status_code == 429:
        raise Exception(f"Rate Limit Error {data}")
    if resp.status_code != 200:
        raise Exception(str(data))
    return data

def _try_json(text: str):
    return json.loads(text)

def _as_step_text(step):
    if isinstance(step, dict):
        return step.get("description") or step.get("desc") or json.dumps(step, ensure_ascii=False)
    return str(step)

# ============ LLM ============
def get_llm_response(llm, url, prompt, prompt_answer_type="default", api_key=None,
                     temperature=0.2, top_p=0.1):
    global counter, tok_prompt_cnt, tok_resp_cnt   # ← 新增 tok_* 全局计数器
    base_payload = {
        "model": llm,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(temperature),
        "top_p": float(top_p),
        "frequency_penalty": 0,
        "presence_penalty": 1.05,
        "max_tokens": 2000,
        "stream": False,
        "response_format": {"type": "json_object"},
    }
    data = _post_chat(url, base_payload, api_key=api_key, timeout=300)
    counter += 1

    # ====== M5: 累加一次调用的 Token 使用量 ======
    try:
        usage = data.get("usage") or {}
        tok_prompt_cnt += int(usage.get("prompt_tokens", 0) or 0)
        tok_resp_cnt   += int(usage.get("completion_tokens", 0) or 0)
    except Exception:
        pass
    # ============================================

    content = data["choices"][0]["message"]["content"]
    try:
        return _try_json(content)
    except Exception:
        pass

    if prompt_answer_type == "solution":
        fmt_prompt = (
            "Please format the result # RESULT # to a strict JSON object.\n"
            "Requirements:\n"
            "1) Do not change the meaning of each candidate tool;\n"
            "2) Return ONE JSON object only; no extra text or code fences;\n"
            '3) Schema: {"best_solution": [the best solution as a list]}\n'
            "# RESULT #:\n{{illegal_result}}"
        )
    else:
        fmt_prompt = (
            "Please format the result # RESULT # to a strict JSON object.\n"
            "Requirements:\n"
            "1) Do not change the meaning of each candidate tool;\n"
            "2) Return ONE JSON object only; no extra text or code fences;\n"
            '3) Schema: {"candidate tool name": score, ...}\n'
            "# RESULT #:\n{{illegal_result}}"
        )
    retry_payload = dict(base_payload)
    retry_payload["messages"] = [{
        "role": "user",
        "content": fmt_prompt.replace("{{illegal_result}}", content)
    }]
    data2 = _post_chat(url, retry_payload, api_key=api_key, timeout=180)
    counter += 1

    # ====== M5: 重试调用的 Token 也计入 ======
    try:
        usage2 = data2.get("usage") or {}
        tok_prompt_cnt += int(usage2.get("prompt_tokens", 0) or 0)
        tok_resp_cnt   += int(usage2.get("completion_tokens", 0) or 0)
    except Exception:
        pass
    # ========================================

    content2 = data2["choices"][0]["message"]["content"]
    try:
        return _try_json(content2)
    except Exception:
        return {'best_solution': []} if prompt_answer_type == "solution" else {}


def prompt_llm_final_solutions(llm, url, user_request, parsed_steps, solution_list,
                               tmp_print=False, api_key=None, temperature=0.2, top_p=0.1):
    if len(solution_list) == 1:
        return {"best_solution": solution_list[0]}
    if len(solution_list) == 0:
        return {"best_solution": []}
    prompt = (
        "\n# GOAL #\nBased on the provided USER REQUEST and the initially inferred STEPS "
        "(to be performed in sequence to solve the user's request), select the best tool solution list from the SOLUTION LIST. "
        "The selected solution should be the one that can perfectly solve the user's request and strictly align with the inferred steps. "
        'The format must be a strict JSON object: {"best_solution": [list of invoked tools]}'
    )
    prompt += (
        "\n\n# REQUIREMENTS #\n1. Your goal is to select the best solution that can best follow the inferred steps and perfectly solve user's request. "
        "Only return the best solution strictly from the provided SOLUTION LIST. Do not change their sequences;"
        "\n2. Carefully analyze both the user's request and the previously inferred task steps."
        f"\n3. Make sure that each tool in the final solution list exists in the valid # TOOL LIST #: {tool_name_list}."
    )
    prompt += (
        f"\n\n# USER REQUEST #: {user_request}"
        f"\n# STEPS #: {parsed_steps}"
        f"\n# SOLUTION LIST #: {solution_list}"
        "\nNow return ONLY a strict JSON object:\n# RESULT #:"
    )
    response = get_llm_response(
        llm, url, prompt, prompt_answer_type="solution",
        api_key=api_key, temperature=temperature, top_p=top_p
    )
    if tmp_print:
        print(response)
    return response

def prompt_llm_candidate_tool_scores(llm, url, step_description: str, provided_demo: int,
                                     tool_candidates: list, tmp_print=False,
                                     api_key=None, temperature=0.2, top_p=0.1):
    prompt = (
        "\n# GOAL #: Based on the provided CANDIDATE TOOL LIST and the user's request described in the STEP, "
        "generate a score dictionary to assess each tool's problem-solving abilities for the given request. "
        'The output must be a strict JSON object like: {"candidate tool name 1": score, ...}'
    )
    prompt += (
        "\n\n# REQUIREMENTS #:\n"
        "1) Keys must align with the provided candidate tools; output scores for ALL candidates;\n"
        "2) 'score' ∈ {1,2,3,4,5}; higher means better match;\n"
        "3) Consider the STEP carefully;\n"
        "4) If the STEP explicitly contains a candidate tool, give it a score ≥ 3."
    )

    # ---- Demo 仅在存在的集合中追加（避免 tmdb KeyError）----
    if provided_demo:
        demos_dict = {
            "huggingface": [
                {"step":"Answer a question related to the depth information from the document",
                 "candidates":["Document Question Answering","Question Answering","Visual Question Answering","Text Generation"],
                 "result":{"Document Question Answering":5,"Question Answering":2,"Visual Question Answering":3,"Text Generation":1}},
                {"step":"Generate a new text based on the translated French text.",
                 "candidates":["Text Generation","Text-to-Image","Text-to-Video","Translation"],
                 "result":{"Text Generation":5,"Text-to-Image":1,"Text-to-Video":1,"Translation":2}}
            ],
            "multimedia": [
                {"step":"Use Image Stitcher to stitch together two images.",
                 "candidates":["Image Search","Image Stitcher","Image Colorizer","Image Style Transfer"],
                 "result":{"Image Search":1,"Image Stitcher":5,"Image Colorizer":1,"Image Style Transfer":1}},
                {"step":"Extract a still image from the video 'example.mp4'",
                 "candidates":["Video-to-Image","Video Search","Video-to-Text"],
                 "result":{"Video-to-Image":5,"Video Search":1,"Video-to-Text":2}},
                {"step":"Use Text Downloader to download the text content ...",
                 "candidates":["Text Search","Text Summarizer","Text Downloader"],
                 "result":{"Text Search":2,"Text Summarizer":2,"Text Downloader":5}},
                {"step":"Extract text from the image obtained in Step 1 using OCR",
                 "candidates":["Image-to-Text","Text Summarizer"],
                 "result":{"Image-to-Text":5,"Text Summarizer":1}}
            ],
            "dailylife": [
                {"step":"Call search_by_engine API with query: 'How to use Microsoft Word' and engine: 'Google'",
                 "candidates":["search_by_engine","apply_for_job"],
                 "result":{"search_by_engine":5,"apply_for_job":1}},
                {"step":"Call buy_insurance API with insurance: 'Health Insurance' and company: 'ABC Insurances'",
                 "candidates":["buy_insurance","stock_operation","online_shopping"],
                 "result":{"buy_insurance":5,"stock_operation":2,"online_shopping":1}},
                {"step":"Call organize_meeting_online API with topic: 'Data Privacy and Security'",
                 "candidates":["organize_meeting_online","attend_meeting_online","make_video_call"],
                 "result":{"organize_meeting_online":5,"attend_meeting_online":3,"make_video_call":2}}
            ]
        }
        demo_string = ""
        if dataset_name in demos_dict:
            for demo in demos_dict[dataset_name]:
                demo_string += (
                    f'\n# EXAMPLE #:\n# STEP #: {demo["step"]}\n'
                    f'# CANDIDATE TOOL LIST #: {demo["candidates"]}\n'
                    f'# RESULT #: {json.dumps(demo["result"])}'
                )
            prompt += "\n" + demo_string

    candidate_tool_string = "# CANDIDATE TOOL LIST #:\n"
    assert len(tool_candidates) <= len(tool_dict)
    for tool in tool_candidates:
        candidate_tool_string += json.dumps(tool, ensure_ascii=False) + "\n"

    step_text = _as_step_text(step_description)
    prompt += "\n\n# STEP #:\n" + step_text + "\nNow return ONLY a strict JSON object:\n# RESULT #:"
    final_prompt = candidate_tool_string + prompt
    response = get_llm_response(
        llm, url, final_prompt, prompt_answer_type='default',
        api_key=api_key, temperature=temperature, top_p=top_p
    )
    if tmp_print:
        print(step_text, response, end=' ')
    return response

# ============ RAG helpers ============
def _build_local_alias_index(tool_nodes):
    idx = {}
    for t in tool_nodes:
        tid = t["id"]
        idx[tid.lower()] = tid
        if "name" in t and isinstance(t["name"], str):
            idx[t["name"].lower()] = tid
        if "aliases" in t and isinstance(t["aliases"], list):
            for a in t["aliases"]:
                if isinstance(a, str):
                    idx[a.lower()] = tid
        # 自动加入下划线展开的短语
        idx[tid.replace("_", " ").lower()] = tid
    return idx

def _extract_from_node_dict(d):
    # 更鲁棒的字段尝试顺序
    for k in ("id","name","title","label","entity","tool","tool_id","text","span"):
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None

def _extract_rag_node_names(rag_result):
    names = []
    if not rag_result or not isinstance(rag_result, dict):
        return names

    # nodes: 支持 str / dict{id|name}
    if isinstance(rag_result.get("nodes"), list):
        for n in rag_result["nodes"]:
            if isinstance(n, str):
                names.append(n)
            elif isinstance(n, dict):
                names.append(n.get("id") or n.get("name"))

    # communities: 收集 seed 与 members
    if isinstance(rag_result.get("communities"), list):
        for c in rag_result["communities"]:
            if not isinstance(c, dict):
                continue
            seed = c.get("seed")
            if isinstance(seed, str):
                names.append(seed)
            mems = c.get("members")
            if isinstance(mems, list):
                for m in mems:
                    if isinstance(m, str):
                        names.append(m)
                    elif isinstance(m, dict):
                        names.append(m.get("id") or m.get("name"))

    # 去掉空白
    return [x for x in names if isinstance(x, str) and x.strip()]

def _link_names_to_tool_ids(names, linker, local_alias_idx):
    """
    names -> tool_ids
    兼容 EntityLinker.link 返回的多种类型:
      - str: 直接当作 tool_id / 别名
      - dict: 取 id 或 name
      - list: 递归展开其中的 str/dict
      - None: 用本地别名表兜底
    """
    out = []

    def _add_one(x):
        if not x:
            return
        if isinstance(x, str):
            out.append(x)
        elif isinstance(x, dict):
            v = x.get("id") or x.get("name")
            if isinstance(v, str):
                out.append(v)

    for n in names:
        cand = None
        if linker is not None:
            try:
                cand = linker.link(n)  # 可能返回 str / dict / list / None
            except Exception:
                cand = None

        if cand is None:
            # 用别名兜底（大小写不敏感）
            v = local_alias_idx.get(str(n).lower())
            if v:
                out.append(v)
        elif isinstance(cand, list):
            for it in cand:
                _add_one(it)
        else:
            _add_one(cand)

    # 统一小写后再用原值去重，保持顺序
    seen = set()
    uniq = []
    for x in out:
        if not isinstance(x, str):
            continue
        key = x.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(x)

    # 只保留图中存在的工具 id / 名称（两者都可映射）
    filtered = []
    for t in uniq:
        tid = t
        # 若是名字而非 id，用别名表转成 id
        tid = local_alias_idx.get(t.lower(), tid)
        if tid in tool_dict:
            filtered.append(tid)
    return filtered

# ============ Triggered RAG (M5) ============
def _should_trigger_rag(num_graph_candidates: int, step_idx: int) -> bool:
    """
    逻辑：
      - 若 rag_triggered_flag=False：始终触发；
      - 若 rag_triggered_flag=True：
          * 若 gs_conf_thres < 0：仍然全部触发（保留负数作为特殊值）；
          * 否则，当 num_graph_candidates <= gs_conf_thres 时触发。
    """
    if not rag_triggered_flag:
        return True
    
    # <<< 修改：允许 0 作为有效阈值，仅负数视为“全开” >>>
    if gs_conf_thres < 0:
        return True
        
    try:
        return int(num_graph_candidates) <= int(gs_conf_thres)
    except Exception:
        return True

# -------------- Param-F1（必填键集合对齐） --------------
def _param_f1_against_required(schema: dict, pred: dict):
    if not schema or not isinstance(pred, dict):
        return 0.0, 0.0, 0.0
    req = [p for p in schema.get("parameters", []) or [] if p.get("required")]
    req_keys = {p.get("name") for p in req if p.get("name")}
    if not req_keys:
        return 1.0, 1.0, 1.0  # 无必填参数时视为完美

    def _typed_ok(pname, ptype, val):
        if ptype == "integer":
            return isinstance(val, int) or (isinstance(val, str) and str(val).isdigit())
        if ptype == "number":
            if isinstance(val, (int, float)):
                return True
            try:
                float(str(val)); return True
            except Exception:
                return False
        if ptype == "boolean":
            return isinstance(val, bool) or str(val).lower() in {"true", "false"}
        return True

    pred_keys = set(pred.keys())
    correct = 0
    for p in req:
        name = p.get("name"); ptype = (p.get("schema") or {}).get("type")
        if name in pred and _typed_ok(name, ptype, pred[name]):
            correct += 1

    tp = correct
    # 预测中命中的必填但类型不对计入 fp；非必填键也计入 fp，使 precision 更严格
    must_hit = pred_keys & req_keys
    fp = max(0, len(must_hit) - tp) + len(pred_keys - req_keys)
    fn = len(req_keys) - tp

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall    = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1        = (2*precision*recall / (precision+recall)) if (precision+recall) > 0 else 0.0
    return precision, recall, f1

# ============ Candidate Generation ============
def generate_candidates(
    score_dict,
    tool_candidates,
    rag_prior_ids=None,
    rag_floor=4,
    rag_bonus=1,
    force_ids=None
):
    if rag_prior_ids is None:
        rag_prior_ids = set()
    else:
        rag_prior_ids = set(rag_prior_ids)

    force_ids = [fid for fid in (force_ids or []) if isinstance(fid, str)]
    force_set = set(force_ids)

    # ---- 容错：LLM 可能用“名称键”打分，这里映射到 id（补丁 F）----
    norm_score = {}
    for k, v in (score_dict or {}).items():
        if k in tool_dict:
            norm_score[k] = v
        else:
            kid = local_alias_idx.get(str(k).lower())
            if kid in tool_dict:
                norm_score[kid] = v
    score_dict = norm_score

    # 组装 (score, idx, tid)；RAG 命中抬到 rag_floor 再 + rag_bonus
    tool_score_list = []
    seen_pool = set()
    for i, tool in enumerate(tool_candidates):
        tid = tool["id"]
        if tid in seen_pool:
            continue
        seen_pool.add(tid)
        base = score_dict.get(tid)
        try:
            base = int(base)
        except Exception:
            base = 1
        if tid in rag_prior_ids:
            base = max(base, int(rag_floor)) + int(rag_bonus)
        tool_score_list.append((base, i, tid))

    # 把 LLM 额外建议但不在 pool 的工具也纳入
    for tid, s in score_dict.items():
        if tid in tool_dict and tid not in seen_pool:
            try:
                base = int(s)
            except Exception:
                base = 1
            if tid in rag_prior_ids:
                base = max(base, int(rag_floor)) + int(rag_bonus)
            tool_score_list.append((base, len(tool_candidates), tid))
            tool_candidates.append(tool_dict[tid])
            seen_pool.add(tid)

    # 按分数降序；同分按较小索引优先
    tool_score_list.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    # === 关键改动：让 beam 也遵守 threshold，避免“随机第二名” ===
    if search_strategy == "greedy":
        selected_idxs = [tool_score_list[0][1]] if tool_score_list else []
    elif search_strategy == "beam":
        strong = [tpl for tpl in tool_score_list if tpl[0] >= score_threshold]
        if strong:
            k = min(beam_width, len(strong))
            selected_idxs = [strong[j][1] for j in range(k)]
        else:
            selected_idxs = [tool_score_list[0][1]] if tool_score_list else []
    elif search_strategy == "adaptive":
        selected_idxs = [idx for sc, idx, tid in tool_score_list if sc >= score_threshold]
        if not selected_idxs and tool_score_list:
            selected_idxs = [tool_score_list[0][1]]
    else:
        raise NotImplementedError()

    # 去重 + 组装
    by_id = {}
    for idx in selected_idxs:
        t = tool_candidates[idx]
        by_id.setdefault(t["id"], t)

    # 强制工具顶到最前
    forced_part = [by_id[fid] for fid in force_ids if fid in by_id]
    rest_part = [t for tid, t in by_id.items() if tid not in force_set]
    return forced_part + rest_part

# ================= DFS =================
def dfs(llm, url, current_idx, steps, solutions, current_tools, tmp_print=False,
        api_key=None, temperature=0.2, top_p=0.1,
        case_meta=None, rag_mode_opt=None, rag_k_nodes=None, rag_k_comms=None,
        rag_use_comms=False, rag_min_score_nodes=35):
    if current_idx == len(steps) or counter >= 30:
        solutions.append(copy.deepcopy(current_tools))
        return

    # 基于图邻接的候选
    if current_idx == 0:
        candidate_tools = copy.deepcopy(tool_nodes)
        graph_ids = {t["id"] for t in candidate_tools}
    else:
        if len(current_tools) == 0:
            return
        last_tool = current_tools[-1]
        candidate_tools = copy.deepcopy(tool_graph[last_tool]["children"])
        if len(candidate_tools) == 0:
            return
        graph_ids = {t["id"] for t in candidate_tools}

    step_text = _as_step_text(steps[current_idx])
    # <<< 新增：抽取显式工具，并加入候选池 >>>
    step_tool = _extract_tool_id_from_step_text(step_text)
    if step_tool and step_tool in tool_dict:
        ids_now = {t["id"] for t in candidate_tools}
        if step_tool not in ids_now:
            candidate_tools.append(tool_dict[step_tool])

    # RAG 候选
    rag_ids = set()
    rag_added = []
    step_text = _as_step_text(steps[current_idx])

    # 默认：不触发 RAG（对于“常开 RAG”模式，会在 _should_trigger_rag 中始终返回 True）
    need_rag = False

    step_tool = _extract_tool_id_from_step_text(step_text)
    if step_tool and step_tool in tool_dict:
        # 显式步骤：直接强制将该工具纳入候选，且跳过 GraphRAG（节省时间、避免噪声）
        if step_tool not in {t["id"] for t in candidate_tools}:
            candidate_tools.append(tool_dict[step_tool])
        rag_ids = {step_tool}  # 仅用于后面“先验加分/force_ids”的逻辑
        if tmp_print:
            print(f"[DBG] step={current_idx} explicit_tool='{step_tool}', skip RAG")
    else:
        # 只有在未显式点名工具时，才根据触发策略决定是否调用 GraphRAG
        need_rag = use_graphrag_flag and (rag_adapter is not None) and _should_trigger_rag(len(graph_ids), current_idx)
        if need_rag:
            try:
                k_nodes = rag_k_nodes if rag_k_nodes is not None else 8
                k_comms = rag_k_comms if rag_k_comms is not None else 3
                mode    = rag_mode_opt if rag_mode_opt else "auto"

                # 用“精简后的关键词”做 RAG 查询，提升命中
                rag_query = _build_rag_query_from_step(step_text)
                rag_res = rag_adapter.search(
                    query=rag_query,
                    mode=mode,
                    topk_nodes=k_nodes,
                    topk_comms=k_comms
                )

                # 只保留分数过阈值的 nodes
                rag_pick = []
                nodes = rag_res.get("nodes", []) if isinstance(rag_res, dict) else []
                for n in nodes:
                    if isinstance(n, dict):
                        tid = n.get("id") or n.get("name")
                        sc  = n.get("score", 0)
                        try:
                            sc = float(sc)
                        except Exception:
                            sc = 0.0
                        if isinstance(tid, str) and tid in tool_dict and sc >= rag_min_score_nodes:
                            rag_pick.append(tid)
                rag_ids = set(rag_pick)

                # 合并 RAG 命中到候选池
                existed = {t["id"] for t in candidate_tools}
                for tid in rag_ids:
                    if tid not in existed:
                        candidate_tools.append(tool_dict[tid])

                # 仅在重叠“有区分度”时，才裁成交集
                inter = graph_ids & rag_ids
                union = graph_ids | rag_ids
                jacc = (len(inter) / max(1, len(union)))
                if 0.15 <= jacc <= 0.85 and len(inter) >= 3:
                    candidate_tools = [tool_dict[tid] for tid in inter]
                    graph_ids = set(inter)
                    if tmp_print:
                        print(f"[DBG] step={current_idx} prune_to_intersection={len(inter)} (jacc={jacc:.2f})")
                else:
                    if tmp_print:
                        print(f"[DBG] step={current_idx} keep_union (jacc={jacc:.2f}, inter={len(inter)}, union={len(union)})")
            except Exception as e:
                if tmp_print:
                    print(f"[RAG-err] step={current_idx} err={e}")

    # 记录来源统计（含是否触发 RAG）
    if case_meta is not None:
        inter = graph_ids & rag_ids
        union = graph_ids | rag_ids
        case_meta.append({
            "step_idx": current_idx,
            "text": step_text,
            "cand_graph": len(graph_ids),
            "cand_rag": len(rag_ids),
            "cand_overlap": len(inter),
            "cand_union": len(union),
            "rag_mode": rag_mode_opt if use_graphrag_flag else None,
            "rag_triggered": bool(need_rag),  # M5: 标记该 step 是否真正触发了 GraphRAG
        })

    inter_ids = graph_ids & rag_ids
    # 如果 RAG 给的东西不全为空，并且交集覆盖比例不太小，就只保留交集作为候选；
    # 阈值随便给个 0.2（20%）起步，太小怕误杀，后续你可以调大/调小。
    if inter_ids and (len(inter_ids) / max(1, len(graph_ids))) >= 0.2:
        candidate_tools = [tool_dict[tid] for tid in inter_ids]
        if tmp_print:
            print(f"[DBG] step={current_idx} prune_to_intersection={len(candidate_tools)}")

    # 让 LLM 打分
    candidate_score = prompt_llm_candidate_tool_scores(
        llm, url, steps[current_idx], 1, candidate_tools, tmp_print,
        api_key=api_key, temperature=temperature, top_p=top_p
    )
    # <<< 新增：显式工具抬到 5 分 >>>
    if step_tool:
        prev = candidate_score.get(step_tool, 1)
        candidate_score[step_tool] = max(5, int(prev) if isinstance(prev, int) else 1)

    candidate_list = generate_candidates(
        candidate_score,
        candidate_tools,
        rag_prior_ids=rag_ids,
        rag_floor=4,
        rag_bonus=1,
        force_ids=[step_tool] if step_tool else None  # <<< 新增：强制放到最前
    )

    if tmp_print:
        print([candidate["id"] for candidate in candidate_list], end='\n')

    for tool in candidate_list:
        tool_name = tool["id"]
        if tool_name in current_tools and search_strategy != 'greedy':
            continue
        current_tools.append(tool_name)
        dfs(llm, url, current_idx+1, steps, solutions, current_tools, tmp_print,
            api_key=api_key, temperature=temperature, top_p=top_p,
            case_meta=case_meta, rag_mode_opt=rag_mode_opt,
            rag_k_nodes=rag_k_nodes, rag_k_comms=rag_k_comms,
            rag_use_comms=rag_use_comms, rag_min_score_nodes=rag_min_score_nodes)
        current_tools.pop(-1)
    return

# ============= ToolLLM 后处理（M4新增） =============
def _toollm_post_steps(adapter, steps, sol_ids, confirm_top=3, sample_gt=None, real_call=None):
    """
    steps     : reformat_steps() 后的步骤列表
    sol_ids   : 最终最佳方案（工具 id 序列）
    confirm_top: 检索前 K 位内命中视作“确认”
    sample_gt : 可传入 data/<ds>/data.json 中该样本的 dict（用于参数 GT 对齐）
    real_call : 可选 bool；若给定则临时覆盖 adapter.real_exec 进行真实/模拟执行
    返回:
      toollm_rows: 每步诊断列表（含参数/执行/指标等）
      exec_flags : 每步执行成功布尔
    """
    # ---- 内部小工具 ----
    def _canon(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(s).lower())

    def _param_prf_by_required(schema: dict, pred: dict):
        """只基于 schema.required 的键做 PRF（不比较取值）。"""
        req = []
        for p in (schema.get("parameters") or []):
            if isinstance(p, dict) and p.get("required") and p.get("name"):
                req.append(str(p["name"]))
        R = set(req)
        P = set(k for k in (pred or {}).keys())
        tp = len(P & R)
        prec = tp / max(1, len(P))
        rec  = tp / max(1, len(R)) if R else 1.0
        f1   = 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))
        return prec, rec, f1, R

    def _param_prf_vs_gt(pred: dict, gt: dict):
        """与 GT 做键+值的精确匹配（数值宽松同值、布尔大小写宽松、字符串去空白大小写）。"""
        if not isinstance(pred, dict): pred = {}
        if not isinstance(gt, dict):   gt = {}
        P, G = set(pred.keys()), set(gt.keys())
        tp = 0
        for k in (P & G):
            a, b = pred.get(k), gt.get(k)
            eq = False
            try:
                # 数值等价
                if isinstance(a, (int, float)) or isinstance(b, (int, float)):
                    eq = float(a) == float(b)
                else:
                    sa = str(a).strip()
                    sb = str(b).strip()
                    # 布尔等价或不区分大小写的字符串等价
                    if sa.lower() in {"true", "false"} or sb.lower() in {"true", "false"}:
                        eq = (sa.lower() == sb.lower())
                    else:
                        eq = (sa.lower() == sb.lower())
            except Exception:
                eq = False
            if eq:
                tp += 1
        prec = tp / max(1, len(P))
        rec  = tp / max(1, len(G)) if G else 1.0
        f1   = 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))
        return prec, rec, f1

    if adapter is None or not sol_ids:
        return [], []

    # ---- 准备 GT（按 step 索引） ----
    gt_map = {}
    if isinstance(sample_gt, dict):
        g_nodes = sample_gt.get("task_nodes") or []
        for i, nd in enumerate(g_nodes):
            if not isinstance(nd, dict):
                continue
            tid = nd.get("task")
            args = nd.get("arguments") or {}
            args_dict = {}
            if isinstance(args, list):
                for a in args:
                    if isinstance(a, dict) and "name" in a:
                        args_dict[a["name"]] = a.get("value")
            elif isinstance(args, dict):
                args_dict = dict(args)
            gt_map[i] = {"tool": tid, "args": args_dict}

    toollm_rows, exec_flags = [], []
    # [M4 Advanced] 维护执行历史，供 adapter.fill_arguments 做上下文感知
    execution_history = []

    for i, tid in enumerate(sol_ids):
        step_text = _as_step_text(steps[i]) if i < len(steps) else ""

        # 1) 工具召回与确认
        retrieved = adapter.retrieve_tools(step_text, topk=getattr(adapter, "_topk", 5))
        confirm = any(_canon(rt.tool_id) == _canon(tid) for rt in (retrieved[:max(1, int(confirm_top))] or []))

        # 2) Schema & 参数拟合 (Context-Aware)
        schema = adapter.get_schema(tid)
        # 传入 execution_history，让 LLM 能够利用上一步的 Output (如 ID)
        param_pred = adapter.fill_arguments(step_text, schema, history=execution_history) if schema else {}
        param_draft = dict(param_pred)  # 兼容旧字段

        # 3) 执行 (支持 real_call 覆盖 + Auto-Retry)
        exec_res = None
        if schema:
            # 确定当前是否需要真实执行
            is_real = bool(real_call) if isinstance(real_call, bool) else getattr(adapter, "real_exec", False)
            
            # 临时切换模式上下文
            prev_mode = getattr(adapter, "real_exec", False)
            if is_real != prev_mode:
                setattr(adapter, "real_exec", is_real)
            
            try:
                # 初次执行
                exec_res = adapter.execute(tid, param_pred or {})

                # [M4 Advanced] 自动重试逻辑 (Self-Correction)
                # 仅在真实执行且失败，且错误看似是参数问题 (如 Invalid id/404) 时触发
                if is_real and (not exec_res.success):
                    err_str = str(exec_res.error or "")
                    # TMDB 常见错误特征
                    if "Invalid id" in err_str or "404" in str(exec_res.output or ""):
                        # 构造包含错误反馈的临时历史
                        temp_hist = execution_history + [{
                            "step_idx": i, 
                            "tool": tid, 
                            "output": f"Execution FAILED: {err_str}. Please check if the ID/Parameters are correct based on previous steps."
                        }]
                        # 重新向 LLM 请求参数
                        print(f"[Auto-Retry] Step {i} ({tid}) failed. Retrying with error context...", flush=True)
                        param_retry = adapter.fill_arguments(step_text, schema, history=temp_hist)
                        
                        # 重新执行
                        retry_res = adapter.execute(tid, param_retry or {})
                        
                        # 若重试成功，采纳新结果
                        if retry_res.success:
                            exec_res = retry_res
                            param_pred = param_retry
                            param_draft["_retry"] = "success"

            finally:
                # 恢复模式
                if is_real != prev_mode:
                    setattr(adapter, "real_exec", prev_mode)

        # 记录执行结果到 History (仅成功或模拟时记录，避免污染 Prompt，除非为了 debug)
        # 这里策略是：无论成败都记，但在 Prompt 构建时可能只取成功的 Output，或者由 _build_arg_prompt 决定
        output_data = exec_res.output if exec_res and exec_res.success else {}
        raw_data = output_data.get("data") if isinstance(output_data, dict) and "data" in output_data else output_data
        
        # 针对 TMDB 的特殊清洗逻辑：只保留 id 和 name/title
        refined_output = raw_data
        if isinstance(raw_data, dict) and "results" in raw_data and isinstance(raw_data["results"], list):
            # 提取前 3 个结果的精简信息
            short_list = []
            for item in raw_data["results"][:3]:
                simple = {}
                for k in ["id", "name", "title", "original_title"]:
                    if k in item:
                        simple[k] = item[k]
                short_list.append(simple)
            refined_output = str(short_list)  # 转为字符串存储
        elif isinstance(raw_data, dict):
             # 单个对象，只留 id/name
             simple = {}
             for k in ["id", "name", "title"]:
                 if k in raw_data:
                     simple[k] = raw_data[k]
             if simple:
                 refined_output = str(simple)
             else:
                 refined_output = str(raw_data)[:200] # 兜底

        execution_history.append({
            "step_idx": i,
            "tool": tid,
            "output": refined_output  # <--- 存入精简后的信息
        })

        # 4) 指标 A：按 required 键做 PRF（不看取值）
        p_prec, p_rec, p_f1, _ = _param_prf_by_required(schema or {}, param_pred or {})

        # 5) 指标 B：若 sample_gt 提供且工具匹配，则与 GT 计算键值 PRF，并回写 param_gt
        param_gt = None
        pg_prec = pg_rec = pg_f1 = None
        gt_entry = gt_map.get(i)
        if gt_entry:
            # 若 GT 的工具标注与当前工具一致，才认为可以做值对齐
            if (gt_entry.get("tool") is None) or (_canon(gt_entry.get("tool")) == _canon(tid)):
                param_gt = gt_entry.get("args") or {}
                pg_prec, pg_rec, pg_f1 = _param_prf_vs_gt(param_pred or {}, param_gt or {})

        row = {
            "step_idx": i,
            "tool": tid,
            "step_text": step_text,
            "tool_retrieval": [{"id": rt.tool_id, "score": round(rt.score, 4)} for rt in (retrieved or [])],
            "tool_confirm": bool(confirm),

            # 参数产出
            "param_draft": param_draft,
            "param_pred": param_pred,

            # 执行结果
            "exec_success": (exec_res.success if exec_res else None),
            "exec_output_sha1": (exec_res.output.get("digest") if (exec_res and exec_res.output and isinstance(exec_res.output, dict)) else None),
            "exec_error": (exec_res.error if exec_res else None),
            # 真实调用的补充记录（若为模拟，这两项通常为空）
            "exec_url": (exec_res.output.get("url") if (exec_res and exec_res.output and isinstance(exec_res.output, dict)) else None),
            "exec_status": (exec_res.output.get("status") if (exec_res and exec_res.output and isinstance(exec_res.output, dict)) else None),

            # 参数质量（required 键）
            "param_precision": round(p_prec, 4),
            "param_recall": round(p_rec, 4),
            "param_f1": round(p_f1, 4),
        }

        # 若存在 GT，则记录以便 eval_aggregate 计算 Param-F1（键+值）
        if param_gt is not None:
            row["param_gt"] = param_gt
            if pg_prec is not None:
                row["param_precision_gt"] = round(pg_prec, 4)
            if pg_rec is not None:
                row["param_recall_gt"] = round(pg_rec, 4)
            if pg_f1 is not None:
                # 方便直接读取，也便于与 required_f1 区分
                row["param_f1_gt"] = round(pg_f1, 4)

        toollm_rows.append(row)
        exec_flags.append(row["exec_success"])

    return toollm_rows, exec_flags

# ============= One Case =============
def graph_search_one_case(input, llm, url, tmp_print=False, maximum_solutions=20,
                          api_key=None, temperature=0.2, top_p=0.1,
                          rag_mode_opt=None, rag_k_nodes=None, rag_k_comms=None,
                          rag_use_comms=False, rag_min_score_nodes=35,
                          adapter=None, confirm_top=3):
    user_request = input["user_request"]
    inferred_steps = reformat_steps(input)

    if tmp_print:
        print(f"# User Request {input['id']} #\n", user_request)
    solutions, current_sol = [], []
    case_meta = []

    dfs(llm, url, 0, inferred_steps, solutions, current_sol, tmp_print,
        api_key=api_key, temperature=temperature, top_p=top_p,
        case_meta=case_meta, rag_mode_opt=rag_mode_opt,
        rag_k_nodes=rag_k_nodes, rag_k_comms=rag_k_comms,
        rag_use_comms=rag_use_comms, rag_min_score_nodes=rag_min_score_nodes)

    if len(solutions) == 0:
        return None, case_meta, []

    solutions = solutions[:maximum_solutions]
    solution_resp = prompt_llm_final_solutions(
        llm, url, user_request, inferred_steps, solutions, tmp_print,
        api_key=api_key, temperature=temperature, top_p=top_p
    )

    # ===== M4: 对最佳方案做 ToolLLM 复核 & 参数填充 & 执行模拟 =====
    toollm_rows = []
    if solution_resp and len(solution_resp.get("best_solution", [])) > 0 and adapter is not None:
        sol = solution_resp["best_solution"]
        if isinstance(sol[0], list):  # 容错外层嵌套
            sol = sol[0]
        # real_call=None → 若设置了 TMDB_BEARER_TOKEN 自动真实调用
        toollm_rows, _ = _toollm_post_steps(
            adapter, inferred_steps, sol, confirm_top=confirm_top,
            sample_gt=input, real_call=None
        )

    return solution_resp, case_meta, toollm_rows

# ============== CLI ==============
@click.command()
@click.option("--dataset", default="huggingface", help="The directory of the data")
@click.option("--api_addr", type=str, default="localhost")
@click.option("--api_port", type=int, default=8008)
@click.option("--llm", type=str, default="CodeLlama-13b-Instruct-hf")
@click.option("--strategy", type=str, default="beam")
@click.option("--width", type=int, default=2)
@click.option("--threshold", type=int, default=3)
@click.option("--mode", type=str, default="single")
@click.option("--request_id", type=str, default="")
@click.option("--temperature", type=float, default=0.2)
@click.option("--top_p", type=float, default=0.1)
@click.option("--use_graphrag", type=int, default=0)
@click.option("--rag_mode", type=str, default="auto")
@click.option("--rag_topk_nodes", type=int, default=8)
@click.option("--rag_topk_comms", type=int, default=3)
@click.option("--overwrite", type=int, default=0)
@click.option("--run_tag", type=str, default="")
@click.option("--rag_use_communities", type=int, default=0)   # 0=忽略社区成员，只用 nodes
@click.option("--rag_min_score_nodes", type=int, default=35)  # 仅保留得分≥阈值的 RAG nodes
# ===== M5: 触发式 RAG 相关 =====
@click.option("--rag_triggered", type=int, default=0,
              help="1: 触发式 RAG（仅在低置信/候选少时调用）；0: 只要 use_graphrag=1 即常开 RAG")
@click.option("--gs_conf_th", type=int, default=0,
              help="触发式 RAG 的候选数阈值（<=0 表示不使用候选数触发）")
# ===== M4: ToolLLM 相关 =====
@click.option("--use_toolllm", type=int, default=0)
@click.option("--tool_schema", type=str, default=None)
@click.option("--toolllm_k", type=int, default=5)
@click.option("--confirm_top", type=int, default=3)
@click.option("--exec_backwrite", type=float, default=0.0)  # >0 写回执行反馈
def main(dataset, api_addr, api_port, llm, strategy, width, threshold, mode, request_id,
         temperature, top_p, use_graphrag, rag_mode, rag_topk_nodes, rag_topk_comms,
         rag_use_communities, rag_min_score_nodes, rag_triggered, gs_conf_th,
         overwrite, run_tag,
         use_toolllm, tool_schema, toolllm_k, confirm_top, exec_backwrite):
    global tool_graph, tool_nodes, tool_dict, tool_name_list
    global search_strategy, beam_width, score_threshold
    global dataset_name, counter
    global rag_adapter, entity_linker, use_graphrag_flag, local_alias_idx
    global rag_triggered_flag, gs_conf_thres  # M5: 触发式 RAG
    global tok_prompt_cnt, tok_resp_cnt       # M5: Token 统计（在 full 模式里会重置）

    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)

    search_strategy = strategy
    beam_width = width
    score_threshold = threshold
    dataset_name = dataset

    # M5: 触发式 RAG 配置
    rag_triggered_flag = bool(rag_triggered)
    try:
        gs_conf_thres = int(gs_conf_th)
    except Exception:
        gs_conf_thres = 0

    # ---- LLM HTTP 入口 ----
    base_url = os.getenv("OPENAI_BASE_URL", None)
    if base_url:
        url = base_url.rstrip("/") + "/chat/completions"
    else:
        url = f"http://{api_addr}:{api_port}/v1/chat/completions"
    api_key = os.getenv("OPENAI_API_KEY", None)

    llm_short_names = {
        "CodeLlama-13b-Instruct-hf": "CodeLlama-13b",
        "vicuna-13b-v1.5": "vicuna-13b",
        "Mistral-7B-Instruct-v0.2": "mistral-7b",
        "CodeLlama-7b-Instruct-hf": "CodeLlama-7b",
        "Baichuan2-13B-Chat": "Baichuan-13b"
    }
    llm_short = llm_short_names.get(llm, llm)

    # ---- 读取 direct.json 作为起始计划 ----
    pred_filename = ROOT / "prediction" / dataset / llm_short / "direct.json"
    if not pred_filename.exists():
        raise Exception(f"Prediction file does not exsists! ({pred_filename})")

    pred_data = []
    with open(pred_filename, "r", encoding="utf-8") as pred_rf:
        for line in pred_rf:
            pred_data.append(json.loads(line))

    # ---- 工具图加载 ----
    graph_data_path = ROOT / "data" / dataset / "graph_desc.json"
    graph_data = json.load(open(graph_data_path, "r", encoding="utf-8"))
    tool_nodes, tool_links = graph_data["nodes"], graph_data["links"]

    tool_graph, tool_dict = {}, {}
    tool_name_list.clear()
    for tool in tool_nodes:
        if 'input-type' in tool.keys() and 'output-type' in tool.keys():
            tool.pop('input-type')
            tool.pop('output-type')
        tool_graph[tool["id"]] = {"id": tool["id"], "children": []}
        tool_dict[tool["id"]] = copy.deepcopy(tool)
        tool_name_list.append(tool["id"])
    for link in tool_links:
        neighbor_tool = tool_dict[link["target"]]
        tool_graph[link["source"]]["children"].append(neighbor_tool)

    # ---- GraphRAG 初始化 ----
    use_graphrag_flag = bool(use_graphrag)
    rag_index_dir = ROOT / "data" / dataset / "graphrag"
    rag_adapter = None
    entity_linker = None
    local_alias_idx = _build_local_alias_index(tool_nodes)

    def _init_graphrag_adapter(rag_dir: Path):
        try:
            adapter = GraphRAGAdapter(index_dir=str(rag_dir))
        except TypeError:
            adapter = GraphRAGAdapter(str(rag_dir))
        # 兼容老版本接口名：query -> search
        if hasattr(adapter, "query") and not hasattr(adapter, "search"):
            adapter.search = adapter.query
        return adapter

    def _init_entity_linker(e2t_path: Path, tools_list):
        try:
            linker = EntityLinker(mapping_path=str(e2t_path), tools=tools_list)
        except TypeError:
            try:
                linker = EntityLinker(str(e2t_path), tools_list)
            except TypeError:
                linker = EntityLinker(str(e2t_path))
        return linker

    if use_graphrag_flag:
        try:
            rag_adapter = _init_graphrag_adapter(rag_index_dir)
        except Exception as e:
            print(f"[WARN] init GraphRAGAdapter failed: {e}")
            use_graphrag_flag = False
        try:
            e2t = rag_index_dir / "entity2tool.json"
            entity_linker = _init_entity_linker(e2t, list(tool_dict.keys()))
        except Exception as e:
            print(f"[WARN] init EntityLinker failed: {e}")
            entity_linker = None

    # ===== M4: ToolLLM 初始化 =====
    adapter = None
    if int(use_toolllm) == 1:
        if ToolLLMAdapter is None:
            raise RuntimeError("ToolLLMAdapter not found. Please put retrieval/toollm_adapter.py in repo.")
        # 默认 schema 路径：data/<dataset>/tool_schema.json；也可以 CLI 显式传入 tmdb_oas.json
        if tool_schema is None:
            tool_schema = str(ROOT / "data" / dataset / "tool_schema.json")
        if not Path(tool_schema).exists():
            raise FileNotFoundError(f"--tool_schema not found: {tool_schema}")
        # —— 补丁 C：让适配器沿用同一 LLM 入口（和 graphsearch 一致）——
        adapter = ToolLLMAdapter(
            tool_schema,
            seed=0,
            api_url=url,
            api_key=api_key,
            model=llm
        )
        setattr(adapter, "_topk", int(toolllm_k))

    # ---- 统一日志文件路径 ----
    suffix = f"_{width}" if strategy == "beam" else ""
    if run_tag:
        suffix += f"_{run_tag}"
    if use_graphrag_flag:
        log_dir = ROOT / "runs" / dataset / "rag"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_fp = open_jsonl(str(log_dir / f"graphsearch_{strategy}{suffix}_{rag_mode}.jsonl"))
    else:
        log_dir = ROOT / "runs" / dataset / "base"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_fp = open_jsonl(str(log_dir / f"graphsearch_{strategy}{suffix}.jsonl"))

    # 可选：执行结果回写（M4 用于后续采样器调权重）
    feedback_fp = None
    if float(exec_backwrite) > 0.0:
        fb_dir = ROOT / "runs" / dataset
        fb_dir.mkdir(parents=True, exist_ok=True)
        feedback_fp = open(fb_dir / "toollm_exec_feedback.jsonl", "a", encoding="utf-8")
        exec_gamma = float(exec_backwrite)

    # ========= 单条调试模式 =========
    if mode == "single":
        # 取单条样本；匹配失败就提示一下
        # 强制转为 string 比较以兼容整数 ID
        item = next((d for d in pred_data if str(d["id"]) == str(request_id)), None)
        
        if not item:
            print(f"[WARN] request_id={request_id} not found in {pred_filename}", flush=True)
        else:
            counter = 0
            tok_prompt_cnt = 0
            tok_resp_cnt = 0
            
            start_time = time.time()
            result, case_meta, toollm_rows = graph_search_one_case(
                item,  # 单个样本 dict
                llm, url, tmp_print=True,
                api_key=api_key, temperature=temperature, top_p=top_p,
                rag_mode_opt=rag_mode,
                rag_k_nodes=rag_topk_nodes,
                rag_k_comms=rag_topk_comms,
                rag_use_comms=bool(rag_use_communities),
                rag_min_score_nodes=rag_min_score_nodes,
                adapter=adapter, confirm_top=int(confirm_top)
            )
            elapsed = round(time.time() - start_time, 4)

            # >>> 新增：组装结果并写入文件 >>>
            task_nodes, task_links = item["task_nodes"], item["task_links"]
            status = "fail"
            if result and len(result.get("best_solution", [])) > 0:
                sol = result["best_solution"]
                if isinstance(sol[0], list): sol = sol[0]
                task_nodes = [{"task": node} for node in sol]
                task_links = [{"source": node, "target": sol[i+1]} for i, node in enumerate(sol[:-1])]
                status = "succ"

            out = {
                "id": item["id"],
                "task_steps": item["task_steps"],
                "task_nodes": task_nodes,
                "task_links": task_links,
                "cost_time": elapsed,
                "llm_query_times": counter,
                "status": status
            }

            # 注入 ToolLLM 结果
            if adapter is not None and toollm_rows:
                out["toollm"] = toollm_rows
                exec_list = [r.get("exec_success") for r in toollm_rows if r is not None]
                if exec_list:
                    out["tool_exec_consistency"] = round(
                        sum(1.0 if v else 0.0 for v in exec_list) / len(exec_list), 4
                    )
                fvals = [r.get("param_f1") for r in toollm_rows if r and isinstance(r.get("param_f1"), (int, float))]
                if fvals:
                    out["param_f1_mean"] = round(sum(fvals) / len(fvals), 4)

            # 确定写入路径
            write_suffix = f"_{rag_mode}" if use_graphrag_flag else ""
            write_filename = ROOT / "prediction" / dataset / llm_short / f"graphsearch_{strategy}{suffix}{write_suffix}.json"
            write_filename.parent.mkdir(parents=True, exist_ok=True)

            # 写入文件（注意：overwrite=1 会覆盖整个文件，仅用于单例调试非常合适）
            open_mode = "w" if overwrite else "a"
            with open(write_filename, open_mode, encoding="utf-8") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
            
            print(f"\n[INFO] Result written to: {write_filename}")
            # <<< 修改结束 <<<

    # ========= 全量评测模式 =========
    elif mode == "full":
        write_suffix = f"_{rag_mode}" if use_graphrag_flag else ""
        write_filename = ROOT / "prediction" / dataset / llm_short / f"graphsearch_{strategy}{suffix}{write_suffix}.json"
        write_filename.parent.mkdir(parents=True, exist_ok=True)
        write_file = open(write_filename, "w" if overwrite else "a", encoding="utf-8")

        has_inferenced = set()
        if (not overwrite) and write_filename.exists():
            with open(write_filename, "r", encoding="utf-8") as rf:
                for line in rf:
                    try:
                        has_inferenced.add(json.loads(line)["id"])
                    except Exception:
                        pass

        split_ids_path = ROOT / "data" / dataset / "split_ids.json"
        # —— 补丁 E：split_ids 缺失时回退到跑满 direct.json —— 
        if split_ids_path.exists():
            alignment_ids = json.load(open(split_ids_path, 'r', encoding="utf-8"))["test_ids"]["chain"]
            def _use_this_id(x): return x in alignment_ids
        else:
            print(f"[WARN] {split_ids_path} not found; fallback to ALL ids in direct.json", flush=True)
            alignment_ids = None
            def _use_this_id(x): return True

        for input_data in pred_data:
            if input_data["id"] in has_inferenced or (alignment_ids is not None and not _use_this_id(input_data["id"])):
                continue

            counter = 0
            tok_prompt_cnt = 0
            tok_resp_cnt = 0

            start = time.time()
            result, case_meta, toollm_rows = graph_search_one_case(
                input_data, llm, url, tmp_print=False,
                api_key=api_key, temperature=temperature, top_p=top_p,
                rag_mode_opt=rag_mode,
                rag_k_nodes=rag_topk_nodes, rag_k_comms=rag_topk_comms,
                rag_use_comms=bool(rag_use_communities),
                rag_min_score_nodes=rag_min_score_nodes,
                adapter=adapter, confirm_top=int(confirm_top)
            )

            task_nodes, task_links = input_data["task_nodes"], input_data["task_links"]
            status = "fail"
            if result and len(result.get("best_solution", [])) > 0:
                sol = result["best_solution"]
                if isinstance(sol[0], list):
                    sol = sol[0]
                task_nodes = [{"task": node} for node in sol]
                task_links = [{"source": node, "target": sol[i+1]} for i, node in enumerate(sol[:-1])]
                status = "succ"

            elapsed = round(time.time() - start, 4)

            out = {
                "id": input_data["id"],
                "task_steps": input_data["task_steps"],
                "task_nodes": task_nodes,
                "task_links": task_links,
                "cost_time": elapsed,
                "llm_query_times": counter,
                "status": status
            }

            # ===== M4: 写入 ToolLLM 诊断 & 样本级指标 =====
            if adapter is not None and toollm_rows:
                out["toollm"] = toollm_rows
                exec_list = [r.get("exec_success") for r in toollm_rows if r is not None]
                if exec_list:
                    out["tool_exec_consistency"] = round(
                        sum(1.0 if v else 0.0 for v in exec_list) / len(exec_list), 4
                    )
                # —— 补丁 D：Param-F1 样本级聚合 —— 
                fvals = [r.get("param_f1") for r in toollm_rows if r and isinstance(r.get("param_f1"), (int, float))]
                if fvals:
                    out["param_f1_mean"] = round(sum(fvals) / len(fvals), 4)

            # ===== GraphRAG 统计信息 =====
            if use_graphrag_flag:
                if case_meta:
                    cg = sum(m["cand_graph"] for m in case_meta) / len(case_meta)
                    cr = sum(m["cand_rag"] for m in case_meta) / len(case_meta)
                    co = sum(m["cand_overlap"] for m in case_meta) / len(case_meta)
                    cu = sum(m["cand_union"] for m in case_meta) / len(case_meta)
                    trig_cnt = sum(1 for m in case_meta if m.get("rag_triggered"))
                    trig_ratio = trig_cnt / len(case_meta)
                else:
                    cg = cr = co = cu = 0.0
                    trig_cnt = 0
                    trig_ratio = 0.0
                out.update({
                    "rag_used": True,
                    "rag_mode": rag_mode,
                    "cand_src_graph_avg": round(cg, 3),
                    "cand_src_rag_avg": round(cr, 3),
                    "cand_src_overlap_avg": round(co, 3),
                    "cand_src_union_avg": round(cu, 3),
                    "rag_triggered_steps": int(trig_cnt),
                    "rag_total_steps": int(len(case_meta) if case_meta else 0),
                    "rag_triggered_ratio": round(trig_ratio, 3),
                })
                # 可选 debug：设置环境变量 RAG_DEBUG=1 开启
                if os.getenv("RAG_DEBUG", "0") == "1" and case_meta:
                    out["rag_debug_steps"] = case_meta[:3]  # 只写前3步以免太长

            write_file.write(json.dumps(out, ensure_ascii=False) + "\n")
            write_file.flush()

            # ===== 统一 JSONL 日志（PSR / Token / Latency 等聚合用）=====
            extra_err = None if status == "succ" else f"search_failed;llm_queries={counter}"
            if use_graphrag_flag:
                rag_note = (
                    f"rag_used=1;mode={rag_mode};"
                    f"cand_graph_avg={cg:.3f};cand_rag_avg={cr:.3f};"
                    f"cand_overlap_avg={co:.3f};cand_union_avg={cu:.3f};"
                    f"rag_trig_steps={trig_cnt};rag_trig_ratio={trig_ratio:.3f}"
                )
                extra_err = (extra_err + ";" if extra_err else "") + rag_note

            log_sample(
                fp=log_fp, ds=dataset, method="graphsearch", llm=llm_short, seed=0,
                sid=input_data["id"], success=(status == "succ"), latency_sec=elapsed,
                tokens_prompt=tok_prompt_cnt, tokens_completion=tok_resp_cnt,  # M5: 写入 Token 数
                temperature=temperature, top_p=top_p, alpha=None, graphsearch=strategy,
                task_steps=input_data["task_steps"], task_nodes=task_nodes, task_links=task_links,
                err=extra_err
            )

            # ===== 可选：执行成败回写（写文件供训练采样器使用） =====
            if feedback_fp is not None and adapter is not None and toollm_rows:
                for r in toollm_rows:
                    succ = r.get("exec_success")
                    if succ is None:
                        continue
                    # 简单规则：exec_gamma>0 时，成功略减权，失败略增权，供采样器重新加权负例/难例
                    mul = (1.0 - 0.5 * exec_gamma) if succ else (1.0 + exec_gamma)
                    feedback = {
                        "case_id": input_data["id"],
                        "step_idx": r.get("step_idx"),
                        "tool_id": r.get("tool"),
                        "exec_success": bool(succ),
                        "weight_mul": round(mul, 4)
                    }
                    feedback_fp.write(json.dumps(feedback, ensure_ascii=False) + "\n")
                feedback_fp.flush()

        write_file.close()

    # ---- 资源清理 ----
    if feedback_fp is not None:
        feedback_fp.close()

    log_fp.close()
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")

if __name__ == "__main__":
    tool_graph, tool_nodes, tool_dict, tool_name_list = None, None, {}, []
    beam_width, search_strategy, score_threshold = 2, "beam", 3
    dataset_name = "huggingface"
    counter = 0
    tok_prompt_cnt = 0
    tok_resp_cnt = 0
    rag_adapter, entity_linker = None, None
    use_graphrag_flag = False
    local_alias_idx = {}
    rag_triggered_flag = False
    gs_conf_thres = 0
    main()