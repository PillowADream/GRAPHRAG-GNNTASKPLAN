# retrieval/toollm_adapter.py
from __future__ import annotations

import json
import os
import re
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import urllib.parse


# ----------------- repo root helper -----------------
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if (p / "data").exists():
            return p
    return here.parent


# ----------------- dataclasses -----------------
@dataclass
class RetrievedTool:
    """Single tool retrieval result."""
    tool_id: str
    score: float
    reason: Optional[str] = None


@dataclass
class ExecResult:
    """Pseudo/real execution result."""
    success: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# ----------------- adapter -----------------
class ToolLLMAdapter:
    """
    Lightweight ToolLLM adapter:
    - Load tool schema from OpenAPI (OAS, supports $ref) or tool_desc.json (nodes).
    - Retrieve top-k tools by simple lexical/keyword scoring.
    - Fill arguments via LLM (OpenAI-compatible endpoint), fallback to regex heuristics.
    - Execute:
        * simulate (required-params & coarse types check), default
        * real TMDB call when real execution is enabled
          (Bearer v4 preferred; v3 api_key fallback)
    - Back-compat aliases: retrieve/fill/simulate/choose.
    """

    def __init__(
        self,
        tool_schema_path: Optional[str] = None,
        schema_path: Optional[str] = None,
        *,
        seed: int = 0,
        # LLM endpoint (OpenAI-compatible) for argument filling
        api_url: Optional[str] = None,       # e.g., http://localhost:11434/v1/chat/completions
        api_key: Optional[str] = None,       # e.g., OPENAI_API_KEY
        model: Optional[str] = None,         # e.g., "gpt-4o" or local name
        llm: Optional[str] = None,           # alias of model
        api_addr: Optional[str] = None,      # if provided together with api_port, builds api_url
        api_port: Optional[int] = None,
        temperature: float = 0.0,
        top_p: float = 0.1,
        # retrieval behavior
        k: int = 5,
        confirm_top: int = 3,
        **kwargs,
    ):
        # ---------- random ----------
        random.seed(seed)

        # ---------- resolve schema path ----------
        path_arg = tool_schema_path or schema_path
        if not path_arg:
            # backward-compat: some callers pass named arg "schema_path" only
            raise ValueError("tool_schema_path or schema_path must be provided")
        p = Path(path_arg)
        if not p.is_absolute():
            p = (_repo_root() / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Tool schema not found: {p}")
        self.schema_path = str(p)

        # ---------- load schema and index tools ----------
        self.schema = self._load_schema(self.schema_path)
        self.tools: Dict[str, Dict[str, Any]] = self._index_tools(self.schema)
        self.tool_ids: List[str] = sorted(self.tools.keys())
        self._canon2opid: Dict[str, str] = {self._canon(op): op for op in self.tool_ids}

        # ---------- LLM endpoint (for argument filling) ----------
        # priority: explicit api_url -> api_addr/api_port -> OPENAI_BASE_URL -> default ollama
        if api_url:
            self.api_url = api_url.rstrip("/")
            if not self.api_url.endswith("/v1/chat/completions"):
                self.api_url = self.api_url.rstrip("/") + "/v1/chat/completions"
        elif api_addr and api_port:
            self.api_url = f"http://{api_addr}:{int(api_port)}/v1/chat/completions"
        else:
            base = os.getenv("OPENAI_BASE_URL")
            if base:
                self.api_url = base.rstrip("/")
                if not self.api_url.endswith("/v1/chat/completions"):
                    self.api_url = self.api_url.rstrip("/") + "/v1/chat/completions"
            else:
                # default to local Ollama style
                self.api_url = "http://localhost:11434/v1/chat/completions"

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or llm or os.getenv("TOOLLLM_MODEL") or os.getenv("OPENAI_MODEL") or "CodeLlama-13b-Instruct-hf"
        self.temperature = float(temperature)
        self.top_p = float(top_p)

        # retrieval behavior
        self._topk: int = int(k) if k else 5
        self.confirm_top: int = int(confirm_top) if confirm_top else 3

        # ---------- Real TMDB execution config ----------
        # v4: Authorization: Bearer <token>
        # v3: ?api_key=xxx
        self.tmdb_base = os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3")
        self.tmdb_api_key = os.getenv("TMDB_API_KEY")          # optional
        self.tmdb_bearer  = os.getenv("TMDB_BEARER_TOKEN")     # optional (preferred)

        # Real exec switch:
        # - If TOOLLLM_REAL_EXEC is set, honor it: 1/true/on => True, others => False
        # - If not set, but we do have bearer or api_key, enable real exec by default
        flag = os.getenv("TOOLLLM_REAL_EXEC", None)
        if flag is None or flag == "":
            self.real_exec = bool(self.tmdb_bearer or self.tmdb_api_key)
        else:
            self.real_exec = flag.strip().lower() in ("1", "true", "yes", "on")

    # ========= Schema =========
    @staticmethod
    def _canon(s: str) -> str:
        """大小写无关+去除非字母数字，便于宽松匹配 operationId。"""
        return re.sub(r"[^a-z0-9]", "", str(s).lower()) if s is not None else ""

    @staticmethod
    def _load_schema(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _get_by_ref(tree: Dict[str, Any], ref: str) -> Optional[Dict[str, Any]]:
        """Resolve a local $ref like '#/components/parameters/lang'."""
        if not ref or not ref.startswith("#/"):
            return None
        cur: Any = tree
        for key in ref[2:].split("/"):
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return cur if isinstance(cur, dict) else None

    def _index_tools(self, schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Build a dict: {operationId: {operationId, path, method, summary, description, parameters:[{name,required,schema,description,in}, ...]}}
        Supports:
          - tool_desc.json: {"nodes":[{"id","desc","parameters":[{"name","type","required"}]}]}
          - OAS: paths + components (parameters/requestBodies with $ref)
        """
        tools: Dict[str, Dict[str, Any]] = {}

        # Case A: repo-style tool_desc.json
        if isinstance(schema, dict) and isinstance(schema.get("nodes"), list):
            for node in schema["nodes"]:
                op_id = node.get("id") or node.get("name") or node.get("task") or "unknown_tool"
                params = []
                for p in node.get("parameters", []) or node.get("params", []) or []:
                    params.append({
                        "name": p.get("name"),
                        "in": "query",
                        "required": bool(p.get("required", False)),
                        "schema": {"type": p.get("type", "string")},
                        "description": p.get("desc", "") or p.get("description", ""),
                    })
                tools[op_id] = {
                    "operationId": op_id,
                    "path": None,
                    "method": None,
                    "summary": node.get("desc", "") or node.get("description", ""),
                    "description": node.get("desc", "") or node.get("description", ""),
                    "parameters": params,
                }
            if tools:
                return tools

        # Case B: OpenAPI / OAS 3.x
        if not isinstance(schema, dict) or "paths" not in schema:
            raise ValueError(f"Unsupported tool schema format: {self.schema_path}")

        def parse_param(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            # resolve $ref at param-level and schema-level
            if "$ref" in entry:
                ref_obj = self._get_by_ref(schema, entry["$ref"])
                if isinstance(ref_obj, dict):
                    entry = ref_obj
            pschema = entry.get("schema") or {}
            if "$ref" in pschema:
                pschema = self._get_by_ref(schema, pschema["$ref"]) or {}
            return {
                "name": entry.get("name"),
                "in": entry.get("in"),
                "required": bool(entry.get("required", False)),
                "schema": {"type": pschema.get("type", "string")},
                "description": entry.get("description", ""),
            }

        def parse_request_body(rb: Dict[str, Any]) -> List[Dict[str, Any]]:
            # resolve $ref on requestBody and schema
            if "$ref" in rb:
                rb = self._get_by_ref(schema, rb["$ref"]) or {}
            required_rb = bool(rb.get("required", False))
            results: List[Dict[str, Any]] = []
            content = rb.get("content") or {}
            app_json = content.get("application/json") or {}
            jschema = app_json.get("schema") or {}
            if "$ref" in jschema:
                jschema = self._get_by_ref(schema, jschema["$ref"]) or {}
            props = jschema.get("properties") or {}
            reqset = set(jschema.get("required") or [])
            for pname, pobj in props.items():
                ptype = pobj.get("type") or "string"
                results.append({
                    "name": pname,
                    "in": "body",
                    "required": (pname in reqset) or required_rb,
                    "schema": {"type": ptype},
                    "description": pobj.get("description", ""),
                })
            return results

        paths = schema.get("paths", {}) or {}
        for path, item in paths.items():
            if not isinstance(item, dict):
                continue
            for method, op in item.items():
                if method.lower() not in {"get", "post", "put", "delete", "patch"}:
                    continue
                if not isinstance(op, dict):
                    continue
                op_id = op.get("operationId") or f"{method.UPPER()} {path}"
                # safeguard: method might not have UPPER; use str
                if not op.get("operationId"):
                    op_id = f"{str(method).upper()} {path}"
                summary = op.get("summary") or ""
                description = op.get("description") or ""
                params: List[Dict[str, Any]] = []

                # parameters[]
                for p in op.get("parameters", []) or []:
                    parsed = parse_param(p)
                    if parsed and parsed.get("name"):
                        params.append(parsed)

                # requestBody
                rb = op.get("requestBody")
                if rb:
                    params.extend(parse_request_body(rb))

                tools[op_id] = {
                    "operationId": op_id,
                    "path": path,
                    "method": str(method).upper(),
                    "summary": summary,
                    "description": description,
                    "parameters": params,
                }

        return tools

    # ========= Retrieval =========
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)  # camelCase -> space
        text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
        return [t.lower() for t in text.split() if t]

    def _simple_rank(self, step: str, candidate_ids: Optional[List[str]] = None) -> List[RetrievedTool]:
        if candidate_ids is None:
            candidate_ids = self.tool_ids
        q_tok = set(self._tokenize(step))

        # keyword bonus
        kw_bonus = {
            "movie": 0.6, "tv": 0.6, "person": 0.6, "company": 0.3,
            "search": 0.8, "find": 0.6, "popular": 0.4, "top rated": 0.5, "trending": 0.4,
            "details": 0.5, "credits": 0.5, "recommend": 0.5, "similar": 0.5, "reviews": 0.4,
            "images": 0.3, "videos": 0.3, "language": 0.2, "year": 0.2, "page": 0.1,
        }

        results: List[RetrievedTool] = []
        for tid in candidate_ids:
            op = self.tools.get(tid)
            mapped_tid = tid
            if op is None:
                # fuzzy match on operationId
                for op_id, spec in self.tools.items():
                    if tid.lower() == op_id.lower() or tid.lower() in op_id.lower():
                        op = spec
                        mapped_tid = op_id
                        break
                if op is None:
                    continue

            text = (mapped_tid + " " + op.get("summary", "") + " " + op.get("description", "")).lower()
            t_tok = set(self._tokenize(text))

            # add param names to tokens
            for p in op.get("parameters", []) or []:
                nm = (p.get("name") or "").lower()
                if nm:
                    t_tok.add(nm)

            inter = q_tok & t_tok
            denom = 1 + len(q_tok)
            s = len(inter) / denom
            for kw, w in kw_bonus.items():
                if (kw in step.lower()) and (kw in text):
                    s += w
            results.append(RetrievedTool(tool_id=mapped_tid, score=float(s)))

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def retrieve_tools(self, step: str, candidate_tools: Optional[List[str]] = None, topk: int = 5) -> List[RetrievedTool]:
        ranked = self._simple_rank(step, candidate_tools)
        k = max(1, int(topk or self._topk or 1))
        return ranked[:k]

    # alias for compatibility with some callers
    def retrieve(self, step: str, k: Optional[int] = None) -> List[RetrievedTool]:
        return self.retrieve_tools(step, None, topk=k or self._topk)

    # ========= Schema query =========
    # ========= Schema query (Enhanced with Jaccard) =========
    def get_schema(self, tool_id: str) -> Optional[Dict[str, Any]]:
        """
        根据图里工具 id 找到 OpenAPI schema：
        1) exact match
        2) canon exact
        3) canon substring
        4) description substring
        5) description token jaccard (新增：容忍 'the', 's' 等差异)
        """
        if not tool_id:
            return None

        # 1) 完全一致
        if tool_id in self.tools:
            return self.tools[tool_id]

        cid = self._canon(tool_id)
        if not cid:
            return None

        # 2) 规范化后一致
        mapped = self._canon2opid.get(cid)
        if mapped and mapped in self.tools:
            return self.tools[mapped]

        # 3) & 4) 子串匹配 (保留之前的逻辑，作为高置信度匹配)
        best_substr = None
        best_len = 10 ** 9
        
        # 5) Jaccard 匹配候选 (新增)
        best_jacc = None
        max_score = -1.0  # 允许负分

        # 预处理 tool_id 的 tokens
        t_tokens = set(self._tokenize(tool_id))
        t_lower = tool_id.lower()

        for op_id, spec in self.tools.items():
            # 构建描述文本
            text_raw = " ".join([
                op_id or "",
                spec.get("summary") or "",
                spec.get("description") or "",
            ])
            
            # 基础分：Jaccard
            d_tokens = set(self._tokenize(text_raw))
            score = 0.0
            if d_tokens and t_tokens:
                score = len(t_tokens & d_tokens) / len(t_tokens | d_tokens)
            
            # >>> 核心修改：基于 Path 的强约束 >>>
            path = spec.get("path", "").lower()
            
            # 规则1: 如果工具名包含 "Movie" 但不含 "Person"，则 Path 必须含 "/movie" 且不含 "/person"
            if "movie" in t_lower and "person" not in t_lower:
                if "/person" in path: 
                    score -= 1.0  # 严重惩罚
                if "/movie" in path:
                    score += 0.2  # 奖励
            
            # 规则2: 如果工具名包含 "Person" 或 "People"，则 Path 必须含 "/person"
            if ("person" in t_lower or "people" in t_lower) and "movie" not in t_lower:
                if "/movie" in path:
                    score -= 1.0
                if "/person" in path:
                    score += 0.2
            
            # 规则3: "Credit" 专用消歧 (GetMovieCredit vs GetPersonMovieCredits)
            # 如果工具名是 GetMovieCredit (电影的演职员)，路径应为 /movie/{id}/credits
            if "credit" in t_lower:
                if "movie" in t_lower and "/movie" in path and "credits" in path:
                    score += 0.5
            # <<< 修改结束 <<<

            # 原有的 operationId 匹配奖励
            cid = self._canon(tool_id)
            if self._canon(op_id) in cid or cid in self._canon(op_id):
                score += 0.3

            if score > max_score:
                max_score = score
                best_jacc = spec

        # 优先返回子串匹配结果，如果没有，且 Jaccard 分数够高(>0.1)，返回 Jaccard 结果
        if best_substr:
            return best_substr
        if best_jacc and max_score > 0.1:
            return best_jacc
            
        return None

    # ========= Arguments =========
    def _build_arg_prompt(self, step: str, api_schema: Optional[Dict[str, Any]], history: List[Dict] = None) -> str:
        if not api_schema:
            return (
                "You are a helpful assistant. "
                "For the given user instruction, propose a JSON object of arguments for calling an API. "
                "If you cannot infer any arguments, return an empty JSON object {}."
            )
        lines = []
        for p in api_schema.get("parameters", []) or []:
            name = p.get("name"); loc = p.get("in"); req = bool(p.get("required", False))
            typ = (p.get("schema") or {}).get("type"); desc = p.get("description", "")
            lines.append(f"- {name} (in={loc}, required={req}, type={typ})" + (f": {desc}" if desc else ""))
        schema_text = "\n".join(lines) if lines else "No parameters."

        context_text = ""
        if history:
            context_text = "Previous Execution Results:\n"
            for i, h in enumerate(history):
                # 只截取前 200 字符避免 Prompt 过长
                out_str = str(h.get('output', ''))[:300]
                context_text += f"Step {i}: Tool='{h.get('tool')}' -> Output={out_str}\n"

        return (
            "You are a tool parameter extraction assistant.\n"
            "Your task is to extract parameters for the current step, potentially using IDs or data from previous results.\n\n"
            f"{context_text}\n"  # 注入历史信息
            f"Current Instruction: {step}\n\n"
            "Target Tool Parameters Schema:\n"
            f"{schema_text}\n\n"
            "Return ONLY a JSON object. Use values from 'Previous Execution Results' if they match the required IDs."
        )

    def _call_llm(self, prompt: str) -> Optional[str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a concise tool-use assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stream": False,
        }
        try:
            resp = requests.post(self.api_url, headers=headers, json=body, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception:
            return None

    @staticmethod
    def _heuristic_args(step: str, api_schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Regex-based fallback when LLM is unavailable."""
        text = step or ""

        def rx(pat, flags=0):
            m = re.search(pat, text, flags)
            return m.group(1) if m else None

        id_val = rx(r"\b(id|movie_id|tv_id|person_id)\s*[:=]\s*([0-9]+)") or rx(r"\b(\d{3,12})\b")
        name_val = rx(r"(?:movie|film|show|tv|person|actor|title|name)\s*[:=]\s*([A-Za-z0-9 :'\-_,.]+)")
        query_val = rx(r"(?:query|search)\s*[:=]\s*([A-Za-z0-9 :'\-_,.]+)") or name_val
        year_val = rx(r"(?:year|y)\s*[:=]\s*(\d{4})")
        lang_val = rx(r"(?:language|lang)\s*[:=]\s*([a-z]{2}(?:-[A-Z]{2})?)")
        page_val = rx(r"(?:page)\s*[:=]\s*(\d{1,3})")

        # fallback from quotes
        if not query_val:
            qm = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
            if qm:
                query_val = max([q[0] or q[1] for q in qm], key=len)

        # default map
        common = {
            "id": id_val, "movie_id": id_val, "tv_id": id_val, "person_id": id_val,
            "query": query_val, "name": name_val or query_val,
            "year": year_val, "primary_release_year": year_val, "first_air_date_year": year_val,
            "language": lang_val or "en-US", "page": page_val or "1", "region": None, "include_adult": None,
        }

        args: Dict[str, Any] = {}
        if api_schema:
            for p in api_schema.get("parameters", []) or []:
                key = p.get("name")
                if not key:
                    continue
                val = common.get(key)
                if val is None:
                    if key in {"title", "query", "name"}:
                        val = query_val or name_val
                    elif key in {"lang", "language"}:
                        val = lang_val or "en-US"
                    elif key == "page":
                        val = page_val or "1"
                    elif key.endswith("_id") and id_val:
                        val = id_val
                if val is not None:
                    args[key] = str(val)

        # prune empties
        for k in list(args.keys()):
            if args[k] is None or str(args[k]).strip() == "":
                args.pop(k, None)
        return args

    def fill_arguments(self, step: str, api_schema: Optional[Dict[str, Any]] = None, history: List[Dict] = None) -> Dict[str, Any]:
        """
        使用 LLM + schema 生成参数；若 LLM 不可用或产出非法，则：
        - 启发式从 step 文本抽取参数；
        - 按 schema 做类型强制与默认值补齐（常见 TMDB 参数）。
        """
        # 1) 先试 LLM
        args: Dict[str, Any] = {}
        try:
            prompt = self._build_arg_prompt(step, api_schema, history)
            raw = self._call_llm(prompt)
            if raw:
                m = re.search(r"\{.*\}", raw, flags=re.S)
                text = m.group(0) if m else raw
                obj = json.loads(text)
                if isinstance(obj, dict):
                    args = obj
        except Exception:
            args = {}

        # 2) 启发式兜底（仅补齐缺失键）
        heur = self._heuristic_from_step(step, api_schema)
        for k, v in (heur or {}).items():
            if k not in args:
                args[k] = v

        # 3) 类型强制 + 常见 TMDB 默认值
        args = self._coerce_by_schema(args, api_schema)
        if api_schema:
            names = { (p.get("name") or "") for p in (api_schema.get("parameters") or []) if isinstance(p, dict) }
            # 语言默认值
            if "language" in names and ("language" not in args or not args["language"]):
                args["language"] = "en-US"
            # 翻页默认值
            if "page" in names and ("page" not in args or args["page"] in ("", None)):
                args["page"] = 1
            # 成人内容默认开关
            if "include_adult" in names and ("include_adult" not in args or args["include_adult"] in ("", None)):
                args["include_adult"] = False
        return args

    # --------- 启发式解析 & 类型强制 ---------
    def _heuristic_from_step(self, step: str, api_schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        从 'Step k: Call xxx API with name: "Inception", page: 2 ...' 这类文本里抽参数。
        仅抽取 schema 中出现的参数名，避免过拟合。
        """
        out: Dict[str, Any] = {}
        if not step or not api_schema:
            return out
        params = [p for p in (api_schema.get("parameters") or []) if isinstance(p, dict)]
        text = str(step)
        text_lc = text.lower()

        def _synonyms(name: str):
            name = str(name or "").lower()
            syn = {
                "query": ["name", "title", "keyword", "keywords", "q", "term", "search"],
                "movie_id": ["id", "movieid"],
                "tv_id": ["id", "tvid"],
                "person_id": ["id", "personid"],
                "include_adult": ["adult"],
            }
            return [name] + syn.get(name, [])

        # 通用三类模式：name: "xxx" / name: 'xxx' / name: 123 / name: true|false
        for p in params:
            name = p.get("name")
            if not name:
                continue
            # 对 name 及其同义名逐一尝试
            tried = False
            for alias in _synonyms(name):
                # 字符串
                m = re.search(rf"{re.escape(alias)}\s*:\s*\"([^\"]+)\"", text, flags=re.I)
                if not m:
                    m = re.search(rf"{re.escape(alias)}\s*:\s*\'([^\']+)\'", text, flags=re.I)
                if m:
                    out[name] = m.group(1).strip()
                    tried = True
                    break
                # 数字
                m = re.search(rf"{re.escape(alias)}\s*:\s*([0-9]+)", text, flags=re.I)
                if m:
                    out[name] = m.group(1).strip()
                    tried = True
                    break
                # 布尔
                m = re.search(rf"{re.escape(alias)}\s*:\s*(true|false)\b", text, flags=re.I)
                if m:
                    out[name] = m.group(1).lower()
                    tried = True
                    break
            if tried:
                continue

        # ---- 针对 query 的兜底：若没抓到，取 Step 中首个引号字符串 ----
        names = { (p.get("name") or "") for p in params }
        if "query" in names and "query" not in out:
            m = re.search(r'\"([^\"]{2,})\"', text) or re.search(r"\'([^\']{2,})\'", text)
            if m:
                out["query"] = m.group(1).strip()

        # ID 兜底：若需要 movie_id/tv_id/person_id 且文本里出现裸数字，取首个数字
        need_ids = {p.get("name") for p in params if p.get("required") and str(p.get("name")).endswith("_id")}
        if need_ids and not any(k in out for k in need_ids):
            m = re.search(r"\b([0-9]{2,})\b", text)
            if m:
                val = m.group(1)
                for k in need_ids:
                    out.setdefault(k, val)
                    break
        return out

    def _coerce_by_schema(self, args: Dict[str, Any], api_schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """按 schema 的 type 做一次宽松类型强制。"""
        if not api_schema or not isinstance(args, dict):
            return args or {}
        params = [p for p in (api_schema.get("parameters") or []) if isinstance(p, dict)]
        for p in params:
            name = p.get("name"); typ = (p.get("schema") or {}).get("type")
            if not name or name not in args or typ is None:
                continue
            v = args[name]
            try:
                if typ == "integer":
                    if isinstance(v, bool):
                        args[name] = int(v)
                    elif isinstance(v, (int,)):
                        pass
                    else:
                        args[name] = int(str(v))
                elif typ == "number":
                    if isinstance(v, (int, float)):
                        pass
                    else:
                        args[name] = float(str(v))
                elif typ == "boolean":
                    if isinstance(v, bool):
                        pass
                    else:
                        args[name] = (str(v).strip().lower() == "true")
                else:
                    # string/array/object 保留
                    args[name] = v
            except Exception:
                # 强制失败时保留原值，交给下游执行期再判错
                pass
        return args

    # alias for compatibility
    def fill(self, step: str, api_schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.fill_arguments(step, api_schema)

    # ========= TMDB real execution helpers =========
    def _tmdb_auth(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Return (headers, query_params). Prefer Bearer; fallback to api_key."""
        headers = {"Accept": "application/json"}
        q = {}
        if self.tmdb_bearer:
            headers["Authorization"] = f"Bearer {self.tmdb_bearer}"
        elif self.tmdb_api_key:
            q["api_key"] = self.tmdb_api_key
        return headers, q

    @staticmethod
    def _substitute_path_params(path: str, args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Substitute /movie/{movie_id} with args['movie_id'] and remove used keys."""
        used = set()

        def repl(m):
            key = m.group(1) # e.g., "movie_id"
            
            # 1. 精确查找
            if key in args:
                used.add(key)
                return str(args[key])
            
            # 2. 别名查找 (关键修复：允许用 'id' 填充 'movie_id' / 'person_id' 等)
            # 如果 key 是 xxx_id 且 args 里有 'id'，则使用 'id'
            if key.endswith("_id") and "id" in args:
                # 记录使用情况，但不一定是 used.add('id')，因为 'id' 可能被多个参数复用?
                # 这里简单起见，认为是消费了 'id'
                used.add("id") 
                return str(args["id"])
            
            # 3. 找不到则保留原样 (这会导致后续 404，但比报错好调试)
            return m.group(0)

        new_path = re.sub(r"\{([^{}]+)\}", repl, path)
        residual = {k: v for k, v in args.items() if k not in used}
        return new_path, residual

    def _real_invoke_tmdb(self, tool_id: str, arguments: Dict[str, Any], timeout: int = 30) -> ExecResult:
        spec = self.get_schema(tool_id)
        if not spec:
            return ExecResult(success=False, error=f"Unknown tool: {tool_id}")
        path = spec.get("path"); method = (spec.get("method") or "GET").upper()
        if not path:
            return ExecResult(success=False, error=f"No path for tool: {tool_id}")

        # 1) path params
        path_filled, rest = self._substitute_path_params(path, dict(arguments or {}))
        url = self.tmdb_base.rstrip("/") + path_filled

        # 2) query/body from spec.in
        query, body = {}, None
        for p in spec.get("parameters", []) or []:
            nm = p.get("name"); loc = p.get("in")
            if not nm or nm not in rest:
                continue
            if loc == "query":
                query[nm] = rest[nm]
            elif loc == "body":
                if body is None:
                    body = {}
                body[nm] = rest[nm]

        # 3) auth
        headers, auth_q = self._tmdb_auth()
        query = {**auth_q, **query}

        try:
            resp = requests.request(method, url, params=query, json=body, headers=headers, timeout=timeout)
            ok = 200 <= resp.status_code < 300
            try:
                data = resp.json()
            except Exception:
                data = {"text": resp.text}
            if ok:
                return ExecResult(success=True, output={"status": resp.status_code, "url": url, "data": data})
            else:
                return ExecResult(success=False, output={"status": resp.status_code, "url": url}, error=json.dumps(data, ensure_ascii=False))
        except Exception as e:
            return ExecResult(success=False, error=str(e))

    # ========= Simulate / Execute =========
    @staticmethod
    def _make_digest(tool_id: str, arguments: Dict[str, Any]) -> str:
        s = json.dumps({"tool": tool_id, "arguments": arguments}, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    def execute(self, tool_id: str, arguments: Dict[str, Any]) -> ExecResult:
        """
        Real TMDB when enabled; otherwise simulate consistency check.
        Real exec is enabled iff:
          - TOOLLLM_REAL_EXEC in {1,true,yes,on}; or
          - TOOLLLM_REAL_EXEC is unset AND (TMDB_BEARER_TOKEN or TMDB_API_KEY) is present.
        """
        # If we can and should do a real call, do it.
        if self.real_exec:
            return self._real_invoke_tmdb(tool_id, arguments)

        # Otherwise do a light schema-based consistency check.
        schema = self.get_schema(tool_id)
        digest = self._make_digest(tool_id, arguments)

        if not schema:
            # No schema: do not penalize
            return ExecResult(success=True, output={"digest": digest})

        required = [p for p in schema.get("parameters", []) or [] if p.get("required")]
        missing: List[str] = []
        mismatch: List[str] = []

        for p in required:
            name = p.get("name")
            if not name:
                continue
            if name not in arguments:
                missing.append(name)
                continue
            val = arguments.get(name)
            typ = (p.get("schema") or {}).get("type")
            # coarse type checks
            if typ == "integer":
                if not isinstance(val, int):
                    if not (isinstance(val, str) and val.isdigit()):
                        mismatch.append(f"{name}:integer")
            elif typ == "number":
                if not isinstance(val, (int, float)):
                    try:
                        float(str(val))
                    except Exception:
                        mismatch.append(f"{name}:number")
            elif typ == "boolean":
                if not isinstance(val, bool) and str(val).lower() not in {"true", "false"}:
                    mismatch.append(f"{name}:boolean")

        if missing or mismatch:
            return ExecResult(
                success=False,
                output={"digest": digest},
                error=json.dumps({"missing": missing, "type_mismatch": mismatch}, ensure_ascii=False),
            )
        return ExecResult(success=True, output={"digest": digest})

    # alias for compatibility
    def simulate(self, tool_name: str, args: Dict[str, Any]) -> Tuple[bool, str]:
        res = self.execute(tool_name, args)
        return res.success, (res.error or "ok")

    # ========= Convenience =========
    def choose(self, step_text: str) -> Dict[str, Any]:
        cands = self.retrieve_tools(step_text, topk=self._topk)
        if not cands:
            return {"tool": None, "args": {}, "candidates": []}
        best = cands[0]
        spec = self.get_schema(best.tool_id)
        args = self.fill_arguments(step_text, spec)
        return {
            "tool": best.tool_id,
            "args": args,
            "candidates": [{"tool_id": c.tool_id, "score": c.score} for c in cands[: self.confirm_top]],
        }


# Optional factory
def load_adapter(tool_schema_path: str, **kwargs) -> ToolLLMAdapter:
    return ToolLLMAdapter(tool_schema_path=tool_schema_path, **kwargs)