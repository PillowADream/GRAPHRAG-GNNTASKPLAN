# retrieval/toollm_adapter.py
import json, re, hashlib, random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

_WORD = re.compile(r"[A-Za-z0-9_]+")
_NUM  = re.compile(r"[-+]?\d+(\.\d+)?")
_DATE = re.compile(r"(\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4})")
_BOOL = re.compile(r"\b(true|false|yes|no)\b", re.I)

def _tok(s: str) -> List[str]:
    return [t.lower() for t in _WORD.findall(s or "")]

def _jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B: return 0.0
    return len(A & B) / max(1, len(A | B))

def _sha1(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

@dataclass
class ParamSpec:
    name: str
    type: str = "string"           # "string"|"integer"|"number"|"boolean"|"array"
    required: bool = False
    enum: Optional[List[Any]] = None
    default: Optional[Any] = None
    description: str = ""

@dataclass
class ToolSchema:
    tool_id: str
    description: str
    params: List[ParamSpec]

@dataclass
class RetrievedTool:
    tool_id: str
    score: float

@dataclass
class ExecResult:
    success: bool
    normalized_args: Dict[str, Any]
    output: Optional[Dict[str, Any]]
    error: Optional[str]
    trace: Dict[str, Any]

class ToolLLMAdapter:
    """
    轻量 ToolLLM 适配器（不依赖外部 LLM）：召回、参数填充、Schema 校验 + 可执行模拟
    """
    def __init__(self, schema_path: str, seed: int = 42):
        self.schemas: Dict[str, ToolSchema] = self._load_schema(schema_path)
        self._rng = random.Random(seed)
        # 预构建倒排 tokens
        self._tool_tokens: Dict[str, List[str]] = {}
        for tid, sc in self.schemas.items():
            toks = _tok(sc.description)
            for p in sc.params:
                toks += _tok(p.name) + _tok(p.description)
            self._tool_tokens[tid] = toks

    # ---------- schema ----------
    def _load_schema(self, path: str) -> Dict[str, ToolSchema]:
        raw = json.load(open(path, "r", encoding="utf-8"))
        # 支持两种格式：
        # A) {"tools":[{"id": "...", "desc":"...", "params":[{"name":..., "type":..., "required":...}, ...]}]}
        # B) {"<tool_id>": {"description":"...", "params":{ "<p>":{"type":"integer","required":true,...}, ...}}}
        out: Dict[str, ToolSchema] = {}
        if isinstance(raw, dict) and "tools" in raw:
            for t in raw["tools"]:
                params = []
                for p in t.get("params", []):
                    params.append(ParamSpec(
                        name=p["name"],
                        type=p.get("type", "string"),
                        required=bool(p.get("required", False)),
                        enum=p.get("enum"),
                        default=p.get("default"),
                        description=p.get("description", "")
                    ))
                out[t["id"]] = ToolSchema(tool_id=t["id"], description=t.get("desc",""), params=params)
        else:
            for tid, t in raw.items():
                plist = []
                psrc = t.get("params", {}) if isinstance(t.get("params"), dict) else {}
                for pn, pd in psrc.items():
                    plist.append(ParamSpec(
                        name=pn,
                        type=pd.get("type", "string"),
                        required=bool(pd.get("required", False)),
                        enum=pd.get("enum"),
                        default=pd.get("default"),
                        description=pd.get("description", "")
                    ))
                out[tid] = ToolSchema(tool_id=tid, description=t.get("description",""), params=plist)
        return out

    def get_schema(self, tool_id: str) -> Optional[ToolSchema]:
        return self.schemas.get(tool_id)

    # ---------- retrieve ----------
    def retrieve_tools(self, step_text: str, topk: int = 5) -> List[RetrievedTool]:
        q = _tok(step_text)
        scores: List[RetrievedTool] = []
        for tid, toks in self._tool_tokens.items():
            s = _jaccard(q, toks)
            # 轻随机打散避免完全并列
            s += 1e-4 * self._rng.random()
            scores.append(RetrievedTool(tool_id=tid, score=s))
        scores.sort(key=lambda x: x.score, reverse=True)
        return scores[:max(1, topk)]

    # ---------- fill arguments ----------
    def _cast_type(self, v: Any, t: str):
        if v is None: return None
        try:
            if t == "integer": return int(float(v))
            if t == "number":  return float(v)
            if t == "boolean":
                if isinstance(v, bool): return v
                return str(v).strip().lower() in ("1","true","yes","y","on")
            if t == "array":
                return v if isinstance(v, list) else [v]
            return str(v)
        except Exception:
            return None

    def _extract_primitives(self, text: str) -> Dict[str, Any]:
        nums  = [n[0] if isinstance(n, tuple) else n for n in _NUM.findall(text)]
        dates = _DATE.findall(text)
        bools = [b.lower() for b in _BOOL.findall(text)]
        return {
            "numbers": [float(n) if "." in n else int(n) for n in nums],
            "dates": dates,
            "bools": [b in ("true","yes") for b in bools]
        }

    def _heuristic_map(self, pname: str, prim: Dict[str, Any], text: str) -> Optional[Any]:
        pl = pname.lower()
        # 粗暴词义映射
        if any(k in pl for k in ["count","num","k","top","limit"]):
            return prim["numbers"][0] if prim["numbers"] else 1
        if any(k in pl for k in ["day","days","date","from","start"]):
            return prim["dates"][0] if prim["dates"] else None
        if any(k in pl for k in ["to","end","until"]):
            return prim["dates"][1] if len(prim["dates"]) >= 2 else None
        if any(k in pl for k in ["id","code"]):
            # 取最靠前的整数当 id
            for n in prim["numbers"]:
                if isinstance(n, int): return n
            return None
        if any(k in pl for k in ["flag","enable","disable","open","close"]):
            return prim["bools"][0] if prim["bools"] else None
        # 兜底：取一句中与参数同词根的片段
        text_toks = _tok(text)
        for i, w in enumerate(text_toks):
            if w == pl and i+1 < len(text_toks):
                return text_toks[i+1]
        return None

    def fill_arguments(self, step_text: str, api_schema: ToolSchema, mode: str = "fill") -> Dict[str, Any]:
        """
        mode: "draft"（只给出可能的值，不做强制转换）
              "fill" （按类型强制转换 + 枚举/默认值修正）
        """
        prim = self._extract_primitives(step_text)
        pred: Dict[str, Any] = {}
        for p in api_schema.params:
            v = self._heuristic_map(p.name, prim, step_text)
            if v is None and p.default is not None:
                v = p.default
            if p.enum and v is not None:
                # 若不在枚举内，尝试最相近（忽略大小写）
                if v not in p.enum:
                    low = str(v).lower()
                    cand = [e for e in p.enum if str(e).lower() == low]
                    v = cand[0] if cand else (p.enum[0] if mode == "fill" else v)
            pred[p.name] = self._cast_type(v, p.type) if mode == "fill" else v
        return pred

    # ---------- execute (simulate) ----------
    def execute(self, tool_id: str, args: Dict[str, Any]) -> ExecResult:
        sc = self.schemas.get(tool_id)
        if not sc:
            return ExecResult(False, args, None, f"unknown tool {tool_id}", {"phase":"schema_check"})
        # 必填校验 + 类型再规范
        norm = {}
        missing = []
        for p in sc.params:
            v = args.get(p.name, None)
            v = self._cast_type(v, p.type)
            if v is None and p.required:
                missing.append(p.name)
            norm[p.name] = v
        if missing:
            return ExecResult(False, norm, None, f"missing required params: {missing}", {"phase":"schema_check"})

        # 简单“成功”模拟：按 (tool_id, norm) 生成稳定摘要；并为少量样例注入失败（可选）
        out = {"digest": _sha1({"tool": tool_id, "args": norm})}
        return ExecResult(True, norm, out, None, {"phase":"simulate", "note":"schema-ok"})

    # ---------- metrics ----------
    @staticmethod
    def param_f1(pred: Dict[str, Any], gt: Optional[Dict[str, Any]]) -> Optional[float]:
        if gt is None: return None
        # 键级别 + 值精确匹配（字符串大小写无关；数字相等）
        def _eq(a,b):
            if a is None or b is None: return False
            try:
                if isinstance(a,(int,float)) or isinstance(b,(int,float)):
                    return float(a) == float(b)
            except Exception:
                pass
            return str(a).strip().lower() == str(b).strip().lower()
        P, G = set(pred.keys()), set(gt.keys())
        if not P and not G: return 1.0
        tp = sum(1 for k in P & G if _eq(pred[k], gt[k]))
        if tp == 0: return 0.0
        prec = tp / max(1, len(P))
        rec  = tp / max(1, len(G))
        return 2*prec*rec/(prec+rec+1e-12)