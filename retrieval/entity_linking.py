# retrieval/entity_linking.py
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

_WORD_RE = re.compile(r"[a-z0-9]+")

def _norm(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _tokens(s: str) -> List[str]:
    return _WORD_RE.findall(_norm(s))

def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()

def _expand_aliases_from_id(tid: str) -> List[str]:
    """
    基于 tool_id 的简单派生：'send_sms' -> ['send sms','sms','send text message']
    仅覆盖通用模式；如需更强覆盖，可在 entity2tool.json 里人工补充。
    """
    base = _norm(tid)             # e.g. "send sms"
    toks = base.split()
    al = {base}

    # 单词还原（例如 book_flight -> 'book a flight', 'flight booking'）
    if len(toks) == 2:
        v, n = toks
        al.add(f"{v} a {n}")
        al.add(f"{n} {v}ing" if not n.endswith("ing") else f"{n} {v}")
        al.add(f"{v} {n}")
    elif len(toks) == 3:
        al.add(" ".join(toks))
        al.add(f"{toks[0]} the {toks[1]} {toks[2]}")
        al.add(f"{toks[0]} a {toks[1]} {toks[2]}")

    # 常见关键词替换
    repl = {
        "sms": ["text message","message","send message"],
        "book": ["reserve","booking","make a reservation for"],
        "flight": ["air ticket","plane ticket"],
        "meeting": ["conference","video conference","online meeting"],
        "doctor": ["physician","online doctor","telemedicine"],
        "weather": ["forecast","weather info"],
        "alarm": ["reminder","set reminder"],
        "print": ["printing","printer","print document","print the document"],
        "lawyer": ["legal","legal consultation","consult a lawyer"],
        "buy": ["purchase"]
    }
    for k, vs in repl.items():
        if k in toks:
            for v in vs:
                al.add(base.replace(k, v))

    # 单词自身也加入
    for t in toks:
        if len(t) >= 3:
            al.add(t)

    return [a for a in al if a.strip()]

class EntityLinker:
    """
    优先读取映射文件；若缺失则基于工具列表自动生成别名并保存。
    支持模糊匹配：score >= threshold 即返回。
    """
    def __init__(self, mapping_path: Optional[str] = None,
                 tools: Optional[List[str]] = None,
                 threshold: float = 0.78):
        self.threshold = threshold
        self.path = Path(mapping_path) if mapping_path else None
        self.alias2tool: Dict[str, str] = {}

        if self.path and self.path.exists():
            try:
                mp = json.load(open(self.path, "r", encoding="utf-8"))
                # 允许两种格式：{alias: tool} 或 {"mapping":{alias:tool}}
                if isinstance(mp, dict) and "mapping" in mp:
                    mp = mp["mapping"]
                if isinstance(mp, dict):
                    for k, v in mp.items():
                        self.alias2tool[_norm(k)] = v
            except Exception:
                pass

        # 如果没有文件或文件为空，尝试基于工具列表自动生成
        if not self.alias2tool and tools:
            for tid in tools:
                # 生成一批 alias
                aliases = set(_expand_aliases_from_id(tid))
                aliases.add(tid)
                aliases.add(tid.replace("_"," "))
                for a in aliases:
                    self.alias2tool[_norm(a)] = tid
            # 回写到磁盘（方便下次直接加载）
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "w", encoding="utf-8") as w:
                    json.dump({"mapping": self.alias2tool}, w, ensure_ascii=False, indent=2)

    def link(self, name: str) -> Optional[str]:
        if not name:
            return None
        q = _norm(name)
        # 直接命中
        if q in self.alias2tool:
            return self.alias2tool[q]
        # 模糊命中（取最高分）
        best_tool, best = None, 0.0
        for a, t in self.alias2tool.items():
            s = _ratio(q, a)
            if s > best:
                best, best_tool = s, t
        if best >= self.threshold:
            return best_tool
        return None