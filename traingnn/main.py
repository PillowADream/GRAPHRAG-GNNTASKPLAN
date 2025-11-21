import argparse
import os
import sys
import time
import json
import pickle
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
from pathlib import Path
from sklearn.neighbors import kneighbors_graph
import prettytable as pt

from model import ModelTrainer

# ---- repo root 抬头注入到 sys.path ----
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in (here.parent, *here.parents):
        if (p / "data").exists() and (p / "prediction").exists():
            return p
    return Path.cwd()

ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import (
    init_random_state, sequence_greedy_tool_selection, prepare_lm_gnn_training_data,
    load_test_data, get_cur_time, prepare_training_ids, load_tool, save_checkpoint
)
from sampler import TrainSampler
from evaluate import f1_score
from utils.runlog import open_jsonl, log_sample


# ========== 工具函数 ==========
def _load_rag_candidate_bank(rag_feat_path: str, index2tool: List[str]) -> Optional[np.ndarray]:
    p = Path(rag_feat_path)
    if not p.exists() or p.suffix.lower() != ".pkl":
        return None
    obj = pickle.load(open(p, "rb"))
    if not isinstance(obj, dict) or "tool_text_vec" not in obj or "id_list" not in obj:
        return None
    id_list = obj["id_list"]
    id2row = {tid: i for i, tid in enumerate(id_list)}
    bank = []
    for i, tid in enumerate(index2tool):
        j = id2row.get(tid, id2row.get(str(tid), None))
        if j is None:
            j = min(i, len(id_list)-1)
        bank.append(np.asarray(obj["tool_text_vec"][j], dtype=np.float32))
    return np.stack(bank, axis=0).astype(np.float32)  # (N, D)


def build_tool_textual_sim_graph(tool_emb, k=5, metric='cosine'):
    text_adj = kneighbors_graph(tool_emb, k, mode='connectivity', metric=metric, include_self=False).toarray()
    text_adj_g = {tool: [] for tool in tool2index.keys()}
    for i in range(text_adj.shape[0]):
        for j in range(text_adj.shape[1]):
            if text_adj[i, j]:
                text_adj_g[index2tool[i]].append(index2tool[j])
    return text_adj_g


def bi_evaluate(id_list, pred_dict, tmp_print=False):
    import numpy as _np
    init_scores, searched_scores = [], []
    high_fix_examples = []

    for data_id in id_list:
        content = pred_dict[data_id]

        gt_node, gt_link = content["gt_nodes"], content["gt_links"]
        pred_node, pred_link = content["pred_nodes"], content["pred_links"]

        search_node = content['search_nodes']
        search_link = [", ".join(link) for link in content["search_links"]]

        node_f1, link_f1 = f1_score(pred_node, gt_node), f1_score(pred_link, gt_link)
        search_node_f1, search_link_f1 = f1_score(search_node, gt_node), f1_score(search_link, gt_link)
        init_succ, search_succ = float(node_f1 >= 0.99), float(search_node_f1 >= 0.99)

        if search_node_f1 > node_f1 and search_link_f1 > link_f1:
            high_fix_examples.append([data_id, round(search_node_f1 - node_f1), round(search_link_f1 - link_f1)])

        init_scores.append([node_f1, link_f1, init_succ])
        searched_scores.append([search_node_f1, search_link_f1, search_succ])

    avg_pred_score = _np.round(_np.mean(_np.array(init_scores), axis=0), 4)
    avg_searched_score = _np.round(_np.mean(_np.array(searched_scores), axis=0), 4)
    if tmp_print:
        print(f"Init   [Node-F1] {avg_pred_score[0]:.4f} [Link-F1] {avg_pred_score[1]:.4f}")
        print(f"Search [Node-F1] {avg_searched_score[0]:.4f} [Link-F1] {avg_searched_score[1]:.4f}")

    high_fix_examples = sorted(high_fix_examples, key=lambda x: x[1], reverse=True)[:50]

    return {
        "base-node-f1": float(avg_pred_score[0]),
        "base-link-f1": float(avg_pred_score[1]),
        "base-acc": float(avg_pred_score[2]),
        "search-node-f1": float(avg_searched_score[0]),
        "search-link-f1": float(avg_searched_score[1]),
        "search-acc": float(avg_searched_score[2])
    }, [example[0] for example in high_fix_examples]


def _default_feat_paths(dataset: str) -> Tuple[str, str, str]:
    base = f"../data/{dataset}/graphrag/features"
    return (
        os.path.join(base, "tool_rag_feat.npy"),
        os.path.join(base, "tool_comm_ids.npy"),
        os.path.join(base, "tool_list.txt"),
    )


def _load_rag_features(
    dataset: str,
    tool2index: Dict[str, int],
    index2tool: List[str],
    rag_feat_path: Optional[str] = None,
    rag_comm_path: Optional[str] = None,
    rag_list_path: Optional[str] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[int]]:
    rf_path, rc_path, rl_path = _default_feat_paths(dataset)
    rag_feat_path = rag_feat_path or rf_path
    rag_comm_path = rag_comm_path or rc_path
    rag_list_path = rag_list_path or rl_path

    p = Path(rag_feat_path)
    if not p.exists():
        if not rag_comm_path or not Path(rag_comm_path).exists():
            print(f"[RAG-Feat] Not found: {rag_feat_path} / {rag_comm_path}. Skip rag_feat.")
            return None, None, None

    obj = None
    if p.suffix.lower() == ".pkl" and p.exists():
        obj = pickle.load(open(p, "rb"))
    elif p.exists():
        tmp = np.load(p, allow_pickle=True)
        if isinstance(tmp, np.ndarray) and tmp.dtype == object and tmp.size == 1 and isinstance(tmp.item(), dict):
            obj = tmp.item()
        else:
            obj = tmp  # 直接矩阵 (N, D)

    def _dict_to_aligned(obj_dict) -> Tuple[np.ndarray, np.ndarray, int]:
        id_list = obj_dict["id_list"]
        id2row = {tid: i for i, tid in enumerate(id_list)}
        sp = obj_dict.get("sp_to_evidence", None)
        src1h = obj_dict.get("source_onehot", None)

        deg_log       = np.asarray(obj_dict["deg_log"], dtype=np.float32)[:, None]
        comm_size_log = np.asarray(obj_dict["comm_size_log"], dtype=np.float32)[:, None]
        tool_text_vec = np.asarray(obj_dict["tool_text_vec"], dtype=np.float32)
        comm_text_vec = np.asarray(obj_dict["comm_text_vec"], dtype=np.float32)
        comm_id_full  = np.asarray(obj_dict["comm_id"], dtype=np.int64)

        sp_arr    = None if sp is None else np.asarray(sp, dtype=np.float32).reshape(len(comm_id_full), -1)
        src1h_arr = None if src1h is None else np.asarray(src1h, dtype=np.float32)

        N = len(index2tool)
        rows: List[np.ndarray] = []
        comm_rows = np.zeros(N, dtype=np.int64)
        missing = []

        for i, tid in enumerate(index2tool):
            j = None
            if isinstance(tid, (int, np.integer)) and 0 <= int(tid) < len(id_list):
                j = int(tid)
            if j is None:
                j = id2row.get(tid, None)
            if j is None:
                j = id2row.get(str(tid), None)
            if j is None or j < 0 or j >= len(id_list):
                missing.append(tid)
                j = min(i, len(id_list) - 1)

            base_feats = [deg_log[j], comm_size_log[j], tool_text_vec[j], comm_text_vec[j]]
            if sp_arr is not None:
                base_feats.append(np.asarray(sp_arr[j]).reshape(-1))
            if src1h_arr is not None:
                base_feats.append(np.asarray(src1h_arr[j]).reshape(-1))
            feat_j = np.concatenate(base_feats, axis=0)

            rows.append(feat_j)
            comm_rows[i] = comm_id_full[j]

        if missing:
            print(f"[RAG-Feat] Warning: {len(missing)} tools could not be matched; used position fallback. (e.g., {missing[:5]})")

        rag_feat = np.stack(rows, axis=0).astype(np.float32)
        num_comms = int(comm_id_full.max()) + 1 if comm_id_full.size > 0 else 0
        return rag_feat, comm_rows, num_comms

    rag_feat_np, comm_ids_np, num_comms = None, None, None
    if isinstance(obj, dict):
        rag_feat_np, comm_ids_np, num_comms = _dict_to_aligned(obj)
    elif isinstance(obj, np.ndarray):
        if obj.ndim != 2 or obj.shape[0] != len(index2tool):
            raise ValueError(f"[RAG-Feat] shape mismatch: got {obj.shape}, expect (N,D) with N={len(index2tool)}")
        rag_feat_np = obj.astype(np.float32)
        if rag_comm_path and Path(rag_comm_path).exists():
            comm_ids_np = np.load(rag_comm_path)
            if comm_ids_np.shape[0] != len(index2tool):
                raise ValueError(f"[RAG-Feat] comm_ids size mismatch: {comm_ids_np.shape} vs N={len(index2tool)}")
            comm_ids_np = comm_ids_np.astype(np.int64)
            num_comms = int(comm_ids_np.max()) + 1
    else:
        if rag_comm_path and Path(rag_comm_path).exists():
            comm_ids_np = np.load(rag_comm_path).astype(np.int64)
            if comm_ids_np.shape[0] == len(index2tool):
                num_comms = int(comm_ids_np.max()) + 1

    rag_feat_t = torch.from_numpy(rag_feat_np).float() if rag_feat_np is not None else None
    comm_ids_t = torch.from_numpy(comm_ids_np).long() if comm_ids_np is not None else None
    return rag_feat_t, comm_ids_t, num_comms


# ======= 评测增强：结构分桶 & 幻觉率 =======

def _precision_recall(pred_list: List[str], gt_list: List[str]) -> Tuple[float, float]:
    ps, gs = set(pred_list), set(gt_list)
    if len(ps) == 0:
        return (1.0, 0.0 if len(gs) > 0 else 1.0)
    tp = len(ps & gs)
    prec = tp / max(1, len(ps))
    rec = tp / max(1, len(gs))
    return prec, rec

def _eval_buckets(id_list, pred_dict, adj_g, index2tool, comm_ids_np: Optional[np.ndarray]):
    from collections import defaultdict
    def node_deg(name: str) -> int:
        return len(adj_g.get(name, []))

    def comm_of(name: str) -> int:
        if comm_ids_np is None:
            return -1
        try:
            idx = index2tool.index(name)
            return int(comm_ids_np[idx])
        except Exception:
            return -1

    stats = defaultdict(list)
    for sid in id_list:
        C = pred_dict[sid]
        gt_node, gt_link = C["gt_nodes"], C["gt_links"]
        pred_node, pred_link = C["search_nodes"], [", ".join(x) for x in C["search_links"]]

        path_len = len(C["search_links"])
        max_branch = 0
        cross = False
        for a, b in C["search_links"]:
            max_branch = max(max_branch, node_deg(a), node_deg(b))
            if comm_of(a) != comm_of(b):
                cross = True

        nf1 = f1_score(pred_node, gt_node)
        lf1 = f1_score(pred_link, gt_link)

        if path_len >= 3:
            stats["long_path"].append((nf1, lf1))
        if max_branch >= 5:
            stats["high_branch"].append((nf1, lf1))
        if cross:
            stats["cross_comm"].append((nf1, lf1))

    def _avg(lst):
        if not lst: return (None, None, 0)
        n = len(lst)
        nf = round(sum(x[0] for x in lst)/n, 4)
        lf = round(sum(x[1] for x in lst)/n, 4)
        return nf, lf, n

    out = {k: _avg(v) for k, v in stats.items()}
    return out


# ======= M3: 反频率权重 =======

def build_anti_freq_weights(adj_g: Dict[str, List[str]],
                            tool2index: Dict[str, int],
                            beta: float = 0.5, eps: float = 1e-6) -> Dict[Tuple[int, int], float]:
    """
    用节点度数近似“热门度”，边频率 ~ deg(u)*deg(v)；权重 = (freq + eps)^(-beta) ，再做 min-max 归一。
    """
    deg = {name: float(len(neigh)) for name, neigh in adj_g.items()}
    edge_w: Dict[Tuple[int, int], float] = {}
    raw_vals = []
    for u, neigh in adj_g.items():
        for v in neigh:
            f = max(eps, deg.get(u, 0.0) * deg.get(v, 0.0))
            w = (f + eps) ** (-beta)
            iu, iv = tool2index[u], tool2index[v]
            edge_w[(iu, iv)] = w
            raw_vals.append(w)

    if not raw_vals:
        return edge_w
    lo, hi = min(raw_vals), max(raw_vals)
    if hi - lo < 1e-12:
        return edge_w
    # 0.3 ~ 1.0 线性拉伸，避免极端权重
    for k, w in list(edge_w.items()):
        edge_w[k] = 0.3 + 0.7 * ((w - lo) / (hi - lo))
    return edge_w


# ======= M3: 同构扰动（训练用 & 评测可复用）=======

def permute_graph_inputs(tool_texts: List[str],
                         tool_adj: Optional[torch.Tensor],
                         rag_feat: Optional[torch.Tensor],
                         comm_ids: Optional[torch.Tensor],
                         device: torch.device):
    """
    给定一次随机置换，返回：
    - perm: torch.LongTensor[N]
    - inv_perm: torch.LongTensor[N]
    - texts_perm / adj_perm / rag_feat_perm / comm_perm
    """
    N = len(tool_texts)
    perm = torch.randperm(N, device=device)
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(N, device=device)

    texts_perm = [tool_texts[i] for i in perm.tolist()]

    adj_perm = None
    if tool_adj is not None:
        # edge_index[2, E] 形式
        adj_perm = torch.stack([perm[tool_adj[0]], perm[tool_adj[1]]], dim=0)

    feat_perm = None
    if rag_feat is not None:
        feat_perm = rag_feat[perm]

    comm_perm = None
    if comm_ids is not None:
        comm_perm = comm_ids[perm]

    return perm, inv_perm, texts_perm, adj_perm, feat_perm, comm_perm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset and training related
    parser.add_argument('--dataset', type=str, default='huggingface', choices=['huggingface', 'multimedia', 'dailylife'])
    parser.add_argument('--load_alignment', type=bool, default=True)
    parser.add_argument('--maximum', type=str, default='')

    # LM related
    parser.add_argument('--lm_name', type=str, default='intfloat/e5-large',
                        choices=['intfloat/e5-large', 'sentence-transformers/all-roberta-large-v1',
                                 'sentence-transformers/all-MiniLM-L6-v2', "intfloat/e5-large-v2"])

    # GNN related
    parser.add_argument('--gnn_name', type=str, default='GCN', choices=['SGC', 'GCN', 'GAT', 'SAGE', 'GIN', 'TransformerConv'])
    parser.add_argument('--gnn_hidden_dim', type=int, default=1024)
    parser.add_argument('--gnn_layer', type=int, default=1)

    # Training models related
    parser.add_argument('--train_num', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_negatives', type=int, default=2)
    parser.add_argument('--lm_frozen', type=int, default=0)
    parser.add_argument('--text_negative', type=int, default=0)

    # Test related
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--measure', type=str, default='dot', choices=['dot', 'distance'])
    parser.add_argument('--save_model', type=int, default=0, choices=[0, 1])
    parser.add_argument('--load_model', type=int, default=0)
    parser.add_argument('--evaluate_only', type=int, default=0, choices=[0, 1])

    # RAG feature & structured negatives
    parser.add_argument('--use_rag_feat', type=int, default=0, choices=[0, 1])
    parser.add_argument('--rag_feat_path', type=str, default='')
    parser.add_argument('--rag_comm_path', type=str, default='')
    parser.add_argument('--rag_list_path', type=str, default='')
    parser.add_argument('--comm_emb_dim', type=int, default=16)
    parser.add_argument('--rag_dropout', type=float, default=0.1)
    parser.add_argument('--structure_hard_negative', type=int, default=1, choices=[0, 1])
    parser.add_argument('--rag_lr', type=float, default=1e-3)
    parser.add_argument('--rag_candidates_topk', type=int, default=0, help='>0 to enable RAG candidate pruning (no fusion, no GNN)')
    # ---- New: RAG(cands) 最小分数阈值（用于评测阶段的子库检索）----
    parser.add_argument('--min_rag_score', type=float, default=0.0,
                        help='RAG-only 检索的最小余弦相似度阈值（0~1）；仅对 --rag_candidates_topk>0 的评测分支生效')

    # ---- New: 融合门正则（将门均值拉到一个目标附近，避免饱和）----
    parser.add_argument('--gate_center', type=float, default=0.65,
                        help='门均值的目标中心（建议 0.6~0.7）')
    parser.add_argument('--gate_reg', type=float, default=0.0,
                        help='门正则系数（>0 开启）；损失 = gate_reg * (mean(gate)-gate_center)^2')


    # === M2 中已有 ===
    parser.add_argument('--reset_rag_head', type=int, default=0, choices=[0, 1])

    # === M3 新增 ===
    parser.add_argument('--anti_freq', type=int, default=0, choices=[0, 1], help='启用反频率加权的 BPR/InfoNCE')
    parser.add_argument('--af_beta', type=float, default=0.5, help='反频率权重的幂指数 beta')
    parser.add_argument('--inv_reg', type=float, default=0.0, help='同构不变性正则系数 λ（0 则关闭）')
    parser.add_argument('--inv_every', type=int, default=10, help='每多少个 step 计算一次不变性正则（避免额外开销过大）')
    parser.add_argument("--use_infonce", type=int, choices=[0, 1], default=0,
                        help="是否使用 InfoNCE 目标（1 开启；0 关闭）。")
    parser.add_argument("--infonce_tau", type=float, default=0.07,
                        help="InfoNCE 温度系数 tau（越小分布越尖锐）。")

    # === M3: 鲁棒性实验（同构扰动）===
    parser.add_argument('--robust_iso', type=int, default=0, choices=[0, 1], help='评测期进行同构扰动鲁棒性测试')
    parser.add_argument('--robust_iso_trials', type=int, default=3, help='同构扰动评测次数')

    args = parser.parse_args()
    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)
    print(args, "\n")

    init_random_state(args.seed)
    device = torch.device(args.device)

    if args.dataset == 'multimedia':
        args.batch_size = 256

    ####################################
    #### Prepare Trainset and Tool #####
    ####################################
    align_path = ROOT / "data" / args.dataset / "split_ids.json"
    alignment_ids = json.load(open(align_path, 'r', encoding="utf-8"))["test_ids"]["chain"]
    train_ids = prepare_training_ids(args.dataset, train_num=args.train_num, alignment_ids=alignment_ids)

    train_data = prepare_lm_gnn_training_data(dataset_name=args.dataset, train_ids=train_ids)
    tool_texts, tool2index, index2tool, edge_index, sgc_edge_index, adj_g = load_tool(dataset_name=args.dataset)

    def build_exp_id(args):
        tag = [
            f"{args.gnn_name}",
            "ragF",
            f"af{args.af_beta}",
            f"inv{args.inv_reg}",
            f"tau{args.infonce_tau}",
            f"neg{args.num_negatives}",
            f"s{args.seed}",
        ]
        return "+".join(tag)

    exp_id  = build_exp_id(args)
    exp_name = f"ragfeat_af_inv/{args.gnn_name}+ragF+af{args.af_beta}+inv{args.inv_reg}+tau{args.infonce_tau}+neg{args.num_negatives}+s{args.seed}"
    exp_dir = os.path.join('runs', args.dataset, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"[EXP] exp_dir = {exp_dir}")
    ####################################
    ##### Prepare and Load model #######
    ####################################
    controller = ModelTrainer(args, device=device, exp_dir=exp_dir)

    # === RAG 特征 ===
    rag_feat_t, comm_ids_t, num_comms = None, None, None
    if args.use_rag_feat:
        rag_feat_t, comm_ids_t, num_comms = _load_rag_features(
            dataset=args.dataset,
            tool2index=tool2index,
            index2tool=index2tool,
            rag_feat_path=args.rag_feat_path or None,
            rag_comm_path=args.rag_comm_path or None,
            rag_list_path=args.rag_list_path or None,
        )
        if hasattr(controller, "set_rag_features"):
            controller.set_rag_features(rag_feat=rag_feat_t, comm_ids=comm_ids_t, num_comms=num_comms)
        else:
            controller.rag_feat = rag_feat_t
            controller.comm_ids = comm_ids_t
            controller.num_comms = num_comms

        if rag_feat_t is not None:
            print(f"[RAG-Feat] rag_feat shape = {tuple(rag_feat_t.shape)}")
        if comm_ids_t is not None:
            print(f"[RAG-Feat] comm_ids shape = {tuple(comm_ids_t.shape)}, num_comms = {num_comms}")

    # === M3: 反频率权重表（边级别）===
    anti_freq_weights = None
    if args.anti_freq:
        anti_freq_weights = build_anti_freq_weights(adj_g, tool2index, beta=args.af_beta)
        print(f"[Anti-Freq] weights prepared for {len(anti_freq_weights)} edges.")

    # === 邻接 ===
    tool_adj = edge_index if args.gnn_name != 'SGC' else sgc_edge_index
    tool_adj = tool_adj.to(device)

    os.makedirs("ckpts", exist_ok=True)
    tag_rag = "_ragF" if args.use_rag_feat else ""
    save_path = f"ckpts/{args.dataset}_lm{'_tune' if not args.lm_frozen else '_frozen'}_{args.gnn_name}{tag_rag}_epoch{args.epoch}_batch{args.batch_size}_{'text' if args.text_negative else 'raw'}_neg.pt"

    # === 断点续训安全加载 ===
    def _safe_load(model, ckpt_dict):
        import math as _math
        model_dict = model.state_dict()
        copy_dict, skipped = {}, []

        if "rag_alpha" in ckpt_dict and "_rag_alpha_raw" in model_dict:
            cap = getattr(model, "_rag_alpha_cap", 0.6)
            with torch.no_grad():
                v = ckpt_dict["rag_alpha"]
                v_float = float(v.item() if torch.is_tensor(v) else v)
                v_float = max(1e-6, min(cap - 1e-6, v_float))
                raw = _math.log(v_float / (cap - v_float))
                copy_dict["_rag_alpha_raw"] = torch.tensor(raw, dtype=model_dict["_rag_alpha_raw"].dtype,
                                                           device=model_dict["_rag_alpha_raw"].device)

        for k, v in ckpt_dict.items():
            if k == "rag_alpha":
                continue
            if k in model_dict and hasattr(model_dict[k], "shape") and hasattr(v, "shape") and model_dict[k].shape == v.shape:
                copy_dict[k] = v
            else:
                shp_old = tuple(v.shape) if hasattr(v, "shape") else None
                shp_new = tuple(model_dict[k].shape) if (k in model_dict and hasattr(model_dict[k], "shape")) else None
                if shp_old is not None:
                    skipped.append((k, shp_old, shp_new))

        model_dict.update(copy_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"[CKPT] copied {len(copy_dict)} keys; skipped {len(skipped)} mismatched; missing {len(model_dict)-len(ckpt_dict)}")
        for k, shp_old, shp_new in skipped[:5]:
            print(f"[CKPT] skip: {k}: ckpt{shp_old} -> model{shp_new}")
        return len(copy_dict), len(skipped)

    did_load = False
    if os.path.exists(save_path) and args.load_model:
        ckpt = torch.load(save_path, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        _safe_load(controller.model, ckpt)
        did_load = True
        print(f"Load Pre-trained Model from {save_path} (filtered)")

    # 可选：冷启动 RAG 头
    if args.reset_rag_head and getattr(controller.model, "use_rag_feat", False):
        print("[INIT] reset_rag_head=1, re-init RAG head to small alpha.")
        controller.model.reset_rag_head()

    ####################################
    ############# Train / Eval #########
    ####################################
    if not args.evaluate_only:
        # 采样器（结构硬负例）
        communities = None
        if args.structure_hard_negative and (comm_ids_t is not None):
            comm_np = comm_ids_t.cpu().numpy()
            communities = {index2tool[i]: int(comm_np[i]) for i in range(len(index2tool))}

        if args.text_negative:
            with torch.no_grad():
                tool_emb_tmp = controller.model.tool_forward(tool_texts).detach().cpu().numpy()
            text_adj_g = build_tool_textual_sim_graph(tool_emb_tmp)
            sample_obj = TrainSampler(
                train_data, args.num_negatives, text_adj_g, tool2index,
                hard_negative=True, batch_size=args.batch_size,
                structure_hard_negative=bool(args.structure_hard_negative),
                communities=communities,
                anti_freq=bool(args.anti_freq),
                af_pow=float(args.af_beta),  
                neg_af_pow=0.3,                      
                neg_edge_pow=1.0,                    
                edge_weight_fn=(lambda a,b: anti_freq_weights.get((tool2index[a], tool2index[b]), 1.0))
                                if args.anti_freq else None,
                seed=args.seed
            )
        else:
            sample_obj = TrainSampler(
                train_data, args.num_negatives, adj_g, tool2index,
                hard_negative=True, batch_size=args.batch_size,
                structure_hard_negative=bool(args.structure_hard_negative),
                communities=communities,
                anti_freq=bool(args.anti_freq),
                af_pow=float(args.af_beta),
                neg_af_pow=0.3, 
                neg_edge_pow=1.0,
                edge_weight_fn=(lambda a,b: anti_freq_weights.get((tool2index[a], tool2index[b]), 1.0))
                                if args.anti_freq else None,
                seed=args.seed
            )

        best_model, total_time = controller.train(
            tool_texts, tool_adj, sample_obj,
            rag_feat=rag_feat_t, comm_ids=comm_ids_t, num_comms=num_comms,
            anti_freq=bool(args.anti_freq),
            inv_reg_lambda=float(args.inv_reg),
            inv_every=int(args.inv_every)
        )
        controller.model = best_model
        if args.save_model:
            save_checkpoint(best_model, save_path)
        print(f"\nFinish Training, Overall time {total_time:.3f}s")
    else:
        controller.model.eval()

    if args.gnn_name == 'SGC':
        alpha = controller.model.gnn_model.alpha
        print(alpha)

    ####################################
    ############# Model Test ###########
    ####################################
    with torch.no_grad():
        # GNN + RAG
        tool_emb = controller.model.tool_forward(
            tool_texts, tool_adj,
            rag_feat=getattr(controller, "rag_feat", None),
            comm_ids=getattr(controller, "comm_ids", None)
        ).detach().cpu().numpy()
        # RAG-only（no-GNN）
        tool_emb_static = controller.model.tool_forward(
            tool_texts, tool_adj=None,
            rag_feat=getattr(controller, "rag_feat", None),
            comm_ids=getattr(controller, "comm_ids", None)
        ).detach().cpu().numpy()

        def _cosine_topk(v: np.ndarray, M: np.ndarray, k: int, min_score: float = 0.0) -> np.ndarray:
            vn = v / (np.linalg.norm(v) + 1e-8)
            Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
            s  = Mn @ vn  # (N,)
            # 先挑出一个稍大的候选集合，再按分数降序+阈值裁剪
            kk  = min(len(s), max(k, k * 4))
            idx = np.argpartition(-s, kth=kk - 1)[:kk]
            idx = idx[np.argsort(-s[idx])]  # 降序
            if min_score > 0.0:
                idx = idx[s[idx] >= float(min_score)]
            return idx[:k]


    # 输出目录：如果启用 M3（anti_freq / inv_reg），写到 ragfeat_af_inv
    if args.anti_freq or args.inv_reg > 0:
        out_dir = f"runs/{args.dataset}/ragfeat_af_inv"
    else:
        out_dir = f"runs/{args.dataset}/ragfeat" if args.use_rag_feat else f"runs/{args.dataset}/lmgnn"
    os.makedirs(out_dir, exist_ok=True)

    table = pt.PrettyTable()
    table.field_names = ["Dataset", "LLM", "LM", "GNN", "N-F1", "L-F1", "Accuracy"]
    lm_name = args.lm_name.split('/')[-1]
    candidate_example_ids = set()
    log_fp = open_jsonl(os.path.join(out_dir, f"{args.gnn_name}.jsonl"))

    # 评测增强：结构分桶 & 幻觉率打印
    def _print_buckets_and_hallu(tag: str, ids: List[str], pred_dict: Dict):
        comm_np = comm_ids_t.cpu().numpy() if comm_ids_t is not None else None
        buckets = _eval_buckets(ids, pred_dict, adj_g, index2tool, comm_np)
        precs = []
        for sid in ids:
            C = pred_dict[sid]
            p = C["search_nodes"]; g = C["gt_nodes"]
            pr, _ = _precision_recall(p, g)
            precs.append(pr)
        halluc_rate = round(1.0 - float(sum(precs)/max(1, len(precs))), 4)
        if buckets:
            print(f"[Buckets][{tag}] long_path={buckets.get('long_path')} | high_branch={buckets.get('high_branch')} | cross_comm={buckets.get('cross_comm')}")
        print(f"[Hallucination~][{tag}] node hallucination rate ≈ {halluc_rate}")

    # GT 跨社区 sanity

    # --- helper: 将各种形态的 gt_links 统一规整为 [(u, v), ...] 二元边 ---
    def _edge_pairs_from_any(gt_links):
        import torch
        pairs = []

        if gt_links is None:
            return pairs

        # torch.Tensor 形态（可能是 [2, E] 的 edge_index，或 [E, 2/3]）
        if isinstance(gt_links, torch.Tensor):
            if gt_links.dim() == 2:
                # edge_index: [2, E]
                if gt_links.size(0) == 2:
                    u = gt_links[0].detach().cpu().tolist()
                    v = gt_links[1].detach().cpu().tolist()
                    return list(zip(u, v))
                # [E, 2] 或 [E, 3]
                if gt_links.size(1) >= 2:
                    for row in gt_links.detach().cpu().tolist():
                        pairs.append((int(row[0]), int(row[1])))
                    return pairs

        # list/iterable 形态
        try:
            for e in gt_links:
                # (u, v) 或 (u, v, attr)
                if isinstance(e, (list, tuple)):
                    if len(e) >= 2:
                        pairs.append((int(e[0]), int(e[1])))
                    continue
                # dict 形态：尝试常见键
                if isinstance(e, dict):
                    a = e.get('u', e.get('src', e.get('from', None)))
                    b = e.get('v', e.get('dst', e.get('to', None)))
                    if a is None or b is None:
                        # 退而求其次：取前两个 value
                        vals = list(e.values())
                        if len(vals) >= 2:
                            a, b = vals[0], vals[1]
                    if a is not None and b is not None:
                        pairs.append((int(a), int(b)))
                    continue
                # 其它类型：忽略
        except TypeError:
            # 非可迭代类型，忽略
            pass

        return pairs


    def _print_gt_cross_comm(pred_content_dict, ids):
        comm_np = comm_ids_t.cpu().numpy() if comm_ids_t is not None else None
        if comm_np is None:
            print("[Sanity] No community ids.")
            return
        name2comm = {index2tool[i]: int(comm_np[i]) for i in range(len(index2tool))}
        edges, cross = 0, 0
        for sid in ids:
            gt_links = pred_content_dict[sid]["gt_task_links"]
            if isinstance(gt_links, list) and all(isinstance(x, str) and "->" in x for x in gt_links):
                pairs = [tuple(x.split("->")) for x in gt_links]
            else:
                pairs = _edge_pairs_from_any(gt_links)
            for a, b in pairs:
                if a in name2comm and b in name2comm:
                    edges += 1
                    if name2comm[a] != name2comm[b]:
                        cross += 1
        print(f"[Sanity] GT cross-community edges: {cross}/{edges}")

    base_llms = ['CodeLlama-13b']
    for base_llm in base_llms:
        methods = ['direct']
        for method in methods:
            new_alignment_ids, pred_content_dict = load_test_data(args.dataset, base_llm, alignment_ids, method=method)
            _print_gt_cross_comm(pred_content_dict, new_alignment_ids)

            # ===== 1) 基线 direct (+GNN+ragF) =====
            final_pred_dict = {}
            for data_id in new_alignment_ids:
                steps = pred_content_dict[data_id]["steps"]
                st_time = time.time()
                steps_emb = controller.model.lm_forward(steps, max_length=64, batch_size=len(steps)+1).detach().cpu().numpy()
                ans = sequence_greedy_tool_selection(steps_emb, tool_emb, index2tool, adj_g, measure=args.measure)
                elapsed = round(time.time() - st_time, 4)

                base_nodes = pred_content_dict[data_id]["pred_task_nodes"]
                base_links = pred_content_dict[data_id]["pred_task_links"]
                gt_nodes = pred_content_dict[data_id]["gt_task_nodes"]
                gt_links = pred_content_dict[data_id]["gt_task_links"]

                final_pred_dict[data_id] = {
                    "steps": steps,
                    "pred_nodes": base_nodes,
                    "pred_links": base_links,
                    "search_nodes": ans["task_nodes"],
                    "search_links": ans["task_links"],
                    "gt_nodes": gt_nodes,
                    "gt_links": gt_links
                }

                succ = bool(f1_score(ans["task_nodes"], gt_nodes) >= 0.99)
                log_sample(
                    fp=log_fp, ds=args.dataset, method=("lmgnn+ragF" if args.use_rag_feat else "lmgnn"),
                    llm=base_llm, seed=args.seed,
                    sid=data_id, success=succ, latency_sec=elapsed,
                    tokens_prompt=None, tokens_completion=None,
                    temperature=None, top_p=None, alpha=None, graphsearch=None,
                    task_steps=steps, task_nodes=ans["task_nodes"], task_links=ans["task_links"],
                    err=None
                )

            score_dict, cur_candidates = bi_evaluate(new_alignment_ids, final_pred_dict)
            candidate_example_ids = set(cur_candidates) if not candidate_example_ids else (candidate_example_ids & set(cur_candidates))
            table.add_row([args.dataset, base_llm, lm_name, 'direct',
                        score_dict['base-node-f1'], score_dict['base-link-f1'], score_dict["base-acc"]])

            # ===== 2) +ragF (no-GNN) =====
            final_pred_dict_static = {}
            for data_id in new_alignment_ids:
                steps = pred_content_dict[data_id]["steps"]
                st_time = time.time()
                steps_emb = controller.model.lm_forward(steps, max_length=64, batch_size=len(steps)+1).detach().cpu().numpy()
                ans = sequence_greedy_tool_selection(steps_emb, tool_emb_static, index2tool, adj_g, measure=args.measure)
                elapsed = round(time.time() - st_time, 4)

                base_nodes = pred_content_dict[data_id]["pred_task_nodes"]
                base_links = pred_content_dict[data_id]["pred_task_links"]
                gt_nodes = pred_content_dict[data_id]["gt_task_nodes"]
                gt_links = pred_content_dict[data_id]["gt_task_links"]

                final_pred_dict_static[data_id] = {
                    "steps": steps,
                    "pred_nodes": base_nodes,
                    "pred_links": base_links,
                    "search_nodes": ans["task_nodes"],
                    "search_links": ans["task_links"],
                    "gt_nodes": gt_nodes,
                    "gt_links": gt_links
                }

            score_dict_static, _ = bi_evaluate(new_alignment_ids, final_pred_dict_static)
            table.add_row([args.dataset, base_llm, lm_name, '+ragF (no-GNN)',
                        score_dict_static['search-node-f1'], score_dict_static['search-link-f1'], score_dict_static["search-acc"]])
            _print_buckets_and_hallu("+ragF (no-GNN)", new_alignment_ids, final_pred_dict_static)

            # ===== 3) +RAG(cands)（可选）=====
            if args.rag_candidates_topk:
                final_pred_dict_cand = {}
                for data_id in new_alignment_ids:
                    steps = pred_content_dict[data_id]["steps"]
                    steps_emb = controller.model.lm_forward(steps, max_length=64, batch_size=len(steps)+1).detach().cpu().numpy()
                    q = steps_emb.mean(axis=0, keepdims=False)
                    cand_bank = tool_emb_static  # 用融合后的 RAG(no-GNN) 作为检索库
                    top_idx = _cosine_topk(
                        q, cand_bank, k=int(args.rag_candidates_topk),
                        min_score=float(getattr(args, "min_rag_score", 0.0))
                    )
                    top_idx = np.unique(top_idx)
                    index2tool_sub = [index2tool[i] for i in top_idx]
                    idx_map = {name: i for i, name in enumerate(index2tool_sub)}
                    tool_emb_sub = tool_emb_static[top_idx]
                    adj_sub = {name: [n for n in adj_g.get(name, []) if n in idx_map] for name in index2tool_sub}
                    ans = sequence_greedy_tool_selection(steps_emb, tool_emb_sub, index2tool_sub, adj_sub, measure=args.measure)

                    base_nodes = pred_content_dict[data_id]["pred_task_nodes"]
                    base_links = pred_content_dict[data_id]["pred_task_links"]
                    gt_nodes   = pred_content_dict[data_id]["gt_task_nodes"]
                    gt_links   = pred_content_dict[data_id]["gt_task_links"]

                    final_pred_dict_cand[data_id] = {
                        "steps": steps,
                        "pred_nodes": base_nodes,
                        "pred_links": base_links,
                        "search_nodes": ans["task_nodes"],
                        "search_links": ans["task_links"],
                        "gt_nodes": gt_nodes,
                        "gt_links": gt_links
                    }

                score_dict_cand, _ = bi_evaluate(new_alignment_ids, final_pred_dict_cand)
                table.add_row([args.dataset, base_llm, lm_name, '+RAG(cands)',
                            score_dict_cand['search-node-f1'], score_dict_cand['search-link-f1'], score_dict_cand["search-acc"]])
                _print_buckets_and_hallu("+RAG(cands)", new_alignment_ids, final_pred_dict_cand)

            # ===== 4) +GCN+ragF（完整）=====
            gnn_tag = f"+{args.gnn_name}{'+ragF' if args.use_rag_feat else ''}"
            table.add_row([args.dataset, base_llm, lm_name, gnn_tag,
                        score_dict['search-node-f1'], score_dict['search-link-f1'], score_dict["search-acc"]])
            _print_buckets_and_hallu(gnn_tag, new_alignment_ids, final_pred_dict)

            # canon_names = list(index2tool)
            # ===== 5) （可选）同构鲁棒性评测（新实现） =====
            if args.robust_iso:
                print("[Robust-Iso] start ...")
                import numpy as np
                import torch

                nf1_list, lf1_list = [], []

                for _trial in range(int(args.robust_iso_trials)):
                    # 1) 对工具图做一次随机置换（节点重标 + 邻接、RAG特征一起置换）
                    perm, inv_perm, texts_p, adj_p, feat_p, comm_p = permute_graph_inputs(
                        tool_texts,
                        tool_adj,  # 注意：这里用的是 edge_index（已经在前面 .to(device) 了）
                        getattr(controller, "rag_feat", None),
                        getattr(controller, "comm_ids", None),
                        device,
                    )

                    # 2) 在置换图上前向一遍，再把输出对齐回原始顺序
                    with torch.no_grad():
                        tool_emb_p = controller.model.tool_forward(
                            texts_p,
                            adj_p,
                            rag_feat=feat_p,
                            comm_ids=comm_p,
                        )  # [N, D]，索引顺序 = perm 后的新顺序
                        # 映回原始顺序：第 i 行对应原始第 i 个工具
                        tool_emb_iso = tool_emb_p[inv_perm].detach().cpu().numpy()

                    # 3) 用“同一套 index2tool / adj_g / eval 逻辑”做检索，只换了 embedding
                    pred_iso = {}
                    for data_id in new_alignment_ids:
                        steps = pred_content_dict[data_id]["steps"]
                        steps_emb = controller.model.lm_forward(
                            steps, max_length=64, batch_size=len(steps) + 1, device=device
                        ).detach().cpu().numpy()

                        ans = sequence_greedy_tool_selection(
                            steps_emb,
                            tool_emb_iso,
                            index2tool,   # 保持原始工具名顺序
                            adj_g,        # 保持原始邻接（按名字）
                            measure=args.measure,
                        )

                        base_nodes = pred_content_dict[data_id]["pred_task_nodes"]
                        base_links = pred_content_dict[data_id]["pred_task_links"]
                        gt_nodes   = pred_content_dict[data_id]["gt_task_nodes"]
                        gt_links   = pred_content_dict[data_id]["gt_task_links"]

                        pred_iso[data_id] = {
                            "steps": steps,
                            "pred_nodes": base_nodes,
                            "pred_links": base_links,
                            "search_nodes": ans["task_nodes"],
                            "search_links": ans["task_links"],
                            "gt_nodes": gt_nodes,
                            "gt_links": gt_links,
                        }

                    # 4) 评测 F1
                    s_iso, _ = bi_evaluate(new_alignment_ids, pred_iso)
                    nf1_list.append(s_iso["search-node-f1"])
                    lf1_list.append(s_iso["search-link-f1"])

                print(
                    f"[Robust-Iso] Node-F1 avg={np.mean(nf1_list):.4f} ± {np.std(nf1_list):.4f} | "
                    f"Link-F1 avg={np.mean(lf1_list):.4f} ± {np.std(lf1_list):.4f}"
                )
                if len(new_alignment_ids) > 0:
                    did = new_alignment_ids[0]
                    print(
                        "[DEBUG] example overlap:",
                        set(pred_iso[did]["search_nodes"]) & set(pred_iso[did]["gt_nodes"]),
                    )





    log_fp.close()
    print(table)

    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")