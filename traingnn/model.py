# model.py
import sys
import time
import copy
import math
import os
import json
from typing import Optional, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from gnn import SGC, GNNEncoder

sys.path.append("../")
from utils import TextDataset, init_random_state
from sampler import TrainSampler


class LMGNNModel(nn.Module):
    """
    LM + （可选）RAG结构特征门控残差融合 + GNN
    融合：fused = LN(tool_lm) + alpha * sigmoid(Wg * LN(feat)) * tanh(Wp * Drop(LN(feat)))
    其中 feat = [rag_feat, comm_emb[comm_ids]]
    """
    def __init__(self, args, emb_dim: int = 1024):
        super().__init__()
        self.args = args
        self.lm_name = args.lm_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.lm_name)
        self.lm_model = AutoModel.from_pretrained(self.lm_name)

        # ===== GNN部分 =====
        self.gnn_name = args.gnn_name
        if self.gnn_name == "SGC":
            gnn = SGC()
        else:
            gnn = GNNEncoder(
                emb_dim,
                hidden_dim=getattr(args, "gnn_hidden_dim", 512),
                output_dim=emb_dim,
                gnn_type=self.gnn_name,
                n_layers=getattr(args, "gnn_layer", 2),
            )
        self.gnn_model = gnn

        # ===== LM 冻结（可选）=====
        if getattr(args, "lm_frozen", False):
            for _, p in self.lm_model.named_parameters():
                p.requires_grad = False

        # ===== RAG 特征相关（懒初始化）=====
        self.use_rag_feat: bool = getattr(args, "use_rag_feat", False)
        self.comm_emb_dim: int = getattr(args, "comm_emb_dim", 16)
        self._emb_dim = emb_dim

        # lazy modules
        self._rag_inited: bool = False
        self.comm_emb: Optional[nn.Embedding] = None
        self.rag_ln_tool: Optional[nn.LayerNorm] = None
        self.rag_ln_feat: Optional[nn.LayerNorm] = None
        self.rag_proj: Optional[nn.Linear] = None
        self.rag_gate: Optional[nn.Linear] = None
        self.rag_dropout = nn.Dropout(getattr(args, "rag_dropout", 0.1))
        self._gate_mean_current: Optional[torch.Tensor] = None  # New: 当前前向批次的 gate 均值（参与反传）

        # alpha 采用 reparam：alpha = cap * sigmoid(_rag_alpha_raw)
        self._rag_alpha_cap: float = 0.6
        self._rag_alpha_raw = nn.Parameter(torch.tensor(-1.38629436))  # ≈ sigmoid^-1(0.2/0.6)

        # 训练时注入的 rag_feat/comm_ids（由 Trainer 设置）
        self.rag_feat: Optional[torch.Tensor] = None  # (N,F)
        self.comm_ids: Optional[torch.Tensor] = None  # (N,)

        # ===== 工具文本 LM 表征缓存（在 LM 冻结时启用）=====
        self._tool_cache_sig: Optional[Tuple[int, int]] = None  # (len, hash)
        self._tool_cache_lm_cpu: Optional[torch.Tensor] = None  # 缓存在 CPU 的 (N, H)
        self._tool_cache_maxlen: int = 128  # 与 tool_forward 中保持一致
        self._tool_cache_batch: int = 64    # 与 tool_forward 中保持一致

        # 运行时强制关闭 RAG 融合（用于自检）
        self._force_no_rag_fuse: bool = False

    # ---------- alpha 工具 ----------
    def rag_alpha(self) -> torch.Tensor:
        return self._rag_alpha_cap * torch.sigmoid(self._rag_alpha_raw)

    @torch.no_grad()
    def set_rag_alpha_value(self, alpha_value: float):
        alpha_value = float(max(1e-6, min(self._rag_alpha_cap - 1e-6, alpha_value)))
        raw = math.log(alpha_value / (self._rag_alpha_cap - alpha_value))
        self._rag_alpha_raw.copy_(torch.tensor(raw, dtype=self._rag_alpha_raw.dtype, device=self._rag_alpha_raw.device))

    # ---------- 可重置 RAG 头（权重重置，shape 不变）----------
    @torch.no_grad()
    def reset_rag_head(self):
        if self.rag_ln_feat is not None:
            nn.init.ones_(self.rag_ln_feat.weight)
            nn.init.zeros_(self.rag_ln_feat.bias)
        if self.rag_ln_tool is not None:
            nn.init.ones_(self.rag_ln_tool.weight)
            nn.init.zeros_(self.rag_ln_tool.bias)
        if self.rag_proj is not None:
            nn.init.xavier_uniform_(self.rag_proj.weight)
            if self.rag_proj.bias is not None:
                nn.init.zeros_(self.rag_proj.bias)
        if self.rag_gate is not None:
            nn.init.xavier_uniform_(self.rag_gate.weight)
            if self.rag_gate.bias is not None:
                nn.init.zeros_(self.rag_gate.bias)
        if self.comm_emb is not None:
            nn.init.normal_(self.comm_emb.weight, std=0.02)
        # alpha 从很小开始
        self.set_rag_alpha_value(0.05)

    # ---------- 工具文本缓存辅助 ----------
    @staticmethod
    def _make_text_sig(texts: list) -> Tuple[int, int]:
        h = 0
        for s in texts:
            h = (h * 1315423911 + hash(s)) & 0xFFFFFFFF
        return (len(texts), h)

    def _can_cache_tool_lm(self) -> bool:
        return not any(p.requires_grad for p in self.lm_model.parameters())

    # ---------- 文本编码 ----------
    def lm_forward(self, plain_text, max_length=128, batch_size=256, device="cuda:0"):
        x = self.tokenizer(plain_text, padding=True, truncation=True, max_length=max_length)
        format_data = TextDataset(x)
        text_emb = None
        dataloader = DataLoader(format_data, shuffle=False, batch_size=batch_size)

        lm_frozen = not any(p.requires_grad for p in self.lm_model.parameters())
        if lm_frozen:
            self.lm_model.eval()

        ctx = torch.no_grad() if lm_frozen else torch.enable_grad()
        with ctx:
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = self.lm_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True
                )
                emb = output['hidden_states'][-1]          # (B, L, H)
                cls_token_emb = emb.permute(1, 0, 2)[0]    # (B, H)
                text_emb = cls_token_emb if text_emb is None else torch.vstack((text_emb, cls_token_emb))
        return text_emb

    # ---------- RAG 模块懒构建 ----------
    def _ensure_rag_modules(self, rag_feat: Optional[torch.Tensor], comm_ids: Optional[torch.Tensor]):
        if self._rag_inited:
            return
        if not self.use_rag_feat:
            self._rag_inited = True
            return

        device = next(self.parameters()).device
        base_dim = getattr(self.lm_model.config, "hidden_size", self._emb_dim)

        rag_dim = 0
        if rag_feat is not None:
            rag_dim += rag_feat.shape[1]

        if comm_ids is not None and comm_ids.numel() > 0:
            num_comms = int(comm_ids.max().item()) + 1
            if num_comms > 0:
                self.comm_emb = nn.Embedding(num_comms, self.comm_emb_dim).to(device)
                rag_dim += self.comm_emb.embedding_dim

        if rag_dim > 0:
            self.rag_ln_tool = nn.LayerNorm(base_dim).to(device)
            self.rag_ln_feat = nn.LayerNorm(rag_dim).to(device)
            self.rag_proj = nn.Linear(rag_dim, base_dim).to(device)
            self.rag_gate = nn.Linear(rag_dim, base_dim).to(device)

        self._rag_inited = True

    # ---------- 融合：门控残差 ----------
    def _fuse_tool_features(
        self,
        tool_emb: torch.Tensor,
        rag_feat: Optional[torch.Tensor] = None,
        comm_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self._force_no_rag_fuse:
            return tool_emb
        if (rag_feat is None) and (comm_ids is None):
            return tool_emb

        self._ensure_rag_modules(rag_feat, comm_ids)
        if self.rag_ln_tool is None or self.rag_ln_feat is None:
            return tool_emb

        device = tool_emb.device
        parts = []
        if rag_feat is not None:
            parts.append(rag_feat.to(device, non_blocking=True).float())
        if comm_ids is not None and self.comm_emb is not None:
            parts.append(self.comm_emb(comm_ids.to(device, non_blocking=True)))
        if not parts:
            return tool_emb

        feat = torch.cat(parts, dim=1)              # (N, rag_dim)
        t = self.rag_ln_tool(tool_emb)              # (N, H)
        f = self.rag_ln_feat(feat)                  # (N, rag_dim)
        f = self.rag_dropout(f)

        delta = torch.tanh(self.rag_proj(f))        # (N, H)
        gate  = torch.sigmoid(self.rag_gate(f))     # (N, H)
        gate  = torch.clamp(gate, 1e-3, 1.0 - 1e-3)  # New: 数值稳健，避免饱和
        self._gate_mean_current = gate.mean()         # New: 记录本次前向的 gate 均值（保留梯度）


        fused = t + self.rag_alpha() * gate * delta
        return fused

    # ---------- 工具 LM 表征（带缓存） ----------
    def _tool_lm_cached(self, tool_text: list, device: str):
        if self._can_cache_tool_lm():
            sig = self._make_text_sig(tool_text)
            if self._tool_cache_sig == sig and self._tool_cache_lm_cpu is not None:
                return self._tool_cache_lm_cpu.to(device, non_blocking=True)

            emb = self.lm_forward(
                tool_text, max_length=self._tool_cache_maxlen, batch_size=self._tool_cache_batch, device=device
            )
            self._tool_cache_lm_cpu = emb.detach().cpu()
            self._tool_cache_sig = sig
            return emb
        else:
            return self.lm_forward(tool_text, max_length=128, batch_size=64, device=device)

    # ---------- 工具编码 + （可选）融合 + （可选）GNN ----------
    def tool_forward(self, tool_text, tool_adj=None, rag_feat=None, comm_ids=None):
        device = next(self.parameters()).device
        init_tool_emb = self._tool_lm_cached(tool_text, device=device)

        if self.use_rag_feat and (rag_feat is not None or comm_ids is not None):
            self._ensure_rag_modules(rag_feat, comm_ids)
            init_tool_emb = self._fuse_tool_features(init_tool_emb, rag_feat=rag_feat, comm_ids=comm_ids)

        if tool_adj is None:
            return init_tool_emb
        tool_emb = self.gnn_model(init_tool_emb, tool_adj)
        return tool_emb

    # ---------- 推理 ----------
    def inference(
        self,
        step_text,
        tool_text,
        tool_adj,
        static: bool = False,
        rag_feat: Optional[torch.Tensor] = None,
        comm_ids: Optional[torch.Tensor] = None,
    ):
        tool_x = self.tool_forward(
            tool_text,
            tool_adj=None if static else tool_adj,
            rag_feat=rag_feat,
            comm_ids=comm_ids,
        )
        bs = 32 if not isinstance(step_text, (list, tuple)) else max(1, min(32, len(step_text)))
        device = next(self.parameters()).device
        step_x = self.lm_forward(step_text, max_length=64, batch_size=bs, device=device)
        return tool_x, step_x

    # ---------- 训练前向（支持加权BPR，兼容多负样本） ----------
    def forward(
        self,
        step_text,
        tool_text,
        tool_adj,
        pos_tools,
        neg_tools,
        rag_feat: Optional[torch.Tensor] = None,
        comm_ids: Optional[torch.Tensor] = None,
        pair_weight: Optional[torch.Tensor] = None,
    ):
        device = next(self.parameters()).device

        # 1) 工具侧编码（可含 RAG + GNN）
        updated_tool_x = self.tool_forward(
            tool_text, tool_adj,
            rag_feat=rag_feat, comm_ids=comm_ids
        )  # [N_tools, H]

        # 2) 步骤侧编码（LM）
        bs = 32 if not isinstance(step_text, (list, tuple)) else max(1, min(32, len(step_text)))
        step_x = self.lm_forward(step_text, max_length=64, batch_size=bs, device=device)  # [B,H]

        # 3) 取正样
        pos_tool_emb = updated_tool_x[pos_tools, :]  # [B,H]

        # 4) 统一构造 logits / scores
        #    - 若 InfoNCE：每行 logits = [pos_score, neg_scores...]
        #    - 若 BPR：保留 pos/neg 两两比较
        if neg_tools.dim() == 2:
            # [B,K,H]
            neg_tool_emb = updated_tool_x[neg_tools, :]
            # 点积
            pos_score = (step_x * pos_tool_emb).sum(dim=-1, keepdim=True)   # [B,1]
            neg_score = (step_x.unsqueeze(1) * neg_tool_emb).sum(dim=-1)    # [B,K]
            if getattr(self, "use_infonce", False):
                # InfoNCE: logits=[pos|neg] / tau, label=0
                tau = float(getattr(self, "infonce_tau", 0.07))
                logits = torch.cat([pos_score, neg_score], dim=1) / max(1e-8, tau)  # [B,1+K]
                target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # 正样索引=0
                ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")      # [B]
                if pair_weight is not None:
                    pw = pair_weight.to(ce.device, dtype=ce.dtype)
                    loss = (ce * pw).mean()
                else:
                    loss = ce.mean()
                return loss
            else:
                # BPR: softplus(neg - pos) 平均
                bpr_vec = torch.nn.functional.softplus(neg_score - pos_score).mean(dim=1)  # [B]
                if pair_weight is not None:
                    pw = pair_weight.to(bpr_vec.device, dtype=bpr_vec.dtype)
                    loss = (bpr_vec * pw).mean()
                else:
                    loss = bpr_vec.mean()
                return loss
        else:
            # [B,H]
            neg_tool_emb = updated_tool_x[neg_tools, :]
            pos_score = (step_x * pos_tool_emb).sum(dim=-1)  # [B]
            neg_score = (step_x * neg_tool_emb).sum(dim=-1)  # [B]
            if getattr(self, "use_infonce", False):
                # 将单负样退化为 K=1 的 InfoNCE
                tau = float(getattr(self, "infonce_tau", 0.07))
                logits = torch.stack([pos_score, neg_score], dim=1) / max(1e-8, tau)  # [B,2]
                target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
                ce = torch.nn.functional.cross_entropy(logits, target, reduction="none")
                if pair_weight is not None:
                    pw = pair_weight.to(ce.device, dtype=ce.dtype)
                    loss = (ce * pw).mean()
                else:
                    loss = ce.mean()
                return loss
            else:
                bpr_vec = torch.nn.functional.softplus(neg_score - pos_score)  # [B]
                if pair_weight is not None:
                    pw = pair_weight.to(bpr_vec.device, dtype=bpr_vec.dtype)
                    loss = (bpr_vec * pw).mean()
                else:
                    loss = bpr_vec.mean()
                return loss



class ModelTrainer:
    def __init__(self, args, device, exp_dir=None):
        self.seed = args.seed
        self.device = device
        self.epoch = args.epoch
        self.patience = args.patience
        self.text_negative = getattr(args, "text_negative", False)
        self.args = args
        self.global_step = 0  # 用于 inv_every 触发
        self.exp_dir = exp_dir or "."
        os.makedirs(self.exp_dir, exist_ok=True)
        # === 新增：每epoch的 JSONL 记录路径 ===
        self._epoch_log_path = os.path.join(self.exp_dir, "metrics_epoch.jsonl")

        init_random_state(args.seed)

        self.model = LMGNNModel(args).to(device)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.model.use_infonce = bool(getattr(args, "use_infonce", 0))
        self.model.infonce_tau = float(getattr(args, "infonce_tau", 0.07))

        # ===== 分组学习率：初始分组（后续 _build_optimizer 会重建）=====
        base_params, rag_params = [], []
        rag_names = ("comm_emb", "rag_ln_tool", "rag_ln_feat", "rag_proj", "rag_gate", "_rag_alpha_raw")
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.split(".")[0] in rag_names or any(name.startswith(n) for n in rag_names):
                rag_params.append(p)
            else:
                base_params.append(p)

        rag_lr = float(getattr(args, "rag_lr", 1e-3))
        rag_wd = 1e-4

        self.optimizer = torch.optim.Adam(
            [
                {"params": base_params, "lr": args.lr, "weight_decay": 0.0},
                {"params": rag_params,  "lr": rag_lr,  "weight_decay": rag_wd},
            ]
        )
        self._rag_params = rag_params
        self.grad_clip = 1.0
        self._build_scheduler()

        # 简单 scheduler：仅 rag 组
        self.total_epochs = max(1, int(self.epoch))
        warmup = max(1, int(0.2 * self.total_epochs))

        def _cos_warm(epoch):
            if epoch < warmup:
                return float(epoch + 1) / float(warmup)
            t = (epoch - warmup) / max(1, self.total_epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * t))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=[lambda _: 1.0, _cos_warm]
        )

        self.eval_steps, self.eval_loader = None, None
        print(f"[Model] LM_GNN Number of parameters: {trainable_params}")

    def _build_scheduler(self):
        """为当前 optimizer 构建 warmup + cosine 的 LR scheduler（只作用于 RAG 组）。"""
        self.total_epochs = max(1, int(self.epoch))
        warmup = max(1, int(0.2 * self.total_epochs))  # 前 20% epoch warmup

        def _cos_warm(epoch):
            if epoch < warmup:
                return float(epoch + 1) / float(warmup)
            t = (epoch - warmup) / max(1, self.total_epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * t))

        # 两个 param_group 各自的 lr_lambda：base 恒 1.0；rag 走 warmup+cos
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=[lambda _: 1.0, _cos_warm]
        )

    def _build_optimizer(self):
        """按当前模型结构重建 Adam；RAG 懒初始化后调用。"""
        rag_modules = (
            self.model.comm_emb,
            self.model.rag_ln_tool,
            self.model.rag_ln_feat,
            self.model.rag_proj,
            self.model.rag_gate,
            self.model._rag_alpha_raw,
        )
        rag_param_ids = set()
        for m in rag_modules:
            if m is None:
                continue
            if isinstance(m, nn.Parameter):
                rag_param_ids.add(id(m))
            elif hasattr(m, "parameters"):
                for p in m.parameters(recurse=True):
                    rag_param_ids.add(id(p))

        base_params, rag_params = [], []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            (rag_params if id(p) in rag_param_ids else base_params).append(p)

        if not rag_params:
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if any(k in name for k in ["comm_emb", "rag_ln_tool", "rag_ln_feat", "rag_proj", "rag_gate", "_rag_alpha_raw"]):
                    if p not in rag_params:
                        rag_params.append(p)
                        if p in base_params:
                            base_params.remove(p)

        rag_lr = float(getattr(self.args, "rag_lr", 1e-3))
        self.optimizer = torch.optim.Adam(
            [
                {"params": base_params, "lr": float(self.args.lr), "weight_decay": 0.0},
                {"params": rag_params,  "lr": rag_lr,               "weight_decay": 0.0},
            ]
        )
        self._rag_params = rag_params
        print(f"[OPT] base_params={sum(p.numel() for p in base_params)} lr={self.args.lr}")
        print(f"[OPT]  rag_params={sum(p.numel() for p in rag_params)} lr={rag_lr}")
        self._build_scheduler()

    # --- 统一解包：同时兼容 dict 批（新sampler）和旧的 tuple 批 ---
    def _unpack_batch(self, batch):
        """
        返回: step_idx(LongTensor[B]), pos_ids(LongTensor[B]), neg_ids(LongTensor[B]), pair_weight(FloatTensor[B] or None)
        """
        if isinstance(batch, dict):
            step_idx = batch["step_idx"].to(self.device).long()
            pos_ids  = batch["pos_ids"].to(self.device).long()
            neg_ids  = batch["neg_ids"].to(self.device).long()
            pair_w   = batch.get("pair_weight", None)
            if pair_w is not None:
                pair_w = pair_w.to(self.device).float()
            return step_idx, pos_ids, neg_ids, pair_w

        # 兼容旧版: DataLoader 返回 (Tensor[B, 3],)
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            x = batch[0].to(self.device)
            # 旧版格式: [step_idx, pos_id, neg_id]
            step_idx = x[:, 0].long()
            pos_ids  = x[:, 1].long()
            neg_ids  = x[:, 2].long()
            pair_w   = None
            # 若你曾把权重并到第四列，这里也兼容一下：
            if x.size(1) > 3:
                pair_w = x[:, 3].float()
            return step_idx, pos_ids, neg_ids, pair_w

    def _quantiles_10_50_90(self, x: torch.Tensor):
        """返回张量 x 的 q10 / q50 / q90（float）。"""
        qs = torch.quantile(x, torch.tensor([0.10, 0.50, 0.90], device=x.device))
        return float(qs[0]), float(qs[1]), float(qs[2])

    # ---- 外部注入 RAG 特征 ----
    def set_rag_features(self, rag_feat=None, comm_ids=None, num_comms=None):
        self.rag_feat = rag_feat.to(self.device) if rag_feat is not None else None
        self.comm_ids = comm_ids.to(self.device) if comm_ids is not None else None
        self.num_comms = int(num_comms) if num_comms is not None else None

        self.model._ensure_rag_modules(self.rag_feat, self.comm_ids)
        self.model.rag_feat = self.rag_feat
        self.model.comm_ids = self.comm_ids
        self._build_optimizer()

    def train_one_epoch(
        self, tool_text, tool_adj, train_loader, train_step_texts,
        anti_freq: bool = False, inv_reg_lambda: float = 0.0, inv_every: int = 10
    ):
        """
        返回: (epoch_loss, extras_dict)
          extras_dict: {"pw_q10": float|None, "pw_q50": float|None, "pw_q90": float|None}
        """
        total_loss = 0.0
        self.model.train()

        # === 新增：记录每个 batch 的 pair_weight 分位数，用于 epoch 汇总 ===
        pw_quantiles_across_batches = []  # [[q10,q50,q90], ...]

        for batch_idx, batch in enumerate(train_loader):
            self.optimizer.zero_grad()

            step_idx, pos_ids, neg_ids, pair_weight = self._unpack_batch(batch)
            step_texts = [train_step_texts[i] for i in step_idx.detach().cpu().tolist()]

            # --- Anti-Freq: 批内归一化 + 可调放大 ---
            if pair_weight is not None:
                # 均值归一化到 1.0，避免全局尺度对 lr 敏感
                denom = pair_weight.mean().clamp_min(1e-8)
                pair_weight = pair_weight / denom

                # 放大系数（命令行可传 --af_scale 2.0 / 3.0）
                af_scale = float(getattr(self.args, "af_scale", 1.0))
                pair_weight = pair_weight * af_scale

                # 软上限（命令行可传 --af_clip 5.0）
                af_clip = float(getattr(self.args, "af_clip", 0.0))
                if af_clip > 0:
                    pair_weight = pair_weight.clamp(max=af_clip)

                # ★ 仅首个 batch 打一条分布诊断日志
                if self.global_step == 0:
                    with torch.no_grad():
                        q10, q50, q90 = self._quantiles_10_50_90(pair_weight)
                        pw_min = float(pair_weight.min().item())
                        pw_max = float(pair_weight.max().item())
                    print(f"[Anti-Freq|batch0] mean=1.00 (norm), min={pw_min:.3f}, p50={q50:.3f}, p90={q90:.3f}, max={pw_max:.3f}, scale={af_scale}, clip={af_clip}")

                # 记录本 batch 的分位数（供 epoch 末平均）
                with torch.no_grad():
                    q10, q50, q90 = self._quantiles_10_50_90(pair_weight)
                    pw_quantiles_across_batches.append([q10, q50, q90])

            # 若 sampler 未注入 pair_weight，但用户开启了 anti_freq，则默认权重为 1（不报错）
            loss_main = self.model(
                step_texts,
                tool_text,
                tool_adj,
                pos_ids,
                neg_ids,
                rag_feat=self.rag_feat,
                comm_ids=self.comm_ids,
                pair_weight=pair_weight if (pair_weight is not None) else None
            )
            # === New: 融合门正则（用“本批前向”的 gate，可反传） ===
            if float(getattr(self.args, "gate_reg", 0.0)) > 0.0:
                gm = getattr(self.model, "_gate_mean_current", None)
                if gm is not None:
                    loss_main = loss_main + float(self.args.gate_reg) * (gm - float(self.args.gate_center)) ** 2


            # === 同构不变性正则（每 inv_every 步触发一次）===
            loss_inv = torch.tensor(0.0, device=self.device)
            if (inv_reg_lambda > 0.0) and (self.global_step % max(1, int(inv_every)) == 0):
                N = len(tool_text)
                perm = torch.randperm(N, device=self.device)
                inv_perm = torch.empty_like(perm); inv_perm[perm] = torch.arange(N, device=self.device)

                # 置换输入
                texts_perm = [tool_text[i] for i in perm.tolist()]
                adj_perm = None
                if tool_adj is not None:
                    # 假设 tool_adj 为形如 [2, E] 的 edge_index
                    adj_perm = torch.stack([perm[tool_adj[0]], perm[tool_adj[1]]], dim=0)
                feat_perm = self.rag_feat[perm] if self.rag_feat is not None else None
                comm_perm = self.comm_ids[perm] if self.comm_ids is not None else None

                # 两次前向：原图 & 置换图，再把置换图输出还原到原顺序
                emb_base = self.model.tool_forward(tool_text, tool_adj, rag_feat=self.rag_feat, comm_ids=self.comm_ids)  # [N,D]
                emb_perm = self.model.tool_forward(texts_perm, adj_perm, rag_feat=feat_perm, comm_ids=comm_perm)        # [N,D]
                emb_perm = emb_perm[inv_perm]

                emb_base = F.normalize(emb_base, dim=-1)
                emb_perm = F.normalize(emb_perm, dim=-1)
                loss_inv = (1.0 - (emb_base * emb_perm).sum(dim=-1)).mean()

            if float(getattr(self.args, "gate_reg", 0.0)) > 0.0:
                gate_mean_batch = None
                if (self.rag_feat is not None) and getattr(self.model, "rag_gate", None) is not None:
                    with torch.no_grad():
                        parts = [self.rag_feat.to(self.device)]
                        if (self.comm_ids is not None) and (self.model.comm_emb is not None):
                            parts.append(self.model.comm_emb(self.comm_ids.to(self.device)))
                        feat = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
                        f = self.model.rag_ln_feat(feat)
                        f = self.model.rag_dropout(f)
                        g = torch.sigmoid(self.model.rag_gate(f))
                        gate_mean_batch = g.mean()

                if gate_mean_batch is not None:
                    tgt = float(getattr(self.args, "gate_center", 0.65))
                    loss_main = loss_main + float(self.args.gate_reg) * (gate_mean_batch - tgt) ** 2

            loss = loss_main + inv_reg_lambda * loss_inv

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._rag_params, self.grad_clip)
            self.optimizer.step()

            total_loss += float(loss.detach().item())
            self.global_step += 1

        # === epoch 末：输出 pair_weight 分位数的“批均值”，方便画曲线 ===
        pw_q10 = pw_q50 = pw_q90 = None
        if pw_quantiles_across_batches:
            tq = torch.tensor(pw_quantiles_across_batches, dtype=torch.float32)
            pw_q10, pw_q50, pw_q90 = tq.mean(dim=0).tolist()
            print(f"[AF|epoch] avg q10={pw_q10:.3f} q50={pw_q50:.3f} q90={pw_q90:.3f}")

        return total_loss / max(1, len(train_loader)), {"pw_q10": pw_q10, "pw_q50": pw_q50, "pw_q90": pw_q90}

    @torch.no_grad()
    def _static_eval_with_toggle(self, tool_text, tool_adj, enable_rag: bool):
        self.model.eval()
        prev = self.model._force_no_rag_fuse
        self.model._force_no_rag_fuse = (not enable_rag)
        total_sample, true_sample = 0, 0

        for batch in self.eval_loader:
            step_idx, pos_ids, _, _ = self._unpack_batch(batch)
            step_texts = [self.eval_steps[i] for i in step_idx.detach().cpu().tolist()]

            tool_emb, step_emb = self.model.inference(
                step_texts, tool_text, tool_adj,
                static=True,  # 不走 GNN
                rag_feat=self.rag_feat,
                comm_ids=self.comm_ids,
            )
            score_matrix = torch.mm(step_emb, tool_emb.t())  # (B, N)
            pred_tool = torch.argmax(score_matrix, dim=1).detach().cpu().tolist()
            gt_tool = pos_ids.detach().cpu().tolist()

            total_sample += len(gt_tool)
            true_sample += sum(1 if a == b else 0 for a, b in zip(pred_tool, gt_tool))

        self.model._force_no_rag_fuse = prev
        return true_sample / max(1, total_sample)

    def train(self, tool_text, tool_adj, sample_obj: TrainSampler,
              rag_feat: Optional[torch.Tensor] = None,
              comm_ids: Optional[torch.Tensor] = None,
              num_comms: Optional[int] = None,
              anti_freq=False, inv_reg_lambda=0.0, inv_every=10):
        """
        anti_freq: 是否启用反频率权重（BPR/InfoNCE）
        inv_reg_lambda: 不变性正则系数（0 则不启用）
        inv_every: 每多少 step 计算一次不变性正则，避免额外开销过大
        """
        device = self.device
        self.model.train()
        if rag_feat is not None or comm_ids is not None:
            self.set_rag_features(rag_feat=rag_feat, comm_ids=comm_ids, num_comms=num_comms)

        best_evaluate_acc, stop_cnt, best_model = 0.0, 0, None

        self.eval_loader, self.eval_steps = sample_obj.sample(maximum=1000, tmp_print=False)

        # —— 自检：RAG 打开 vs 关闭（静态）
        base0 = self._static_eval_with_toggle(tool_text, tool_adj, enable_rag=False)
        rag0  = self._static_eval_with_toggle(tool_text, tool_adj, enable_rag=True)
        print(f"[RAG-Sanity] static eval: LM-only={base0:.4f} | LM+RAG={rag0:.4f}")
        if rag0 + 0.15 < base0:
            print("[RAG-Sanity] RAG harms static eval; reset rag head and start with small alpha.")
            self.model.reset_rag_head()

        init_acc = self.evaluate(tool_text, tool_adj, static_mode=True)
        print(f"Static Evaluation {init_acc:.4f}")

        init_timer = time.time()
        for epoch in range(self.epoch):
            start_time = time.time()

            train_loader, train_step_texts = sample_obj.sample(shuffle=True)
            bpr_loss, extras = self.train_one_epoch(
                tool_text, tool_adj, train_loader, train_step_texts,
                anti_freq=bool(anti_freq),
                inv_reg_lambda=float(inv_reg_lambda),
                inv_every=int(inv_every)
            )

            evaluate_acc = self.evaluate(tool_text, tool_adj, static_mode=False)

            if evaluate_acc >= best_evaluate_acc:
                best_evaluate_acc = evaluate_acc
                stop_cnt = 0
                best_model = copy.deepcopy(self.model)
            else:
                stop_cnt += 1

            self.scheduler.step()
            cur_rag_lr = self.optimizer.param_groups[1]["lr"]

            # 计算 gate_mean 用于日志与 JSONL
            gate_mean = None
            if (self.rag_feat is not None) and getattr(self.model, "rag_gate", None) is not None:
                with torch.no_grad():
                    parts = [self.rag_feat.to(self.device)]
                    if (self.comm_ids is not None) and (self.model.comm_emb is not None):
                        parts.append(self.model.comm_emb(self.comm_ids.to(self.device)))
                    feat = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
                    f = self.model.rag_ln_feat(feat)
                    f = self.model.rag_dropout(f)
                    g = torch.sigmoid(self.model.rag_gate(f))
                    gate_mean = float(g.mean().item())

            gate_str = f", Gate(mean)={gate_mean:.4f}" if gate_mean is not None else ""
            print(
                f"Epoch: {epoch:3d}, Time {time.time() - start_time:.4f}s, "
                f"Loss: {bpr_loss:.4f}, Eval Acc: {evaluate_acc:.4f}, "
                f"rag_lr={cur_rag_lr:.6f}, alpha={self.model.rag_alpha().item():.4f}{gate_str}"
            )

            # === 新增：把 epoch 级指标写入 JSONL（metrics_epoch.jsonl）===
            log_row = {
                "time": time.time(),
                "epoch": int(epoch),
                "train_loss": float(bpr_loss),
                "eval_acc": float(evaluate_acc),
                "gate_mean": gate_mean,
                "rag_lr": float(cur_rag_lr),
                "alpha": float(self.model.rag_alpha().item()),
            }
            if extras:
                log_row.update(extras)
            with open(self._epoch_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_row, ensure_ascii=False) + "\n")

            if stop_cnt >= self.patience:
                break

        # === 新增：导出训练摘要（方便聚合脚本统计耗时与最佳ACC）===
        summary_path = os.path.join(self.exp_dir, "train_summary.json")
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "best_eval_acc": float(best_evaluate_acc),
                        "total_time_sec": float(time.time() - init_timer),
                        "seed": int(self.seed),
                        "exp_dir": self.exp_dir,
                        "args": vars(self.args) if hasattr(self, "args") else None,
                    },
                    f, ensure_ascii=False, indent=2
                )
            print(f"[SAVE] train_summary.json -> {summary_path}")
        except Exception as e:
            print(f"[WARN] failed to save train_summary.json: {e}")

        return best_model, time.time() - init_timer

    @torch.no_grad()
    def evaluate(self, tool_text, tool_adj, static_mode=False, tmp_prinpt=False):
        self.model.eval()
        total_sample, true_sample = 0, 0

        for batch in self.eval_loader:
            step_idx, pos_ids, _, _ = self._unpack_batch(batch)
            step_texts = [self.eval_steps[i] for i in step_idx.detach().cpu().tolist()]

            tool_emb, step_emb = self.model.inference(
                step_texts, tool_text, tool_adj,
                static=static_mode,
                rag_feat=self.rag_feat,
                comm_ids=self.comm_ids,
            )

            score_matrix = torch.mm(step_emb, tool_emb.t())  # (B, N)
            pred_tool = torch.argmax(score_matrix, dim=1).detach().cpu().tolist()
            gt_tool = pos_ids.detach().cpu().tolist()

            total_sample += len(gt_tool)
            true_sample += sum(1 if a == b else 0 for a, b in zip(pred_tool, gt_tool))

        return true_sample / max(1, total_sample)