import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, SAGEConv, GINConv


def build_conv(conv_type: str):
    if conv_type == "GCN":
        return GCNConv
    elif conv_type == "GIN":
        return lambda i, h: GINConv(
            nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h))
        )
    elif conv_type == "GAT":
        return GATConv
    elif conv_type == "TransformerConv":
        return TransformerConv
    elif conv_type == "SAGE":
        return SAGEConv
    else:
        raise KeyError("GNN_TYPE can only be GAT, GCN, SAGE, GIN, and TransformerConv")


class SGC(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x, sp_edge_index):
        neighbor_emb = torch.sparse.mm(sp_edge_index, x)
        return self.alpha * x + (1.0 - self.alpha) * neighbor_emb


def _identity_like_(linear: nn.Linear):
    """把 Linear 初始化成尽可能接近“矩形单位阵”的投影。"""
    if not isinstance(linear, nn.Linear):
        return
    with torch.no_grad():
        linear.weight.zero_()
        m, n = linear.weight.shape  # [out, in]
        for i in range(min(m, n)):
            linear.weight[i, i] = 1.0
        if linear.bias is not None:
            linear.bias.zero_()


def _zeros_(linear: nn.Linear):
    if not isinstance(linear, nn.Linear):
        return
    with torch.no_grad():
        linear.weight.zero_()
        if linear.bias is not None:
            linear.bias.zero_()


class GNNEncoder(nn.Module):
    """
    Safe-init 版（小图稳定）：
      - proj_in 用“矩形单位阵”初始化（尽量保持输入语义）
      - 每层：LayerNorm → LeakyReLU → Dropout → Conv；残差项用 sigmoid 门，门初值≈0（几乎不引入消息）
      - 输出：out = W_h·h + W_skip·x0，其中 W_h（proj_out）零初始化、W_skip（proj_skip）单位初始化
        => 初始 out ≈ x0（严格不劣于 no-GNN）
    """
    def __init__(self, input_dim, hidden_dim, output_dim,
                 n_layers=1, gnn_type="GAT", dropout=0.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.n_layers = max(1, int(n_layers))
        self.dropout = float(dropout)
        self.gnn_type = gnn_type

        # 输入先对齐到 hidden 维
        self.proj_in = nn.Identity() if input_dim == self.hidden_dim else nn.Linear(input_dim, self.hidden_dim, bias=True)

        # 卷积都在 hidden 维内进行；SAGE 显式 normalize=False 以避免幅度漂移
        self.convs = nn.ModuleList()
        for _ in range(self.n_layers):
            if gnn_type == "SAGE":
                self.convs.append(SAGEConv(self.hidden_dim, self.hidden_dim, normalize=False))
            else:
                Conv = build_conv(gnn_type)
                self.convs.append(Conv(self.hidden_dim, self.hidden_dim))

        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.n_layers)])
        self.act = nn.LeakyReLU(0.2)

        # 残差门（标量），初值很小→几乎不引入消息（sigmoid(-4)≈0.018）
        self.res_gates = nn.ParameterList([nn.Parameter(torch.tensor(-4.0)) for _ in range(self.n_layers)])

        # 输出映射：W_h 零，W_skip 单位（让初始 out == x0）
        need_out_map = not (self.output_dim <= 0 or self.output_dim == self.hidden_dim)
        self.proj_out  = nn.Identity() if not need_out_map else nn.Linear(self.hidden_dim, self.output_dim, bias=True)
        self.proj_skip = nn.Identity() if not need_out_map else nn.Linear(self.hidden_dim, self.output_dim, bias=True)

        # —— 初始化 —— #
        if isinstance(self.proj_in, nn.Linear):
            _identity_like_(self.proj_in)
        if isinstance(self.proj_out, nn.Linear):
            _zeros_(self.proj_out)
        if isinstance(self.proj_skip, nn.Linear):
            _identity_like_(self.proj_skip)

    def reset_parameters(self):
        # 重新 safe-init
        for c in self.convs:
            c.reset_parameters()
        for n in self.norms:
            if hasattr(n, "reset_parameters"):
                n.reset_parameters()
        for i in range(len(self.res_gates)):
            with torch.no_grad():
                self.res_gates[i].fill_(-4.0)
        if isinstance(self.proj_in, nn.Linear):
            _identity_like_(self.proj_in)
        if isinstance(self.proj_out, nn.Linear):
            _zeros_(self.proj_out)
        if isinstance(self.proj_skip, nn.Linear):
            _identity_like_(self.proj_skip)

    def forward(self, x, edge_index):
        x0 = self.proj_in(x)  # [N, hidden]
        h = x0
        for i, conv in enumerate(self.convs):
            h_norm = self.norms[i](h)
            h_act = self.act(h_norm)
            if self.dropout > 0.0:
                h_act = F.dropout(h_act, p=self.dropout, training=self.training)
            try:
                h_new = conv(h_act, edge_index)
            except TypeError:
                h_new = conv(h_act)
            gate = torch.sigmoid(self.res_gates[i])  # (0,1)
            h = h + gate * h_new  # 残差注入

        h_out = self.proj_out(h)  if isinstance(self.proj_out, nn.Linear) else h
        s_out = self.proj_skip(x0) if isinstance(self.proj_skip, nn.Linear) else x0
        return h_out + s_out  # 初始等于 x0（保真），训练再慢慢偏离