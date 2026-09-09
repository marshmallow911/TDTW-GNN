from typing import Callable, Dict, Tuple

import torch
from torch import Tensor
import time

import math
from torch_geometric.nn import TransformerConv,BatchNorm, Linear, PNAConv,GINEConv
import torch.nn.functional as F
import torch.nn as nn
import timeit
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from flow import MultiDimPlanarFlowStack,MultiDimRealNVP


class AncestorBias(nn.Module):
    """
    Return attention bias matrix B of shape (B, n_head, N, N)
    given MPTT (l,r).
    """
    def __init__(self, n_head=2, init_val_l=-0.5, init_val_r=0.5, init_val_le=0.5):
        super().__init__()
        # 让每个 head 都有独立 α, β（广播时维度匹配）
        self.alpha = nn.Parameter(torch.full((n_head, 1, 1),
                                             init_val_l))
        self.beta = nn.Parameter(torch.full((n_head, 1, 1),
                                             init_val_r))
        self.theta = nn.Parameter(torch.full((n_head, 1, 1),
                                            init_val_le))

    @staticmethod
    def _relation_mask(l, r, level):
        # l, r: (B, N)
        N=l.size(1)
        l_lag=l.unsqueeze(-1).expand(-1,-1,N)-l.unsqueeze(-2).expand(-1,N,-1)
        r_lag=r.unsqueeze(-1).expand(-1,-1,N)-r.unsqueeze(-2).expand(-1,N,-1)
        level_lag=level.unsqueeze(-1).expand(-1,-1,N)-level.unsqueeze(-2).expand(-1,N,-1)
        return l_lag/(2*N), r_lag/(2*N), level_lag/level_lag.max()  # (B,N,N) float

    def forward(self, l, r,level):
        """
        Inputs
        ------
        l, r : LongTensor, shape (B, N)
        Returns
        -------
        bias : FloatTensor, shape (B, n_head, N, N)
        """
        # N = l.size(1)
        # scale = math.log(max(N, 2))  # N↑ ⇒ bias↓
        l_lag, r_lag, level_lag = self._relation_mask(l, r,level)  # (B,N,N) bool
        # broadcast to heads
        l_lag = l_lag.unsqueeze(1)  # (B,1,N,N)
        r_lag = r_lag.unsqueeze(1)
        level_lag = level_lag.unsqueeze(1)
        bias = l_lag * self.alpha+ r_lag * self.beta+level_lag*self.theta  # broadcasting
        return bias.view(-1,l_lag.size(-2),l_lag.size(-1))


class GraphTreeWalkEmbedding(torch.nn.Module):
    """
    Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    """

    def __init__(self, in_channels, out_channels, msg_dim,time_enc,deg,in_out_flag='in'):
        super().__init__()

        self.in_ch=in_channels
        self.time_enc = time_enc
        self.time_dim = time_enc.out_channels
        self.in_out_flag = in_out_flag
        edge_dim =self.time_dim + msg_dim+in_channels
        self.embed_dim =self.in_ch+2*self.time_dim
        ff_hidden_dim = 2 * out_channels

        self.conv = PNA(in_channels, out_channels, 2, edge_dim=edge_dim, deg=deg)

        self.bias=AncestorBias(n_head=2)
        self.posencoder = nn.Sequential(nn.Linear(3, ff_hidden_dim), nn.LeakyReLU(),
                                       nn.Linear(ff_hidden_dim, self.time_dim))
        self.feat_linear = nn.Sequential(nn.Linear(self.embed_dim, ff_hidden_dim), nn.LeakyReLU(),nn.Dropout(0.1),
                                       nn.Linear(ff_hidden_dim, in_channels))
        self.seq_encoder = nn.MultiheadAttention(embed_dim=in_channels, num_heads=2, batch_first=True)
        # 前馈神经网络
        self.linear1 = nn.Linear(in_channels, ff_hidden_dim)
        self.linear2 = nn.Linear(ff_hidden_dim, in_channels)
        # 层归一化
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        # Dropout
        self.dropout = nn.Dropout(0.1)
        self.output_linear = nn.Sequential(nn.Linear(in_channels, ff_hidden_dim), nn.ReLU(),nn.Dropout(0.1),
                                           nn.Linear(ff_hidden_dim, out_channels))
        self.feat_net=nn.Sequential(nn.Linear(7, 2*in_channels), nn.ReLU(),
                                           nn.Linear(2*in_channels, in_channels))

        # 参数相对固定，但层级在变，先聚合再序列？或者不重排，相对参数固定

    def forward(self, nodes_list, root_list, z, n_id_neighbors, last_update,edge_index, msg,t):
        # 直接调用优化后的tree_walk_encoding函数
        tree_seq_feat, tree_feat_list = self.tree_walk_encoding(nodes_list, root_list, z, n_id_neighbors,t, last_update)
        tree_seq_feat = tree_seq_feat.view(-1, tree_seq_feat.size(-2), tree_seq_feat.size(-1))
        mask = tree_seq_feat[:, :, 0] != 0
        feat_pos=self.posencoder(tree_seq_feat[:,:,:3])
        attn_bias=self.bias(tree_seq_feat[:,:,0],tree_seq_feat[:,:,1],tree_seq_feat[:,:,2])
        rel_t_enc = self.time_enc(tree_seq_feat[:, :, -1]).view(-1, tree_seq_feat.size(1),
                                                                self.time_dim)  # [B, N, time_dim]

        tree_seq_feat_emb = torch.cat([tree_seq_feat[:, :, 3:-1], feat_pos, rel_t_enc], dim=-1)  # [B, N, in_ch+time_dim]
        tree_seq_feat_pos = self.feat_linear(tree_seq_feat_emb)

        key_padding_mask = torch.zeros((tree_seq_feat_pos.size(0), tree_seq_feat_pos.size(1)),
                                       device=tree_seq_feat_pos.device)
        key_padding_mask[~mask] = float('-inf')
        # 多头注意力处理
        tree_z, _ = self.seq_encoder(tree_seq_feat_pos, tree_seq_feat_pos, tree_seq_feat_pos,
                                     key_padding_mask=key_padding_mask,attn_mask=attn_bias)
        # 残差连接 + 归一化
        tree_z = self.norm1(tree_seq_feat[:, :, 3:-1] + self.dropout(tree_z))
        # 前馈网络
        ff_output = self.linear2(F.relu(self.linear1(tree_z)))
        # 残差连接 + 归一化
        tree_z = self.norm2(tree_z + self.dropout(ff_output))

        tree_z = tree_z.view(-1,nodes_list.size(0), tree_seq_feat_pos.size(-2), tree_seq_feat_pos.size(-1))
        mask_expand = mask.view(-1, nodes_list.size(0), tree_seq_feat_pos.size(-2), 1).expand(-1, -1, -1,
                                                                                          tree_seq_feat_pos.size(
                                                                                              -1)).float()
        tree_z= tree_z * mask_expand  # 应用掩码
        sum_embeddings = tree_z.sum(-2)  # 聚合特征
        sum_mask = torch.clamp(mask_expand.sum(-2), min=1e-9) # 平均池化
        tree_z = sum_embeddings / sum_mask # nodes samples length
        tree_z = tree_z.mean(1)
        tree_z = self.output_linear(tree_z)

        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(z.dtype))
        tree_feat=self.feat_net(tree_feat_list)
        # 图注意力卷积
        edge_attr = torch.cat([msg,tree_feat,rel_t_enc], dim=-1)
        gat_z = self.conv(z, edge_index, edge_attr)

        return tree_z, gat_z

    def tree_walk_encoding(self, nodes_list, root_list, z, n_id_neighbors, t, last_update):
        """
        使用torch.scatter_的完全向量化MPTT编码实现，返回按左值排序的节点张量

        现在支持连接点在任意位置（排序后的结果），而非固定在第0位

        参数:
            nodes_list: 形状为 (num_roots, num_samples, num_hops, max_nodes_per_hop) 的张量
            root_list: 形状为 (num_roots, num_samples, num_hops) 的张量，第0个是根，1到num_hops-1是前num_hops-1层的连接点

        返回:
            ordered_values: 形状为 (num_roots, num_samples, num_hops * max_nodes_per_hop+1, 2) 的张量，
                           其中第三维按左值排序存放所有节点，最后一维存放左值和右值
        """

        dimen = 4 + self.in_ch + 1
        nodes_list = nodes_list.permute(2, 0, 1, 3)  # [batch_size, num_walk, num_hop, num_nodes]
        root_list = root_list.permute(2, 0, 1)  # [batch_size, num_walk, num_hop]
        num_roots, num_samples, num_hops, max_nodes_per_hop = nodes_list.shape
        device = nodes_list.device

        left_values = torch.zeros((num_roots, num_samples, num_hops, max_nodes_per_hop, 2), dtype=torch.long,
                                  device=device)
        hop_values = torch.arange(1, num_hops + 1, device=device, dtype=torch.long).view(1, 1, -1, 1).expand(num_roots, num_samples, num_hops, max_nodes_per_hop)
        max_nodes = num_hops * max_nodes_per_hop + 1
        tree_seq_feat = torch.zeros((num_roots, num_samples, max_nodes, dimen), device=device) - 1

        if (not hasattr(self, "_assoc_buf")) or self._assoc_buf.numel() < (int(n_id_neighbors.max()) + 1):
            self._assoc_buf = torch.zeros(int(n_id_neighbors.max()) + 1, dtype=torch.long, device=device)

        self._assoc_buf[n_id_neighbors] = torch.arange(
            n_id_neighbors.size(0), dtype=torch.long, device=device
        )

        # 1. 创建有效节点掩码
        leaf_mask = (nodes_list != -1)
        flat_leaf_mask = leaf_mask.reshape(num_roots, num_samples, -1)

        # 2. 计算每个(根节点,样本)对的有效叶子节点总数
        valid_leaves_count = leaf_mask.sum(dim=(2, 3))

        # ============ 新增：找连接点位置并计算子树大小 ============
        # 找每层连接点在 nodes_list 中的位置（向量化）
        # root_list[:,:,h+1] 是第h层的连接点，h从0到num_hops-2
        # 只有前 num_hops-1 层有连接点，最后一层是叶子层
        connect_nodes = root_list[:, :, 1:num_hops].unsqueeze(-1)  # [num_roots, num_samples, num_hops-1, 1]
        match = (nodes_list[:, :, :num_hops - 1,
                 :] == connect_nodes)  # [num_roots, num_samples, num_hops-1, max_nodes_per_hop]
        connect_pos = match.long().argmax(dim=-1)  # [num_roots, num_samples, num_hops-1]

        # 计算子树大小（从底向上累积）
        # subtree_size[h, i] 表示第h层第i个节点的子树包含多少个后代节点（包括自己）
        subtree_size = torch.ones(num_roots, num_samples, num_hops, max_nodes_per_hop, dtype=torch.long, device=device)
        subtree_size.masked_fill_(~leaf_mask, 0)  # 无效节点子树大小为0

        for h in range(num_hops - 2, -1, -1):  # 从倒数第二层向上
            # 下一层的总节点数
            next_total = subtree_size[:, :, h + 1, :].sum(dim=-1, keepdim=True)  # [num_roots, num_samples, 1]
            # 加到当前层连接点上
            cp = connect_pos[:, :, h:h + 1]  # [num_roots, num_samples, 1]
            subtree_size[:, :, h, :].scatter_add_(-1, cp, next_total)

        # ============ 计算左右值（基于子树大小） ============
        # 每个节点占用的编号空间 = 2 * subtree_size
        space_per_node = subtree_size * 2  # [num_roots, num_samples, num_hops, max_nodes_per_hop]

        # 3. 每个(根节点,样本)对的编码都从1开始
        # 根节点的左值始终为1
        left_values_roots = torch.ones((num_roots, num_samples), dtype=torch.long, device=device)

        # 根节点的右值 = 2 * 叶子节点数 + 2 (一个根节点占两个位置)
        right_values_roots = 2 * valid_leaves_count + 2
        lr_values_roots = torch.cat([left_values_roots.unsqueeze(-1), right_values_roots.unsqueeze(-1)],
                                    dim=-1).unsqueeze(2)

        # 4. 计算每层的起始左值
        layer_start = torch.zeros(num_roots, num_samples, num_hops, dtype=torch.long, device=device)
        layer_start[:, :, 0] = 2  # 第0层从2开始（根是1）

        # 计算每层内部的累积偏移（在连接点之前的节点占用的空间）
        cumsum_space = space_per_node.cumsum(dim=-1)  # [num_roots, num_samples, num_hops, max_nodes_per_hop]
        prev_space = torch.cat([
            torch.zeros(num_roots, num_samples, num_hops, 1, dtype=torch.long, device=device),
            cumsum_space[:, :, :, :-1]
        ], dim=-1)  # [num_roots, num_samples, num_hops, max_nodes_per_hop]

        for h in range(1, num_hops):
            # 上一层连接点的左值
            cp_prev = connect_pos[:, :, h - 1]  # [num_roots, num_samples]
            prev_left = layer_start[:, :, h - 1] + prev_space[:, :, h - 1, :].gather(-1, cp_prev.unsqueeze(-1)).squeeze(
                -1)
            layer_start[:, :, h] = prev_left + 1

        # 5. 计算所有节点的左值和右值
        left_vals = layer_start.unsqueeze(-1) + prev_space  # [num_roots, num_samples, num_hops, max_nodes_per_hop]
        right_vals = left_vals + space_per_node - 1

        # 组合左右值
        left_values[:, :, :, :, 0] = left_vals
        left_values[:, :, :, :, 1] = right_vals

        left_values.masked_fill_(~leaf_mask.unsqueeze(-1), 0)
        left_values = left_values.view(num_roots, num_samples, -1, 2) / (2 * (num_hops * max_nodes_per_hop + 1))

        index_root = torch.arange(0, max_nodes - 1, max_nodes_per_hop)[:-1]
        lr_root_list = torch.repeat_interleave(
            torch.cat([lr_values_roots / (2 * (num_hops * max_nodes_per_hop + 1)), left_values[:, :, index_root, :]],
                      dim=2),
            repeats=max_nodes_per_hop, dim=2
        )
        root_node_list = self._assoc_buf[root_list[:, :, 0].long().clamp(min=0)].unsqueeze(-1).expand(-1, -1,
                                                                                                      max_nodes - 1)
        root_node_list = root_node_list / root_node_list.max()
        root_sample_num_list = torch.arange(num_samples).repeat(num_roots).view(num_roots, num_samples).unsqueeze(
            -1).expand(-1, -1, max_nodes - 1).to(device)
        root_sample_num_list = root_sample_num_list / num_samples
        hop_values = hop_values.reshape(num_roots, num_samples, -1)
        hop_values = hop_values / hop_values.max()

        final_list = torch.cat(
            [left_values, lr_root_list, root_node_list.unsqueeze(-1), root_sample_num_list.unsqueeze(-1),
             hop_values.unsqueeze(-1)], dim=-1)
        final_list = final_list[flat_leaf_mask]

        nodes_list = nodes_list.reshape(num_roots, num_samples, -1)

        # 计算每个样本的有效节点数量
        valid_counts = flat_leaf_mask.sum(dim=-1, keepdim=True)  # [num_roots, num_samples, 1]

        # 计算有效节点的新位置（从0开始的连续位置）
        valid_positions = torch.zeros_like(flat_leaf_mask, dtype=torch.long)
        cumsum_mask = torch.cumsum(flat_leaf_mask.long(), dim=-1) - 1
        valid_positions[flat_leaf_mask] = cumsum_mask[flat_leaf_mask]

        # 创建目标索引
        target_indices = torch.where(
            flat_leaf_mask,
            valid_positions,
            valid_counts.expand_as(flat_leaf_mask) + torch.cumsum((~flat_leaf_mask).long(), dim=-1) - 1
        )

        # 使用scatter操作重新排列（比gather更高效）
        left_values_new = torch.zeros_like(left_values)
        left_values_new.scatter_(2, target_indices.unsqueeze(-1).expand(-1, -1, -1, 2), left_values)

        sorted_valid_mask = torch.zeros_like(flat_leaf_mask)
        sorted_valid_mask.scatter_(2, target_indices, flat_leaf_mask)

        left_values = left_values_new
        left_values.masked_fill_(~sorted_valid_mask.unsqueeze(-1), 0)

        nodes_list_new = torch.zeros_like(nodes_list)
        nodes_list_new.scatter_(2, target_indices, nodes_list)

        nodes_list = nodes_list_new
        nodes_list.masked_fill_(~sorted_valid_mask, -1)

        tree_seq_feat[:, :, 1:, 0] = nodes_list
        tree_seq_feat[:, :, 1:, 1:3] = left_values * (2 * (num_hops * max_nodes_per_hop + 1))

        tree_seq_feat[:, :, 0, 0] = root_list[:, :, 0]
        tree_seq_feat[:, :, 0, 1] = left_values_roots
        tree_seq_feat[:, :, 0, 2] = right_values_roots
        tree_seq_feat[:, :, 1:, 3] = hop_values.view(num_roots, num_samples, -1)
        tree_seq_feat[:, :, 0, 3] = 0

        leaf_mask_1d = flat_leaf_mask.view(-1)
        temp_t_vec = torch.zeros_like(leaf_mask_1d, dtype=torch.float, device=device)
        temp_t_vec[leaf_mask_1d] = t
        temp_t_vec = temp_t_vec.view(num_roots, num_samples, -1)
        temp_t_vec_root = last_update[self._assoc_buf[root_list[:, :, 0]]].view(num_roots, num_samples, 1)
        temp_t_vec = torch.cat([temp_t_vec_root, temp_t_vec], dim=-1)
        relative_t = temp_t_vec[:, :, 0].unsqueeze(-1) - temp_t_vec

        idx_all = tree_seq_feat[:, :, :, 0].long().clamp(min=0)
        tree_seq_feat[:, :, :, 4:-1] = z[self._assoc_buf[idx_all]]
        tree_seq_feat[:, :, :, -1] = relative_t

        # 把填充位置保持 -1 → 0
        filler = torch.zeros(dimen, device=device)
        tree_seq_feat[tree_seq_feat[:, :, :, 0] == -1] = filler

        return tree_seq_feat[:, :, :, 1:], final_list

class ViewFusion(nn.Module):

    def __init__(self, in_channels, out_channels,views):
        super().__init__()
        self.fusion_linear = nn.Sequential(nn.Linear(in_channels * views, in_channels * views),nn.LeakyReLU(),nn.Linear(in_channels * views, out_channels))
    def forward(self, views_list, view_weights=None):
        if view_weights is not None:
            views_list = [v * w for v, w in zip(views_list, view_weights)]
        fused_views = torch.cat(views_list, dim=-1)
        return self.fusion_linear(fused_views)

class AttentionViewFusion(nn.Module):
    def __init__(self, in_channels, out_channels, views):
        super().__init__()
        # 1. 计算注意力得分的网络 (将每个 view 映射为一个标量得分)
        self.attn_net = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.Tanh(),
            nn.Linear(in_channels // 2, 1, bias=False)
        )

        # 2. 融合后的特征映射层 (保持与你原版类似的非线性结构，确保输出维度一致)
        # self.out_proj = nn.Sequential(
        #     nn.Linear(in_channels, in_channels),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_channels, out_channels)
        # )

    def forward(self, views_list, view_weights=None):
        # 保持你原有的 view_weights 逻辑，防止外部调用报错
        if view_weights is not None:
            views_list = [v * w for v, w in zip(views_list, view_weights)]

        # 1. 堆叠特征 -> shape: (Batch, views, in_channels)
        stacked_views = torch.stack(views_list, dim=1)

        # 2. 计算注意力得分 -> shape: (Batch, views, 1)
        scores = self.attn_net(stacked_views)

        # 3. Softmax 归一化权重 -> shape: (Batch, views, 1)
        weights = F.softmax(scores, dim=1)

        # 4. 加权求和融合 -> shape: (Batch, in_channels)
        fused_views = (weights * stacked_views).sum(dim=1)

        # 5. 通过 MLP 映射到目标输出维度 -> shape: (Batch, out_channels)
        # return self.out_proj(fused_views)
        return fused_views


class GatingViewFusion(nn.Module):
    def __init__(self, in_channels, out_channels, views):
        super().__init__()
        # 1. 门控生成网络：输入拼接后的全局特征，输出 views 个门控值
        self.gate_net = nn.Sequential(
            nn.Linear(in_channels * views, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, views),
            nn.Sigmoid()  # 确保门控值在 0~1 之间
        )

        # # 2. 融合后的特征映射层 (保持与你原版类似的非线性结构)
        # self.out_proj = nn.Sequential(
        #     nn.Linear(in_channels, in_channels),
        #     nn.LeakyReLU(),
        #     nn.Linear(in_channels, out_channels)
        # )

    def forward(self, views_list, view_weights=None):
        # 保持你原有的 view_weights 逻辑
        if view_weights is not None:
            views_list = [v * w for v, w in zip(views_list, view_weights)]

        # 1. 拼接获取全局上下文 -> shape: (Batch, in_channels * views)
        concat_views = torch.cat(views_list, dim=-1)

        # 2. 生成门控值 -> shape: (Batch, views)
        gates = self.gate_net(concat_views)

        # 扩展维度以便于广播相乘 -> shape: (Batch, views, 1)
        gates = gates.unsqueeze(-1)

        # 3. 堆叠原始特征 -> shape: (Batch, views, in_channels)
        stacked_views = torch.stack(views_list, dim=1)

        # 4. 门控加权求和 -> shape: (Batch, in_channels)
        fused_views = (gates * stacked_views).sum(dim=1)

        # 5. 通过 MLP 映射到目标输出维度 -> shape: (Batch, out_channels)
        # return self.out_proj(fused_views)
        return fused_views

class PNA(torch.nn.Module):
    def __init__(self, num_features, out_features, num_gnn_layers,
                n_hidden=100, edge_dim=None,dropout=0.1, deg=None):
        super().__init__()
        n_hidden = int((n_hidden // 5) * 5)
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = PNAConv(in_channels=n_hidden, out_channels=n_hidden,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           edge_dim=n_hidden, towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

            self.mlp = nn.Sequential(Linear(n_hidden, n_hidden * 2), nn.LeakyReLU(), nn.Dropout(dropout), Linear(n_hidden * 2, out_features))

    def forward(self, x, edge_index, edge_attr):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
        return self.mlp(x)


class GINe(torch.nn.Module):
    def __init__(self, num_features,out_features, num_gnn_layers,n_hidden=128, edge_dim=None, dropout=0.1):
        super().__init__()
        self.n_hidden = n_hidden
        self.num_gnn_layers = num_gnn_layers

        self.node_emb = nn.Linear(num_features, n_hidden)
        self.edge_emb = nn.Linear(edge_dim, n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = GINEConv(nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden)
            ), edge_dim=self.n_hidden)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(n_hidden))

        self.mlp = nn.Sequential(Linear(n_hidden, n_hidden * 2), nn.ReLU(), nn.Dropout(dropout), Linear(n_hidden * 2, out_features))

    def forward(self, x, edge_index, edge_attr):

        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2

        return self.mlp(x)


class TreeNeighborsampler:
    def __init__(self, num_nodes: int, size: int, device=None):
        self.size = size

        self.neighbors_in = torch.zeros((num_nodes + 1, size), dtype=torch.long, device=device) - 1
        self.e_id_in = torch.zeros((num_nodes + 1, size), dtype=torch.long, device=device) - 1
        self.t_in = torch.zeros((num_nodes + 1, size), dtype=torch.long, device=device) - 1

        self.neighbors_out = torch.zeros((num_nodes + 1, size), dtype=torch.long, device=device) - 1
        self.e_id_out = torch.zeros((num_nodes + 1, size), dtype=torch.long, device=device) - 1
        self.t_out = torch.zeros((num_nodes + 1, size), dtype=torch.long, device=device) - 1

        self._assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
        self.d_out = torch.zeros(num_nodes, dtype=torch.long, device=device)
        self.d_in = torch.zeros(num_nodes, dtype=torch.long, device=device)

        self.reset_state()

    def get_din_out(self, n_id: Tensor, link_ids: Tensor,in_out_flag: str):
        if in_out_flag == 'in':
            temp_j=self.d_in[n_id]
            temp_j[n_id<0]=0
            temp_k = self.d_in[link_ids]
            temp_k[link_ids < 0] = 0
            return temp_j, temp_k
        elif in_out_flag == 'out':
            temp_j = self.d_out[n_id]
            temp_j[n_id < 0] = 0
            temp_k = self.d_out[link_ids]
            temp_k[link_ids < 0] = 0
            return temp_j, temp_k


    def get_time_decay(self,src,dst,cutoff_time,gamma,in_out_flag):
        temp_root = self.t_in[dst].float()
        temp_root[temp_root == -1] = torch.nan
        temp_leaf = self.t_out[src].float()
        temp_leaf[temp_leaf == -1] = torch.nan
        if in_out_flag == 'in':
            temp_out_root=self.t_out[dst].float()
            temp_out_root[temp_out_root == -1] = torch.nan

            time_prob = (gamma * cutoff_time + torch.nansum((cutoff_time - temp_root), dim=-1) + torch.nansum(
                (cutoff_time - temp_out_root), dim=-1)) / (gamma * cutoff_time + torch.nansum((cutoff_time - temp_root),dim=-1) + torch.nansum(
                (cutoff_time - temp_leaf), dim=-1) + 1e-6)

            # time_prob = (gamma * cutoff_time + torch.nansum((cutoff_time - temp_root),dim=-1) + torch.nansum((
            #     cutoff_time - temp_leaf),dim=-1))/(gamma * cutoff_time + torch.nansum((cutoff_time - temp_root),dim=-1) + torch.nansum((
            #     cutoff_time - temp_out_root),dim=-1)+1e-6)

        else:
            temp_in_root = self.t_in[src].float()
            temp_in_root[temp_in_root == -1] = torch.nan

            time_prob = (gamma * cutoff_time + torch.nansum((cutoff_time - temp_leaf),dim=-1) + torch.nansum((
                cutoff_time - temp_in_root),dim=-1)) / (gamma * cutoff_time + torch.nansum((cutoff_time - temp_leaf),dim=-1) + torch.nansum((
                cutoff_time - temp_root),dim=-1)+1e-6)

            # time_prob = (gamma * cutoff_time + torch.nansum((cutoff_time - temp_leaf),dim=-1) + torch.nansum((
            #     cutoff_time - temp_root),dim=-1)) / (gamma * cutoff_time + torch.nansum((cutoff_time - temp_leaf),dim=-1) + torch.nansum((
            #     cutoff_time - temp_in_root),dim=-1)+1e-6)

        return time_prob

    def walk_tree(self, nid, cutoff_time, num_walk, num_hop, k, gamma):
        batch_size = nid.size(0)
        device = nid.device

        # 预分配内存
        nodes_list_in = torch.zeros((num_walk, num_hop, batch_size, k), dtype=torch.long, device=device)
        root_list_in = torch.zeros((num_walk, num_hop, batch_size), dtype=torch.long, device=device)
        nodes_list_out = torch.zeros((num_walk, num_hop, batch_size, k), dtype=torch.long, device=device)
        root_list_out = torch.zeros((num_walk, num_hop, batch_size), dtype=torch.long, device=device)

        # 边信息收集列表
        all_edge_list_in = []
        all_edge_list_out = []
        all_mem_edge_index_in = []
        all_mem_edge_index_out = []

        # 重塑数据以支持批量处理
        # 将 [num_walk, batch_size] 重塑为 [num_walk * batch_size]
        expanded_batch_size = num_walk * batch_size

        # 初始化所有walks
        temp_id_in = nid.unsqueeze(0).expand(num_walk, -1).reshape(-1)  # [num_walk * batch_size]
        temp_id_out = nid.unsqueeze(0).expand(num_walk, -1).reshape(-1)  # [num_walk * batch_size]

        # 创建原始nid的扩展版本用于后续计算
        expanded_nid = nid.unsqueeze(0).expand(num_walk, -1).reshape(-1)  # [num_walk * batch_size]

        for i in range(num_hop):
            # 创建统一的mask
            mk_in = torch.ones(expanded_batch_size, device=device)

            # 批量处理所有walks的采样
            one_hop_in_dict, mem_edge_index_in, e_list_in, times_in_dict  = self.sample_valid_elements(
                temp_id_in, mk_in, 1, 'in')
            all_edge_list_in.append(e_list_in)
            all_mem_edge_index_in.append(mem_edge_index_in)

            one_hop_out_dict, mem_edge_index_out, e_list_out,times_out_dict  = self.sample_valid_elements(
                temp_id_out, mk_in, 1, 'out')
            all_edge_list_out.append(e_list_out)
            all_mem_edge_index_out.append(mem_edge_index_out)

            one_hop_in = one_hop_in_dict[:, 0]  # [num_walk * batch_size]
            one_hop_out = one_hop_out_dict[:, 0]  # [num_walk * batch_size]
            times_in_first = times_in_dict[:, 0]
            times_out_first = times_out_dict[:, 0]

            # 批量计算时间衰减
            time_decay_in = self.get_time_decay(one_hop_in, expanded_nid, cutoff_time, gamma, 'in')
            time_decay_out = self.get_time_decay(expanded_nid, one_hop_out, cutoff_time, gamma, 'out')

            # 批量计算度
            doutj, doutk = self.get_din_out(expanded_nid, one_hop_in, 'out')
            dinj, dink = self.get_din_out(expanded_nid, one_hop_out, 'in')

            # 批量计算采样权重
            mk_in = (doutj / (doutk + 1e-6)).squeeze(-1) * time_decay_in
            mk_out = (dinj / (dink + 1e-6)).squeeze(-1) * time_decay_out

            # 批量限制最大值
            mk_in = torch.clamp(mk_in, 0, k)
            mk_out = torch.clamp(mk_out, 0, k)

            mk_in_hat = torch.floor(mk_in)
            mk_out_hat = torch.floor(mk_out)

            # 批量计算采样概率
            res_in = mk_in - mk_in_hat
            res_out = mk_out - mk_out_hat

            # 批量伯努利采样
            in_list = torch.bernoulli(res_in)
            out_list = torch.bernoulli(res_out)

            # 批量更新采样数量
            mk_in_hat = in_list + mk_in_hat - 1
            mk_in_hat = torch.clamp(mk_in_hat, min=0)
            mk_out_hat = out_list + mk_out_hat - 1
            mk_out_hat = torch.clamp(mk_out_hat, min=0)

            # 批量获取最终采样结果
            if mk_in_hat.sum() > 0:
                one_hop_in_final, mem_edge_index_in, e_list_in,times_in_final = self.sample_valid_elements(
                    temp_id_in, mk_in_hat, k, 'in')
                all_edge_list_in.append(e_list_in)
                all_mem_edge_index_in.append(mem_edge_index_in)
                one_hop_in_final = torch.cat([one_hop_in.unsqueeze(-1), one_hop_in_final[:, 0:k - 1]], dim=-1)
                times_in_final = torch.cat([times_in_first.unsqueeze(-1), times_in_final[:, :k - 1]], dim=-1)
                sort_indices = torch.argsort(times_in_final, dim=-1, descending=True)
                one_hop_in_final = torch.gather(one_hop_in_final, -1, sort_indices)
            else:
                one_hop_in_final = torch.cat([one_hop_in_dict, torch.zeros(num_walk*batch_size,k-1,device=device)-1], dim=-1)

            if mk_out_hat.sum() > 0:
                one_hop_out_final, mem_edge_index_out, e_list_out,times_out_final  = self.sample_valid_elements(
                    temp_id_out, mk_out_hat, k, 'out')
                all_edge_list_out.append(e_list_out)
                all_mem_edge_index_out.append(mem_edge_index_out)
                one_hop_out_final = torch.cat([one_hop_out.unsqueeze(-1), one_hop_out_final[:, 0:k - 1]], dim=-1)
                times_out_final = torch.cat([times_out_first.unsqueeze(-1), times_out_final[:, :k - 1]], dim=-1)
                sort_indices = torch.argsort(times_out_final, dim=-1, descending=True)
                one_hop_out_final = torch.gather(one_hop_out_final, -1, sort_indices)
            else:
                one_hop_out_final = torch.cat([one_hop_out_dict, torch.zeros(num_walk*batch_size,k-1,device=device)-1], dim=-1)

            # 重塑回原始形状并存储结果
            # [num_walk * batch_size, k] -> [num_walk, batch_size, k]
            nodes_list_in[:, i, :, :] = one_hop_in_final.reshape(num_walk, batch_size, k)
            nodes_list_out[:, i, :, :] = one_hop_out_final.reshape(num_walk, batch_size, k)
            root_list_in[:, i, :] = temp_id_in.reshape(num_walk, batch_size)
            root_list_out[:, i, :] = temp_id_out.reshape(num_walk, batch_size)

            # 更新当前节点
            temp_id_in = one_hop_in
            temp_id_out = one_hop_out

        # 合并所有边信息
        final_all_edge_list_in = torch.cat(all_edge_list_in) if all_edge_list_in else torch.empty((0,),
                                                                                                  dtype=torch.int32,
                                                                                                  device=device)
        final_all_edge_list_out = torch.cat(all_edge_list_out) if all_edge_list_out else torch.empty((0,),
                                                                                                     dtype=torch.int32,
                                                                                                     device=device)
        final_all_mem_edge_index_in = torch.cat(all_mem_edge_index_in,
                                                dim=-1) if all_mem_edge_index_in else torch.empty((2, 0),
                                                                                                  dtype=torch.int32,
                                                                                                  device=device)
        final_all_mem_edge_index_out = torch.cat(all_mem_edge_index_out,
                                                 dim=-1) if all_mem_edge_index_out else torch.empty((2, 0),
                                                                                                    dtype=torch.int32,
                                                                                                    device=device)

        return nodes_list_in, nodes_list_out, root_list_in, root_list_out, final_all_mem_edge_index_in, final_all_mem_edge_index_out, final_all_edge_list_in, final_all_edge_list_out

    def sample_valid_elements(self, n_id: Tensor, num_nodes: Tensor, k: int, in_out_flag: str):
        """
        获取一跳邻居的优化版本

        Args:
            n_id (Tensor): 节点 ID 的张量，形状为 [batch_size]。
            num_nodes (Tensor): 每个节点需要采样的数量，形状为 [batch_size]。
            k (int): 最大采样数量
            in_out_flag (str): 'in'表示入边邻居，'out'表示出边邻居
            add_dict (Tensor, optional): 额外的邻居字典

        Returns:
            - one_hop_dict: 一跳邻居字典
            - mem_edge_index: 内存边索引
            - edge_list: 边列表
        """
        batch_size = n_id.size(0)
        device = n_id.device

        if in_out_flag == 'in':
            # 获取入边邻居
            neighbors = self.neighbors_in[n_id]  # [batch_size, max_neighbors]
            e_id = self.e_id_in[n_id]
            t_neighbors = self.t_in[n_id]

        else:
            neighbors = self.neighbors_out[n_id]
            e_id = self.e_id_out[n_id]
            t_neighbors = self.t_out[n_id]

        mask = neighbors != -1
        # 为每一行创建概率分布 - 有效值位置概率相等，无效值位置概率为0
        probs = torch.arange(neighbors.size(1),0,-1, device=device, dtype=torch.float).unsqueeze(0).expand(batch_size, -1)  # [batch_size, max_neighbors] 按照位置概率采样
        # probs = torch.ones_like(neighbors, device=device, dtype=torch.float)  # [batch_size, max_neighbors] 按照位置概率采样
        probs = probs * mask
        # 归一化每行的概率，使其和为1
        row_sums = probs.sum(dim=1, keepdim=True)  # 保持形状为 [batch_size, 1]
        valid_rows = row_sums > 0
        # 使用广播进行除法，避免索引错误
        # 只处理有效行，其他行保持0概率
        normalized_probs = torch.zeros_like(probs)
        normalized_probs[valid_rows.squeeze()] = probs[valid_rows.squeeze()] / row_sums[valid_rows.squeeze()]

        sampled_indices=torch.zeros(batch_size,k,dtype=torch.long,device=device)
        # 使用归一化后的概率进行采样
        sampled_indices[valid_rows.squeeze()] = torch.multinomial(normalized_probs[valid_rows.squeeze()], num_samples=k, replacement=True)
        row_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, k)
        one_hop_dict = neighbors[row_indices, sampled_indices]
        edge_list = e_id[row_indices, sampled_indices]
        times_list = t_neighbors[row_indices, sampled_indices]

        # 处理全无效行的情况
        all_invalid_rows = (row_sums.squeeze() == 0)
        if all_invalid_rows.any():
            one_hop_dict[all_invalid_rows] = -1
            edge_list[all_invalid_rows] = -1
            times_list[all_invalid_rows] = -1
        # 删掉多余采样值
        col_indices = torch.arange(one_hop_dict.shape[1])
        # Create a comparison mask
        expanded_cols = col_indices.unsqueeze(0).expand(one_hop_dict.shape).to(device)
        expanded_a = num_nodes.unsqueeze(1).expand(-1, one_hop_dict.shape[1])
        # Create mask where column index > corresponding a value
        mask = expanded_cols >= expanded_a
        one_hop_dict[mask] = -1
        edge_list[mask] = -1
        times_list[mask] = -1

        edge_list=edge_list.view(-1)
        if in_out_flag == 'in':
            mem_edge_index = torch.vstack([one_hop_dict.view(-1), n_id.repeat_interleave(k)])
        else:
            mem_edge_index = torch.vstack([n_id.repeat_interleave(k), one_hop_dict.view(-1)])
        mask = (mem_edge_index != -1).all(dim=0)
        mem_edge_index = mem_edge_index[:,mask]
        edge_list=edge_list[mask]

        return one_hop_dict.int(),mem_edge_index.int(), edge_list.int(), times_list

    def __call__(self, mem_edge_index_in, mem_edge_index_out,n_id):

        # Relabel node indices.
        n_id_in = torch.cat([mem_edge_index_in[0], mem_edge_index_in[1], n_id]).unique()
        n_id_out = torch.cat([mem_edge_index_out[0], mem_edge_index_out[1], n_id]).unique()
        self._assoc[n_id_in] = torch.arange(n_id_in.size(0), device=self._assoc.device)
        in_neighbors, nodes_in = self._assoc[mem_edge_index_in[0]], self._assoc[mem_edge_index_in[1]]
        self._assoc[n_id_out] = torch.arange(n_id_out.size(0), device=self._assoc.device)
        out_neighbors, nodes_out = self._assoc[mem_edge_index_out[0]], self._assoc[mem_edge_index_out[1]]

        return n_id_in, n_id_out, torch.stack([in_neighbors, nodes_in]), torch.stack([nodes_out, out_neighbors])

    def insert(self, src: Tensor, dst: Tensor, t: Tensor):

        # Inserts newly encountered interactions into an ever-growing
        # (undirected) temporal graph.

        # Collect central nodes, their neighbors and the current event ids.
        neighbors = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        e_id = torch.arange(
            self.cur_e_id, self.cur_e_id + src.size(0), device=src.device
        ).repeat(2)
        t_nodes=torch.cat([t,t] ,dim=0)
        #src->dst:1(coming in edges); dst->src:-1(coming out edges)
        in_out_flag=torch.cat([nodes.new_full((len(src),),1),nodes.new_full((len(dst),),-1)])
        self.cur_e_id += src.numel()

        # Convert newly encountered interaction ids so that they point to
        # locations of a "dense" format of shape [num_nodes, size].
        nodes, perm = nodes.sort()
        neighbors, e_id, in_out_flag,t_nodes = neighbors[perm], e_id[perm],in_out_flag[perm],t_nodes[perm]

        n_id = nodes.unique()
        self._assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        # The following code is a bit tricky. It first assigns a unique id to each node in the graph and then assigns a unique id to each edge in the graph.
        # The unique id of each node is calculated by the following code:
        dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.size
        dense_id += self._assoc[nodes].mul_(self.size)

        dense_e_id = e_id.new_full((n_id.numel() * self.size,), -1)
        dense_e_flag = in_out_flag.new_full((n_id.numel() * self.size,), 0)
        dense_e_id[dense_id] = e_id
        dense_e_flag[dense_id] = in_out_flag
        dense_e_id = dense_e_id.view(-1, self.size)
        dense_e_flag = dense_e_flag.view(-1, self.size)

        dense_t = t_nodes.new_full((n_id.numel() * self.size,), -1)
        dense_t[dense_id] = t_nodes
        dense_t = dense_t.view(-1, self.size)

        in_index = dense_e_flag==1
        out_index = dense_e_flag==-1
        dense_e_flag[out_index]=0
        in_num = torch.sum(dense_e_flag,dim=1)
        dense_e_flag[out_index]=1
        dense_e_flag[in_index]=0
        out_num = torch.sum(dense_e_flag,dim=1)

        self.d_out[n_id] = self.d_out[n_id]+out_num
        self.d_in[n_id] = self.d_in[n_id]+in_num

        dense_neighbors = e_id.new_empty(n_id.numel() * self.size)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.view(-1, self.size)

        # Collect new and old interactions...
        temp_dense_e_id = dense_e_id.detach().clone()
        temp_dense_e_id[out_index] = -1
        e_id_in = torch.cat([self.e_id_in[n_id, : self.size], temp_dense_e_id], dim=-1)
        dense_e_id[in_index] = -1
        e_id_out = torch.cat([self.e_id_out[n_id, : self.size], dense_e_id], dim=-1)

        t_in = torch.cat([self.t_in[n_id, : self.size], dense_t], dim=-1)
        t_out = torch.cat([self.t_out[n_id, : self.size], dense_t], dim=-1)

        neighbors_in = torch.cat(
            [self.neighbors_in[n_id, : self.size], dense_neighbors], dim=-1
        )
        neighbors_out = torch.cat(
            [self.neighbors_out[n_id, : self.size], dense_neighbors], dim=-1
        )

        # And sort them based on `e_id`.
        e_id_in, perm_in = e_id_in.topk(self.size, dim=-1)
        e_id_out, perm_out = e_id_out.topk(self.size, dim=-1)
        self.e_id_in[n_id] = e_id_in
        self.e_id_out[n_id] = e_id_out
        self.t_in[n_id] = torch.gather(t_in, 1, perm_in)
        self.t_out[n_id] = torch.gather(t_out, 1, perm_out)
        self.neighbors_in[n_id] = torch.gather(neighbors_in, 1, perm_in)
        self.neighbors_out[n_id] = torch.gather(neighbors_out, 1, perm_out)

    def reset_state(self):
        torch.cuda.empty_cache()
        self.cur_e_id = 0
        self.e_id_in.fill_(-1)
        self.e_id_out.fill_(-1)
        self.neighbors_out.fill_(-1)
        self.neighbors_in.fill_(-1)
        self.t_in.fill_(-1)
        self.t_out.fill_(-1)
        self.d_in.fill_(0)
        self.d_out.fill_(0)


class NodeEdgePredictor_ori(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin_node =nn.Sequential(Linear(in_dim, in_dim), nn.BatchNorm1d(in_dim), nn.LeakyReLU(), Linear(in_dim, out_dim))

    def forward(self, src_embed,dst_emd,edge_attr):
        x= torch.cat([src_embed,dst_emd,edge_attr],dim=1)
        h = self.lin_node(x)
        # h = F.log_softmax(h, dim=-1)
        # h = F.softmax(h, dim=-1)
        return h



class DiceFocalLoss(nn.Module):
    def __init__(self, alpha_fl=1, gamma=2.0, weight_fl=None, beta=1.0, smooth=1e-6):
        super(DiceFocalLoss, self).__init__()
        self.alpha_fl = alpha_fl  # Focal Loss 的 alpha (少数类权重)
        self.gamma = gamma  # Focal Loss 的 gamma (聚焦参数)
        self.weight_fl = weight_fl  # optional: class weights for Focal, e.g., torch.tensor([1.0, 10.0])
        self.beta = beta  # weight for Dice (相对于 Focal 的权重，e.g., 0.5 以减少 Dice 影响)
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 假设 inputs 是 logits (batch_size, 2)，targets 是 (batch_size,) 的 0/1
        assert inputs.size(1) == 2, "Inputs should be (batch, 2) for binary classification."

        # Focal Loss (binary version, adapted for class imbalance)
        probs = F.softmax(inputs, dim=1)  # 转换为概率
        probs_t = inputs[torch.arange(inputs.size(0)), targets.long()]  # p_t: 正确类的概率

        focal_factor = (1 - probs_t).pow(self.gamma)  # (1 - p_t)^γ
        ce = F.cross_entropy(inputs, targets, weight=self.weight_fl, reduction='none')  # per-sample CE
        focal_loss = self.alpha_fl * focal_factor * ce
        focal_loss = focal_loss.mean()  # mean over batch

        # Dice: binary 版本，只关注异常类 (与之前相同)
        preds = probs[:, 1]  # 异常概率
        targets_float = targets.float()  # 0/1

        # 处理全正常 batch
        intersection = torch.sum(preds * targets_float)  # sum over batch
        cardinality = torch.sum(preds.pow(3)) + torch.sum(targets_float) + 0.01*torch.sum(preds*(1-targets_float))
        # cardinality = torch.sum(preds.pow(3)) + torch.sum(targets_float) + 0.1 * torch.sum((1-preds) * targets_float)
        dice = (2. * intersection + self.smooth) / (cardinality + 1)
        dice_loss = 1 - dice

        # 组合: Focal + beta * Dice
        total_loss = self.beta * focal_loss + dice_loss
        return total_loss


class DiceFocalLoss_split(nn.Module):

    def __init__(self, alpha_fl=1, gamma=2.0, weight_fl=None, beta=1.0, smooth=1e-6):
        super(DiceFocalLoss_split, self).__init__()
        self.alpha_fl = alpha_fl  # Focal Loss 的 alpha (少数类权重)
        self.gamma = gamma  # Focal Loss 的 gamma (聚焦参数)
        self.weight_fl = weight_fl  # optional: class weights for Focal, e.g., torch.tensor([1.0, 10.0])
        self.beta = beta  # weight for Dice (相对于 Focal 的权重，e.g., 0.5 以减少 Dice 影响)
        self.smooth = smooth

    def forward(self, inputs, targets):
        # 假设 inputs 是 logits (batch_size, 2)，targets 是 (batch_size,) 的 0/1
        assert inputs.size(1) == 2, "Inputs should be (batch, 2) for binary classification."

        # Focal Loss (binary version, adapted for class imbalance)
        probs = F.softmax(inputs, dim=1)  # 转换为概率
        probs_t = inputs[torch.arange(inputs.size(0)), targets.long()]  # p_t: 正确类的概率

        focal_factor = (1 - probs_t).pow(self.gamma)  # (1 - p_t)^γ
        ce = F.cross_entropy(inputs, targets, weight=self.weight_fl, reduction='none')  # per-sample CE
        focal_loss = self.alpha_fl * focal_factor * ce
        focal_loss = focal_loss.mean()  # mean over batch

        # Dice: binary 版本，只关注异常类 (与之前相同)
        preds = probs[:, 1]  # 异常概率
        targets_float = targets.float()  # 0/1

        # 处理全正常 batch
        intersection = torch.sum(preds * targets_float)  # sum over batch
        cardinality = torch.sum(preds.pow(3)) + torch.sum(targets_float) + 0.01*torch.sum(preds*(1-targets_float))
        # cardinality = torch.sum(preds.pow(3)) + torch.sum(targets_float) + 0.1 * torch.sum((1-preds) * targets_float)
        dice = (2. * intersection + self.smooth) / (cardinality + 1)
        dice_loss = 1 - dice

        # 组合: Focal + beta * Dice
        total_loss = self.beta * focal_loss + dice_loss
        return total_loss, focal_loss, dice_loss