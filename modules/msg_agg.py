"""
Message Aggregator Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""


import torch
from torch import Tensor
from torch_geometric.utils import scatter
from torch_scatter import scatter_max,scatter_sum,scatter_min


class LastAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        _, argmax = scatter_max(t, index, dim=0, dim_size=dim_size)
        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax < msg.size(0)  # Filter items with at least one entry.
        out[mask] = msg[argmax[mask]]
        return out


class MeanAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):

        return scatter(msg, index, dim=0, dim_size=dim_size, reduce="mean")


class TimeAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int):
        group_mins, _ = scatter_min(t, index, dim=0, dim_size=dim_size)
        # 步骤2：转换为相对时间
        relative_t = (t - group_mins[index]+1)/3600# 避免除以0
        group_sums = scatter_sum(relative_t, index, dim=0, dim_size=dim_size)
        # 归一化，使每组内元素和为1
        normalized_t = relative_t / group_sums[index]
        msg = msg * normalized_t.unsqueeze(-1)
        return scatter(msg, index, dim=0, dim_size=dim_size, reduce="sum")
