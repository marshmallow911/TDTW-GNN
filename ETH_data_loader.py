from pathlib import Path

DEFAULT_GRAPH_PATH = Path(__file__).resolve().parent / "data" / "subgraph.pkl"

import pandas as pd
import numpy as np
import torch
import logging
import itertools
from torch_geometric.data import Data,TemporalData
import pickle
import networkx as nx
import math



def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
    return (data - data.mean(0).unsqueeze(0)) / std


def get_data():
    '''Loads the AML transaction data.

    1. The data is loaded from the csv and the necessary features are chosen.
    2. The data is split into training, validation and test data.
    3. PyG Data objects are created with the respective data splits.
    '''


    # Load the graph
    with open(DEFAULT_GRAPH_PATH, 'rb') as f:
        subgraph2 = pickle.load(f)
        print("Graph read successfully")

    node_mapping = {node: i for i, node in enumerate(subgraph2.nodes())}

    # Extract edges with numerical indices
    edge_index = []
    edge_attr = []

    for u, v, key, data in subgraph2.edges(keys=True, data=True):
        edge_index.append([node_mapping[u], node_mapping[v]])  # Convert to numerical indices
        edge_attr.append([data.get("amount", 0), data.get("timestamp", 0)])  # Edge features

    # Convert to PyTorch tensors
    edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long)  # Transpose for PyG format
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    e_tr=math.floor(0.6*edge_index.size(1))
    e_val=math.floor(e_tr+0.2*edge_index.size(1))
    e_te=edge_index.size(1)

    timestamps=edge_attr[:, 1]  # Assuming the second column is the timestamp
    timestamps,index=torch.sort(timestamps, descending=False)  # Sort timestamps in ascending order
    timestamps = timestamps-timestamps[0]  # Normalize timestamps to start from 0
    edge_attr= edge_attr[index, 0].unsqueeze(1)  # Assuming the first column is the amount, reshape to 2D
    edge_attr = z_norm(edge_attr)  # Normalize the edge features
    y_labels = torch.tensor([subgraph2.nodes[node].get("isp", 0) for node in subgraph2.nodes()], dtype=torch.long) # Classification labels should be long dtype
    edge_index=edge_index[:,index]

    full_data = TemporalData(src=edge_index[0,:],dst=edge_index[1,:],t=timestamps,msg=edge_attr)  # Create a PyG Data object with the graph data


    return full_data,e_tr,e_val,e_te,y_labels


def get_semi_data(lable_bias=None, graph_path=DEFAULT_GRAPH_PATH):
    # Load the graph
    with open(graph_path, 'rb') as f:
        subgraph2 = pickle.load(f)
        print("Graph read successfully")

    node_mapping = {node: i for i, node in enumerate(subgraph2.nodes())}

    # Extract edges with numerical indices
    edge_index = []
    edge_attr = []

    for u, v, key, data in subgraph2.edges(keys=True, data=True):
        edge_index.append([node_mapping[u], node_mapping[v]])  # Convert to numerical indices
        edge_attr.append([data.get("amount", 0), data.get("timestamp", 0)])  # Edge features

    # Convert to PyTorch tensors
    edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long)  # Transpose for PyG format
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)


    tr_ratio = 0.65
    val_ratio = 0.15
    test_ratio = 0.2

    e_tr = math.floor(tr_ratio * edge_index.size(1))
    e_val = math.floor(e_tr + val_ratio * edge_index.size(1))
    e_te = edge_index.size(1)

    timestamps = edge_attr[:, 1]  # Assuming the second column is the timestamp
    timestamps, index = torch.sort(timestamps, descending=False)  # Sort timestamps in ascending order
    timestamps = timestamps - timestamps[0]  # Normalize timestamps to start from 0
    edge_attr = edge_attr[index, 0].unsqueeze(1)  # Assuming the first column is the amount, reshape to 2D
    edge_attr = z_norm(edge_attr)  # Normalize the edge features
    y_labels = torch.tensor([subgraph2.nodes[node].get("isp", 0) for node in subgraph2.nodes()],
                            dtype=torch.long)  # Classification labels should be long dtype
    edge_index = edge_index[:, index]
    ill_nodes=torch.where(y_labels == 1)[0]
    mask_src=torch.isin(edge_index[0,:], ill_nodes)
    mask_dst = torch.isin(edge_index[1, :], ill_nodes)
    if lable_bias == 'src':
        mask_ill = mask_src
    elif lable_bias == 'dst':
        mask_ill = mask_dst
    else:
        mask_ill = mask_src | mask_dst

    edge_lable=torch.zeros_like(timestamps)
    edge_lable[mask_ill]=1

    full_data = TemporalData(src=edge_index[0, :], dst=edge_index[1, :], t=timestamps,
                             msg=edge_attr, y=edge_lable)  # Create a PyG Data object with the graph data

    full_inds = torch.arange(full_data.t.size(0))
    train_mask = full_inds < e_tr
    val_mask = (full_inds >= e_tr) & (full_inds < e_val)
    test_mask = full_inds >= e_val
    train_data = full_data[train_mask]
    val_data = full_data[val_mask]
    test_data = full_data[test_mask]
    # Print the distribution

    print("illicit rate in train data: ", train_data.y[train_data.y == 1].size(0)/train_data.y.size(0))
    print("illicit rate in val data: ", val_data.y[val_data.y == 1].size(0)/val_data.y.size(0))
    print("illicit rate in test data: ", test_data.y[test_data.y == 1].size(0)/test_data.y.size(0))

    remove_label_rate = 0.00

    illict_pos = torch.where(train_data.y == 1)[0]
    licit_pos = torch.where(train_data.y == 0)[0]
    num_cols = illict_pos.shape[0]
    num_cols_to_select = int(num_cols * remove_label_rate)
    indices_to_select = torch.randperm(num_cols)[:num_cols_to_select]
    changed_illict_pos = illict_pos[indices_to_select]
    num_cols = licit_pos.shape[0]
    num_cols_to_select = int(num_cols * remove_label_rate)
    indices_to_select = torch.randperm(num_cols)[:num_cols_to_select]
    changed_licit_pos = licit_pos[indices_to_select]
    changed_pos = torch.cat([changed_illict_pos, changed_licit_pos])
    y_labels_train=train_data.y.clone()
    y_labels_train[changed_pos] = 2
    y_labels_train[y_labels_train == 1] = -1
    y_labels_train[y_labels_train == 0] = 1
    train_data.sy= y_labels_train

    return full_data, train_data, val_data, test_data

def get_semi_data_cold_start(
    lable_bias=None,
    tr_ratio=0.65,
    val_ratio=0.15,
    test_ratio=0.20,
    cold_start=True,
    unseen_until='train',      # 'train' or 'val'
    test_cold_only=True,       # True: test_data 直接输出为 cold正样本+负样本
    neg_pos_ratio=None,        # 仅在 test_cold_only=True 时生效；如 5 表示负样本=正样本5倍
    remove_label_rate=0.00,
    graph_path=DEFAULT_GRAPH_PATH
):
    assert abs(tr_ratio + val_ratio + test_ratio - 1.0) < 1e-8, "ratios must sum to 1"
    assert unseen_until in ['train', 'val']

    # ========= 1) 读图 =========
    with open(graph_path, 'rb') as f:
        subgraph2 = pickle.load(f)
    print("Graph read successfully")

    nodes = list(subgraph2.nodes())
    node_mapping = {node: i for i, node in enumerate(nodes)}
    num_nodes = len(nodes)

    node_labels = torch.tensor(
        [subgraph2.nodes[node].get("isp", 0) for node in nodes],
        dtype=torch.long
    )

    # ========= 2) 提取边并按时间排序 =========
    src_list, dst_list, amt_list, ts_list = [], [], [], []
    for u, v, key, data in subgraph2.edges(keys=True, data=True):
        src_list.append(node_mapping[u])
        dst_list.append(node_mapping[v])
        amt_list.append(float(data.get("amount", 0)))
        ts_list.append(float(data.get("timestamp", 0)))

    src = torch.tensor(src_list, dtype=torch.long)
    dst = torch.tensor(dst_list, dtype=torch.long)
    amount = torch.tensor(amt_list, dtype=torch.float).unsqueeze(1)
    t_raw = torch.tensor(ts_list, dtype=torch.float)

    sort_idx = torch.argsort(t_raw, descending=False)
    src = src[sort_idx]
    dst = dst[sort_idx]
    amount = amount[sort_idx]
    t_raw = t_raw[sort_idx]

    t = t_raw - t_raw[0]
    msg = z_norm(amount)

    E = t.size(0)
    e_tr = math.floor(tr_ratio * E)
    e_val = math.floor((tr_ratio + val_ratio) * E)

    # ========= 3) 边标签 =========
    ill_nodes = torch.where(node_labels == 1)[0]
    mask_src_ill = torch.isin(src, ill_nodes)
    mask_dst_ill = torch.isin(dst, ill_nodes)

    if lable_bias == 'src':
        mask_ill = mask_src_ill
    elif lable_bias == 'dst':
        mask_ill = mask_dst_ill
    else:
        mask_ill = mask_src_ill | mask_dst_ill

    edge_label = torch.zeros(E, dtype=torch.long)
    edge_label[mask_ill] = 1

    # ========= 4) 时间切分 =========
    idx_all = torch.arange(E)
    train_time_mask = idx_all < e_tr
    val_time_mask = (idx_all >= e_tr) & (idx_all < e_val)
    test_time_mask = idx_all >= e_val

    # ========= 5) 节点首次出现时间 =========
    src_np = src.cpu().numpy()
    dst_np = dst.cpu().numpy()
    t_np = t.cpu().numpy()

    first_seen_np = np.full((num_nodes,), np.inf, dtype=np.float64)
    np.minimum.at(first_seen_np, src_np, t_np)
    np.minimum.at(first_seen_np, dst_np, t_np)
    first_seen = torch.from_numpy(first_seen_np).to(t.dtype)

    t_train_end = t[e_tr - 1] if e_tr > 0 else torch.tensor(float('-inf'), dtype=t.dtype)
    t_val_end = t[e_val - 1] if e_val > 0 else torch.tensor(float('-inf'), dtype=t.dtype)

    # ========= 6) cold-start =========
    if cold_start:
        cutoff = t_train_end if unseen_until == 'train' else t_val_end
        cold_phish_nodes = ill_nodes[first_seen[ill_nodes] > cutoff]
        warm_phish_nodes = ill_nodes[~torch.isin(ill_nodes, cold_phish_nodes)]  # NEW

        cold_node_mask = torch.isin(src, cold_phish_nodes) | torch.isin(dst, cold_phish_nodes)
        warm_phish_node_mask = torch.isin(src, warm_phish_nodes) | torch.isin(dst, warm_phish_nodes)  # NEW

        # 训练去泄漏
        train_mask = train_time_mask & (~cold_node_mask)

        # val 是否去泄漏
        if unseen_until == 'val':
            val_mask = val_time_mask & (~cold_node_mask)
        else:
            val_mask = val_time_mask

        test_pos_cold_mask = test_time_mask & cold_node_mask & (edge_label == 1) & (~warm_phish_node_mask)
        test_neg_mask = test_time_mask & (edge_label == 0) & (~warm_phish_node_mask)

        if test_cold_only:
            # 直接让 test_data = cold正样本 + 负样本(已去掉warm-phish相关边)
            if (neg_pos_ratio is not None) and (neg_pos_ratio > 0):
                pos_idx = torch.where(test_pos_cold_mask)[0]
                neg_idx = torch.where(test_neg_mask)[0]
                n_pos = pos_idx.numel()
                n_neg_keep = min(neg_idx.numel(), int(n_pos * neg_pos_ratio))

                test_mask = torch.zeros_like(test_time_mask, dtype=torch.bool)
                if n_pos > 0:
                    test_mask[pos_idx] = True
                if n_neg_keep > 0:
                    keep_neg = neg_idx[torch.randperm(neg_idx.numel())[:n_neg_keep]]
                    test_mask[keep_neg] = True
            else:
                test_mask = test_pos_cold_mask | test_neg_mask
        else:
            # 若不过滤成cold-only，也建议至少去掉warm-phish相关边（可按需改回 test_time_mask）
            test_mask = test_time_mask & (~warm_phish_node_mask)
    else:
        cold_phish_nodes = torch.tensor([], dtype=torch.long)
        warm_phish_nodes = torch.tensor([], dtype=torch.long)
        train_mask, val_mask, test_mask = train_time_mask, val_time_mask, test_time_mask
        test_pos_cold_mask = torch.zeros(E, dtype=torch.bool)
        test_neg_mask = test_time_mask & (edge_label == 0)

    # ========= 7) 打印统计 =========
    def count_nodes(mask):
        n_edges = mask.sum().item()
        if n_edges == 0:
            return 0, 0, torch.tensor([], dtype=torch.long)
        nset = torch.unique(torch.cat([src[mask], dst[mask]], dim=0))
        n_nodes = nset.numel()
        n_phish = torch.isin(nset, ill_nodes).sum().item()
        return n_nodes, n_phish, nset

    before_edges = test_time_mask.sum().item()
    before_nodes, before_phish_nodes, _ = count_nodes(test_time_mask)

    after_edges = test_mask.sum().item()
    after_nodes, after_phish_nodes, after_node_set = count_nodes(test_mask)

    print("\n===== Test Split Statistics (Before vs After) =====")
    print(f"[Before] test edges (time-only): {before_edges}")
    print(f"[Before] test unique nodes:      {before_nodes}")
    print(f"[Before] test phishing nodes:    {before_phish_nodes}")
    print(f"[Before] illicit edges:          {(edge_label[test_time_mask] == 1).sum().item()}")

    print(f"[After ] test edges (final):     {after_edges}")
    print(f"[After ] test unique nodes:      {after_nodes}")
    print(f"[After ] test phishing nodes:    {after_phish_nodes}")
    print(f"[After ] illicit edges:          {(edge_label[test_mask] == 1).sum().item()}")

    if cold_start:
        num_cold_total = cold_phish_nodes.numel()
        num_cold_in_test_after = torch.isin(cold_phish_nodes, after_node_set).sum().item() if after_nodes > 0 else 0
        num_warm_in_test_after = torch.isin(warm_phish_nodes, after_node_set).sum().item() if after_nodes > 0 else 0  # NEW
        print(f"Cold phishing nodes total:       {num_cold_total}")
        print(f"Cold phishing nodes in test:     {num_cold_in_test_after}")
        print(f"Warm phishing nodes in test:     {num_warm_in_test_after}")  # NEW
        print("----- Cold subset composition in test-time window -----")
        print(f"cold positive edges:             {test_pos_cold_mask.sum().item()}")
        print(f"negative edges(after filter):    {test_neg_mask.sum().item()}")

    # ========= 8) 组装 TemporalData =========
    base_data = TemporalData(src=src, dst=dst, t=t, msg=msg, y=edge_label)
    train_data = base_data[train_mask]
    val_data = base_data[val_mask]
    test_data = base_data[test_mask]

    def safe_rate(y):
        return 0.0 if y.numel() == 0 else (y == 1).sum().item() / y.numel()

    print(f"\n#Edges total/train/val/test: {E}/{train_data.y.numel()}/{val_data.y.numel()}/{test_data.y.numel()}")
    print("illicit rate in train data:", safe_rate(train_data.y))
    print("illicit rate in val data:", safe_rate(val_data.y))
    print("illicit rate in test data:", safe_rate(test_data.y))
    print(f"#Cold phishing nodes: {cold_phish_nodes.numel()} (unseen_until={unseen_until}, test_cold_only={test_cold_only})")

    # 泄漏检查
    if cold_start and train_data.y.numel() > 0 and cold_phish_nodes.numel() > 0:
        train_nodes = torch.unique(torch.cat([train_data.src, train_data.dst], dim=0))
        leak = torch.isin(cold_phish_nodes, train_nodes).any().item()
        print("Leakage check (cold nodes in train?):", bool(leak))

    # ========= 9) 半监督标签处理 =========
    illict_pos = torch.where(train_data.y == 1)[0]
    licit_pos = torch.where(train_data.y == 0)[0]

    num_illict_select = int(illict_pos.shape[0] * remove_label_rate)
    num_licit_select = int(licit_pos.shape[0] * remove_label_rate)

    changed_illict_pos = (
        illict_pos[torch.randperm(illict_pos.shape[0])[:num_illict_select]]
        if illict_pos.numel() > 0 and num_illict_select > 0 else torch.tensor([], dtype=torch.long)
    )
    changed_licit_pos = (
        licit_pos[torch.randperm(licit_pos.shape[0])[:num_licit_select]]
        if licit_pos.numel() > 0 and num_licit_select > 0 else torch.tensor([], dtype=torch.long)
    )

    changed_pos = (
        torch.cat([changed_illict_pos, changed_licit_pos], dim=0)
        if (changed_illict_pos.numel() + changed_licit_pos.numel()) > 0 else torch.tensor([], dtype=torch.long)
    )

    y_labels_train = train_data.y.clone()
    if changed_pos.numel() > 0:
        y_labels_train[changed_pos] = 2
    y_labels_train[y_labels_train == 1] = -1
    y_labels_train[y_labels_train == 0] = 1
    train_data.sy = y_labels_train

    # 你需要的对齐版 full_data：train + val + test
    full_data = TemporalData(
        src=torch.cat([train_data.src, val_data.src, test_data.src], dim=0),
        dst=torch.cat([train_data.dst, val_data.dst, test_data.dst], dim=0),
        t=torch.cat([train_data.t, val_data.t, test_data.t], dim=0),
        msg=torch.cat([train_data.msg, val_data.msg, test_data.msg], dim=0),
        y=torch.cat([train_data.y, val_data.y, test_data.y], dim=0),
    )

    return full_data, train_data, val_data, test_data, cold_phish_nodes


def get_semi_data_by_nodes():
    # Load the graph
    with open(DEFAULT_GRAPH_PATH, 'rb') as f:
        subgraph2 = pickle.load(f)
        print("Graph read successfully")

    node_mapping = {node: i for i, node in enumerate(subgraph2.nodes())}

    # Extract edges with numerical indices
    edge_index = []
    edge_attr = []

    for u, v, key, data in subgraph2.edges(keys=True, data=True):
        edge_index.append([node_mapping[u], node_mapping[v]])  # Convert to numerical indices
        edge_attr.append([data.get("amount", 0), data.get("timestamp", 0)])  # Edge features

    # Convert to PyTorch tensors
    edge_index = torch.tensor(np.array(edge_index).T, dtype=torch.long)  # Transpose for PyG format
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    timestamps = edge_attr[:, 1]  # Assuming the second column is the timestamp
    timestamps, index = torch.sort(timestamps, descending=False)  # Sort timestamps in ascending order
    timestamps = timestamps - timestamps[0]  # Normalize timestamps to start from 0
    edge_attr = edge_attr[index, 0].unsqueeze(1)  # Assuming the first column is the amount, reshape to 2D
    edge_attr = z_norm(edge_attr)  # Normalize the edge features
    edge_index = edge_index[:, index]
    y_labels = torch.tensor([subgraph2.nodes[node].get("isp", 0) for node in subgraph2.nodes()],
                            dtype=torch.long)  # Classification labels should be long dtype

    node_first_time = torch.full((y_labels.size(0),), float('-inf'), dtype=torch.float32)
    nodes_flat = edge_index.flatten()
    times_expanded = timestamps.repeat_interleave(2)
    node_first_time.scatter_reduce_(0, nodes_flat, times_expanded.float(), reduce='amax')
    node_first_time[torch.isinf(node_first_time)] = -1

    tr_ratio = 0.65
    val_ratio = 0.15
    test_ratio = 0.20

    e_tr = math.floor(tr_ratio * node_first_time.size(0))
    e_val = e_tr + math.floor(val_ratio * node_first_time.size(0))
    node_time_index = torch.argsort(node_first_time,
                                    descending=False)  # Get indices that would sort the first occurrence times in ascending order
    node_first_time = node_first_time[node_time_index]  # Sort nodes by their first occurrence time
    tr_time = node_first_time[e_tr]
    val_time = node_first_time[e_val]
    full_data = TemporalData(src=edge_index[0, :], dst=edge_index[1, :], t=timestamps,
                             msg=edge_attr)  # Create a PyG Data object with the graph data

    train_mask = timestamps < tr_time
    val_mask = (timestamps >= tr_time) & (timestamps < val_time)
    test_mask = timestamps >= val_time
    train_data = full_data[train_mask]
    val_data = full_data[val_mask]
    test_data = full_data[test_mask]

    train_nodes = torch.unique(torch.cat([train_data.src, train_data.dst]))
    val_nodes = torch.unique(torch.cat([val_data.src, val_data.dst]))
    test_nodes = torch.unique(torch.cat([test_data.src, test_data.dst]))
    y_labels_train = y_labels[train_nodes]
    y_labels_val = y_labels[val_nodes]
    y_labels_test = y_labels[test_nodes]

    unique_train, counts_train = np.unique(y_labels_train, return_counts=True)
    unique_val, counts_val = np.unique(y_labels_val, return_counts=True)
    unique_test, counts_test = np.unique(y_labels_test, return_counts=True)
    # Print the distribution
    print("Class Distribution in train:")
    for cls, count in zip(unique_train, counts_train):
        print(f"Class {cls}: {count} nodes")

    print("Class Distribution in val:")
    for cls, count in zip(unique_val, counts_val):
        print(f"Class {cls}: {count} nodes")

    print("Class Distribution in test:")
    for cls, count in zip(unique_test, counts_test):
        print(f"Class {cls}: {count} nodes")

    remove_label_rate = 0.90

    illict_pos = torch.where(y_labels_train == 1)[0]
    licit_pos = torch.where(y_labels_train == 0)[0]
    num_cols = illict_pos.shape[0]
    num_cols_to_select = int(num_cols * remove_label_rate)
    indices_to_select = torch.randperm(num_cols)[:num_cols_to_select]
    changed_illict_pos = illict_pos[indices_to_select]
    num_cols = licit_pos.shape[0]
    num_cols_to_select = int(num_cols * remove_label_rate)
    indices_to_select = torch.randperm(num_cols)[:num_cols_to_select]
    changed_licit_pos = licit_pos[indices_to_select]
    changed_pos = torch.cat([changed_illict_pos, changed_licit_pos])
    y_labels_train[changed_pos] = 2

    unique_train, counts_train = np.unique(y_labels_train, return_counts=True)
    print("Class Distribution in train:")
    for cls, count in zip(unique_train, counts_train):
        print(f"Class {cls}: {count} nodes")

    return full_data,train_data, val_data, test_data, y_labels_train, y_labels_val, y_labels_test, train_nodes, val_nodes, test_nodes



