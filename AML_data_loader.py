import pandas as pd
import numpy as np
import torch
import logging
import itertools
from torch_geometric.data import TemporalData
import math


def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
    return (data - data.mean(0).unsqueeze(0)) / std


def add_ports(num_nodes,edge_index,timestamps,edge_attr):
    '''Adds port numberings to the edge features'''
    # 虽然按照了时间排序，但实际上用了全图的端口，batch时也是取了全图的
    adj_list_in, adj_list_out = to_adj_nodes_with_times(num_nodes,edge_index,timestamps)
    in_ports = ports(edge_index, adj_list_in)
    out_ports = [ports(edge_index.flipud(), adj_list_out)]
    edge_attr = torch.cat(out_ports + [in_ports,edge_attr] , dim=1)
    return edge_attr

def to_adj_nodes_with_times(num_nodes,edge_index,timestamps):
    timestamps = torch.zeros((edge_index.shape[1], 1)) if timestamps is None else timestamps.reshape((-1,1))
    edges = torch.cat((edge_index.T, timestamps), dim=1)
    adj_list_out = dict([(i, []) for i in range(num_nodes)])
    adj_list_in = dict([(i, []) for i in range(num_nodes)])
    for u,v,t in edges:
        u,v,t = int(u), int(v), int(t)
        adj_list_out[u] += [(v, t)]
        adj_list_in[v] += [(u, t)]
    return adj_list_in, adj_list_out


def ports(edge_index, adj_list):
    ports = torch.zeros(edge_index.shape[1], 1)
    ports_dict = {}
    for v, nbs in adj_list.items():
        if len(nbs) < 1: continue
        a = np.array(nbs)
        a = a[a[:, -1].argsort()]
        _, idx = np.unique(a[:,[0]],return_index=True,axis=0)
        nbs_unique = a[np.sort(idx)][:,0]
        for i, u in enumerate(nbs_unique):
            ports_dict[(u,v)] = i
    for i, e in enumerate(edge_index.T):
    # 上面是按照时间排序拍的，满足不使用未来信息的要求，可用
        ports[i] = ports_dict[tuple(e.numpy())]
    return ports

def get_data(path):
    '''Loads the AML transaction data.

    1. The data is loaded from the csv and the necessary features are chosen.
    2. The data is split into training, validation and test data.
    3. PyG Data objects are created with the respective data splits.
    '''

    transaction_file = f"{path}/formatted_transactions.csv"  # replace this with your path to the respective AML data objects
    df_edges = pd.read_csv(transaction_file)

    logging.info(f'Available Edge Features: {df_edges.columns.tolist()}')

    df_edges['Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()

    max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
    df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})
    timestamps = torch.LongTensor(df_edges['Timestamp'].to_numpy())
    y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())

    logging.info(f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
    logging.info(f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}")
    logging.info(f"Number of transactions = {df_edges.shape[0]}")

    edge_features = ['Sent Currency', 'Amount Received', 'Received Currency', 'Payment Format']
    node_features = ['Feature']

    logging.info(f'Edge features being used: {edge_features}')
    logging.info(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

    edge_index = torch.LongTensor(df_edges.loc[:, ['from_id', 'to_id']].to_numpy().T)
    edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()

    n_days = int(timestamps.max() / (3600 * 24) + 1)
    n_samples = y.shape[0]
    logging.info(f'number of days and transactions in the data: {n_days} days, {n_samples} transactions')

    # data splitting
    daily_irs, weighted_daily_irs, daily_inds, daily_trans = [], [], [], []  # irs = illicit ratios, inds = indices, trans = transactions
    for day in range(n_days):
        l = day * 24 * 3600
        r = (day + 1) * 24 * 3600
        day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
        daily_irs.append(y[day_inds].float().mean())
        weighted_daily_irs.append(y[day_inds].float().mean() * day_inds.shape[0] / n_samples)
        daily_inds.append(day_inds)
        daily_trans.append(day_inds.shape[0])

    split_per = [0.6, 0.2, 0.2]
    daily_totals = np.array(daily_trans)
    d_ts = daily_totals
    I = list(range(len(d_ts)))
    split_scores = dict()
    for i, j in itertools.combinations(I, 2):
        if j >= i:
            split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
            split_totals_sum = np.sum(split_totals)
            split_props = [v / split_totals_sum for v in split_totals]
            split_error = [abs(v - t) / t for v, t in zip(split_props, split_per)]
            score = max(split_error)  # - (split_totals_sum/total) + 1
            split_scores[(i, j)] = score
        else:
            continue

    i, j = min(split_scores, key=split_scores.get)
    # split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
    split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]
    logging.info(f'Calculate split: {split}')

    # Now, we seperate the transactions based on their indices in the timestamp array
    split_inds = {k: [] for k in range(3)}
    for i in range(3):
        for day in split[i]:
            split_inds[i].append(daily_inds[
                                     day])  # split_inds contains a list for each split (tr,val,te) which contains the indices of each day seperately

    tr_inds = torch.cat(split_inds[0])
    val_inds = torch.cat(split_inds[1])
    te_inds = torch.cat(split_inds[2])

    logging.info(f"Total train samples: {tr_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
                 f"{y[tr_inds].float().mean() * 100 :.2f}% || Train days: {split[0][:5]}")
    logging.info(f"Total val samples: {val_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
                 f"{y[val_inds].float().mean() * 100:.2f}% || Val days: {split[1][:5]}")
    logging.info(f"Total test samples: {te_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
                 f"{y[te_inds].float().mean() * 100:.2f}% || Test days: {split[2][:5]}")

    # Creating the final data objects
    e_tr = tr_inds.numpy()
    e_val = val_inds.numpy()
    e_te = te_inds.numpy()

    full_data = TemporalData(src=edge_index[0,:],dst=edge_index[1,:],t=timestamps.float(),msg=edge_attr,y=y)
    # full_data_edge_index= torch.stack([full_data.src, full_data.dst], dim=0)
    # full_data.msg=add_ports(max_n_id, full_data_edge_index, full_data.t, full_data.msg)
    full_data.msg[:, :-1] = z_norm(full_data.msg[:, :-1])

    full_inds = torch.arange(full_data.t.size(0))
    train_mask = torch.isin(full_inds, torch.tensor(e_tr))
    val_mask = torch.isin(full_inds, torch.tensor(e_val))
    test_mask = torch.isin(full_inds, torch.tensor(e_te))
    train_data = full_data[train_mask]
    val_data = full_data[val_mask]
    test_data = full_data[test_mask]



    return full_data, train_data, val_data, test_data

def get_data_mask(path):
    '''Loads the AML transaction data.

    1. The data is loaded from the csv and the necessary features are chosen.
    2. The data is split into training, validation and test data.
    3. PyG Data objects are created with the respective data splits.
    '''

    transaction_file = f"{path}/formatted_transactions.csv"  # replace this with your path to the respective AML data objects
    df_edges = pd.read_csv(transaction_file)

    logging.info(f'Available Edge Features: {df_edges.columns.tolist()}')

    df_edges['Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()

    max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
    df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})
    timestamps = torch.LongTensor(df_edges['Timestamp'].to_numpy())
    y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())

    logging.info(f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
    logging.info(f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}")
    logging.info(f"Number of transactions = {df_edges.shape[0]}")

    edge_features = ['Sent Currency', 'Amount Received', 'Received Currency', 'Payment Format']
    node_features = ['Feature']

    logging.info(f'Edge features being used: {edge_features}')
    logging.info(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

    edge_index = torch.LongTensor(df_edges.loc[:, ['from_id', 'to_id']].to_numpy().T)
    edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()

    n_days = int(timestamps.max() / (3600 * 24) + 1)
    n_samples = y.shape[0]
    logging.info(f'number of days and transactions in the data: {n_days} days, {n_samples} transactions')

    # data splitting
    daily_irs, weighted_daily_irs, daily_inds, daily_trans = [], [], [], []  # irs = illicit ratios, inds = indices, trans = transactions
    for day in range(n_days):
        l = day * 24 * 3600
        r = (day + 1) * 24 * 3600
        day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
        daily_irs.append(y[day_inds].float().mean())
        weighted_daily_irs.append(y[day_inds].float().mean() * day_inds.shape[0] / n_samples)
        daily_inds.append(day_inds)
        daily_trans.append(day_inds.shape[0])

    split_per = [0.6, 0.2, 0.2]
    daily_totals = np.array(daily_trans)
    d_ts = daily_totals
    I = list(range(len(d_ts)))
    split_scores = dict()
    for i, j in itertools.combinations(I, 2):
        if j >= i:
            split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
            split_totals_sum = np.sum(split_totals)
            split_props = [v / split_totals_sum for v in split_totals]
            split_error = [abs(v - t) / t for v, t in zip(split_props, split_per)]
            score = max(split_error)  # - (split_totals_sum/total) + 1
            split_scores[(i, j)] = score
        else:
            continue

    i, j = min(split_scores, key=split_scores.get)
    # split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
    split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]
    logging.info(f'Calculate split: {split}')

    # Now, we seperate the transactions based on their indices in the timestamp array
    split_inds = {k: [] for k in range(3)}
    for i in range(3):
        for day in split[i]:
            split_inds[i].append(daily_inds[
                                     day])  # split_inds contains a list for each split (tr,val,te) which contains the indices of each day seperately

    tr_inds = torch.cat(split_inds[0])
    val_inds = torch.cat(split_inds[1])
    te_inds = torch.cat(split_inds[2])

    logging.info(f"Total train samples: {tr_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
                 f"{y[tr_inds].float().mean() * 100 :.2f}% || Train days: {split[0][:5]}")
    logging.info(f"Total val samples: {val_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
                 f"{y[val_inds].float().mean() * 100:.2f}% || Val days: {split[1][:5]}")
    logging.info(f"Total test samples: {te_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
                 f"{y[te_inds].float().mean() * 100:.2f}% || Test days: {split[2][:5]}")

    # Creating the final data objects
    e_tr = tr_inds.numpy()
    e_val = val_inds.numpy()
    e_te = te_inds.numpy()

    full_data = TemporalData(src=edge_index[0,:],dst=edge_index[1,:],t=timestamps.float(),msg=edge_attr,y=y)
    # full_data_edge_index= torch.stack([full_data.src, full_data.dst], dim=0)
    # full_data.msg=add_ports(max_n_id, full_data_edge_index, full_data.t, full_data.msg)
    full_data.msg[:, :-1] = z_norm(full_data.msg[:, :-1])

    full_inds = torch.arange(full_data.t.size(0))
    train_mask = torch.isin(full_inds, torch.tensor(e_tr))
    val_mask = torch.isin(full_inds, torch.tensor(e_val))
    test_mask = torch.isin(full_inds, torch.tensor(e_te))
    train_data = full_data[train_mask]
    val_data = full_data[val_mask]
    test_data = full_data[test_mask]



    return full_data, train_mask, val_mask, test_mask



def get_bitcoin_data(path):
    '''Loads the AML transaction data.

    1. The data is loaded from the csv and the necessary features are chosen.
    2. The data is split into training, validation and test data.
    3. PyG Data objects are created with the respective data splits.
    '''

    transaction_file = f"{path}/soc-sign-bitcoinalpha.csv"  # replace this with your path to the respective AML data objects
    df_edges = pd.read_csv(transaction_file)

    logging.info(f'Available Edge Features: {df_edges.columns.tolist()}')

    df_edges['time'] = df_edges['time'] - df_edges['time'].min()

    nodes=np.unique(df_edges.loc[:, ['src', 'dst']].values.flatten())
    node_mapping = {node: i for i, node in enumerate(nodes)}
    df_edges['src'] = df_edges['src'].map(node_mapping)
    df_edges['dst'] = df_edges['dst'].map(node_mapping)
    timestamps = torch.LongTensor(df_edges['time'].to_numpy())
    timestamps, index = torch.sort(timestamps, descending=False)  # Sort timestamps in ascending order
    y = torch.LongTensor(df_edges.iloc[index,2].to_numpy())  # Convert negative rates to 1 (illicit), positive rates to 0 (licit)
    y[y >= 0.0] = 0
    y[y < 0.0] = 1

    edge_index = torch.LongTensor(df_edges.loc[index, ['src', 'dst']].to_numpy().T)
    edge_attr = torch.Tensor(df_edges.iloc[index, 2].abs().to_numpy()).unsqueeze(1)  # Placeholder for edge attributes, can be modified later

    split_per = [0.65, 0.15, 0.20]

    e_tr = math.floor(split_per[0] * edge_index.size(1))
    e_val = math.floor(e_tr + split_per[1] * edge_index.size(1))
    full_inds = torch.arange(timestamps.size(0))
    train_mask = full_inds < e_tr
    val_mask = (full_inds >= e_tr) & (full_inds < e_val)
    test_mask = full_inds >= e_val

    full_data = TemporalData(src=edge_index[0,:],dst=edge_index[1,:],t=timestamps.float(),msg=edge_attr,y=y)

    train_data = full_data[train_mask]
    val_data = full_data[val_mask]
    test_data = full_data[test_mask]

    print("illicit rate in train data: ", train_data.y[train_data.y == 1].size(0)/train_data.y.size(0))
    print("illicit rate in val data: ", val_data.y[val_data.y == 1].size(0)/val_data.y.size(0))
    print("illicit rate in test data: ", test_data.y[test_data.y == 1].size(0)/test_data.y.size(0))

    return full_data, train_data, val_data, test_data


def get_semi_bitcoin_data(path):
    '''Loads the AML transaction data.

    1. The data is loaded from the csv and the necessary features are chosen.
    2. The data is split into training, validation and test data.
    3. PyG Data objects are created with the respective data splits.
    '''

    transaction_file = f"{path}/soc-sign-bitcoinalpha.csv"  # replace this with your path to the respective AML data objects
    df_edges = pd.read_csv(transaction_file)

    logging.info(f'Available Edge Features: {df_edges.columns.tolist()}')

    df_edges['time'] = df_edges['time'] - df_edges['time'].min()

    nodes=np.unique(df_edges.loc[:, ['src', 'dst']].values.flatten())
    node_mapping = {node: i for i, node in enumerate(nodes)}
    df_edges['src'] = df_edges['src'].map(node_mapping)
    df_edges['dst'] = df_edges['dst'].map(node_mapping)
    timestamps = torch.Tensor(df_edges['time'].to_numpy())/60/60
    timestamps, index = torch.sort(timestamps, descending=False)  # Sort timestamps in ascending order
    y = torch.LongTensor(df_edges.iloc[index,2].to_numpy())  # Convert negative rates to 1 (illicit), positive rates to 0 (licit)
    y[y > 0.5] = 0
    y[y < -0.5] = 1

    edge_index = torch.LongTensor(df_edges.loc[index, ['src', 'dst']].to_numpy().T)
    edge_attr = torch.zeros_like(timestamps, dtype=torch.float32).unsqueeze(1)  # Placeholder for edge attributes, can be modified later

    split_per = [0.70, 0.15, 0.15]

    e_tr = math.floor(split_per[0] * edge_index.size(1))
    e_val = math.floor(e_tr + split_per[1] * edge_index.size(1))
    full_inds = torch.arange(timestamps.size(0))
    train_mask = full_inds < e_tr
    val_mask = (full_inds >= e_tr) & (full_inds < e_val)
    test_mask = full_inds >= e_val

    full_data = TemporalData(src=edge_index[0,:],dst=edge_index[1,:],t=timestamps,msg=edge_attr,y=y)

    train_data = full_data[train_mask]
    val_data = full_data[val_mask]
    test_data = full_data[test_mask]

    print("illicit rate in train data: ", train_data.y[train_data.y == 1].size(0)/train_data.y.size(0))
    print("illicit rate in val data: ", val_data.y[val_data.y == 1].size(0)/val_data.y.size(0))
    print("illicit rate in test data: ", test_data.y[test_data.y == 1].size(0)/test_data.y.size(0))

    remove_label_rate = 0.90

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


if __name__ == "__main__":
    data, train_inds, val_inds, test_inds = get_bitcoin_data('../DataSet/bitcoin')