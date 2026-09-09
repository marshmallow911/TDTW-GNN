"""Train and evaluate the paper model on the Ethereum transaction graph."""
from runtime_config import parse_args

# Parse before importing ML packages so --help works without the training environment.
args = parse_args()

import timeit
from tqdm import tqdm
import torch
from torch_geometric.loader import TemporalDataLoader
import matplotlib.pyplot as plt
import os
from torch_geometric.utils import degree
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, \
    roc_curve, classification_report,average_precision_score

# internal imports
from tgb.utils.utils import set_random_seed
from nodeproppred.evaluate import Evaluator
from modules.msg_func import IdentityMessage
from modules.msg_agg import LastAggregator,TimeAggregator
# from Tree_final import ViewFusion,NodeEdgePredictor_ori,DiceFocalLoss,DiceFocalLoss_bk,TreeNeighborsampler,GraphTreeWalkEmbedding
from Tree_final_time_mptt import ViewFusion,NodeEdgePredictor_ori,DiceFocalLoss,TreeNeighborsampler,GraphTreeWalkEmbedding,GatingViewFusion,AttentionViewFusion
from torch_geometric.nn import TGNMemory
from ETH_data_loader import get_semi_data,get_semi_data_cold_start
from AML_data_loader import get_semi_bitcoin_data
import numpy as np
import torch.nn.functional as F



def plot_curve(scores, out_name):
    plt.plot(scores, color="#e34a33")
    plt.ylabel("score")
    plt.savefig(out_name + ".pdf")
    plt.close()

def process_edges(src, dst, t, msg):
    if src.nelement() > 0:
        model['memory'].update_state(src, dst, t, msg)
        # neighbor_loader.insert(src, dst)
        neighbor_loader.insert(src, dst, t)

# ==========
# ========== Define helper function...
# ==========

def train():
    model['memory'].train()
    model['gnn1'].train()
    model['gnn2'].train()
    model['node_pred'].train()
    model['fusion'].train()

    model['memory'].reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    num_label_ts = 0
    total_score = 0
    i=0
    for batch in tqdm(train_loader):
        i+=1
        batch = batch.to(device)
        optimizer.zero_grad()

        src, dst, t, msg, label,true_label = batch.src, batch.dst, batch.t.to(torch.int64), batch.msg, batch.sy.to(torch.int64),batch.y.to(torch.int64)
        # 但是我们此处不用等待，有什么训练什么，不用判断是否到了下一天
        process_edges(src, dst, t, msg)
        """
        modified for node property prediction
        1. sample neighbors from the neighbor loader for all nodes to be predicted
        2. extract memory from the sampled neighbors and the nodes
        3. run gnn with the extracted memory embeddings and the corresponding time and message
        """

        n_id, z_inds = torch.cat([src,dst]).unique(return_inverse=True)
        n_id=n_id.to(torch.int)
        cutoff_time = max(t)
        nodes_list_in, nodes_list_out, root_list_in, root_list_out, mem_edge_index_in, mem_edge_index_out, edge_list_in, edge_list_out = neighbor_loader.walk_tree(
            n_id, cutoff_time, num_walk=NUM_WALK, num_hop=NUM_HOP, k=K, gamma=1)
        n_id_neighbors_in,n_id_neighbors_out, mem_edge_index_in, mem_edge_index_out = neighbor_loader(mem_edge_index_in,
                                                                                    mem_edge_index_out,n_id)

        assoc[n_id_neighbors_in] = torch.arange(n_id_neighbors_in.size(0)).to(assoc.device)
        z, last_update = model['memory'](n_id_neighbors_in)
        tree_in, z_in = model['gnn1'](nodes_list_in, root_list_in, z, n_id_neighbors_in, last_update, mem_edge_index_in, data.msg[edge_list_in.cpu()].to(device),
                                     data.t[edge_list_in.cpu()].to(device))

        z_in= z_in[assoc[n_id]]
        assoc[n_id_neighbors_out] = torch.arange(n_id_neighbors_out.size(0)).to(assoc.device)
        z, last_update = model['memory'](n_id_neighbors_out)
        tree_out,z_out = model['gnn2'](nodes_list_out, root_list_out, z, n_id_neighbors_out, last_update, mem_edge_index_out, data.msg[edge_list_out.cpu()].to(device),
                                       data.t[edge_list_out.cpu()].to(device))
        z_out = z_out[assoc[n_id]]
        view_list = [tree_in, z_in, tree_out, z_out]
        z = model['fusion'](view_list)
        src_z = z[z_inds][0:src.size(0)]
        dst_z = z[z_inds][src.size(0):]

        pred = model['node_pred'](src_z, dst_z, msg)
        label_index=label!=2
        pred_hat= pred[label_index]
        true_label_hat = true_label[label_index]
        loss = criterion(pred_hat, true_label_hat)
        np_pred = pred.argmax(dim=-1).cpu().detach().numpy().astype(int)

        np_true = true_label.cpu().detach().numpy().astype(int)
        input_dict = {
            "y_true": np_true,
            "y_pred": np_pred,
            "eval_metric": [metric],
        }
        result_dict = evaluator.eval(input_dict)
        score = result_dict[metric]
        total_score += score
        num_label_ts += 1

        loss.backward()
        optimizer.step()
        total_loss += float(loss)

        model['memory'].detach()

    metric_dict = {
        "ce": total_loss / num_label_ts,
    }
    metric_dict[metric] = total_score / num_label_ts
    return metric_dict


@torch.no_grad()
def test(loader):
    model['memory'].eval()
    model['gnn1'].eval()
    model['gnn2'].eval()
    model['node_pred'].eval()
    model['fusion'].eval()

    total_score = 0
    num_label_ts = 0

    pred_prob_vec = []

    for batch in tqdm(loader):
        batch = batch.to(device)
        src, dst, t, msg, label = batch.src, batch.dst, batch.t.to(torch.int64), batch.msg, batch.y.to(torch.int64)
        # 但是我们此处不用等待，有什么训练什么，不用判断是否到了下一天
        process_edges(src, dst, t, msg)
        """
        modified for node property prediction
        1. sample neighbors from the neighbor loader for all nodes to be predicted
        2. extract memory from the sampled neighbors and the nodes
        3. run gnn with the extracted memory embeddings and the corresponding time and message
        """

        n_id, z_inds = torch.cat([src, dst]).unique(return_inverse=True)
        n_id = n_id.to(torch.int)
        cutoff_time = max(t)
        nodes_list_in, nodes_list_out, root_list_in, root_list_out, mem_edge_index_in, mem_edge_index_out, edge_list_in, edge_list_out = neighbor_loader.walk_tree(
            n_id, cutoff_time, num_walk=NUM_WALK, num_hop=NUM_HOP, k=K, gamma=1)
        n_id_neighbors_in,n_id_neighbors_out, mem_edge_index_in, mem_edge_index_out = neighbor_loader(mem_edge_index_in,
                                                                                    mem_edge_index_out,n_id)
        assoc[n_id_neighbors_in] = torch.arange(n_id_neighbors_in.size(0)).to(assoc.device)
        z, last_update = model['memory'](n_id_neighbors_in)
        tree_in,z_in = model['gnn1'](nodes_list_in, root_list_in, z, n_id_neighbors_in, last_update, mem_edge_index_in, data.msg[edge_list_in.cpu()].to(device),
                                     data.t[edge_list_in.cpu()].to(device))

        z_in= z_in[assoc[n_id]]
        assoc[n_id_neighbors_out] = torch.arange(n_id_neighbors_out.size(0)).to(assoc.device)
        z, last_update = model['memory'](n_id_neighbors_out)
        tree_out,z_out = model['gnn2'](nodes_list_out, root_list_out, z, n_id_neighbors_out, last_update, mem_edge_index_out, data.msg[edge_list_out.cpu()].to(device),
                                       data.t[edge_list_out.cpu()].to(device))
        z_out = z_out[assoc[n_id]]
        view_list = [tree_in, z_in, tree_out, z_out]
        z = model['fusion'](view_list)
        src_z = z[z_inds][0:src.size(0)]
        dst_z = z[z_inds][src.size(0):]
        pred_prob = F.softmax(model['node_pred'](src_z, dst_z, msg), dim=1)
        pred = pred_prob.argmax(dim=-1)

        np_pred = pred.cpu().detach().numpy().astype(int)
        np_true = label.cpu().detach().numpy().astype(int)

        input_dict = {
            "y_true": np_true,
            "y_pred": np_pred,
            "eval_metric": [metric],
        }
        result_dict = evaluator.eval(input_dict)
        score = result_dict[metric]
        total_score += score
        num_label_ts += 1
        pred_prob_vec.append(pred_prob.cpu().detach().numpy())

    metric_dict = {}
    metric_dict[metric] = total_score / num_label_ts
    return metric_dict, np.vstack(pred_prob_vec)


# ==========
# ==========
# ==========

# Start...
start_overall = timeit.default_timer()
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ========== set parameters...
print("INFO: Arguments:", args)

DATA = "tgbn-aml"
LR = args.lr
BATCH_SIZE = args.batch_size
NUM_EPOCH = args.epochs
SEED = args.seed
MEM_DIM = args.mem_dim
TIME_DIM = args.time_dim
EMB_DIM = args.emb_dim
NUM_NEIGHBORS = args.num_neighbors
NUM_HOP = args.num_hop
K = args.k
NUM_WALK = args.num_walk

# setting random seed
torch.manual_seed(SEED)
set_random_seed(SEED)

MODEL_NAME = 'TGN'
USE_SRC_EMB_IN_MSG = False
USE_DST_EMB_IN_MSG = True
evaluator = Evaluator(name=DATA)
# ==========
if torch.cuda.is_available():
    print("GPU可用！")
else:
    print("GPU不可用，将使用CPU进行计算。")
device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu")
                      if args.device == "auto" else args.device)
print("INFO: Device:", device)
if not args.data_path.is_file():
    raise FileNotFoundError(f"Dataset not found: {args.data_path}. See data/README.md.")
args.output_dir.mkdir(parents=True, exist_ok=True)
# set the device
# data, train_data, val_data, test_data = get_semi_bitcoin_data('../DataSet/bitcoin') #
data, train_data, val_data, test_data = get_semi_data(graph_path=args.data_path) #lable_bias='src'
# data, train_data, val_data, test_data, cold_nodes = get_semi_data_cold_start(
#     cold_start=True,
#     unseen_until='train',
#     test_cold_only=True,   # 关键：直接输出冷启动正负样本 test_data
#     neg_pos_ratio=None     # 不下采样负样本；若想平衡可设 3/5/10
# )

d=degree(train_data.edge_index[1,:], dtype=torch.long)
metric = "f1"
num_classes = 2

train_loader = TemporalDataLoader(train_data, batch_size=BATCH_SIZE)
val_loader = TemporalDataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = TemporalDataLoader(test_data, batch_size=BATCH_SIZE)


# neighborhood sampler
neighbor_loader = TreeNeighborsampler(data.num_nodes, size=NUM_NEIGHBORS, device=device)

# define the model end-to-end
memory = TGNMemory(
    data.num_nodes,
    data.msg.size(-1),
    MEM_DIM,
    TIME_DIM,
    message_module=IdentityMessage(data.msg.size(-1), MEM_DIM, TIME_DIM),
    aggregator_module=TimeAggregator(),
).to(device)

gnn_1 = (
    GraphTreeWalkEmbedding(
        in_channels=MEM_DIM,
        out_channels=EMB_DIM,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
        deg=d,
    )
    .to(device)
    .float()
)
gnn_2 = (
    GraphTreeWalkEmbedding(
        in_channels=MEM_DIM,
        out_channels=EMB_DIM,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
        deg=d,
    )
    .to(device)
    .float()
)


view_fusion = ViewFusion(in_channels=EMB_DIM, out_channels=EMB_DIM, views=4).to(device)
# view_fusion = AttentionViewFusion(in_channels=EMB_DIM, out_channels=EMB_DIM, views=4).to(device)
# view_fusion = GatingViewFusion(in_channels=EMB_DIM, out_channels=EMB_DIM, views=4).to(device)
node_pred = NodeEdgePredictor_ori(in_dim=2*EMB_DIM+data.msg.size(-1),out_dim=num_classes).to(device)

model = {'memory': memory,
         'gnn1': gnn_1,
         'gnn2': gnn_2,
         'fusion': view_fusion,
         'node_pred': node_pred}

optimizer = torch.optim.Adam(
    set(memory.parameters()) | set(gnn_1.parameters()) | set(gnn_2.parameters()) | set(node_pred.parameters()) | set(view_fusion.parameters()),
    lr=LR,
)


criterion = DiceFocalLoss(beta=1)
assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
train_curve = []
val_curve = []
test_curve = []
train_time_list = []
test_time_list = []
train_mem_list = []
test_mem_list = []
max_val_score = 0  #find the best test score based on validation score
best_test_idx = 0

for epoch in range(1, NUM_EPOCH + 1):
    start_time = timeit.default_timer()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_dict = train()
    train_time = timeit.default_timer() - start_time
    if device.type == "cuda":
        train_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        train_mem = 0.0
    train_time_list.append(train_time)
    train_mem_list.append(train_mem)
    print("------------------------------------")
    print(f"training Epoch: {epoch:02d}")
    print(train_dict)
    train_curve.append(train_dict["ce"])
    print("Training takes--- %s seconds ---" % train_time)



    start_time = timeit.default_timer()
    val_dict, pred_prob_vec = test(val_loader)
    y_pred_bi = pred_prob_vec.argmax(axis=1)
    print(val_dict)
    AP = round(
        average_precision_score(val_data.y[:len(pred_prob_vec)], pred_prob_vec[:, 1], average='macro', pos_label=1,
                                sample_weight=None), 5)
    val_curve.append(AP)
    if (val_dict[metric] > max_val_score):
        max_val_score = val_dict[metric]
        best_test_idx = epoch - 1
        torch.save({
            'memory': model['memory'].state_dict(),
            'gnn1': model['gnn1'].state_dict(),
            'gnn2': model['gnn2'].state_dict(),
            'fusion': model['fusion'].state_dict(),
            'node_pred': model['node_pred'].state_dict(),
        }, args.output_dir / 'model_parameters_ETH.pth')
    print("Validation takes--- %s seconds ---" % (timeit.default_timer() - start_time))

    start_time = timeit.default_timer()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    test_dict, pred_prob_vec = test(test_loader)
    test_time = timeit.default_timer() - start_time
    if device.type == "cuda":
        test_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        test_mem = 0.0
    test_time_list.append(test_time)
    test_mem_list.append(test_mem)
    y_pred_bi=pred_prob_vec.argmax(axis=1)
    test_precision = round(precision_score(test_data.y[:len(pred_prob_vec)], y_pred_bi, average="binary") * 100, 5)
    test_recall = round(recall_score(test_data.y[:len(pred_prob_vec)], y_pred_bi, average="binary") * 100, 5)
    test_f1 = round(f1_score(test_data.y[:len(pred_prob_vec)], y_pred_bi) * 100, 5)
    print(f"Test Precision: {test_precision}%")
    print(f"Test Recall: {test_recall}%")
    print(f"Test F1 Score: {test_f1}%")
    AP = round(average_precision_score(test_data.y[:len(pred_prob_vec)], pred_prob_vec[:, 1], average='macro', pos_label=1, sample_weight=None)* 100, 5)
    print(f"AP Score: {AP}%")
    print(test_dict)
    test_dict["Precision"]=test_precision
    test_dict["Recall"] = test_recall
    test_dict["F1"] = test_f1
    test_dict["AP"] = AP
    test_curve.append(test_f1)
    print("Test takes--- %s seconds ---" % (timeit.default_timer() - start_time))
    print("------------------------------------")
    latest_ts= data.t.min().item()

    torch.cuda.empty_cache()


# import pickle
#
# plot_curve(train_curve, "train_curve_eth")
# with open("train_curve_eth.pkl", "wb") as f:
#     pickle.dump(train_curve, f)
# plot_curve(val_curve, "val_curve_eth")
# with open("val_curve_eth.pkl", "wb") as f:
#     pickle.dump(val_curve, f)
# plot_curve(test_curve, "test_curve_eth")

max_test_score = test_curve[best_test_idx]
print("------------------------------------")
print("------------------------------------")
print ("best val score: ", max_val_score)
print ("best validation epoch   : ", best_test_idx + 1)
print ("best test score: ", max_test_score)
if train_time_list:
    avg_train_time = sum(train_time_list) / len(train_time_list)
    avg_train_mem = sum(train_mem_list) / len(train_mem_list)
    print("avg train epoch time (s): ", round(avg_train_time, 4))
    print("avg train peak mem (MB): ", round(avg_train_mem, 2))
if test_time_list:
    avg_test_time = sum(test_time_list) / len(test_time_list)
    avg_test_mem = sum(test_mem_list) / len(test_mem_list)
    print("avg test time (s): ", round(avg_test_time, 4))
    print("avg test peak mem (MB): ", round(avg_test_mem, 2))

