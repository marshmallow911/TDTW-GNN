# TDTW-GNN

Code for **Temporal-Directional Tree-Walk Graph Neural Network for Illicit Transaction Detection**.

**DOI:** [10.1109/TNNLS.2026.3732968](https://doi.org/10.1109/TNNLS.2026.3732968)

The training and evaluation entry point is `Tree_gnn_smalltest_2.py`.

## Repository layout

```text
Tree_gnn_smalltest_2.py    Training and evaluation entry point
runtime_config.py        Command-line options and experiment defaults
Tree_final_time_mptt.py   Tree sampler, embeddings, fusion, predictor, loss
flow.py                  Normalizing-flow components imported by the model
ETH_data_loader.py       Ethereum graph loading and temporal splits
AML_data_loader.py       Bitcoin/AML loaders (imported by the original entry point)
modules/                 Message function and aggregation
nodeproppred/            Local evaluation implementation
requirements.txt         Dependencies extracted from the original environment
data/                   Dataset placement instructions (data excluded from Git)
outputs/                 Generated checkpoints (excluded from Git)
docs/                    Preparation notes and source provenance
LICENSE                  Preserved upstream license
```

## Environment

Use Python 3.10 for the reference dependency versions. The dependency list comes from the original repository; the cluster-specific `+computecanada` suffixes have been removed. It is not a newly validated environment lock.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install `torch==1.13.1` and `torch-scatter==2.1.1` builds matching your operating system and CUDA/CPU setup, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

`py-tgb` supplies `tgb.utils.utils.set_random_seed`; there is no missing local `tgb/` directory. PyTorch/PyG extension binaries must match the chosen PyTorch and CUDA versions.

## Data

Place the original experiment's `subgraph.pkl` in `data/`, or pass its path using `--data-path`. See [data/README.md](data/README.md) for the expected graph format. The GitHub repository also contains the existing `subgraph.7z` archive. Extract it separately and place the resulting `subgraph.pkl` at the path above. The archive contents have not been validated during code preparation; the graph preprocessing pipeline is not included.

## Run

From this directory:

```bash
python Tree_gnn_smalltest_2.py --help
python Tree_gnn_smalltest_2.py --data-path /path/to/subgraph.pkl --device cuda:0
```

Default device selection uses `cuda:0` when CUDA is available, otherwise CPU. To use the original experiment's GPU index, specify `--device cuda:2`. GPU visibility can be configured through the shell's `CUDA_VISIBLE_DEVICES` environment variable.

For a shorter execution check with real data:

```bash
python Tree_gnn_smalltest_2.py --data-path /path/to/subgraph.pkl --epochs 1
```

This still processes the full dataset; it is not a paper reproduction run. Defaults retain 100 epochs, learning rate 0.0015, batch size 8192, seed 10, memory/time/embedding dimensions 32/16/64, 50 neighbors, 2 hops, K=4, and 9 walks. See `--help` for overrides.

The default checkpoint location is `outputs/model_parameters_ETH.pth`; change it with `--output-dir`. Reusing an output directory can replace its checkpoint. The checkpoint stores model states, not a complete resumable training session.

## Evaluation behavior

The original training, sampling, loss, and evaluation logic is retained. Ethereum edges are sorted by timestamp and split 65%/15%/20%. By default, an edge is positive if either endpoint has node attribute `isp=1`.

The local evaluator's `f1` is **micro F1**, averaged across batches and used for selecting the best validation epoch. Printed test precision/recall/F1 use binary classification metrics; test AP uses positive-class probabilities. These are different aggregations. Test data are evaluated each epoch, and the final reported test score corresponds to the selected validation epoch.

## Provenance

The original repository identifies TGB Baselines and DyGLib as upstream sources. The existing MIT license and copyright notice are preserved in `LICENSE`. Preparation changes and remaining publication metadata are listed in [docs/PREPARATION.md](docs/PREPARATION.md). Original file hashes are recorded in `docs/source_manifest.json`.

## Validation status

Syntax, local dependency coverage, command-line help, and preservation of core source code were checked during preparation. End-to-end training has not been validated in the preparation environment, which lacks the ML dependencies and the required `subgraph.pkl`.
