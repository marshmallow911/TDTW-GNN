"""Portable command-line configuration; defaults match the original experiment."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Train the temporal tree GNN on Ethereum transactions.")
    parser.add_argument("--data-path", type=Path, default=ROOT / "data" / "subgraph.pkl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, cuda:2, etc.")
    parser.add_argument("--lr", type=float, default=0.0015)
    parser.add_argument("--seed", type=int, default=10)
    for name, default in [("batch-size", 8192), ("epochs", 100), ("mem-dim", 32),
                          ("time-dim", 16), ("emb-dim", 64), ("num-neighbors", 50),
                          ("num-hop", 2), ("k", 4), ("num-walk", 9)]:
        parser.add_argument("--" + name, type=positive_int, default=default)
    args = parser.parse_args()
    if args.lr <= 0:
        parser.error("--lr must be positive")
    return args
