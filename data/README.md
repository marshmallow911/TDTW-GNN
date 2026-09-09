# Ethereum dataset

The default training entry point expects `data/subgraph.pkl`. Alternatively, use `--data-path /absolute/path/to/subgraph.pkl`.

The file must contain the original experiment's pickled NetworkX directed multigraph, compatible with `graph.edges(keys=True, data=True)`:

- Nodes: `isp` classification attribute, with 1 denoting an illicit node and 0 a normal node.
- Edges: numeric `amount` and `timestamp` attributes.

The loader remaps node IDs, sorts edges by timestamp, normalizes amounts, derives edge labels from endpoint labels, and produces temporal train/validation/test splits of 65%/15%/20%. Use the same graph and node/edge ordering as the original experiment for reproduction. Missing attributes default to zero in the existing loader; a correctly prepared graph should explicitly provide them.

The GitHub repository contains an existing [`subgraph.7z`](../subgraph.7z) archive. Extract it separately and place the resulting `subgraph.pkl` in this directory (or use `--data-path`). The archive contents have not been validated during preparation.

No `subgraph.pkl` was found in the source workspace during preparation. The loader originally referenced an external Linux path. That data file is not bundled, and this release does not reconstruct it from raw transactions. Add the dataset source and exact graph preprocessing instructions before public release. Only load pickle files from a trusted source.

`AML_data_loader.py` is retained because the entry point imports it. The active experiment uses Ethereum; Bitcoin loading remains an alternative in the original commented code, not a command-line dataset option.
