#!/usr/bin/env python3
"""
batch_extract_circuit_features.py

Batch process many circuit_id.pkl (comp_graph) files, extract node-level features
using extract_all_node_features(model, comp_graph_path, ...), attach features to
the graph, and save results.

Usage example:

python3 batch_extract_circuit_features.py \
  --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
  --circuits_dir ./results/circuits_before_new \
  --out_dir ./results/circuits_before_new \
  --pattern "circuit_*.pkl" \
  --pattern "*.pkl" \
  --save_raw_npz

"""

import argparse
import os
import glob
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

# import your extractor (ensure extract_all_node_features.py is in PYTHONPATH or same dir)
try:
    from extract_all_node_features import extract_all_node_features
except Exception as e:
    raise RuntimeError("Could not import extract_all_node_features from extract_all_node_features.py: " + str(e))

# transformers model loader
def load_model_once(model_name_or_path):
    """
    Load model using transformers AutoModelForCausalLM into CPU as float32.
    This avoids BFloat16 issues on CPU-only setups.
    """
    import torch
    from transformers import AutoModelForCausalLM

    print(f"[load_model] loading model {model_name_or_path} (CPU, float32)")

    # Force float32 to avoid BFloat16 unsupported dtype on CPU
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True  # if model repo has custom code
    )

    # ensure everything on CPU and in float32 dtype
    model.to(torch.device("cpu"))
    for p in model.parameters():
        if p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)

    return model


def attach_features_to_graph_pickle(comp_graph_path, features_csv_path, out_graph_path):
    """
    Load pickled comp_graph, load features CSV, attach features as node attributes,
    and save augmented graph pickle to out_graph_path.
    """
    # load graph object
    with open(comp_graph_path, "rb") as f:
        comp_graph = pickle.load(f)

    # load features
    df = pd.read_csv(features_csv_path)
    df = df.fillna(value=np.nan)

    # attach each row as attributes for node key == df.node
    # support both networkx and dict/list style graphs: we'll try to set attributes if nodes exist
    for _, r in df.iterrows():
        node_key = r["node"]
        # prepare attributes (skip 'node' column)
        attr = {k: (None if pd.isna(v) else v) for k, v in r.items() if k != "node"}
        try:
            # networkx graph: nodes as keys
            if hasattr(comp_graph, "nodes") and node_key in comp_graph.nodes:
                comp_graph.nodes[node_key].update(attr)
            # dict-like
            elif isinstance(comp_graph, dict) and node_key in comp_graph:
                if isinstance(comp_graph[node_key], dict):
                    comp_graph[node_key].update(attr)
                else:
                    comp_graph[node_key] = {"_features": attr}
            else:
                # best-effort: store in a top-level 'external_node_features' map
                if not hasattr(comp_graph, "_external_node_features"):
                    try:
                        comp_graph._external_node_features = {}
                    except Exception:
                        comp_graph["_external_node_features"] = {}
                # support both attribute or dict
                if hasattr(comp_graph, "_external_node_features"):
                    comp_graph._external_node_features[node_key] = attr
                else:
                    comp_graph["_external_node_features"][node_key] = attr
        except Exception as e:
            print(f"[warn] attaching features for node {node_key} in {comp_graph_path} failed: {e}")

    # save augmented graph
    with open(out_graph_path, "wb") as f:
        pickle.dump(comp_graph, f)

def find_circuit_files(circuits_dir, pattern):
    search_pattern = os.path.join(circuits_dir, pattern)
    files = sorted(glob.glob(search_pattern))
    return files

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True, help="local model directory or HF id")
    ap.add_argument("--circuits_dir", required=True, help="directory containing circuit_id .pkl files")
    ap.add_argument("--out_dir", required=True, help="where to write per-circuit outputs (csv/pkl/augmented graphs)")
    ap.add_argument("--pattern", default="*.pkl", help="glob pattern to select circuit files (default *.pkl)")
    ap.add_argument("--save_raw_npz", action="store_true", help="let extractor save per-node raw arrays (.npy)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing per-circuit outputs")
    ap.add_argument("--combine_csv", action="store_true", help="create combined CSV of all circuits")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load model once
    model = load_model_once(args.model_name_or_path)

    circuit_files = find_circuit_files(args.circuits_dir, args.pattern)
    if not circuit_files:
        print(f"[error] no files found in {args.circuits_dir} matching {args.pattern}")
        return

    combined_csvs = []  # list of per-file csv paths for optional combine
    summary = []
    for cf in tqdm(circuit_files, desc="circuits"):
        base = os.path.splitext(os.path.basename(cf))[0]
        out_prefix = os.path.join(args.out_dir, base + "_node_features")
        csv_path = out_prefix + ".csv"
        rows_pkl = out_prefix + "_rows.pkl"
        augmented_graph_path = os.path.join(args.out_dir, base + "_with_features.pkl")

        if (not args.overwrite) and os.path.exists(csv_path) and os.path.exists(augmented_graph_path):
            print(f"[skip] outputs for {cf} exist (use --overwrite to force).")
            combined_csvs.append(csv_path)
            summary.append((cf, "skipped"))
            continue

        try:
            # call extractor (this writes csv/pkl itself)
            print(f"[run] extracting features for {cf} -> {csv_path}")
            df_nodes, rows = extract_all_node_features(
                model,
                cf,
                out_prefix=out_prefix,
                save_raw_npz=args.save_raw_npz
            )

            # attach and save augmented graph
            print(f"[attach] attaching features back to graph and saving -> {augmented_graph_path}")
            attach_features_to_graph_pickle(cf, csv_path, augmented_graph_path)

            combined_csvs.append(csv_path)
            summary.append((cf, "ok", len(df_nodes)))
        except Exception as e:
            print(f"[error] extraction failed for {cf}: {e}")
            summary.append((cf, "error", str(e)))
            continue

    # optionally combine CSVs
    if args.combine_csv and combined_csvs:
        print("[combine] concatenating per-circuit CSVs into combined_node_features.csv")
        try:
            dfs = []
            for p in combined_csvs:
                try:
                    d = pd.read_csv(p)
                    d["circuit_file"] = os.path.basename(p).replace("_node_features.csv", "")
                    dfs.append(d)
                except Exception as e:
                    print(f"[warn] failed to read {p}: {e}")
            if dfs:
                big = pd.concat(dfs, ignore_index=True)
                combined_path = os.path.join(args.out_dir, "combined_node_features.csv")
                big.to_csv(combined_path, index=False)
                print(f"[done] combined CSV written -> {combined_path}")
        except Exception as e:
            print(f"[error] combine failed: {e}")

    # print summary
    print("\nSummary:")
    for s in summary:
        print(s)

if __name__ == "__main__":
    main()

