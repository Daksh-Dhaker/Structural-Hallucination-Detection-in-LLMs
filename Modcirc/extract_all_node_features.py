#!/usr/bin/env python3
"""
Extract all node-level features from a transformer model using a comp_graph node list.
- No forward pass by default: extracts raw parameters (weights, biases, layernorms) and summaries.
- Produces CSV with many descriptive columns + optional per-node .npz raw dumps.
Assumes comp_graph.pkl contains a NetworkX graph OR a pickled list of node-name strings.
Node naming expected like: "mlp,3,456,down_proj" or "self_attn,5,8,attn_head".
"""

import argparse
import pickle
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ----------------- Helpers -----------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def parse_node_name(node_name):
    """
    Parse node strings expected like: "mlp,3,456,down_proj"
    Returns dict with keys: node (str), comp (str), layer (int or None), idx (int or None), comp_type (str or None)
    """
    s = str(node_name)
    parts = [p.strip() for p in s.split(",")]
    meta = {"node": s, "comp": None, "layer": None, "idx": None, "comp_type": None}
    if len(parts) >= 1:
        meta["comp"] = parts[0] if parts[0] else None
    if len(parts) >= 2:
        try:
            meta["layer"] = int(parts[1])
        except Exception:
            meta["layer"] = None
    if len(parts) >= 3:
        try:
            meta["idx"] = int(parts[2])
        except Exception:
            meta["idx"] = None
    if len(parts) >= 4:
        meta["comp_type"] = parts[3]
    return meta

def find_best_module_match(model, meta, named_modules):
    """
    Heuristic search:
    1) Try to find module with substring that contains layer index and comp or comp_type.
    2) Try exact match if nodes store exact module path.
    3) Fallback to any module name that contains comp_type or comp.
    Returns (module_name, module) or (None, None)
    """
    comp = (meta.get("comp") or "").lower()
    comp_type = (meta.get("comp_type") or "").lower()
    layer = meta.get("layer")

    # 1) exact match
    if meta["node"] in named_modules:
        return meta["node"], named_modules[meta["node"]]

    # 2) look for names that contain both layer number and comp or comp_type
    candidates = []
    for nm, m in named_modules.items():
        low = nm.lower()
        if layer is not None and str(layer) in low:
            if comp and comp in low:
                candidates.append((nm, m, 2))
            elif comp_type and comp_type in low:
                candidates.append((nm, m, 2))
            else:
                # contains layer but not comp: lower score
                candidates.append((nm, m, 1))

    # prefer high score candidates that also contain comp_type
    if candidates:
        # rank by presence of comp_type and length (prefer longer match)
        def score(c):
            nm, _, base = c
            nm_low = nm.lower()
            bonus = 1 if (comp_type and comp_type in nm_low) else 0
            bonus += 1 if (comp and comp in nm_low) else 0
            return (base + bonus, len(nm))
        candidates.sort(key=score, reverse=True)
        return candidates[0][0], candidates[0][1]

    # 3) no layer match: try substring comp_type or comp
    for nm, m in named_modules.items():
        low = nm.lower()
        if comp_type and comp_type in low:
            return nm, m
    for nm, m in named_modules.items():
        low = nm.lower()
        if comp and comp in low:
            return nm, m

    return None, None

def tensor_stats(np_arr, topk_svd=3):
    arr = np_arr.ravel()
    n = arr.size
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "l2": 0.0, "max_abs": 0.0, "sparsity": 0.0, "svd": [None]*topk_svd}
    mean = float(arr.mean())
    std = float(arr.std())
    l2 = float(np.linalg.norm(arr))
    max_abs = float(np.abs(arr).max())
    sparsity = float((arr == 0).sum()) / n
    svd_vals = [None]*topk_svd
    if np_arr.ndim == 2 and max(np_arr.shape) <= 4096:
        try:
            s = np.linalg.svd(np_arr, compute_uv=False)
            for i in range(min(topk_svd, s.size)):
                svd_vals[i] = float(s[i])
        except Exception:
            pass
    return {"mean": mean, "std": std, "l2": l2, "max_abs": max_abs, "sparsity": sparsity, "svd": svd_vals}

def per_head_norms_if_possible(np_arr, head_guess):
    """Return list of per-head L2 norms if arr shape matches head splits, else None."""
    if head_guess is None or np_arr.ndim != 2:
        return None
    n_heads = head_guess.get("n_heads")
    head_dim = head_guess.get("head_dim")
    if n_heads is None or head_dim is None:
        return None
    out_dim, in_dim = np_arr.shape
    if out_dim == n_heads * head_dim:
        try:
            reshaped = np_arr.reshape(n_heads, head_dim, in_dim)
            return [float(np.linalg.norm(reshaped[h])) for h in range(n_heads)]
        except Exception:
            return None
    if in_dim == n_heads * head_dim:
        try:
            reshaped = np_arr.reshape(out_dim, n_heads, head_dim)
            return [float(np.linalg.norm(reshaped[:, h, :])) for h in range(n_heads)]
        except Exception:
            return None
    return None

def guess_heads_from_model(model):
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None) or getattr(cfg, "n_heads", None)
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None) or getattr(cfg, "dim", None)
    if n_heads and hidden_size:
        try:
            head_dim = int(hidden_size // n_heads)
            return {"n_heads": int(n_heads), "head_dim": head_dim}
        except Exception:
            return None
    return None

# ----------------- Main extraction -----------------
def extract_all_node_features(
    model,
    comp_graph_path,
    out_prefix="node_features_all",
    save_raw_npz=False,
    raw_npz_folder=None
):
    # load nodes
    obj = load_pickle(comp_graph_path)
    if hasattr(obj, "nodes"):
        nodes = list(obj.nodes)
    elif isinstance(obj, (list, tuple)):
        nodes = list(obj)
    elif isinstance(obj, dict) and "nodes" in obj:
        nodes = list(obj["nodes"])
    else:
        try:
            nodes = list(obj.keys())
        except Exception:
            raise ValueError("comp_graph pickle format not recognized. Provide NetworkX graph or list.")

    named_modules = dict(model.named_modules())
    head_guess = guess_heads_from_model(model)

    # collect comp_types to one-hot
    comp_types_set = set()
    metas = {}
    for n in nodes:
        meta = parse_node_name(n)
        metas[n] = meta
        if meta.get("comp_type"):
            comp_types_set.add(str(meta["comp_type"]))
        elif meta.get("comp"):
            comp_types_set.add(str(meta["comp"]))
    comp_types = sorted(list(comp_types_set))

    # prepare result rows
    rows = []
    raw_store = {}  # for optional global npz; keys will be node__paramname

    if save_raw_npz and raw_npz_folder is None:
        raw_npz_folder = out_prefix + "_raw"
        os.makedirs(raw_npz_folder, exist_ok=True)

    for node in tqdm(nodes, desc="Nodes"):
        meta = metas[node]
        row = {
            "node": str(node),
            "comp": meta.get("comp"),
            "layer": meta.get("layer"),
            "idx": meta.get("idx"),
            "comp_type": meta.get("comp_type")
        }
        # add one-hot type columns
        for t in comp_types:
            row[f"type_{t}"] = 1 if (meta.get("comp_type")==t or meta.get("comp")==t) else 0

        # find module
        module_name, module = find_best_module_match(model, meta, named_modules)
        row["module_name"] = module_name
        row["found"] = bool(module is not None)

        if not module:
            # nothing else to collect
            rows.append(row)
            continue

        # summarize parameters (non-recursive to only include this specific submodule's params)
        param_count = 0
        for pname, p in module.named_parameters(recurse=False):
            param_count += p.numel()
            arr = p.detach().cpu().numpy()
            stats = tensor_stats(arr, topk_svd=3)
            # add stat columns with prefix
            prefix = pname.replace(".", "_")
            row[f"{prefix}_mean"] = stats["mean"]
            row[f"{prefix}_std"] = stats["std"]
            row[f"{prefix}_l2"] = stats["l2"]
            row[f"{prefix}_max_abs"] = stats["max_abs"]
            row[f"{prefix}_sparsity"] = stats["sparsity"]
            # svd topk
            for i, sv in enumerate(stats["svd"]):
                row[f"{prefix}_svd{i+1}"] = sv

            # per-head norms
            ph = per_head_norms_if_possible(arr, head_guess)
            if ph is not None:
                for i, v in enumerate(ph):
                    row[f"{prefix}_headnorm_{i}"] = v

            # optionally save raw arrays for heavy analysis
            if save_raw_npz:
                # save per-node small npz file
                safe_name = str(node).replace("/", "_").replace(" ", "_").replace(",", "_")
                npz_path = os.path.join(raw_npz_folder, f"{safe_name}__{prefix}.npy")
                # save in float32 to save space
                try:
                    np.save(npz_path, arr.astype(np.float32))
                    row[f"{prefix}_raw_saved"] = npz_path
                except Exception as e:
                    row[f"{prefix}_raw_saved"] = None

        row["param_count"] = int(param_count)
        rows.append(row)

    # DataFrame and save CSV
    df = pd.DataFrame(rows)
    csv_path = out_prefix + ".csv"
    df.to_csv(csv_path, index=False)
    # save full rows pickled as well
    pkl_path = out_prefix + "_rows.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(rows, f)

    print(f"Saved CSV -> {csv_path}")
    print(f"Saved rows pickle -> {pkl_path}")
    return df, rows

# ----------------- CLI -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True, help="local model dir or HF id")
    ap.add_argument("--comp_graph_path", required=True, help="pickled comp_graph or pickled node list")
    ap.add_argument("--out_prefix", default="node_features_all")
    ap.add_argument("--save_raw_npz", action="store_true", help="save numeric raw arrays to per-node .npy files")
    args = ap.parse_args()

    # load model (use transformers AutoModel if available)
    model = None
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    except Exception as e:
        # try torch.load fallback (user may have raw state dict)
        print("transformers load failed or unavailable:", e)
        try:
            model = torch.load(args.model_name_or_path, map_location="cpu")
        except Exception as e2:
            raise RuntimeError("Unable to load model via transformers or torch.load: " + str(e2))

    df, rows = extract_all_node_features(
        model,
        args.comp_graph_path,
        out_prefix=args.out_prefix,
        save_raw_npz=args.save_raw_npz
    )
    print(df.head())

if __name__ == "__main__":
    main()
