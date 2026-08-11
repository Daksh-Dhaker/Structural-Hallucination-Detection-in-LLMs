#!/usr/bin/env python3
"""
batch_extract_dynamic_aggregate.py

Batch process many circuit_id.pkl (comp_graph) files, extract node-level features
(reuses extract_all_node_features), save full node tensors for PCA, and (optionally)
run model forward on a prompts file to compute dataset-level aggregated attention
and hidden-state statistics (mean, variance, norms, entropy).

Usage example:
python3 batch_extract_dynamic_aggregate.py \
  --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
  --circuits_dir ./results/circuits_before_new \
  --out_dir ./results/circuits_before_new \
  --pattern "circuit_*.pkl" \
  --prompts_file ./text_dataset/hallucination_data/numerical_dataset.jsonl \
  --save_raw_npz \
  --overwrite
"""

import argparse
import os
import glob
import pickle
import json
import math
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch

# reuse your extractor
try:
    from extract_all_node_features import extract_all_node_features, load_pickle
except Exception as e:
    raise RuntimeError("Could not import extract_all_node_features: " + str(e))


# ---------------------------
# Model loader (like your existing function)
# ---------------------------
def load_model_once(model_name_or_path):
    import torch
    from transformers import AutoModelForCausalLM

    print(f"[load_model] loading model {model_name_or_path} (CPU, float32)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(torch.device("cpu"))
    for p in model.parameters():
        if p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)
    model.eval()
    return model


# ---------------------------
# Helpers: prompts loader
# ---------------------------
def load_prompts(prompts_path):
    """
    Support:
     - JSONL: each line JSON with 'prompt' key (or top-level string)
     - plain text: one prompt per line
    """
    prompts = []
    if not prompts_path:
        return prompts
    with open(prompts_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                # If JSON object contains 'prompt' key, use it; else if it's a string use that
                if isinstance(obj, dict) and "prompt" in obj:
                    prompts.append(obj["prompt"])
                elif isinstance(obj, str):
                    prompts.append(obj)
                else:
                    # fallback: whole JSON string
                    prompts.append(s)
            except Exception:
                # plain text line
                prompts.append(s)
    return prompts


# ---------------------------
# Online aggregator for arrays (Welford's algorithm)
# ---------------------------
class OnlineAggregate:
    def __init__(self, shape, dtype=np.float64):
        self.shape = tuple(shape)
        self.count = 0
        self.mean = np.zeros(self.shape, dtype=dtype)
        self.M2 = np.zeros(self.shape, dtype=dtype)

    def update(self, x):
        x = np.array(x, dtype=self.mean.dtype)
        if x.shape != self.shape:
            raise ValueError(f"shape mismatch: expected {self.shape}, got {x.shape}")
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2  # incremental variance

    def variance(self):
        if self.count < 2:
            return np.zeros_like(self.mean)
        return self.M2 / (self.count - 1)

    def get_mean(self):
        return self.mean

    def get_var(self):
        return self.variance()

    def get_count(self):
        return self.count


# ---------------------------
# Small utilities
# ---------------------------
def attention_entropy(attn_matrix):
    """
    attn_matrix: numpy array with shape (..., seq_len) representing a distribution along last axis
    returns entropy averaged over preceding dims
    """
    # clip for numerical stability
    p = np.clip(attn_matrix, 1e-12, 1.0)
    ent = -np.sum(p * np.log(p), axis=-1)
    # return mean entropy
    return float(np.mean(ent))


def pool_attention_to_scalar(attn):
    """
    Reduce attention tensor (seq_len x seq_len) to a scalar summary:
      - we average over source and target positions (i.e., global mean attention weight)
    Input: attn numpy array shape (seq_len, seq_len)
    """
    return float(np.mean(attn))


# ---------------------------
# Main processing for a single circuit file
# ---------------------------
def process_circuit(cf, model, args):
    base = os.path.splitext(os.path.basename(cf))[0]
    out_prefix = os.path.join(args.out_dir, base + "_node_features")
    csv_path = out_prefix + ".csv"
    rows_pkl = out_prefix + "_rows.pkl"
    augmented_graph_path = os.path.join(args.out_dir, base + "_with_features.pkl")
    raw_npz_folder = out_prefix + "_raw" if args.save_raw_npz else None

    # If outputs exist and not overwrite -> skip
    if (not args.overwrite) and os.path.exists(csv_path) and os.path.exists(augmented_graph_path):
        print(f"[skip] outputs for {cf} exist (use --overwrite to force).")
        return csv_path

    # 1) static node extraction (parameters etc.) via existing function
    print(f"[run] extracting static node features for {cf} -> {csv_path}")
    df_nodes, rows = extract_all_node_features(
        model,
        cf,
        out_prefix=out_prefix,
        save_raw_npz=args.save_raw_npz,
        raw_npz_folder=raw_npz_folder if raw_npz_folder else None
    )

    # 2) attach features back to graph and save augmented graph
    # replicate attach_features_to_graph_pickle behavior (lightweight)
    print(f"[attach] attaching static features to graph -> {augmented_graph_path}")
    try:
        with open(cf, "rb") as f:
            comp_graph = pickle.load(f)
    except Exception as e:
        print(f"[warn] failed to load comp_graph {cf}: {e}")
        comp_graph = None

    # attach CSV attributes to graph (best-effort)
    try:
        df = pd.read_csv(csv_path).fillna(value=np.nan)
        for _, r in df.iterrows():
            node_key = r["node"]
            attr = {k: (None if pd.isna(v) else v) for k, v in r.items() if k != "node"}
            try:
                if comp_graph is None:
                    continue
                if hasattr(comp_graph, "nodes") and node_key in comp_graph.nodes:
                    comp_graph.nodes[node_key].update(attr)
                elif isinstance(comp_graph, dict) and node_key in comp_graph:
                    if isinstance(comp_graph[node_key], dict):
                        comp_graph[node_key].update(attr)
                    else:
                        comp_graph[node_key] = {"_features": attr}
                else:
                    if not hasattr(comp_graph, "_external_node_features"):
                        try:
                            comp_graph._external_node_features = {}
                        except Exception:
                            comp_graph["_external_node_features"] = {}
                    if hasattr(comp_graph, "_external_node_features"):
                        comp_graph._external_node_features[node_key] = attr
                    else:
                        comp_graph["_external_node_features"][node_key] = attr
            except Exception as ex:
                print(f"[warn] failed attach node {node_key}: {ex}")
    except Exception as e:
        print(f"[warn] reading csv to attach failed: {e}")

    # Save augmented graph
    try:
        with open(augmented_graph_path, "wb") as f:
            pickle.dump(comp_graph, f)
    except Exception as e:
        print(f"[warn] failed to save augmented graph: {e}")

    # 3) Save full node activation tensors for PCA (if present)
    if args.save_raw_npz and comp_graph is not None:
        node_activation_folder = os.path.join(raw_npz_folder, "activations")
        os.makedirs(node_activation_folder, exist_ok=True)
        try:
            nodes = list(comp_graph.nodes) if hasattr(comp_graph, "nodes") else list(comp_graph.keys())
            for n in nodes:
                try:
                    act = comp_graph.nodes[n].get("activation", None) if hasattr(comp_graph, "nodes") else comp_graph[n].get("activation", None)
                    if act is None:
                        continue
                    arr = np.array(act)
                    # Save as .npy (float32)
                    safe_name = str(n).replace("/", "_").replace(" ", "_").replace(",", "_")
                    dest = os.path.join(node_activation_folder, f"{safe_name}__activation.npy")
                    np.save(dest, arr.astype(np.float32))
                except Exception as e:
                    print(f"[warn] saving activation for node {n} failed: {e}")
        except Exception as e:
            print(f"[warn] iterating nodes to save activation failed: {e}")

    # 4) If prompts provided: compute aggregate attention/hidden stats across prompts
    if args.prompts_file:
        prompts = load_prompts(args.prompts_file)
        if len(prompts) == 0:
            print(f"[info] prompts file {args.prompts_file} contained no prompts. Skipping dynamic aggregation.")
            return csv_path

        # We'll compute per-layer x per-head aggregations of a scalar summary of attention
        # For each prompt: run forwards with output_attentions & output_hidden_states
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

        # We need to run on a device (CPU used because model loaded to CPU earlier)
        device = next(model.parameters()).device

        # For the first prompt, inspect shapes to allocate aggregators
        print(f"[attn] Running {len(prompts)} prompts to compute aggregated attention stats...")
        sample_prompt = prompts[0]
        enc = tokenizer(sample_prompt, return_tensors="pt", truncation=True, max_length=args.max_seq_len)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=True, return_dict=True)

        # attentions: tuple(#layers) of (batch, num_heads, seq_len, seq_len)
        n_layers = len(out.attentions)
        b, n_heads, s1, s2 = out.attentions[0].shape  # batch usually 1
        # We'll collapse seq dims by taking average (global scalar per head)
        # aggregator shape: (n_layers, n_heads)
        agg_shape = (n_layers, n_heads)
        attn_agg = OnlineAggregate(agg_shape)
        attn_entropy_agg = OnlineAggregate((n_layers,))  # per-layer entropy
        # hidden states: tuple(num_layers+1) of (batch, seq_len, hidden_size)
        n_hidden_layers = len(out.hidden_states)
        # We'll pool hidden states across seq by mean -> shape per layer = (hidden_size,)
        hidden_dim = out.hidden_states[0].shape[-1]
        hidden_mean_agg = OnlineAggregate((n_hidden_layers, hidden_dim))
        # Also keep running norm stats across prompts for hidden means per layer (scalar L2)
        hidden_norm_agg = OnlineAggregate((n_hidden_layers,))

        # iterate all prompts and update aggregators
        for p in tqdm(prompts, desc="prompts"):
            try:
                enc = tokenizer(p, return_tensors="pt", truncation=True, max_length=args.max_seq_len)
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True, output_hidden_states=True, return_dict=True)
                # per-layer, per-head scalar summary
                for li, att in enumerate(out.attentions):
                    # att shape (batch, n_heads, seq_len, seq_len)
                    att_np = att[0].cpu().numpy()  # (n_heads, seq, seq)
                    # compute per-head scalar: mean across both seq dims
                    head_means = np.array([pool_attention_to_scalar(att_np[h]) for h in range(att_np.shape[0])], dtype=np.float64)
                    attn_agg.update(head_means)
                    # layer entropy: flatten to (n_heads, seq*seq) and compute mean entropy across heads then avg across heads
                    # but simpler: compute mean across source positions: first normalize last axis and compute entropy
                    # compute entropy per head by averaging entropies of rows, then mean across heads
                    head_entropies = []
                    for h in range(att_np.shape[0]):
                        # normalize rows -> each row is distribution over targets
                        rows = att_np[h]
                        # normalize per-row
                        row_sums = rows.sum(axis=-1, keepdims=True)
                        normed = rows / np.clip(row_sums, 1e-12, None)
                        # entropy per row
                        ent = -np.sum(np.clip(normed, 1e-12, 1.0) * np.log(np.clip(normed, 1e-12, 1.0)), axis=-1)
                        head_entropies.append(float(np.mean(ent)))
                    layer_entropy = float(np.mean(head_entropies))
                    attn_entropy_agg.update(np.array([layer_entropy for _ in range(n_layers)])[li:li+1] if False else np.array([layer_entropy]))  # we'll store per-layer directly below

                # hidden states pooling
                for hi, hstate in enumerate(out.hidden_states):
                    # hstate: (batch, seq_len, hidden_dim)
                    h_np = hstate[0].cpu().numpy()
                    pooled = np.mean(h_np, axis=0)  # (hidden_dim,)
                    hidden_mean_agg.update(pooled)
                    hidden_norm_agg.update(np.array([float(np.linalg.norm(pooled))]))
            except Exception as e:
                print(f"[warn] forward failed for prompt: {e}")
                continue

        # Post-process aggregated stats and save to disk
        attn_mean = attn_agg.get_mean()  # (n_layers, n_heads)
        attn_var = attn_agg.get_var()
        hidden_mean = hidden_mean_agg.get_mean()  # (n_hidden_layers, hidden_dim)
        hidden_var = hidden_mean_agg.get_var()
        hidden_norm_mean = hidden_norm_agg.get_mean().reshape(-1)
        # attn_entropy_agg was updated per-layer differently; to keep consistent compute per-layer entropy separately:
        # For simplicity compute per-layer entropy across the first prompt's attentions averaged across heads (we still saved attn_mean)
        # Save aggregates as compressed npz
        agg_npz_path = os.path.join(args.out_dir, base + "_attn_aggregates.npz")
        np.savez_compressed(
            agg_npz_path,
            attn_mean=attn_mean.astype(np.float32),
            attn_var=attn_var.astype(np.float32),
            hidden_mean=hidden_mean.astype(np.float32),
            hidden_var=hidden_var.astype(np.float32),
            hidden_norm_mean=hidden_norm_mean.astype(np.float32),
            prompts_count=np.array([attn_agg.get_count()], dtype=np.int32),
        )
        print(f"[done] saved attention aggregates -> {agg_npz_path}")

        # Append a few scalar summary columns to CSV for convenience (mean L2 norm across layers, mean attn mean)
        try:
            extra = {}
            extra["agg_prompts_count"] = int(attn_agg.get_count())
            extra["agg_attn_mean_mean"] = float(np.mean(attn_mean))
            extra["agg_attn_mean_std"] = float(np.std(attn_mean))
            extra["agg_hidden_mean_norm_mean"] = float(np.mean(hidden_norm_mean))
            # append to the CSV by adding a one-row 'circuit-level' file
            circuit_level_path = os.path.join(args.out_dir, base + "_circuit_level.csv")
            pd.DataFrame([extra]).to_csv(circuit_level_path, index=False)
            print(f"[done] saved circuit-level summary -> {circuit_level_path}")
        except Exception as e:
            print(f"[warn] saving summary csv failed: {e}")

    return csv_path


# ---------------------------
# Command-line
# ---------------------------
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
    ap.add_argument("--prompts_file", default=None, help="path to prompts file (JSONL or plain text) to aggregate dynamic attentions")
    ap.add_argument("--save_raw_npz", action="store_true", help="save numeric raw arrays to per-node .npy files (including activations)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing per-circuit outputs")
    ap.add_argument("--max_seq_len", type=int, default=1024, help="max token length for tokenizer/truncation")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # load model once
    model = load_model_once(args.model_name_or_path)

    circuit_files = find_circuit_files(args.circuits_dir, args.pattern)
    if not circuit_files:
        print(f"[error] no files found in {args.circuits_dir} matching {args.pattern}")
        return

    summary = []
    for cf in tqdm(circuit_files, desc="circuits"):
        try:
            csv = process_circuit(cf, model, args)
            summary.append((cf, "ok"))
        except Exception as e:
            print(f"[error] processing {cf} failed: {e}")
            summary.append((cf, "error", str(e)))
            continue

    print("\nSummary:")
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
