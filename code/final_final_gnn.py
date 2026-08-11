#!/usr/bin/env python3
import os
import re
import time
import pickle
import random
import argparse
from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, GlobalAttention, SAGEConv, GINConv

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ---------------------------
# Helper: fallback node features
# ---------------------------
def extract_node_features_from_name(node_name, G, comp_id, comp_size, degree):
    """
    Hand-crafted fallback features parsed from node name (keeps backward compatibility).
    Returns small fixed-length list of floats.
    """
    block_type = "unknown"
    layer_id = 0.0
    position = 0.0
    subtype = "unknown"

    tokens = [t.strip() for t in str(node_name).split(',') if t.strip()]
    if len(tokens) >= 4:
        block_type = tokens[0]
        try:
            layer_id = float(tokens[1])
        except:
            layer_id = 0.0
        try:
            position = float(tokens[2])
        except:
            position = 0.0
        subtype = tokens[3]
    elif len(tokens) == 3:
        block_type = tokens[0]
        try:
            layer_id = float(tokens[1])
        except:
            layer_id = 0.0
        subtype = tokens[2]
    else:
        parts = re.split(r'[\\/\.]', str(node_name))
        if parts:
            subtype = parts[-1]
            block_type = parts[0]

    block_vocab = {"self_attn": 0, "mlp": 1, "embedding": 2, "output": 3, "unknown": 4}
    subtype_vocab = {
        "attn_head": 0, "mlp_in": 1, "mlp_out": 2,
        "o_proj": 3, "q_proj": 4, "k_proj": 5, "v_proj": 6, "unknown": 9
    }

    block_id = block_vocab.get(block_type, block_vocab["unknown"])
    subtype_id = subtype_vocab.get(subtype, subtype_vocab["unknown"])

    layer_id_norm = layer_id / 100.0
    position_norm = position / 50000.0

    features = [
        float(block_id),
        layer_id_norm,
        position_norm,
        float(subtype_id),
        float(degree),
        float(comp_id),
        float(comp_size)
    ]
    return features

# ---------------------------
# Subgraph canonicalization / mining utilities
# ---------------------------
def canonicalize_subgraph_simple(G, nodes):
    """
    Simple canonical signature using node type (prefix before comma) and adjacency upper triangle.
    """
    nodes = list(nodes)
    node_types = []
    for n in nodes:
        nstr = str(n)
        node_types.append(nstr.split(",")[0].strip() if "," in nstr else nstr)
    sorted_labels = sorted(node_types)
    subG = G.subgraph(nodes)
    adj = nx.to_numpy_array(subG, nodelist=nodes)
    adj_upper = "".join(str(int(x)) for x in adj[np.triu_indices(len(nodes), k=1)])
    return "|".join(sorted_labels) + "#" + adj_upper

def mine_frequent_subgraphs(pyg_dataset, min_size=2, max_size=3):
    """
    Enumerate connected induced subgraphs of sizes [min_size, max_size] and count signature occurrences per graph.
    """
    freq_counter = Counter()
    for data in pyg_dataset:
        if data.num_nodes < min_size:
            continue
        node_names = getattr(data, "orig_node_names", [f"n{i}" for i in range(data.num_nodes)])
        G = nx.Graph()
        G.add_nodes_from(node_names)
        if data.edge_index is not None and data.edge_index.numel() > 0:
            ei = data.edge_index.cpu().numpy()
            for u, v in ei.T:
                if u < len(node_names) and v < len(node_names):
                    G.add_edge(node_names[u], node_names[v])
        present = set()
        for k in range(min_size, min(max_size, len(node_names)) + 1):
            for combo in combinations(node_names, k):
                subG = G.subgraph(combo)
                if nx.is_connected(subG):
                    sig = canonicalize_subgraph_simple(G, combo)
                    present.add(sig)
        for s in present:
            freq_counter[s] += 1
    return freq_counter

def select_top_subgraphs(counter_obj, top_pct=0.1):
    unique = len(counter_obj)
    if unique == 0:
        return []
    k = max(1, int(unique * top_pct))
    return [sig for sig, _ in counter_obj.most_common(k)]

def recover_subgraph_objects_from_pyg(pyg_dataset, top_sigs, min_size=2, max_size=3):
    sigs_set = set(top_sigs)
    found = {}
    for data in pyg_dataset:
        node_names = getattr(data, "orig_node_names", [f"n{i}" for i in range(data.num_nodes)])
        G = nx.Graph()
        G.add_nodes_from(node_names)
        if data.edge_index is not None and data.edge_index.numel() > 0:
            ei = data.edge_index.cpu().numpy()
            for u, v in ei.T:
                if u < len(node_names) and v < len(node_names):
                    G.add_edge(node_names[u], node_names[v])
        node_list = list(node_names)
        for k in range(min_size, min(max_size, len(node_list)) + 1):
            for combo in combinations(node_list, k):
                subG = G.subgraph(combo)
                if not nx.is_connected(subG): continue
                sig = canonicalize_subgraph_simple(G, combo)
                if sig in sigs_set and sig not in found:
                    found[sig] = subG.copy()
                    if len(found) == len(sigs_set):
                        return found
    return found

def attach_subgraph_features(pyg_dataset, selected_subgraphs_union):
    feat_dim = len(selected_subgraphs_union)
    if feat_dim == 0:
        for data in pyg_dataset:
            data.subgraph_feats = torch.zeros((1, 0), dtype=torch.float)
        return 0
    sig_to_idx = {s: i for i, s in enumerate(selected_subgraphs_union)}
    for data in pyg_dataset:
        node_names = getattr(data, "orig_node_names", [f"n{i}" for i in range(data.num_nodes)])
        G = nx.Graph()
        G.add_nodes_from(node_names)
        if data.edge_index is not None and data.edge_index.numel() > 0:
            ei = data.edge_index.cpu().numpy()
            for u, v in ei.T:
                if u < len(node_names) and v < len(node_names):
                    G.add_edge(node_names[u], node_names[v])
        present = set()
        node_list = list(node_names)
        for k in range(2, min(3, len(node_list)) + 1):
            for combo in combinations(node_list, k):
                subG = G.subgraph(combo)
                if nx.is_connected(subG):
                    sig = canonicalize_subgraph_simple(G, combo)
                    if sig in sig_to_idx:
                        present.add(sig_to_idx[sig])
        vec = torch.zeros((feat_dim,), dtype=torch.float)
        for idx in present:
            vec[idx] = 1.0
        data.subgraph_feats = vec.unsqueeze(0)
    return feat_dim

# ---------------------------
# Graph loader with CSV feature unification
# ---------------------------
def load_graph_data(subgraphs_path, label_value, node_features_dir=None, feature_columns="all", auto_unify_columns=True, selected_csv_cols=None, use_name_feats=True):
    """
    Loads graphs and returns list[Data]. Ensures node feature width consistent across graphs:
     - If selected_csv_cols provided, uses that fixed list (zero-fills missing cells).
     - Else if auto_unify_columns True, pre-scans CSVs in node_features_dir and uses union of numeric columns (sorted).
     - Else only uses name-based features.
    """
    pyg_dataset = []
    graph_files = [f for f in os.listdir(subgraphs_path) if f.endswith('_with_features.pkl')]
    print(f"Found {len(graph_files)} circuit files in '{subgraphs_path}'.")

    search_dir = node_features_dir if node_features_dir is not None else subgraphs_path

    # Determine global numeric columns if not explicitly provided
    global_numeric_cols = []
    if selected_csv_cols is not None:
        global_numeric_cols = list(selected_csv_cols)
    else:
        if auto_unify_columns and os.path.isdir(search_dir):
            numeric_set = set()
            for fname in os.listdir(search_dir):
                if fname.lower().endswith(".csv") and fname.startswith("circuit_") and "node" in fname:
                    csvf = os.path.join(search_dir, fname)
                    try:
                        df_tmp = pd.read_csv(csvf, nrows=2)
                        for c in df_tmp.columns:
                            if c.lower() in ("node", "node_name"): continue
                            if np.issubdtype(df_tmp[c].dtype, np.number):
                                numeric_set.add(c)
                    except Exception:
                        pass
            global_numeric_cols = sorted(list(numeric_set))

    if feature_columns != "all" and isinstance(feature_columns, str) and feature_columns.strip():
        specified = [c.strip() for c in feature_columns.split(",") if c.strip()]
        if specified:
            global_numeric_cols = specified

    print("Using CSV numeric columns:", global_numeric_cols)

    # Now process each graph
    for graph_file in graph_files:
        match = re.search(r'circuit_(\d+)', graph_file)
        if not match:
            print("Warning: skipping file with no circuit id:", graph_file)
            continue
        circuit_id = int(match.group(1))
        file_path = os.path.join(subgraphs_path, graph_file)
        with open(file_path, "rb") as f:
            G = pickle.load(f)
        if G.number_of_nodes() == 0:
            print(f"Warning: circuit {circuit_id} empty - skipping")
            continue

        original_nodes = list(G.nodes())
        node_map = {n: i for i, n in enumerate(original_nodes)}

        # components
        if G.number_of_edges() > 0:
            if nx.is_directed(G):
                components = list(nx.weakly_connected_components(G))
            else:
                components = list(nx.connected_components(G))
        else:
            components = [[n] for n in original_nodes]
        node_to_comp_id = {n: i for i, comp in enumerate(components) for n in comp}
        node_to_comp_size = {n: len(comp) for comp in components for n in comp}

        # try to find per-circuit CSV
        features_df = None
        if os.path.isdir(search_dir):
            possible = [
                f"circuit_{circuit_id}_node_features.csv",
                f"circuit_{circuit_id}_node_circuits.csv"
            ]
            found_csv = None
            for fname in os.listdir(search_dir):
                if fname in possible:
                    found_csv = os.path.join(search_dir, fname)
                    break
            if not found_csv:
                for fname in os.listdir(search_dir):
                    if fname.startswith(f"circuit_{circuit_id}") and "node" in fname and fname.endswith(".csv"):
                        found_csv = os.path.join(search_dir, fname)
                        break
            if found_csv:
                try:
                    features_df = pd.read_csv(found_csv)
                    if 'node' in features_df.columns:
                        features_df = features_df.set_index('node')
                    elif 'node_name' in features_df.columns:
                        features_df = features_df.set_index('node_name')
                    else:
                        # not indexed by node -> we'll attempt to match by node name column or skip CSV usage
                        if features_df.shape[1] > 0 and 'node' in features_df.columns:
                            features_df = features_df.set_index('node')
                        else:
                            print(f"Warning: CSV {found_csv} has no 'node' index column; ignoring for circuit {circuit_id}")
                            features_df = None
                except Exception as e:
                    print("Warning reading CSV", found_csv, e)
                    features_df = None

        node_features_list = []
        for node_name in original_nodes:
            degree = G.degree(node_name)
            comp_id = node_to_comp_id[node_name]
            comp_size = node_to_comp_size[node_name]

            # name feats
            name_feats = extract_node_features_from_name(node_name, G, comp_id, comp_size, degree) if use_name_feats else []

            # csv feats consistent order
            csv_feats = []
            if len(global_numeric_cols) > 0:
                if features_df is not None and node_name in features_df.index:
                    row = features_df.loc[node_name]
                    for col in global_numeric_cols:
                        if col in features_df.columns:
                            v = row[col]
                            try:
                                csv_feats.append(float(v) if not pd.isna(v) else 0.0)
                            except Exception:
                                csv_feats.append(0.0)
                        else:
                            csv_feats.append(0.0)
                else:
                    # CSV missing or node missing in CSV -> zero fill
                    csv_feats = [0.0] * len(global_numeric_cols)

            combined = name_feats + csv_feats
            node_features_list.append(combined)

        # convert to tensor (guaranteed consistent length across graphs)
        node_features = torch.tensor(node_features_list, dtype=torch.float)

        # edge_index
        edges = []
        for u, v in G.edges():
            if u in node_map and v in node_map:
                edges.append((node_map[u], node_map[v]))
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        label = torch.tensor([label_value], dtype=torch.long)

        data = Data(x=node_features, edge_index=edge_index, y=label)
        data.circuit_id = circuit_id
        data.orig_node_names = original_nodes
        pyg_dataset.append(data)

    return pyg_dataset

# ---------------------------
# GNN model (GAT/GCN switch) with GlobalAttention pooling and optional appended graph-level features
# ---------------------------
# class GNNGraphClassifier(torch.nn.Module):
#     def __init__(self, num_node_features, hidden_channels, num_classes,
#                  use_gat=False, gat_heads=4, graph_feat_dim=0):
#         super(GNNGraphClassifier, self).__init__()
#         torch.manual_seed(42)
#         embedding_size = 32
#         self.node_encoder = Linear(num_node_features, embedding_size)
#         self.use_gat = use_gat
#         self.graph_feat_dim = graph_feat_dim

#         if use_gat:
#             # GAT: attention heads; implement output dimension as hidden_channels
#             self.conv1 = GATConv(embedding_size, hidden_channels // max(1, gat_heads), heads=gat_heads, concat=True)
#             self.conv2 = GATConv(hidden_channels, hidden_channels // max(1, gat_heads), heads=gat_heads, concat=True)
#             self.conv3 = GATConv(hidden_channels, hidden_channels // max(1, gat_heads), heads=gat_heads, concat=True)
#         else:
#             self.conv1 = GCNConv(embedding_size, hidden_channels)
#             self.conv2 = GCNConv(hidden_channels, hidden_channels)
#             self.conv3 = GCNConv(hidden_channels, hidden_channels)

#         gate_nn = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, 1))
#         self.att_pool = GlobalAttention(gate_nn=gate_nn)

#         total_final_dim = hidden_channels + (graph_feat_dim if graph_feat_dim is not None else 0)
#         self.lin = Linear(total_final_dim, num_classes)

#     def forward(self, x, edge_index, batch, graph_feats=None):
#         x = self.node_encoder(x)
#         x = F.relu(x)

#         if edge_index is not None and edge_index.numel() > 0:
#             x = self.conv1(x, edge_index)
#             x = F.relu(x)
#             x = self.conv2(x, edge_index)
#             x = F.relu(x)
#             x = self.conv3(x, edge_index)
#         else:
#             # no edges: project to correct dimension if needed
#             pass

#         x = self.att_pool(x, batch)  # [num_graphs, hidden_channels]

#         if graph_feats is not None:
#             if graph_feats.dim() == 3 and graph_feats.size(1) == 1:
#                 graph_feats = graph_feats.squeeze(1)
#             x = torch.cat([x, graph_feats.to(x.device)], dim=1)

#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin(x)
#         return x


class GNNGraphClassifier(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes,
                 use_gat=False, gat_heads=4, graph_feat_dim=0, model_type = "gsage"):
        super(GNNGraphClassifier, self).__init__()
        torch.manual_seed(42)
        
        embedding_size = 32
        self.node_encoder = Linear(num_node_features, embedding_size)
        self.use_gat = use_gat
        self.graph_feat_dim = graph_feat_dim

        if model_type == "gat":
            self.conv1 = GATConv(embedding_size, hidden_channels // gat_heads, heads=gat_heads, concat=True)
            self.conv2 = GATConv(hidden_channels, hidden_channels // gat_heads, heads=gat_heads, concat=True)
            self.conv3 = GATConv(hidden_channels, hidden_channels // gat_heads, heads=gat_heads, concat=True)
            # self.conv4 = GATConv(hidden_channels, hidden_channels // gat_heads, heads=gat_heads, concat=True)
        elif model_type == "gcn":
            self.conv1 = GCNConv(embedding_size, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            # self.conv4 = GCNConv(hidden_channels, hidden_channels)
        elif model_type == "sage":
            self.conv1 = SAGEConv(embedding_size, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, hidden_channels)
            self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        else : 
            nn1 = Sequential(Linear(embedding_size, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
            nn2 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
            nn3 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
            self.conv1 = GINConv(nn1)
            self.conv2 = GINConv(nn2)
            self.conv3 = GINConv(nn3)

        gate_nn = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, 1))
        self.att_pool = GlobalAttention(gate_nn=gate_nn)

        total_final_dim = hidden_channels + (graph_feat_dim if graph_feat_dim is not None else 0)
        self.lin = Linear(total_final_dim, num_classes)

    def forward(self, x, edge_index, batch, graph_feats=None):
        x = self.node_encoder(x)
        x = x.relu()

        if edge_index is not None and edge_index.numel() > 0:
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = self.conv2(x, edge_index)
            x = x.relu()
            x = self.conv3(x, edge_index)
            # x = x.relu()
            # x = self.conv4(x, edge_index)
        else:
            if x.shape[1] != self.lin.in_features - (graph_feats.shape[1] if graph_feats is not None else 0):
                proj = getattr(self, "_no_edge_proj", None)
                if proj is None:
                    target_dim = (self.lin.in_features - (graph_feats.shape[1] if graph_feats is not None else 0))
                    self._no_edge_proj = Linear(x.shape[1], target_dim).to(x.device)
                    proj = self._no_edge_proj
                x = proj(x)

        x = self.att_pool(x, batch)  # shape [num_graphs, hidden_channels]

        if graph_feats is not None:
            # ensure graph_feats matches batch ordering and shapes: graph_feats should be [num_graphs, feat_dim]
            if graph_feats.dim() == 3 and graph_feats.size(1) == 1:
                # if stored as (1, feat_dim), DataLoader will stack to (num_graphs, 1, feat_dim)
                graph_feats = graph_feats.squeeze(1)
            x = torch.cat([x, graph_feats.to(x.device)], dim=1)

        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


# ---------------------------
# Training and evaluation helpers
# ---------------------------
def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    model.to(device)
    total_loss = 0.0
    total_graphs = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch, data.subgraph_feats if hasattr(data, "subgraph_feats") else None)
        loss = criterion(out, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * data.num_graphs
        total_graphs += data.num_graphs
    return total_loss / total_graphs if total_graphs > 0 else 0.0

def evaluate(model, loader, device):
    model.eval()
    model.to(device)
    all_preds, all_labels = [], []
    total_loss = 0.0
    total_graphs = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch, data.subgraph_feats if hasattr(data, "subgraph_feats") else None)
            loss = torch.nn.functional.cross_entropy(out, data.y.view(-1), reduction='sum')
            total_loss += float(loss.item())
            total_graphs += data.num_graphs
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(data.y.view(-1).cpu().numpy().tolist())
    avg_loss = total_loss / total_graphs if total_graphs > 0 else 0.0
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    return avg_loss, acc, all_labels, all_preds

def train_and_evaluate(model, train_loader, val_loader, optimizer, criterion, device, run_dir, epochs=20):
    os.makedirs(run_dir, exist_ok=True)
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        _, train_acc, _, _ = evaluate(model, train_loader, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()

        print(f"[{epoch}/{epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # persist metrics
        pd.DataFrame(history).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

        # save epoch plot
        try:
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.plot(history["epoch"], history["train_loss"], label="train_loss")
            plt.plot(history["epoch"], history["val_loss"], label="val_loss")
            plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(history["epoch"], history["train_acc"], label="train_acc")
            plt.plot(history["epoch"], history["val_acc"], label="val_acc")
            plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, "train_plot.png"))
            plt.close()
        except Exception as e:
            print("Plot failed:", e)

    if best_state is not None:
        torch.save(best_state, os.path.join(run_dir, "best_model.pt"))
    return best_val_acc, history

def append_run_summary(results_csv, run_metadata):
    df_new = pd.DataFrame([run_metadata])
    if os.path.exists(results_csv):
        df_old = pd.read_csv(results_csv)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(results_csv, index=False)

# ---------------------------
# Feature selection (aggregate -> rank)
# ---------------------------
def aggregate_csv_features_for_selection(hallu_dir, nonhallu_dir, node_features_dir=None, agg_funcs=None):
    if agg_funcs is None: agg_funcs = ['mean', 'std', 'min', 'max']

    def find_csv_for(cid, search_dir):
        if not os.path.isdir(search_dir): return None
        candidates = [
            f"circuit_{cid}_node_features.csv",
            f"circuit_{cid}_node_circuits.csv"
        ]
        for fname in os.listdir(search_dir):
            if fname in candidates:
                return os.path.join(search_dir, fname)
        for fname in os.listdir(search_dir):
            if fname.startswith(f"circuit_{cid}") and "node" in fname and fname.endswith(".csv"):
                return os.path.join(search_dir, fname)
        return None

    rows, labels = [], []
    def collect(folder, label):
        files = [f for f in os.listdir(folder) if f.endswith('_with_features.pkl')]
        for f in files:
            m = re.search(r'circuit_(\d+)', f)
            if not m: continue
            cid = int(m.group(1))
            csvp = find_csv_for(cid, node_features_dir if node_features_dir else folder)
            if csvp:
                try:
                    df = pd.read_csv(csvp)
                    if 'node' in df.columns:
                        df = df.set_index('node')
                    elif 'node_name' in df.columns:
                        df = df.set_index('node_name')
                    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
                    if len(numeric_cols) == 0: continue
                    aggr = {}
                    for col in numeric_cols:
                        s = df[col].dropna()
                        if len(s) == 0: continue
                        if 'mean' in agg_funcs: aggr[f"{col}__mean"] = s.mean()
                        if 'std' in agg_funcs: aggr[f"{col}__std"] = s.std()
                        if 'min' in agg_funcs: aggr[f"{col}__min"] = s.min()
                        if 'max' in agg_funcs: aggr[f"{col}__max"] = s.max()
                    if len(aggr) == 0: continue
                    rows.append(aggr)
                    labels.append(label)
                except Exception as e:
                    print("CSV read failed:", e)
    collect(hallu_dir, 1)
    collect(nonhallu_dir, 0)
    if len(rows) == 0:
        raise RuntimeError("No aggregated CSV rows found. Check CSV locations.")
    X_df = pd.DataFrame(rows).fillna(0.0)
    y = pd.Series(labels)
    return X_df, y

def rank_csv_columns_by_importance(X_df, y, top_k=None):
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)
    X = X_df.fillna(0.0).values
    Xs = StandardScaler().fit_transform(X)
    rf.fit(Xs, y.values)
    feat_imp = sorted(zip(X_df.columns.tolist(), rf.feature_importances_), key=lambda x: x[1], reverse=True)
    if top_k is not None:
        feat_imp = feat_imp[:top_k]
    return feat_imp, rf

def select_top_k_columns(hallu_dir, nonhallu_dir, node_features_dir=None, k=10):
    X_df, y = aggregate_csv_features_for_selection(hallu_dir, nonhallu_dir, node_features_dir=node_features_dir)
    feat_imp, rf = rank_csv_columns_by_importance(X_df, y, top_k=None)
    top_base = []
    for name, _ in feat_imp:
        base = name.split("__")[0]
        if base not in top_base:
            top_base.append(base)
        if len(top_base) >= k:
            break
    return top_base

# ---------------------------
# Loader that uses selected CSV columns and attaches subgraph features
# ---------------------------
def load_all_folders_with_options(hallu_folders, non_hallu_folders, node_features_dir=None, selected_csv_cols=None, use_name_feats=True, auto_unify_columns=True):
    all_train, all_test = [], []
    for folder in hallu_folders:
        ds = load_graph_data(folder, 1, node_features_dir=node_features_dir, feature_columns="all", auto_unify_columns=auto_unify_columns, selected_csv_cols=selected_csv_cols, use_name_feats=use_name_feats)
        if not ds:
            print("No graphs in", folder)
            continue
        ds_sorted = sorted(ds, key=lambda d: getattr(d, "circuit_id", 0))
        split = int(0.8 * len(ds_sorted))
        all_train.extend(ds_sorted[:split])
        all_test.extend(ds_sorted[split:])
    for folder in non_hallu_folders:
        ds = load_graph_data(folder, 0, node_features_dir=node_features_dir, feature_columns="all", auto_unify_columns=auto_unify_columns, selected_csv_cols=selected_csv_cols, use_name_feats=use_name_feats)
        if not ds:
            print("No graphs in", folder)
            continue
        ds_sorted = sorted(ds, key=lambda d: getattr(d, "circuit_id", 0))
        split = int(0.8 * len(ds_sorted))
        all_train.extend(ds_sorted[:split])
        all_test.extend(ds_sorted[split:])
    print(f"Loaded train: {len(all_train)} test: {len(all_test)}")
    return all_train, all_test

# ---------------------------
# CLI / main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hallu_folders", nargs="+", required=True, help="List of hallucination folders")
    parser.add_argument("--nonhallu_folders", nargs="+", required=True, help="List of non-halluc folders")
    parser.add_argument("--node_features_dir", default=None)
    parser.add_argument("--selected_csv_cols", default="", help="Comma-separated CSV column names to use (base names).")
    parser.add_argument("--top_k_csv", type=int, default=0, help="If >0, automatically select top-k CSV columns via RandomForest ranking")
    parser.add_argument("--use_name_feats", action="store_true", default=True)
    parser.add_argument("--use_gat", action="store_true", default=False)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--results_csv", default="results_summary.csv")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--auto_unify_columns", action="store_true", default=True)
    args = parser.parse_args()

    selected_cols = None
    if args.selected_csv_cols:
        selected_cols = [c.strip() for c in args.selected_csv_cols.split(",") if c.strip()]

    if args.top_k_csv > 0 and selected_cols is None:
        print("Selecting top-k CSV columns using RandomForest on aggregated graph-level stats...")
        top = select_top_k_columns(args.hallu_folders[0], args.nonhallu_folders[0], node_features_dir=args.node_features_dir, k=args.top_k_csv)
        selected_cols = top
        print("Selected CSV columns:", selected_cols)

    # load datasets
    train_data, test_data = load_all_folders_with_options(args.hallu_folders, args.nonhallu_folders,
                                                         node_features_dir=args.node_features_dir,
                                                         selected_csv_cols=selected_cols,
                                                         use_name_feats=args.use_name_feats,
                                                         auto_unify_columns=args.auto_unify_columns)

    # mine subgraphs on train only (optionally, you can pass different lists of folders)
    train_hallu = [d for d in train_data if d.y.item() == 1]
    train_nonhallu = [d for d in train_data if d.y.item() == 0]
    hallu_counter = mine_frequent_subgraphs(train_hallu, min_size=2, max_size=3)
    nonhallu_counter = mine_frequent_subgraphs(train_nonhallu, min_size=2, max_size=3)
    hallu_top = select_top_subgraphs(hallu_counter, top_pct=0.20)
    nonhallu_top = select_top_subgraphs(nonhallu_counter, top_pct=0.20)
    selected_union = list(dict.fromkeys(hallu_top + nonhallu_top))
    graph_feat_dim = attach_subgraph_features(train_data, selected_union)  # modifies Data in-place
    _ = attach_subgraph_features(test_data, selected_union)

    if len(train_data) == 0 or len(test_data) == 0:
        print("No data loaded. Exiting.")
        return

    # DataLoaders
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    num_node_features = train_data[0].num_node_features
    print("num_node_features:", num_node_features, "graph_feat_dim:", graph_feat_dim)

    model = GNNGraphClassifier(num_node_features=num_node_features,
                               hidden_channels=args.hidden_channels,
                               num_classes=2,
                               use_gat=args.use_gat,
                               gat_heads=args.gat_heads,
                               graph_feat_dim=graph_feat_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    run_dir = args.run_dir or f"runs/run_{int(time.time())}"
    os.makedirs(run_dir, exist_ok=True)

    print("Starting training...")
    best_val_acc, history = train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, device, run_dir, epochs=args.epochs)

    # final test eval
    val_loss, val_acc, y_true, y_pred = evaluate(model, test_loader, device)
    print("Final test accuracy:", val_acc)
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification report:\n", classification_report(y_true, y_pred, zero_division=0))

    # save summary
    run_meta = {
        "run_name": os.path.basename(run_dir),
        "use_gat": args.use_gat,
        "gat_heads": args.gat_heads,
        "hidden_channels": args.hidden_channels,
        "selected_csv_cols": ",".join(selected_cols) if selected_cols else "",
        "top_k_csv": args.top_k_csv,
        "use_name_feats": args.use_name_feats,
        "epochs": args.epochs,
        "final_val_acc": float(val_acc),
        "run_dir": run_dir
    }
    append_run_summary(args.results_csv, run_meta)
    print("Run metadata appended to", args.results_csv)
    print("Done.")

if __name__ == "__main__":
    main()
