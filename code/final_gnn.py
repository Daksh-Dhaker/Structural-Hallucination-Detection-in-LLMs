import os
import pickle
import pandas as pd
import networkx as nx
import torch
from torch.nn import Linear, Sequential, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, GlobalAttention, SAGEConv, GINConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import random
from sklearn.model_selection import train_test_split
from itertools import combinations
from collections import Counter, defaultdict
import re

def load_node_features(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in ['node', 'comp', 'layer', 'idx', 'comp_type',
                                                       'type_down_proj', 'type_mlp_in', 'module_name', 'found']]
    node_feats = {}
    for _, row in df.iterrows():
        node_name = str(row['node']).strip('"')
        node_feats[node_name] = row[feature_cols].astype(float).to_numpy()

    print("features : ", len(node_feats))
    return node_feats

def canonicalize_subgraph(G, nodes=None):
    """
    Create a canonical string signature for a subgraph.
    Uses only the node's high-level type (e.g., 'mlp') as its identity.
    """
    if nodes is None:
        nodes = list(G.nodes)

    # ✅ Use only the first part (before comma) as node type
    node_types = []
    for n in nodes:
        if isinstance(n, str):
            node_type = n.split(",")[0].strip()  # keep only 'mlp' from 'mlp,4,12322,mlp_in'
        else:
            node_type = str(n)
        node_types.append(node_type)

    # Build adjacency bitstring (upper-triangular)
    subgraph = G.subgraph(nodes)
    adj = nx.to_numpy_array(subgraph, nodelist=nodes)
    adj_upper = "".join(str(int(x)) for x in adj[np.triu_indices(len(nodes), k=1)])

    # Canonical order by sorting node type labels (to avoid permutation duplicates)
    sorted_labels = sorted(node_types)

    return "|".join(sorted_labels) + "#" + adj_upper

def select_top_subgraphs(counter_obj, top_pct=0.1):
    """
    Select top `top_pct` fraction (by unique signature count) from the counter.
    If top_pct is small and results in 0, return at least one.
    """
    unique = len(counter_obj)
    if unique == 0:
        return []
    k = max(1, int(unique * top_pct))
    most_common = [sig for sig, c in counter_obj.most_common(k)]
    return most_common

def attach_subgraph_features(pyg_dataset, selected_subgraphs_union):
    """
    For each Data in pyg_dataset, compute a binary presence vector for the selected_subgraphs_union
    and attach it as data.subgraph_feats (torch.float tensor of shape [num_graphs, feat_dim]).
    Also preserve original node names to allow re-canonicalization later if needed.
    """
    feat_dim = len(selected_subgraphs_union)
    if feat_dim == 0:
        # attach zeros
        for data in pyg_dataset:
            data.subgraph_feats = torch.zeros((1, 0), dtype=torch.float)
        return feat_dim

    sig_to_idx = {s: i for i, s in enumerate(selected_subgraphs_union)}
    for data in pyg_dataset:
        node_names = getattr(data, "orig_node_names", None)
        if node_names is None:
            node_names = [f"n{i}" for i in range(data.num_nodes)]
        # rebuild small graph as above
        G = nx.Graph()
        G.add_nodes_from(node_names)
        if data.edge_index is not None and data.edge_index.numel() > 0:
            ei = data.edge_index.cpu().numpy()
            for u, v in ei.T:
                if u < len(node_names) and v < len(node_names):
                    G.add_edge(node_names[u], node_names[v])

        present = set()
        node_list = list(node_names)
        # enumerate up to size 3 (same as mining settings)
        for k in range(2, min(3, len(node_list)) + 1):
            for combo in combinations(node_list, k):
                subG = G.subgraph(combo)
                if nx.is_connected(subG):
                    sig = canonicalize_subgraph(G, combo)
                    if sig in sig_to_idx:
                        present.add(sig_to_idx[sig])
        vec = torch.zeros((feat_dim,), dtype=torch.float)
        for idx in present:
            vec[idx] = 1.0
        # store as shape [1, feat_dim] so batching gives correct dims
        data.subgraph_feats = vec.unsqueeze(0)
    return feat_dim

def recover_subgraph_objects_from_pyg(pyg_dataset, top_sigs, min_size=2, max_size=3):
    """
    For each signature in top_sigs, find one representative NetworkX induced subgraph
    from the graphs in pyg_dataset. Returns dict(sig -> networkx.Graph).
    """
    sigs_set = set(top_sigs)
    found = {}
    for data in pyg_dataset:
        # rebuild node name list and networkx graph like in mining
        node_names = getattr(data, "orig_node_names", None)
        if node_names is None:
            node_names = [f"n{i}" for i in range(data.num_nodes)]
        G = nx.Graph()
        G.add_nodes_from(node_names)
        if data.edge_index is not None and data.edge_index.numel() > 0:
            ei = data.edge_index.cpu().numpy()
            for u, v in ei.T:
                if u < len(node_names) and v < len(node_names):
                    G.add_edge(node_names[u], node_names[v])

        node_list = list(node_names)
        # enumerate subgraphs (same sizes used during mining)
        for k in range(min_size, min(max_size, len(node_list)) + 1):
            for combo in combinations(node_list, k):
                subG = G.subgraph(combo)
                if not nx.is_connected(subG):
                    continue
                sig = canonicalize_subgraph(G, combo)
                if sig in sigs_set and sig not in found:
                    # store a copy so subsequent graph rebuilds don't mutate this
                    found[sig] = subG.copy()
                    if len(found) == len(sigs_set):
                        return found
    return found


def load_all_folders(hallu_folders, non_hallu_folders):
    """
    Loads data from hallucination and non-hallucination folders,
    labels them (1 for hallucination, 0 for non-hallucination),
    and splits 80/20 per folder.
    """
    all_train, all_test = [], []

    # Load hallucination folders
    for folder in hallu_folders:
        print(f"\n🔹 Loading hallucination graphs from: {folder}")
        folder_dataset = load_graph_data(folder, label_value=1)
        if not folder_dataset:
            print(f"⚠️ Skipped empty folder: {folder}")
            continue

        folder_dataset = sorted(folder_dataset, key=lambda d: getattr(d, 'circuit_id', 0))
        split_idx = int(len(folder_dataset) * 0.8)
        all_train.extend(folder_dataset[:split_idx])
        all_test.extend(folder_dataset[split_idx:])

    # Load non-hallucination folders
    for folder in non_hallu_folders:
        print(f"\n🔹 Loading non-hallucination graphs from: {folder}")
        folder_dataset = load_graph_data(folder, label_value=0)
        if not folder_dataset:
            print(f"⚠️ Skipped empty folder: {folder}")
            continue

        folder_dataset = sorted(folder_dataset, key=lambda d: getattr(d, 'circuit_id', 0))
        split_idx = int(len(folder_dataset) * 0.8)
        all_train.extend(folder_dataset[:split_idx])
        all_test.extend(folder_dataset[split_idx:])

    print(f"\n✅ Total hallucination graphs: {sum([1 for d in all_train+all_test if d.y.item()==1])}")
    print(f"✅ Total non-hallucination graphs: {sum([1 for d in all_train+all_test if d.y.item()==0])}")
    return all_train, all_test



def canonicalize_subgraph(G, nodes):
    # Return canonical string for induced connected subgraph (simple deterministic encoding)
    nodes = tuple(sorted(nodes, key=str))
    node_label_strs = []
    for n in nodes:
        feats = extract_node_features_from_name(n, G, comp_id=0, comp_size=0, degree=G.degree(n))
        node_label_strs.append(f"{int(feats[0])}:{int(feats[3])}")
    labels_part = "|".join(node_label_strs)
    idx_map = {n: i for i, n in enumerate(nodes)}
    edges_part = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            u, v = nodes[i], nodes[j]
            edges_part.append("1" if G.has_edge(u, v) or G.has_edge(v, u) else "0")
    return labels_part + ";" + ",".join(edges_part)

def extract_node_features_from_name(node_name, G, comp_id, comp_size, degree):
    """
    Parses node names like 'self_attn,2,2,attn_head' or similar and returns a small handcrafted feature vector.
    This keeps backward compatibility when CSV per-node features are missing.
    """
    # default values
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
        # fallback: try splitting by '/' or '.' if weird naming
        parts = re.split(r'[\/\.]', str(node_name))
        if parts:
            subtype = parts[-1]
            block_type = parts[0]

    # small vocab maps (extend as necessary)
    block_vocab = {"self_attn": 0, "mlp": 1, "embedding": 2, "output": 3, "unknown": 4}
    subtype_vocab = {
        "attn_head": 0, "mlp_in": 1, "mlp_out": 2,
        "o_proj": 3, "q_proj": 4, "k_proj": 5, "v_proj": 6, "unknown": 9
    }

    block_id = block_vocab.get(block_type, block_vocab["unknown"])
    subtype_id = subtype_vocab.get(subtype, subtype_vocab["unknown"])

    # Normalize numeric fields (heuristic)
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

def mine_frequent_subgraphs(pyg_dataset, min_size=2, max_size=3):
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
                G.add_edge(node_names[u], node_names[v])
        present = set()
        for k in range(min_size, min(max_size, len(node_names)) + 1):
            for combo in combinations(node_names, k):
                subG = G.subgraph(combo)
                if nx.is_connected(subG):
                    sig = canonicalize_subgraph(G, combo)
                    present.add(sig)
        for s in present:
            freq_counter[s] += 1
    return freq_counter

# --------------------------
# Loading graphs + incorporating per-node CSV features
# --------------------------

# def load_graph_data(subgraphs_path, label_value, node_features_dir=None, feature_columns="all", auto_unify_columns=True):
#     """
#     Loads graph .pkl files from a given folder and assigns a fixed label.

#     - node_features_dir: where per-circuit CSVs live (defaults to subgraphs_path).
#     - feature_columns: "all" or comma-separated list of column names to use. If "all" and auto_unify_columns=True,
#                        the loader will compute the union of numeric columns across all CSVs and use that fixed set.
#     - auto_unify_columns: when True, unify CSV numeric columns across circuits by taking the union.
#     """
#     pyg_dataset = []
#     graph_files = [f for f in os.listdir(subgraphs_path) if '_with_features' in f]
#     print(f"Found {len(graph_files)} circuit files in '{subgraphs_path}' directory.")

#     search_dir = node_features_dir if node_features_dir is not None else subgraphs_path

#     # --- Pre-scan CSVs to determine global numeric columns (if requested) ---
#     global_numeric_cols = []
#     numeric_cols_set = set()
#     csv_files = []
#     if auto_unify_columns:
#         for fname in os.listdir(search_dir):
#             if fname.lower().endswith(".csv") and fname.startswith("circuit_") and "node" in fname:
#                 csv_files.append(os.path.join(search_dir, fname))
#         for csvf in csv_files:
#             try:
#                 df_tmp = pd.read_csv(csvf, nrows=2)  # small read for dtype detection
#                 # ignore index columns like 'node' / 'node_name' when collecting numeric columns
#                 for c in df_tmp.columns:
#                     if c.lower() in ("node", "node_name"):
#                         continue
#                     if np.issubdtype(df_tmp[c].dtype, np.number):
#                         if c not in numeric_cols_set:
#                             numeric_cols_set.add(c)
#                             global_numeric_cols.append(c)
#             except Exception as e:
#                 print(f"Warning: could not read {csvf} during pre-scan: {e}")
#         # keep order stable: sort alphabetically if you prefer deterministic order:
#         global_numeric_cols = sorted(global_numeric_cols)

#     # If user explicitly specified feature_columns (comma separated list), use that instead
#     if feature_columns != "all":
#         specified = [c.strip() for c in feature_columns.split(",") if c.strip()]
#         if len(specified) > 0:
#             global_numeric_cols = specified

#     if len(global_numeric_cols) == 0:
#         print("No numeric CSV columns found or specified. CSV numeric features will be empty (only name-feats used).")

#     # --- Now iterate graphs and build consistent node feature vectors ---
#     for graph_file in graph_files:
#         match = re.search(r'circuit_(\d+)', graph_file)
#         if not match:
#             print(f"Warning: Could not parse circuit ID from filename '{graph_file}'. Skipping.")
#             continue
#         circuit_id = int(match.group(1))

#         file_path = os.path.join(subgraphs_path, graph_file)
#         with open(file_path, 'rb') as f:
#             G = pickle.load(f)

#         if G.number_of_nodes() == 0:
#             print(f"Warning: Circuit {circuit_id} has no nodes. Skipping.")
#             continue

#         original_nodes = list(G.nodes())
#         node_map = {node_name: i for i, node_name in enumerate(original_nodes)}

#         # Components handling
#         if G.number_of_edges() > 0:
#             if nx.is_directed(G):
#                 components = list(nx.weakly_connected_components(G))
#             else:
#                 components = list(nx.connected_components(G))
#         else:
#             components = [[node] for node in original_nodes]

#         node_to_comp_id = {node: i for i, comp in enumerate(components) for node in comp}
#         node_to_comp_size = {node: len(comp) for comp in components for node in comp}

#         # Try to read the specific CSV for this circuit (if present)
#         features_df = None
#         possible_names = [
#             f"circuit_{circuit_id}_node_features.csv",
#             f"circuit_{circuit_id}_node_circuits.csv",
#             f"circuit_{circuit_id}_node_features.tsv",
#             f"circuit_{circuit_id}_node_circuits.tsv"
#         ]
#         found_csv = None
#         for fname in os.listdir(search_dir):
#             if fname in possible_names:
#                 found_csv = os.path.join(search_dir, fname)
#                 break
#         if not found_csv:
#             # flexible fallback
#             for fname in os.listdir(search_dir):
#                 if fname.startswith(f"circuit_{circuit_id}") and "node" in fname and fname.endswith(".csv"):
#                     found_csv = os.path.join(search_dir, fname)
#                     break
#         if found_csv:
#             try:
#                 features_df = pd.read_csv(found_csv)
#                 if 'node' in features_df.columns:
#                     features_df = features_df.set_index('node')
#                 elif 'node_name' in features_df.columns:
#                     features_df = features_df.set_index('node_name')
#                 else:
#                     # no node index column — try to find a column that matches node names
#                     # If there's no match, we'll not use the CSV
#                     print(f"Warning: CSV {found_csv} has no 'node' or 'node_name' column; ignoring CSV for circuit {circuit_id}.")
#                     features_df = None
#             except Exception as e:
#                 print(f"Warning: Failed to read CSV {found_csv}: {e}")
#                 features_df = None

#         node_features_list = []
#         for node_name in original_nodes:
#             degree = G.degree(node_name)
#             comp_id = node_to_comp_id[node_name]
#             comp_size = node_to_comp_size[node_name]

#             # CSV features in ordered global_numeric_cols; fill with 0.0 if missing
#             csv_feats = []
#             if features_df is not None:
#                 if node_name in features_df.index:
#                     row = features_df.loc[node_name]
#                     for col in global_numeric_cols:
#                         # if column not present in this file, put 0.0
#                         if col in features_df.columns:
#                             try:
#                                 val = row[col]
#                                 csv_feats.append(float(val) if (not pd.isna(val)) else 0.0)
#                             except Exception:
#                                 csv_feats.append(0.0)
#                         else:
#                             csv_feats.append(0.0)
#                 else:
#                     # CSV exists but node missing — zero-fill numeric cols
#                     csv_feats = [0.0] * len(global_numeric_cols)
#             else:
#                 # CSV not available for this circuit — zero-fill numeric cols
#                 csv_feats = [0.0] * len(global_numeric_cols)

#             # handcrafted name-based features (keeps backward compat)
#             name_feats = extract_node_features_from_name(node_name, G, comp_id, comp_size, degree)

#             combined = name_feats + csv_feats
#             node_features_list.append(combined)

#         # Convert to tensor
#         node_features = torch.tensor(node_features_list, dtype=torch.float)

#         # Build edge_index (0-based indices)
#         edges = [(node_map[u], node_map[v]) for u, v in G.edges()]
#         if len(edges) > 0:
#             edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
#         else:
#             edge_index = torch.empty((2, 0), dtype=torch.long)

#         label = torch.tensor([label_value], dtype=torch.long)

#         data = Data(x=node_features, edge_index=edge_index, y=label)
#         data.circuit_id = circuit_id
#         data.orig_node_names = original_nodes
#         pyg_dataset.append(data)

#     return pyg_dataset

def load_graph_data(subgraphs_path, label_value, node_features_dir=None, feature_columns="all", auto_unify_columns=True):
    """
    Loads graph .pkl files from a given folder and assigns
    a fixed label (1 for hallucination, 0 for non-hallucination).

    Handles filenames like 'circuit_14_with_features.pkl' robustly.
    Retains edge-less graphs by encoding node degree, component ID, and size.
    """
    pyg_dataset = []
    graph_files = [f for f in os.listdir(subgraphs_path) if '_with_features' in f]
    print(f"Found {len(graph_files)} circuit files in '{subgraphs_path}' directory.")

    search_dir = node_features_dir if node_features_dir is not None else subgraphs_path

    for graph_file in graph_files:
        # ✅ Robustly extract circuit ID using regex
        match = re.search(r'circuit_(\d+)', graph_file)
        if not match:
            print(f"Warning: Could not parse circuit ID from filename '{graph_file}'. Skipping.")
            continue
        circuit_id = int(match.group(1))

        file_path = os.path.join(subgraphs_path, graph_file)
        with open(file_path, 'rb') as f:
            G = pickle.load(f)

        if G.number_of_nodes() == 0:
            print(f"Warning: Circuit {circuit_id} has no nodes. Skipping.")
            continue

        original_nodes = list(G.nodes())
        node_map = {node_name: i for i, node_name in enumerate(original_nodes)}

        # ✅ Handle both directed and undirected graphs (retain even edge-less ones)
        if G.number_of_edges() > 0:
            if nx.is_directed(G):
                components = list(nx.weakly_connected_components(G))
            else:
                components = list(nx.connected_components(G))
        else:
            components = [[node] for node in original_nodes]

        node_to_comp_id = {node: i for i, comp in enumerate(components) for node in comp}
        node_to_comp_size = {node: len(comp) for comp in components for node in comp}

        node_features_list = []
        for node_name in original_nodes:
            degree = G.degree(node_name)
            comp_id = node_to_comp_id[node_name]
            comp_size = node_to_comp_size[node_name]
            feats = extract_node_features_from_name(node_name, G, comp_id, comp_size, degree)
            node_features_list.append(feats)

        node_features = torch.tensor(node_features_list, dtype=torch.float)

        edges = [(node_map[u], node_map[v]) for u, v in G.edges()]
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
#             self.conv1 = GATConv(embedding_size, hidden_channels // gat_heads, heads=gat_heads, concat=True)
#             self.conv2 = GATConv(hidden_channels, hidden_channels // gat_heads, heads=gat_heads, concat=True)
#             self.conv3 = GATConv(hidden_channels, hidden_channels // gat_heads, heads=gat_heads, concat=True)
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
#         x = x.relu()

#         if edge_index is not None and edge_index.numel() > 0:
#             x = self.conv1(x, edge_index)
#             x = x.relu()
#             x = self.conv2(x, edge_index)
#             x = x.relu()
#             x = self.conv3(x, edge_index)
#         else:
#             if x.shape[1] != self.lin.in_features - (graph_feats.shape[1] if graph_feats is not None else 0):
#                 proj = getattr(self, "_no_edge_proj", None)
#                 if proj is None:
#                     target_dim = (self.lin.in_features - (graph_feats.shape[1] if graph_feats is not None else 0))
#                     self._no_edge_proj = Linear(x.shape[1], target_dim).to(x.device)
#                     proj = self._no_edge_proj
#                 x = proj(x)

#         x = self.att_pool(x, batch)  # shape [num_graphs, hidden_channels]

#         if graph_feats is not None:
#             # ensure graph_feats matches batch ordering and shapes: graph_feats should be [num_graphs, feat_dim]
#             if graph_feats.dim() == 3 and graph_feats.size(1) == 1:
#                 # if stored as (1, feat_dim), DataLoader will stack to (num_graphs, 1, feat_dim)
#                 graph_feats = graph_feats.squeeze(1)
#             x = torch.cat([x, graph_feats.to(x.device)], dim=1)

#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.lin(x)
#         return x

class GNNGraphClassifier(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes,
                 use_gat=False, gat_heads=4, graph_feat_dim=0, model_type = "gat"):
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


# --- 3. Training Function ---
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    model.to(device)
    for data in train_loader:
        data = data.to(device)
        # out = model(data.x, data.edge_index, data.batch)
        out = model(data.x, data.edge_index, data.batch, data.subgraph_feats if hasattr(data, "subgraph_feats") else None)
        loss = criterion(out, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

# --- 4. Testing (Evaluation) Function ---
def test(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    model.to(device)
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # out = model(data.x, data.edge_index, data.batch)
            out = model(data.x, data.edge_index, data.batch, data.subgraph_feats if hasattr(data, "subgraph_feats") else None)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    print("\n--- Evaluation Results on Test Set ---")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, zero_division=0))

# --- Main Execution Block ---
if __name__ == '__main__':

    BASE_PATH = 'circuits_data'
    SCORES_CSV_PATH = 'hallucination_scores.csv'


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # HALLU_FOLDERS = [
    #     os.path.join("hallucinating_circuits", 'code_dataset'),
    #     os.path.join("hallucinating_circuits", 'medal_1500_mc'),
    #     os.path.join("hallucinating_circuits", 'medical_abstract_1500'),
    #     os.path.join("hallucinating_circuits", 'numerical_dataset'),
    #     os.path.join("hallucinating_circuits", 'prompts_rationalization_binary'),
    # ]

    # NON_HALLU_FOLDERS = [
    #     os.path.join("Non_hallucinating_circuits", 'code_dataset'),
    #     os.path.join("Non_hallucinating_circuits", 'medal_1500_mc'),
    #     os.path.join("Non_hallucinating_circuits", 'medical_abstract_1500'),
    #     os.path.join("Non_hallucinating_circuits", 'numerical_dataset'),
    #     os.path.join("Non_hallucinating_circuits", 'prompts_rationalization_binary'),
    # ]
    HALLU_FOLDERS = [
        os.path.join("hallucinating_circuits", 'combined_folder'),
    ]

    NON_HALLU_FOLDERS = [
        os.path.join("Non_hallucinating_circuits", 'combined_folder'),
    ]

    # full_dataset = load_graph_data(SUBGRAPHS_PATH, SCORES_CSV_PATH)

    train_data, test_data = load_all_folders(HALLU_FOLDERS, NON_HALLU_FOLDERS)

    train_hallu = [d for d in train_data if d.y.item() == 1]
    train_nonhallu = [d for d in train_data if d.y.item() == 0]

    # 2) mine counters (use conservative max_size=3 to avoid explosion)
    hallu_counter =[]
    nonhallu_counter = []

    # 3) select top-k (top_pct default 0.1 -> top 10%)
    hallu_top = select_top_subgraphs(hallu_counter, top_pct=0.20)
    nonhallu_top = select_top_subgraphs(nonhallu_counter, top_pct=0.20)

    print(f"Selected {len(hallu_top)} frequent subgraphs for hallucination class")
    print(f"Selected {len(nonhallu_top)} frequent subgraphs for non-hallucination class")

    # hallu_subgraph_objs = recover_subgraph_objects_from_pyg(train_hallu, hallu_top, min_size=2, max_size=3)
    # nonhallu_subgraph_objs = recover_subgraph_objects_from_pyg(train_nonhallu, nonhallu_top, min_size=2, max_size=3)

    # # Attach frequency information to be used for visualization
    # hallu_freqs = {sig: hallu_counter[sig] for sig in hallu_top}
    # nonhallu_freqs = {sig: nonhallu_counter[sig] for sig in nonhallu_top}

    # Save both structures and their frequencies
    # with open("hallucination_subgraphs_struct.pkl", "wb") as f:
    #     pickle.dump({"graphs": hallu_subgraph_objs, "freqs": hallu_freqs}, f)

    # with open("non_hallucination_subgraphs_struct.pkl", "wb") as f:
    #     pickle.dump({"graphs": nonhallu_subgraph_objs, "freqs": nonhallu_freqs}, f)

    print("💾 Saved subgraph structures + frequencies to:")
    print("   - hallucination_subgraphs_struct.pkl")
    print("   - non_hallucination_subgraphs_struct.pkl")

    # 4) union and attach features
    selected_union = list(dict.fromkeys(hallu_top + nonhallu_top))  # preserves order and unique
    graph_feat_dim = len(selected_union)
    print(f"Using {graph_feat_dim} union subgraph features (will be appended to pooled embedding)")

    # attach to train and test
    attach_subgraph_features(train_data, selected_union)
    attach_subgraph_features(test_data, selected_union)


    if not train_data or not test_data:
        print("❌ Dataset empty after loading. Exiting.")
        exit()

    print(f"\nTraining set size: {len(train_data)}")
    print(f"Testing set size: {len(test_data)}")


    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        
    num_features = train_data[0].num_node_features

    print(f"Number of node features: {num_features}")
    print("node feature width (num_node_features) =", train_data[0].num_node_features)

    # Choose attention config here:
    # use_gat=True -> GATConv message passing; use_gat=False -> GCNConv
    # model = GNNGraphClassifier(num_node_features=num_features, hidden_channels=64, num_classes=2,
    #                         use_gat=True, gat_heads=4)

    model = GNNGraphClassifier(num_node_features=num_features, hidden_channels=64, num_classes=2,
                            use_gat=True, gat_heads=4, graph_feat_dim=graph_feat_dim)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print("\n--- Starting Training ---")
    for epoch in range(1, 201):
        loss = train(model, train_loader, optimizer, criterion, device)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, Avg. Loss: {loss:.4f}')

    test(model, test_loader, device)
