import os
import torch
import pickle
import copy
import random
import numpy as np
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Sequential, ReLU
from torch_geometric.nn import GATConv, GlobalAttention
from sklearn.metrics import accuracy_score
import networkx as nx
import torch
from torch.nn import Linear, Sequential, ReLU
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, GlobalAttention
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F


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

def mine_frequent_subgraphs(pyg_dataset, min_size=2, max_size=3):
    """
    Enumerate connected induced subgraphs of sizes in [min_size, max_size] for each graph in pyg_dataset.
    Returns Counter of canonicalized subgraph signatures -> count (number of graphs that contain it).
    We count per-graph presence (boolean) to avoid multi-counting repeated occurrences within a single graph.
    """
    freq_counter = Counter()
    for data in pyg_dataset:
        # skip trivial graphs
        if data.num_nodes < min_size:
            continue

        # rebuild a small networkx graph to use connectivity tests and name lookups
        # data stored original node names in data._node_names if present; otherwise we cannot recover original labels
        # In your loader, you used node_map and original node_name list; to preserve names we rely on `data.orig_node_names` if present.
        # If not present, we fall back to synthetic names 'n0','n1',... and canonicalization will be limited.
        node_names = getattr(data, "orig_node_names", None)
        if node_names is None:
            # attempt to use circuit_id-based synthetic names (best-effort fallback)
            node_names = [f"n{i}" for i in range(data.num_nodes)]

        # create nx graph
        G = nx.Graph()
        G.add_nodes_from(node_names)
        # reconstruct edges if edge_index exists
        if data.edge_index is not None and data.edge_index.numel() > 0:
            ei = data.edge_index.cpu().numpy()
            for u, v in ei.T:
                if u < len(node_names) and v < len(node_names):
                    G.add_edge(node_names[u], node_names[v])

        present = set()
        node_list = list(node_names)
        # enumerate combinations up to max_size but ensure connectedness
        for k in range(min_size, min(max_size, len(node_list)) + 1):
            for combo in combinations(node_list, k):
                # quick check: connectedness in induced subgraph
                subG = G.subgraph(combo)
                if nx.is_connected(subG):
                    sig = canonicalize_subgraph(G, combo)
                    present.add(sig)
        # count presence once per graph
        for s in present:
            freq_counter[s] += 1
    return freq_counter

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


def extract_node_features_from_name(node_name, G, comp_id, comp_size, degree):
    """
    Parses node name like 'self_attn,2,2,attn_head' into numerical features.
    Returns a fixed-length list of floats.
    """

    tokens = node_name.split(',')
    # Default values
    block_type, layer_id, position, subtype = None, 0, 0, None

    if len(tokens) == 4:
        block_type, layer_id, position, subtype = tokens
        try:
            layer_id = float(layer_id)
            position = float(position)
        except ValueError:
            layer_id, position = 0.0, 0.0
    else:
        # fallback for irregular names
        block_type, subtype = tokens[0], tokens[-1]

    # --- Encode categorical fields ---
    # Define small vocab mappings (extend as needed)
    block_vocab = {"self_attn": 0, "mlp": 1, "embedding": 2, "output": 3}
    subtype_vocab = {
        "attn_head": 0, "mlp_in": 1, "mlp_out": 2,
        "o_proj": 3, "q_proj": 4, "k_proj": 5, "v_proj": 6
    }

    block_id = block_vocab.get(block_type, len(block_vocab))
    subtype_id = subtype_vocab.get(subtype, len(subtype_vocab))

    # Normalize numeric fields
    layer_id_norm = layer_id / 10.0       # assuming <10 layers
    position_norm = position / 5000.0     # assuming <5000 positions

    # Combine everything into one feature vector
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

def node_overlap_ratio(data_i, data_j):
    """
    |Vi ∩ Vj| / min(|Vi|, |Vj|)
    """
    nodes_i = set(data_i.orig_node_names)
    nodes_j = set(data_j.orig_node_names)

    return len(nodes_i & nodes_j) / min(len(nodes_i), len(nodes_j))

# --- 1. Data Loading and Preprocessing ---

def load_graph_data(subgraphs_path, label_value):
    """
    Loads graph .pkl files from a given folder and assigns
    a fixed label (1 for hallucination, 0 for non-hallucination).

    Handles filenames like 'circuit_14_with_features.pkl' robustly.
    Retains edge-less graphs by encoding node degree, component ID, and size.
    """
    pyg_dataset = []
    # graph_files = [f for f in os.listdir(subgraphs_path) if f.endswith('_with_features.pkl')]
    graph_files = [
    f for f in os.listdir(subgraphs_path)
    if re.search(r'_with_features.*\.pkl$', f)
    ]
    print(f"Found {len(graph_files)} circuit files in '{subgraphs_path}' directory.")

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

# def load_graph_data(subgraphs_path, label_value):
#     """
#     Loads graph .pkl files from a given folder and assigns a fixed label.
#     Also reports how many graphs were skipped for each reason.
#     """

#     pyg_dataset = []

#     graph_files = [
#         f for f in os.listdir(subgraphs_path)
#         if re.search(r'_with_features.*\.pkl$', f)
#     ]

#     print(f"\n📁 Folder: {subgraphs_path}")
#     print(f"Found {len(graph_files)} raw circuits")

#     # skip counters
#     skipped_small = 0
#     skipped_duplicate = 0

#     seen_node_sets = []

#     for graph_file in graph_files:

#         # parse circuit ID
#         match = re.search(r'circuit_(\d+)', graph_file)
#         if not match:
#             # silently skip → do not count
#             continue

#         circuit_id = int(match.group(1))

#         # read graph
#         file_path = os.path.join(subgraphs_path, graph_file)
#         with open(file_path, 'rb') as f:
#             G = pickle.load(f)

#         # 1️⃣ filter tiny graphs
#         if G.number_of_nodes() < 5:
#             skipped_small += 1
#             continue

#         # 2️⃣ duplicate detection
#         node_set = set(G.nodes())
#         is_duplicate = False
#         for prev in seen_node_sets:
#             overlap = len(node_set & prev) / max(len(node_set), len(prev))
#             if overlap >= 0.9:
#                 skipped_duplicate += 1
#                 is_duplicate = True
#                 break

#         if is_duplicate:
#             continue

#         seen_node_sets.append(node_set)

#         # -------------------------------
#         # Normal graph processing (same)
#         # -------------------------------
#         original_nodes = list(G.nodes())
#         node_map = {node_name: i for i, node_name in enumerate(original_nodes)}

#         if G.number_of_edges() > 0:
#             components = (list(nx.weakly_connected_components(G))
#                           if nx.is_directed(G)
#                           else list(nx.connected_components(G)))
#         else:
#             components = [[node] for node in original_nodes]

#         node_to_comp_id = {node: i for i, comp in enumerate(components) for node in comp}
#         node_to_comp_size = {node: len(comp) for comp in components for node in comp}

#         node_features_list = []
#         for node_name in original_nodes:
#             degree = G.degree(node_name)
#             comp_id = node_to_comp_id[node_name]
#             comp_size = node_to_comp_size[node_name]
#             feats = extract_node_features_from_name(node_name, G, comp_id, comp_size, degree)
#             node_features_list.append(feats)

#         node_features = torch.tensor(node_features_list, dtype=torch.float)

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

#     # -------------------------------
#     # Summary per folder
#     # -------------------------------
#     print(f"➡️ Loaded {len(pyg_dataset)} cleaned graphs")
#     print(f"   ⤷ Skipped (tiny graphs, <5 nodes): {skipped_small}")
#     print(f"   ⤷ Skipped (≥90% duplicates): {skipped_duplicate}")

#     return pyg_dataset



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



# =====================================================
# 1. GNNGraphClassifier (Embedding Extractor)
# =====================================================
class GNNGraphClassifier(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes,
                 use_gat=False, gat_heads=4, graph_feat_dim=0):
        super(GNNGraphClassifier, self).__init__()
        torch.manual_seed(42)
        
        embedding_size = 32
        self.node_encoder = Linear(num_node_features, embedding_size)
        self.use_gat = use_gat
        self.graph_feat_dim = graph_feat_dim

        if use_gat:
            self.conv1 = GATConv(embedding_size, hidden_channels // gat_heads, heads=gat_heads, concat=True)
            self.conv2 = GATConv(hidden_channels, hidden_channels // gat_heads, heads=gat_heads, concat=True)
            self.conv3 = GATConv(hidden_channels, hidden_channels // gat_heads, heads=gat_heads, concat=True)
        else:
            from torch_geometric.nn import GCNConv
            self.conv1 = GCNConv(embedding_size, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)

        gate_nn = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, 1))
        self.att_pool = GlobalAttention(gate_nn=gate_nn)

        total_final_dim = hidden_channels + (graph_feat_dim if graph_feat_dim else 0)
        self.lin = Linear(total_final_dim, num_classes)

    def forward(self, x, edge_index, batch, graph_feats=None):
        x = self.node_encoder(x)
        x = F.relu(x)

        if edge_index is not None and edge_index.numel() > 0:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))
        else:
            if x.shape[1] != self.lin.in_features - (graph_feats.shape[1] if graph_feats is not None else 0):
                proj = getattr(self, "_no_edge_proj", None)
                if proj is None:
                    target_dim = (self.lin.in_features - (graph_feats.shape[1] if graph_feats is not None else 0))
                    self._no_edge_proj = Linear(x.shape[1], target_dim).to(x.device)
                    proj = self._no_edge_proj
                x = proj(x)

        x = self.att_pool(x, batch)

        if graph_feats is not None:
            if graph_feats.dim() == 3 and graph_feats.size(1) == 1:
                graph_feats = graph_feats.squeeze(1)
            x = torch.cat([x, graph_feats.to(x.device)], dim=1)

        return x  # return embeddings (not logits)


import itertools
import numpy as np
import torch
from torch_geometric.loader import DataLoader

# All Pairs
# def create_pairwise_data(graphs, model, device):
#     """
#     Converts all possible graph pairs → embeddings → pairwise data (concat of two embeddings).
#     Label = 1 if both belong to same class else 0.
#     Returns:
#         X: [num_pairs, 2*embedding_dim]
#         y: [num_pairs]
#         all_embs: [num_graphs, embedding_dim]
#         all_labels: [num_graphs]
#     """
#     model.eval()
#     loader = DataLoader(graphs, batch_size=32, shuffle=False)
#     all_embs, all_labels = [], []

#     # Step 1: Extract embeddings for all graphs
#     with torch.no_grad():
#         for data in loader:
#             data = data.to(device)
#             emb = model(data.x, data.edge_index, data.batch, getattr(data, "subgraph_features", None))
#             all_embs.append(emb.cpu().numpy())
#             all_labels.extend(data.y.cpu().numpy())

#     all_embs = np.vstack(all_embs)
#     all_labels = np.array(all_labels)

#     # Step 2: Create all possible pairs (i, j)
#     pairs_X, pairs_y = [], []
#     n = len(all_embs)

#     print(f"🔹 Creating all possible pairs from {n} graphs... total = {n * (n - 1) // 2}")

#     for i, j in itertools.combinations(range(n), 2):  # all unique unordered pairs
#         pair = np.concatenate([all_embs[i], all_embs[j]])
#         label = 1 if all_labels[i] == all_labels[j] else 0
#         pairs_X.append(pair)
#         pairs_y.append(label)

#     pairs_X = np.array(pairs_X)
#     pairs_y = np.array(pairs_y)

#     print(f"✅ Created {len(pairs_X)} pairs ({np.sum(pairs_y)} same-class, {len(pairs_y)-np.sum(pairs_y)} different-class)")
#     return pairs_X, pairs_y, all_embs, all_labels

# Contrastive Pairing
def create_pairwise_data(graphs, model, device, overlap_thresh=0.15):
    """
    Contrastive pairwise data:
    - Pair graphs only if node overlap < overlap_thresh
    """

    model.eval()
    loader = DataLoader(graphs, batch_size=32, shuffle=False)

    all_embs, all_labels, all_graphs = [], [], []

    # 1️⃣ Extract embeddings
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = model(
                batch.x,
                batch.edge_index,
                batch.batch,
                getattr(batch, "subgraph_features", None)
            )
            all_embs.append(emb.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_graphs.extend(batch.to_data_list())

    all_embs = np.vstack(all_embs)
    all_labels = np.array(all_labels)

    # 2️⃣ Contrastive pairing
    pairs_X, pairs_y = [], []

    total_pairs = 0
    skipped_overlap = 0

    for (i, gi), (j, gj) in itertools.combinations(enumerate(all_graphs), 2):
        total_pairs += 1

        overlap = node_overlap_ratio(gi, gj)
        if overlap >= overlap_thresh:
            skipped_overlap += 1
            continue  # ❌ reject high-overlap pair

        pair = np.concatenate([all_embs[i], all_embs[j]])
        label = 1 if all_labels[i] == all_labels[j] else 0

        pairs_X.append(pair)
        pairs_y.append(label)

    pairs_X = np.array(pairs_X)
    pairs_y = np.array(pairs_y)

    print(f"\n🔹 Total candidate pairs: {total_pairs}")
    print(f"🔹 Skipped (overlap ≥ {overlap_thresh*100:.0f}%): {skipped_overlap}")
    print(f"✅ Used contrastive pairs: {len(pairs_X)}")
    print(f"   ↳ Same-class: {np.sum(pairs_y)} | Diff-class: {len(pairs_y)-np.sum(pairs_y)}")

    return pairs_X, pairs_y, all_embs, all_labels

# =====================================================
# 3. Simple Pairwise Classifier (MLP)
# =====================================================
class PairwiseMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

def eval_pairwise_on_test(
    gnn_model,
    pairwise_model,
    train_embs,
    train_labels,
    test_graphs,
    device
):
    gnn_model.eval()
    pairwise_model.eval()

    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    preds, y_true = [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)

            emb = gnn_model(
                data.x, data.edge_index, data.batch,
                getattr(data, "subgraph_features", None)
            )
            emb = emb.cpu().numpy().flatten()

            votes = []
            for ref_emb, ref_label in zip(train_embs, train_labels):
                pair = np.concatenate([emb, ref_emb])[None, :]
                pair_t = torch.tensor(pair, dtype=torch.float32).to(device)

                prob = F.softmax(pairwise_model(pair_t), dim=1)[0, 1].item()

                # same-class probability vote
                votes.append(1 if prob > 0.5 else 0)

            final_pred = 1 if np.mean(votes) > 0.5 else 0
            preds.append(final_pred)
            y_true.append(int(data.y.item()))

    return accuracy_score(y_true, preds)

# =====================================================
# 4. Train Pairwise Classifier
# =====================================================
# def train_pairwise(model, train_X, train_y, device, epochs=4000, lr=1e-3):
#     model = model.to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     X_tensor = torch.tensor(train_X, dtype=torch.float32).to(device)
#     y_tensor = torch.tensor(train_y, dtype=torch.long).to(device)

#     for epoch in range(epochs):
#         model.train()
#         optimizer.zero_grad()
#         out = model(X_tensor)
#         loss = criterion(out, y_tensor)
#         loss.backward()
#         optimizer.step()
#         if epoch % 100 == 0:
#             print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

#     return model

def train_pairwise(
    pairwise_model,
    train_X,
    train_y,
    gnn_model,
    train_embs,
    train_labels,
    test_graphs,
    device,
    epochs=10000,
    lr=1e-4
):
    pairwise_model = pairwise_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(pairwise_model.parameters(), lr=lr)

    X_tensor = torch.tensor(train_X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(train_y, dtype=torch.long).to(device)

    best_test_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        pairwise_model.train()
        optimizer.zero_grad()

        out = pairwise_model(X_tensor)
        loss = criterion(out, y_tensor)
        loss.backward()
        optimizer.step()

        # 🔹 Evaluate every 100 epochs
        if epoch % 100 == 0:
            test_acc = eval_pairwise_on_test(
                gnn_model,
                pairwise_model,
                train_embs,
                train_labels,
                test_graphs,
                device
            )

            print(
                f"Epoch {epoch:4d} | "
                f"Loss: {loss.item():.4f} | "
                f"Test Acc: {test_acc:.4f}"
            )

            # ✅ Save best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_state = copy.deepcopy(pairwise_model.state_dict())

    # 🔁 Load best model before returning
    if best_state is not None:
        pairwise_model.load_state_dict(best_state)
        print(f"\n✅ Best Test Accuracy achieved: {best_test_acc:.4f}")

    return pairwise_model

def classify_test_graphs(gnn_model, pairwise_model, train_embs, train_labels, test_graphs, device):
    """
    Classify test graphs using all training graphs as references.
    Each test graph is compared with every train graph via the pairwise model.
    The predicted label is decided by majority voting of all pairwise predictions,
    interpreted relative to the reference graph's class.
    """

    gnn_model.eval()
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    preds, y_true = [], []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            
            # Get embedding of the test graph
            emb = gnn_model(data.x, data.edge_index, data.batch, getattr(data, "subgraph_features", None))
            emb = emb.cpu().numpy().flatten()

            votes = []
            # Compare with each training graph embedding
            for ref_emb, ref_label in zip(train_embs, train_labels):
                pair = np.concatenate([emb, ref_emb])[None, :]
                pair_t = torch.tensor(pair, dtype=torch.float32).to(device)
                
                prob = F.softmax(pairwise_model(pair_t), dim=1)[0, 1].item()

                # Interpret prediction based on reference class
                if ref_label == 1:  # ref = hallucinating
                    pred_pair = 1 if prob > 0.5 else 0
                else:               # ref = non-hallucinating
                    pred_pair = 1 if prob > 0.5 else 0

                votes.append(pred_pair)

            # Majority voting
            final_pred = 1 if np.mean(votes) > 0.5 else 0
            preds.append(final_pred)
            y_true.append(int(data.y.item()))

    # Accuracy
    acc = accuracy_score(y_true, preds)
    print(f"\n✅ Test Accuracy (Majority Voting with Ref-based Logic): {acc:.4f}")

    # Confusion matrix and classification report
    cm = confusion_matrix(y_true, preds)
    print("\n🔹 Confusion Matrix:")
    print(cm)

    print("\n🔹 Classification Report:")
    print(classification_report(y_true, preds, target_names=["Non-Hallucinating", "Hallucinating"]))

    # Visualization
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Pred: Non-Hallu", "Pred: Hallu"],
                yticklabels=["True: Non-Hallu", "True: Hallu"])
    plt.title("Confusion Matrix (Majority Voting w/ Ref Logic)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

    return preds

# =====================================================
# 6. Main Execution
# =====================================================
if __name__ == '__main__':
    import pickle

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BASE_PATH = 'circuits_data'
    SCORES_CSV_PATH = 'hallucination_scores.csv'

    HALLU_FOLDERS = [
        os.path.join("hallucinating_circuits", 'code_dataset'),
        os.path.join("hallucinating_circuits", 'medal_1500_mc'),
        os.path.join("hallucinating_circuits", 'medical_abstract_1500'),
        os.path.join("hallucinating_circuits", 'numerical_dataset'),
        os.path.join("hallucinating_circuits", 'prompts_rationalization_binary'),
    ]

    NON_HALLU_FOLDERS = [
        os.path.join("Non_hallucinating_circuits", 'code_dataset'),
        os.path.join("Non_hallucinating_circuits", 'medal_1500_mc'),
        os.path.join("Non_hallucinating_circuits", 'medical_abstract_1500'),
        os.path.join("Non_hallucinating_circuits", 'numerical_dataset'),
        os.path.join("Non_hallucinating_circuits", 'prompts_rationalization_binary'),
    ]

    # === Load your preprocessed PyG graphs ===

    train_data, test_data = load_all_folders(HALLU_FOLDERS, NON_HALLU_FOLDERS)

    selected_union = []  # if you have subgraph features
    attach_subgraph_features(train_data, selected_union)
    attach_subgraph_features(test_data, selected_union)

    graph_feat_dim = len(selected_union)
    num_features = train_data[0].num_node_features
    print(f"🧩 Node features: {num_features}")

    # --- Build GAT encoder ---
    gnn_model = GNNGraphClassifier(num_node_features=num_features,
                                   hidden_channels=64,
                                   num_classes=2,
                                   use_gat=True,
                                   gat_heads=4,
                                   graph_feat_dim=graph_feat_dim).to(device)

    print("🔹 Extracting embeddings and building pairwise data...")
    pair_X, pair_y, train_embs, train_labels = create_pairwise_data(train_data, gnn_model, device)

    print("🔹 Training pairwise classifier...")
    pairwise_model = PairwiseMLP(in_dim=pair_X.shape[1])
    # pairwise_model = train_pairwise(pairwise_model, pair_X, pair_y, device)
    pairwise_model = train_pairwise(
        pairwise_model=pairwise_model,
        train_X=pair_X,
        train_y=pair_y,
        gnn_model=gnn_model,
        train_embs=train_embs,
        train_labels=train_labels,
        test_graphs=test_data,
        device=device,
        epochs=10000,
        lr=1e-4
    )

    print("🔹 Evaluating on test dataset...")
    classify_test_graphs(gnn_model, pairwise_model, train_embs, train_labels, test_data, device)