import os
import pickle
import pandas as pd
import networkx as nx
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np

# --- 1. Data Loading and Preprocessing ---

def load_graph_data(subgraphs_path, scores_csv_path):
    """
    Loads graph .pkl files and their scores.
    This version INCLUDES graphs with no edges by creating features
    that describe their disconnected nature.
    """
    try:
        scores_df = pd.read_csv(scores_csv_path)
        score_dict = scores_df.set_index('circuit_id')['hallucination_score'].to_dict()
    except FileNotFoundError:
        print(f"Error: Scores CSV not found at {scores_csv_path}")
        return []

    threshold = scores_df['hallucination_score'].mean()
    print(f"Using hallucination score MEAN as threshold: {threshold:.4f}")

    pyg_dataset = []
    graph_files = [f for f in os.listdir(subgraphs_path) if f.endswith('.pkl')]
    print(f"Found {len(graph_files)} circuit files in '{subgraphs_path}' directory.")

    for graph_file in graph_files:
        try:
            circuit_id_str = os.path.splitext(graph_file)[0].split('_')[-1]
            circuit_id = int(circuit_id_str)
        except (IndexError, ValueError):
            print(f"Warning: Could not parse circuit ID from filename '{graph_file}'. Skipping.")
            continue

        if circuit_id not in score_dict:
            print(f"Warning: No score found for circuit {circuit_id}. Skipping.")
            continue

        with open(os.path.join(subgraphs_path, graph_file), 'rb') as f:
            G = pickle.load(f)

        if G.number_of_nodes() == 0:
            print(f"Warning: Circuit {circuit_id} has no nodes. Skipping.")
            continue

        original_nodes = list(G.nodes())
        node_map = {node_name: i for i, node_name in enumerate(original_nodes)}
        
        # --- KEY CHANGE: DO NOT DISCARD GRAPHS WITH NO EDGES ---
        # Instead, we will rely on node features to describe them.
        
        # Feature Engineering (works for graphs with or without edges)
        node_features_list = []
        if G.number_of_edges() > 0:
            # If the graph is directed, use weakly_connected_components
            if nx.is_directed(G):
                components = list(nx.weakly_connected_components(G))
            else: # If undirected, use connected_components
                components = list(nx.connected_components(G))
        else:
            # For edge-less graphs, each node is its own component of size 1
            components = [[node] for node in original_nodes]

        node_to_comp_id = {node: i for i, comp in enumerate(components) for node in comp}
        node_to_comp_size = {node: len(comp) for comp in components for node in comp}
                
        for node_name in original_nodes:
            degree = G.degree(node_name)
            comp_id = node_to_comp_id[node_name]
            comp_size = node_to_comp_size[node_name]
            node_features_list.append([node_name, degree, comp_id, comp_size])
            
        node_features = torch.tensor(node_features_list, dtype=torch.float)

        edges = [(node_map[u], node_map[v]) for u, v in G.edges()]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        score = score_dict[circuit_id]
        label = torch.tensor([1 if score > threshold else 0], dtype=torch.long)

        data = Data(x=node_features, edge_index=edge_index, y=label)
        data.circuit_id = circuit_id
        pyg_dataset.append(data)

    return pyg_dataset

# --- 2. GNN Model Definition with Node Embedding ---

class GNNGraphClassifier(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes):
        super(GNNGraphClassifier, self).__init__()
        torch.manual_seed(42)
        
        # --- KEY CHANGE: Add a Node Embedding Layer ---
        # This layer creates a rich representation from the initial node features.
        embedding_size = 32
        self.node_encoder = Linear(num_node_features, embedding_size)
        
        self.conv1 = GCNConv(embedding_size, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Encode node features into a higher-dimensional embedding space.
        x = self.node_encoder(x)
        x = x.relu()
        
        # 2. Perform message passing (this part is skipped for edge-less graphs).
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        
        # 3. Readout layer to get a graph-level embedding.
        x = global_mean_pool(x, batch)
        
        # 4. Apply a final classifier.
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

# --- 3. Training Function (remains the same) ---
def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
         out = model(data.x, data.edge_index, data.batch)
         loss = criterion(out, data.y)
         loss.backward()
         optimizer.step()
         optimizer.zero_grad()
         total_loss += loss.item() * data.num_graphs
    return total_loss / len(train_loader.dataset)

# --- 4. Testing (Evaluation) Function (remains the same) ---
def test(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data in loader:
            out = model(data.x, data.edge_index, data.batch)
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
    SUBGRAPHS_PATH = 'subgraphs'
    SCORES_CSV_PATH = 'hallucination_scores.csv'

    full_dataset = load_graph_data(SUBGRAPHS_PATH, SCORES_CSV_PATH)

    if not full_dataset:
        print("Dataset is empty after processing. Check warnings. Exiting.")
    else:
        print(f"\nLoaded {len(full_dataset)} valid graphs successfully.")
        
        labels = [data.y.item() for data in full_dataset]
        if len(set(labels)) < 2:
            print("\nError: The dataset has only one class after labeling. Cannot train a classifier.")
            print("This can happen if all hallucination scores are on one side of the mean,")
            print("or if there are too few graphs to form a balanced split.")
            exit()
        
        train_data, test_data = train_test_split(
            full_dataset, test_size=0.4, random_state=42, stratify=labels
        )
        print(f"Training set size: {len(train_data)}")
        print(f"Testing set size: {len(test_data)}")

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        
        num_features = full_dataset[0].num_node_features
        print(f"Number of node features: {num_features}")
        
        model = GNNGraphClassifier(num_node_features=num_features, hidden_channels=64, num_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        print("\n--- Starting Training ---")
        for epoch in range(1, 201):
            loss = train(model, train_loader, optimizer, criterion)
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Avg. Loss: {loss:.4f}')

        test(model, test_loader)