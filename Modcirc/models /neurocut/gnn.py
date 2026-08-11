import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GNN(nn.Module):
    """Graph encoder using PyG's GCN layers."""

    def __init__(
        self,
        node_activation_dims,
        input_dim,
        hidden_dim,
        emb_dim,
        num_layers=3,
    ):
        super().__init__()
        self.activations = {}
        for dim in node_activation_dims:
            setattr(self, 'activation_' + str(dim), nn.Linear(dim, input_dim))
            self.activations[dim] = getattr(self, 'activation_' + str(dim))

        # Input layer
        self.convs = nn.ModuleList([GCNConv(input_dim, hidden_dim)])

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Output layer
        self.convs.append(GCNConv(hidden_dim, emb_dim))
        self.emb_dim = emb_dim

    def create_partition_mlp(self, num_partitions):
        self.num_partitions = num_partitions
        self.partition_mlp = nn.Sequential(
            nn.Linear(self.emb_dim * 2, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, 1),
        )

    def get_embedding(self, data):
        """
        Forward pass using PyG data object.
        Args:
            data: PyG Data object with x and edge_index
        """
        feats = [a.to(data.edge_index.device) for a in data.activation]
        feats = [self.activations[feat.shape[-1]](feat).mean((0,1)) for feat in feats]
        x = torch.stack(feats)
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, data.edge_index))
        x = self.convs[-1](x, data.edge_index)
        del feats
        return x

    def get_partition_scores(self, node_embs, selected_node, partitions):
        """
        Compute partition scores for selected nodes.
        Args:
            node_embeddings: Node embeddings from GNN
            selected_nodes: Indices of selected nodes for each graph
            data: PyG Data object
        """
        partition_embs = torch.zeros(self.num_partitions, node_embs.shape[1]).to(node_embs.device)
        for i in range(self.num_partitions):
            if i in partitions:
                partition_embs[i] = node_embs[partitions == i].mean(dim=0)

        # Combine selected node and neighborhood embeddings
        combined = torch.cat([node_embs[selected_node].repeat(self.num_partitions, 1), partition_embs], dim=-1)

        # Compute partition scores
        scores = self.partition_mlp(combined).squeeze()

        del partition_embs, combined
        return scores
