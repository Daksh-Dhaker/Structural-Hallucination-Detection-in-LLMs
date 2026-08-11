# test data generation: should be removed after circuits are acquired.

import torch
from torch_geometric.data import Data, Dataset, DataLoader
import networkx as nx
import numpy as np

class ModularGraphDataset(Dataset):
    """Dataset of synthetic graphs with modular structure."""

    def __init__(self, num_graphs=100, min_nodes=20, max_nodes=50,
                 num_feats=32, num_modules=3, module_size=5):
        """
        Args:
            num_graphs: Number of graphs to generate
            min_nodes: Minimum number of nodes per graph
            max_nodes: Maximum number of nodes per graph
            num_feats: Dimension of node features
            num_modules: Number of distinct modular structures to use
            module_size: Approximate size of each module
        """
        # super().__init__(None, None, None)
        super().__init__(None, None, None)
        self.num_graphs = num_graphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.num_feats = num_feats
        self.num_modules = num_modules
        self.module_size = module_size

        # Generate and store graphs
        self.graphs = []

        self._generate_graphs()

    def _generate_module_template(self):
        """Generate a template for a modular structure."""
        # Create dense subgraph
        num_nodes = self.module_size
        density = 0.7  # High internal density for modules

        G = nx.erdos_renyi_graph(num_nodes, density)
        return G

    def _generate_module_features(self):
        """Generate distinct feature pattern for a module."""
        # Create base pattern
        base = torch.randn(self.module_size, self.num_feats)
        # Add some structure by emphasizing certain dimensions
        mask = torch.zeros(self.num_feats)
        mask[:self.module_size] = 1
        weighted = base * mask
        return torch.nn.functional.normalize(weighted, dim=1)

    def _generate_single_graph(self):
        """Generate a single graph with modular structure."""
        # Determine graph size
        num_nodes = np.random.randint(self.min_nodes, self.max_nodes + 1)

        # Initialize empty graph
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        # Generate node features
        features = torch.randn(num_nodes, self.num_feats)

        # Place modules in graph
        modules_placed = 0
        nodes_used = 0
        module_templates = [self._generate_module_template() for _ in range(self.num_modules)]
        module_features = [self._generate_module_features() for _ in range(self.num_modules)]

        while nodes_used + self.module_size <= num_nodes and modules_placed < self.num_modules:
            # Select a module template
            module_idx = np.random.randint(len(module_templates))
            module = module_templates[module_idx]
            module_feat = module_features[module_idx]

            # Map module to current graph
            module_nodes = list(range(nodes_used, nodes_used + self.module_size))
            module_mapping = dict(zip(range(self.module_size), module_nodes))

            # Add module edges
            for edge in module.edges():
                u, v = edge
                G.add_edge(module_mapping[u], module_mapping[v])

            # Add module features
            features[module_nodes] = module_feat

            # Add some random connections to rest of graph
            if nodes_used > 0:
                num_external = np.random.randint(1, 4)
                for _ in range(num_external):
                    module_node = np.random.choice(module_nodes)
                    other_node = np.random.randint(0, nodes_used)
                    G.add_edge(module_node, other_node)

            nodes_used += self.module_size
            modules_placed += 1

        # Add some random edges for remaining nodes
        for i in range(nodes_used, num_nodes):
            num_edges = np.random.randint(1, 4)
            for _ in range(num_edges):
                j = np.random.randint(0, i)
                G.add_edge(i, j)

        # Convert to PyG format
        edge_index = torch.tensor(list(G.edges())).t().contiguous()

        # Create bidirectional edges
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # Normalize features
        features = torch.nn.functional.normalize(features, dim=1)

        return Data(
            x=features,
            edge_index=edge_index,
            num_nodes=num_nodes
        )

    def _generate_graphs(self):
        """Generate all graphs in the dataset."""
        for _ in range(self.num_graphs):
            self.graphs.append(self._generate_single_graph())

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

def create_test_dataloader(
    num_graphs=100,
    min_nodes=20,
    max_nodes=50,
    num_feats=32,
    num_modules=3,
    module_size=5,
    batch_size=32
):
    """
    Create a dataloader with synthetic graphs for testing.

    Args:
        num_graphs: Number of graphs to generate
        min_nodes: Minimum number of nodes per graph
        max_nodes: Maximum number of nodes per graph
        num_feats: Dimension of node features
        num_modules: Number of distinct modular structures
        module_size: Size of each module
        batch_size: Batch size for the dataloader

    Returns:
        PyG DataLoader object
    """
    # Create dataset
    dataset = ModularGraphDataset(
        num_graphs=num_graphs,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        num_feats=num_feats,
        num_modules=num_modules,
        module_size=module_size
    )

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return loader
