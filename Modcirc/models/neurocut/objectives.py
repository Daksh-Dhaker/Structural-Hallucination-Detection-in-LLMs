import torch

class Objectives:
    """Class implementing various objective functions for graph partitioning."""

    def __init__(self, alpha=0.5, beta=10, gamma=5, infty_val=1e6):
        """
        Initialize Objectives calculator.
        Args:
            alpha: Weight factor for composability calculation
            beta: Maximum allowed total unique modular circuits
            gamma: Maximum allowed modular circuits per individual circuit
            infty_val: Value to use for infinity in penalty
        """
        self.alpha = alpha
        self.beta = beta  # Threshold for total unique circuits
        self.gamma = gamma  # Threshold for circuits per graph
        self.infty_val = infty_val

    def get_partition_in_out_edges(self, partitions, edge_idx, partition_idx):
        in_edges = (partitions[edge_idx[0]] != partition_idx) & (
            partitions[edge_idx[1]] == partition_idx
        )
        out_edges = (partitions[edge_idx[0]] == partition_idx) & (
            partitions[edge_idx[1]] != partition_idx
        )
        return in_edges, out_edges

    def calculate_node_vals(self, node_embs, data, bins=20, epsilon=1e-10):
        if "emb_dist" in dir(self):
            del self.emb_dist, self.layer_dist, self.mutual_info
        node_one = node_embs.repeat_interleave(len(node_embs), 0).reshape(len(node_embs),len(node_embs),-1)
        node_two = node_embs.repeat(len(node_embs), 1,1)
        self.emb_dist = (node_one - node_two).sum(-1).pow(2)

        layer_one = data.layer.repeat_interleave(len(data.layer), 0).reshape(len(data.layer),len(data.layer),-1)
        layer_two = data.layer.repeat(len(data.layer), 1,1)
        self.layer_dist = (layer_one - layer_two).sum(-1).pow(2)

        idx_one = torch.arange(0, len(node_embs)).repeat_interleave(len(node_embs), 0).reshape(len(node_embs),len(node_embs),-1)
        idx_one = idx_one.repeat(1,1, node_embs.shape[-1])
        node_one = (node_one - node_one.min(-1, keepdim=True)[0]) / (node_one.max(-1, keepdim=True)[0] - node_one.min(-1, keepdim=True)[0] + epsilon)
        node_one = torch.floor(node_one * (bins - 1)).long()

        idx_two = torch.arange(0, len(node_embs)).repeat(len(node_embs), 1)
        idx_two = idx_two.unsqueeze(-1).repeat(1,1, node_embs.shape[-1])
        node_two = (node_two - node_two.min(-1, keepdim=True)[0]) / (node_two.max(-1, keepdim=True)[0] - node_two.min(-1, keepdim=True)[0] + epsilon)
        node_two = torch.floor(node_two * (bins - 1)).long()

        joint_hist = torch.zeros(len(node_embs), len(node_embs), bins, bins, device=node_embs.device)
        joint_hist.index_put_(
            (idx_one.flatten(), idx_two.flatten(), node_one.flatten(), node_two.flatten()),
            torch.ones_like(node_one.flatten(), device=node_embs.device).float(),
            accumulate=True
        )
        joint_hist = joint_hist / joint_hist.sum((-1,-2), keepdim=True)
        p_x = joint_hist.sum(-1).unsqueeze(-1)
        p_y = joint_hist.sum(-2).unsqueeze(-2)
        ratio = joint_hist / (p_x * p_y + epsilon)
        self.mutual_info = (joint_hist * torch.log2(ratio + epsilon)).sum((-1, -2))
        del node_one, node_two, layer_one, layer_two, idx_one, idx_two, joint_hist, p_x, p_y, ratio
    
    def calculate_locality(self, partitions):
        sem_loc_in, sem_loc_across = 0, 0
        phys_loc_in, phys_loc_across = 0, 0

        for p_idx in partitions.unique():
            partition_nodes = (partitions == p_idx).nonzero().squeeze(1)
            other_nodes = (partitions != p_idx).nonzero().squeeze(1)
            sem_loc_in += self.emb_dist[partition_nodes][:, partition_nodes].sum()
            sem_loc_across += self.emb_dist[partition_nodes][:, other_nodes].sum()
            phys_loc_in += self.layer_dist[partition_nodes][:, partition_nodes].sum()
            phys_loc_across += self.layer_dist[partition_nodes][:, other_nodes].sum()
        locality = (sem_loc_in + phys_loc_in) / (sem_loc_across + phys_loc_across + 1e-10)
        return locality / len(partitions.unique())
    
    def calculate_composability(self, partitions, data):
        composability = 0
        for p_idx in partitions.unique():
            in_edges, _ = self.get_partition_in_out_edges(partitions, data.edge_index, p_idx)
            in_edges = data.edge_index[:, in_edges] # 2* num_in_edges
            composability += self.mutual_info[in_edges[0], in_edges[1]].sum()
        return composability

    def __call__(self, data, partitions):
        """
        Compute total objective value for the graph.
        Args:
            data: PyG Data object
            partitions: Current partition assignments
            node_embs: Node embeddings/features
            dataset_partitions: List of all partitions in dataset for globality computation
        """
        # Calculate independence and composability
        ave_locality = self.calculate_locality(partitions)
        ave_composability = self.calculate_composability(partitions, data)
        total_objective = ave_locality + ave_composability
        return ave_locality, ave_composability, total_objective