import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from models.circuit_discovery import nx_to_pyg
from models.neurocut.reward import RewardCalculator
from models.neurocut.gnn import GNN
from models.neurocut.objectives import Objectives
from torch_geometric.utils import degree
import networkx as nx


class NeuroCUTTrainer:
    """Trainer for NeuroCUT using PyG data structures."""

    def __init__(
        self,
        gnn: GNN,
        reward_calculator: RewardCalculator,
        objective: Objectives,
        device: str,
        results_path: str,
        comp_graph: nx.DiGraph,
        lr: float
    ):
        self.comp_graph = comp_graph.copy()
        self.starting_partitions = self.get_starting_partitions()
        # self.num_partitions = max(self.starting_partitions) + 1
        self.num_partitions = int(self.starting_partitions.max().item()) + 1  # ensure python int
        self.aps = [self.get_aps(self.starting_partitions, idx) for idx in range(self.num_partitions)]
        gnn.create_partition_mlp(self.num_partitions)
        self.gnn = gnn.to(device)
        self.reward_calculator = reward_calculator
        self.objective = objective
        self.device = device
        self.optimizer = optim.Adam(self.gnn.parameters(), lr=lr)
        self.results_path = results_path

    def get_aps(self,partitions, part_idx):
        def get_aps_utils(u, visited, ap, parent, low, disc, steps, graph):
                visited[u], disc[u], low[u] = True, steps, steps
                children = 0
                for v in graph.edge_index[1, graph.edge_index[0] == u]:
                    if not visited[v] :
                        parent[v], children = u, children + 1
                        get_aps_utils(v, visited, ap, parent, low, disc, steps + 1, graph)
                        low[u] = min(low[u], low[v])
                        if parent[u] == -1 and children > 1:
                            ap[u] = True
                        if parent[u] != -1 and low[v] >= disc[u]:
                            ap[u] = True     
                    elif v != parent[u]: 
                        low[u] = min(low[u], disc[v])
        cur_part = self.comp_graph.subgraph([n for p, n in zip(partitions, self.comp_graph) if p == part_idx])
        if len(cur_part) == 0:
            return []
        graph = nx_to_pyg(cur_part)
        graph.edge_index = torch.cat([graph.edge_index,graph.edge_index.flip((0))], dim=1)
        visited = [False] * graph.num_nodes
        disc = [float("Inf")] * graph.num_nodes
        low = [float("Inf")] * graph.num_nodes
        parent = [-1] * graph.num_nodes
        ap = [False for _ in range(graph.num_nodes)]
        for i in range(graph.num_nodes):
            if not visited[i]:
                get_aps_utils(i, visited, ap, parent, low, disc, 0, graph)
        return [list(self.comp_graph.nodes).index(list(cur_part.nodes)[i]) for i, v in enumerate(ap) if v]

    def train(self, exp_type, comp_graph, epochs):
        comp_graph = nx_to_pyg(comp_graph)
        for e in range(epochs):
            print("Epoch " + str(e))
            if exp_type == 'ablation_initparti':
                # starting_partition = torch.randint(0, self.num_partitions, (comp_graph.num_nodes,)).to(self.device)
                starting_partition = torch.randint(0, self.num_partitions, (comp_graph.num_nodes,)).to(self.device).long
            else:
                # starting_partition = self.starting_partitions.to(self.device)
                starting_partition = self.starting_partitions.to(self.device).clone().long().view(-1)
            g = comp_graph.clone().to(self.device)
            best_part, loss = self.train_graph(g, starting_partition)
            final_partition = best_part.cpu()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            del g.edge_index, g.layer, g.activation, g, loss, best_part
            torch.cuda.empty_cache()
        return final_partition

    def train_graph(self, cur_graph, partitions):
        rewards, partition_probs, cur_obj = [], [], None
        node_embs = self.gnn.get_embedding(cur_graph)

        self.objective.calculate_node_vals(node_embs, cur_graph)
        self.calculate_node_scores(partitions, cur_graph)

        for _ in tqdm(range(cur_graph.num_nodes)):
            partitions, reward, partition_prob, cur_obj = self.train_step(partitions, cur_graph, node_embs, cur_obj) # save objective to save computations.
            rewards.append(reward), partition_probs.append(partition_prob)
        loss = -(torch.log(torch.stack(partition_probs) + 1e-8) * torch.stack(rewards)).mean()
        del node_embs, rewards, partition_probs
        return partitions, loss

    def get_starting_partitions(self):
        temp_graph = self.comp_graph.copy()
        temp_graph.remove_edges_from([e for e in temp_graph.edges() if abs(int(e[0].split(",")[1]) - int(e[1].split(",")[1])) >=2])
        partition, nodes_list, part_count = 0, [], []
        centers = [n for n in temp_graph.nodes if temp_graph.nodes[n]['activation'].shape[0] == 4]
        center_sub = temp_graph.subgraph(centers).copy()
        for comp in nx.weakly_connected_components(center_sub):
            for n in comp:
                temp_graph.nodes[n]["part"] = partition
            nodes_list.append(random.sample(list(comp), 1)[0])
            part_count.append(len(comp))
            partition += 1
        if partition < len(list(nx.weakly_connected_components(temp_graph))):
            for comp in nx.weakly_connected_components(temp_graph):
                sampled_node = random.sample(list(comp), 1)[0]
                temp_graph.nodes[sampled_node]["part"] = partition
                nodes_list.append(sampled_node)
                part_count.append(1)
                partition += 1
        while len(nodes_list) > 0:
            node = nodes_list.pop(0)
            neighbors = list([o[1] for o in temp_graph.out_edges(node)])
            neighbors += list([i[0] for i in temp_graph.in_edges(node)])
            if len(neighbors) > 0:
                possible_neighbors = [n for n in neighbors if "part" not in temp_graph.nodes[n]]
                if len(possible_neighbors) == 0:
                    node_parts = [(n,part_count[temp_graph.nodes[n]['part']]) for n in neighbors]
                    neighbor, neighbor_part_count = min(node_parts, key=lambda x: x[1])
                    if neighbor_part_count > part_count[temp_graph.nodes[node]["part"]]:
                        neighbor, _ = max(node_parts, key=lambda x: x[1])
                    else:
                        continue
                else:
                    neighbor = possible_neighbors.pop(random.sample(range(len(possible_neighbors)), 1)[0])
                    nodes_list.append(neighbor)
                temp_graph.nodes[neighbor]["part"] = temp_graph.nodes[node]["part"]
                part_count[temp_graph.nodes[node]["part"]] += 1
                nodes_list.append(neighbor)
        for n in temp_graph.nodes:
            if 'part' not in temp_graph.nodes[n]:
                neighbors = [o[1] for o in temp_graph.out_edges(n) if "part" in temp_graph.nodes[o[1]]]
                neighbors += [i[0] for i in temp_graph.in_edges(n) if "part" in temp_graph.nodes[i[0]]]
                temp_graph.nodes[n]['part'] = temp_graph.nodes[random.sample(neighbors,1)[0]]['part']
                part_count[temp_graph.nodes[n]['part']] += 1
        partitions = []
        for n in temp_graph.nodes:
            partitions.append(temp_graph.nodes[n].pop("part"))

        print( "size of partitions :" , len(partitions) )
        # return torch.tensor(partitions)
        return torch.tensor(partitions, dtype=torch.long)


    def get_partitions(self, exp_type, comp_graph):
        comp_graph = nx_to_pyg(comp_graph)
        if exp_type == 'ablation_initparti':
            starting_partition = torch.randint(0, self.num_partitions, (comp_graph.num_nodes,)).to(self.device)
        else:
            starting_partition = self.starting_partitions.to(self.device)
        g = comp_graph.clone().to(self.device)
        best_part, loss = self.train_graph(g, starting_partition)
        final_partition = best_part.cpu()
        del g.edge_index, g.layer, g.activation, g, loss, best_part
        return final_partition
    
    def calculate_node_scores(self, partitions, graph):
        '''
        NodeScore[v] = max_p∈P\PART(P,v) |u|u ∈ N(v) ∋ PART(P,u) = p| /
                   |u|u ∈ N(v) ∋ PART(P,u) = PART(P,v)| × 1/degree(v)
        '''
        edge_index = graph.edge_index
        self.degrees = degree(graph.edge_index[0])
        self.scores = torch.zeros(graph.num_nodes).to(edge_index.device)
        for v in range(graph.num_nodes):
            # Get neighbors of node v
            neighbors = edge_index[1, edge_index[0] == v]
            if len(neighbors) == 0:  # Skip isolated nodes
                continue
            neighbors = torch.concat(
                [neighbors, torch.tensor([v]).to(neighbors.device)]
            )
            partition, count = partitions[neighbors].unique(return_counts=True)
            max_other_count = count[partition != partitions[v]]
            max_other_count = max_other_count.max() if len(max_other_count) > 0 else 0
            same_part_count = count[partition == partitions[v]][0]
            # Compute score using the formula
            self.scores[v] = (max_other_count / same_part_count) / self.degrees[v]
            del neighbors, partition, count, max_other_count, same_part_count
    
    def update_step(self, partitions, old_part, new_part, node_idx, graph):
        self.aps[old_part] = self.get_aps(partitions, old_part)
        self.aps[new_part] = self.get_aps(partitions, new_part)
        nodes = torch.concat([graph.edge_index[1, graph.edge_index[0] == node_idx], graph.edge_index[0, graph.edge_index[1] == node_idx]])
        nodes = torch.concat([nodes, torch.tensor([node_idx]).to(partitions.device)])
        for v in nodes:
            neighbors = graph.edge_index[1, graph.edge_index[0] == v]
            if len(neighbors) == 0:  # Skip isolated nodes
                continue
            neighbors = torch.concat([neighbors, torch.tensor([v]).to(partitions.device)])
            partition, count = partitions[neighbors].unique(return_counts=True)
            max_other_count = count[partition != partitions[v]]
            max_other_count = max_other_count.max() if len(max_other_count) > 0 else 0
            same_part_count = count[partition == partitions[v]][0]
            self.scores[v] = (max_other_count / same_part_count) / self.degrees[v]

    def train_step(self, partitions, graph, node_embs, cur_obj):

        # Make sure partitions is 1-D long tensor on same device as graph.edge_index
        partitions = partitions.to(graph.edge_index.device).long().view(-1)
        selected_node = int(torch.multinomial(F.softmax(self.scores, dim=0), 1).detach().cpu().item())

        # selected_node = torch.multinomial(F.softmax(self.scores, dim=0), 1).detach().cpu().item()
        partition_scores = self.gnn.get_partition_scores(
            node_embs, selected_node, partitions
        )
        # print("partition scores: ",partition_scores)
        if partition_scores.dim() == 0:
            partition_scores = partition_scores.view(1)

        partition_probs = F.softmax(partition_scores, dim=0)
        mask = torch.zeros_like(partition_probs)

        #########################################
        def _mask_for_indices(idxs):
            if idxs.numel() == 0:
                return torch.tensor([], dtype=torch.long, device=partitions.device)
            return partitions[idxs].to(partitions.device).long().view(-1)
        ############################################

        # if selected_node in self.aps[partitions[selected_node]]:
        #     mask[partitions[selected_node]] = 1
        # else:
        #     mask[partitions[graph.edge_index[1, graph.edge_index[0] == selected_node]]] = 1
        #     mask[partitions[graph.edge_index[0, graph.edge_index[1] == selected_node]]] = 1


        # is selected_node an articulation point in its partition?
        if selected_node in self.aps[int(partitions[selected_node].item())]:
            mask[int(partitions[selected_node].item())] = 1
        else:
            # neighbors outgoing
            idxs_out = graph.edge_index[1, graph.edge_index[0] == selected_node]
            parts_out = _mask_for_indices(idxs_out).unique()
            # print(parts_out)
            if parts_out.numel() > 0:
                mask[parts_out] = 1

            # neighbors incoming
            idxs_in = graph.edge_index[0, graph.edge_index[1] == selected_node]
            parts_in = _mask_for_indices(idxs_in).unique()
            # print(parts_in)
            if parts_in.numel() > 0:
                mask[parts_in] = 1
        
        partition_probs = partition_probs * mask
        if partition_probs.sum() <= 0:
            new_part = partitions[selected_node]
        else:
            new_part = torch.multinomial(partition_probs, 1).item()
        # Compute reward and update partition
        if partitions[selected_node] != new_part:
            if cur_obj is None:
                cur_obj = self.objective(graph, partitions)
            old_obj, old_part = cur_obj, partitions[selected_node].item()
            partitions[selected_node] = new_part
            cur_obj = self.objective(graph, partitions)
            reward = self.reward_calculator.compute_reward(old_obj[2], cur_obj[2]).detach()
            self.update_step(partitions, old_part, new_part, selected_node, graph)
            del old_obj, old_part
        else:
            reward = torch.tensor(0).to(node_embs.device)
        del partition_scores
        return partitions, reward, partition_probs[new_part], cur_obj