import torch
import networkx as nx
from typing import Dict, Set, Any
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
from ..circuit_discovery import visualize_circuit, visualize_circuit2
import pickle
import re

class ActivationExtractor:
    """Extracts activations from specified computational nodes in a model."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        comp_graph: nx.DiGraph,
        partition_attr: str = "partition",
    ):
        """
        Initialize the activation extractor.

        Args:
            model: The Llama model to analyze
            comp_graph: NetworkX directed graph representing computational dependencies
            partition_attr: Name of the node attribute containing partition information
        """
        self.model = model
        self.comp_graph = comp_graph
        self.partition_attr = partition_attr
        self.activation_cache = {}
        self.hooks = []
        # self._register_hooks() dont use this 

    def _get_module_path(self, node_name: str) -> str:
        """Convert node name to module path."""
        parts = node_name.split(",")
        if len(parts) != 4:
            return ""

        component, layer_idx, idx, component_type = parts
        layer_idx = int(layer_idx)-1

        if component == "self_attn":
            if component_type == "attn_head":
                # For attention heads, we need q_proj, k_proj, v_proj
                return f"model.layers.{layer_idx}.self_attn.o_proj" # we then get its input and reverse it to each attention head, this avoids getting all those q_proj, k_proj, v_proj
            elif component_type == "o_proj":
                return f"model.layers.{layer_idx}.self_attn.o_proj"
        elif component == "mlp":
            if component_type in ["gate_proj", "up_proj", "down_proj"]:
                return f"model.layers.{layer_idx}.mlp.{component_type}"
            elif component_type == "mlp_in":
                return [f"model.layers.{layer_idx}.mlp.gate_proj", f"model.layers.{layer_idx}.mlp.up_proj"]

        return ""

    def _activation_hook(self, name, head_nums):
        """Create a hook function to cache activations for a specific computation node."""
        # def hook(module, input, output): # the output are norms where each token corresponds to a number
        #     if 'attn_head' in name:
        #         head_idx = int(name.split(",")[2])
        #         output_shape = input[0].shape
        #         # print('output_shape', output_shape)
        #         output = input[0].reshape(output_shape[0], output_shape[1], head_nums, -1)[:,:,head_idx, :].detach()
        #         output_norm = torch.norm(output, dim=2)/output.shape[-1]
        #         self.activation_cache[name] = {
        #             "input": [],
        #             "output": output_norm # the following o_proj's input is the output of the previous layer's self_attn
        #         }
        #     else:
        #         if name in self.activation_cache:
        #             # print(self.activation_cache[name])
        #             output_norm = (self.activation_cache[name]['output'] + torch.norm(output, dim=2).detach()/output.shape[-1])/2
        #             self.activation_cache[name] = {
        #                 "input": [x.detach() if isinstance(x, torch.Tensor) else x for x in input],
        #                 "output": output_norm,
        #             }
        #         else:
        #             if name in self.activation_cache:
        #                 # print(self.activation_cache[name])
        #                 output_norm = (self.activation_cache[name]['output'] + torch.norm(output, dim=2).detach()/output.shape[-1])/2
        #                 self.activation_cache[name] = {
        #                     "input": [x.detach() if isinstance(x, torch.Tensor) else x for x in input],
        #                     "output": output_norm,
        #                 }
        #             else:
        #                 output_norm = torch.norm(output, dim=2).detach()/output.shape[-1]
        #                 self.activation_cache[name] = {
        #                     "input": [x.detach() if isinstance(x, torch.Tensor) else x for x in input],
        #                     "output": output_norm,
        #                 }
        # return hook
        # new updated hook
        def hook(module, input, output):
            # This logic is now safe and stateless. It just captures the activation.
            if 'attn_head' in name:
                head_idx = int(name.split(",")[2])
                output_shape = input[0].shape
                # Detach the tensor to prevent holding onto the computation graph
                head_output = input[0].reshape(output_shape[0], output_shape[1], head_nums, -1)[:,:,head_idx, :].detach()
                output_norm = torch.norm(head_output, dim=2) / head_output.shape[-1]
            else:
                # For all other layers, just get the norm of the output tensor
                output_norm = torch.norm(output, dim=2).detach() / output.shape[-1]

            # Overwrite the cache. Don't try to average or accumulate.
            self.activation_cache[name] = {"output": output_norm}
        return hook

    def _register_hooks(self):
        """Register hooks for all attention heads and feedforward layers."""
        self.hooks.clear()
        for node_name in self.comp_graph.nodes():
            module_path = self._get_module_path(node_name)
            if module_path:
                try:
                    if type(module_path) is list:
                        for path in module_path:
                            module = self.model.get_submodule(path)
                            hook = module.register_forward_hook(self._activation_hook(node_name, None))
                            self.hooks.append(hook)
                    else:
                        module = self.model.get_submodule(module_path)
                        if 'attn_head' in node_name:
                            head_nums = self.model.config.num_attention_heads
                            hook = module.register_forward_hook(self._activation_hook(node_name, head_nums))
                        else:
                            hook = module.register_forward_hook(self._activation_hook(node_name, None))
                        self.hooks.append(hook)
                except Exception as e:
                    print(f"Failed to register hook for {node_name}: {str(e)}")

    def extract_activations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Runs a forward pass and returns a dict mapping node_name -> metadata dict:
        {
           'node_name': str,
           'module_path': str,
           'layer': int,
           'component': str,
           'component_type': str,
           'idx': int,
           'activation': Tensor or None,
           'activation_norms': Tensor or None,
           'shape': tuple
        }
        """
        # clear old cache and register hooks
        self.activation_cache.clear()
        self._register_hooks()

        try:
            with torch.no_grad():
                # do forward pass
                self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )

            # build output structure
            result = {}
            for node_name, cache in self.activation_cache.items():
                meta = cache.get('meta', {})
                out = cache.get('output', None)
                out_norm = cache.get('output_norm', None)
                shape = tuple(out.shape) if isinstance(out, torch.Tensor) else None

                node_info = {
                    'node_name': node_name,
                    'module_path': meta.get('module_path'),
                    'component': meta.get('component'),
                    'component_type': meta.get('component_type'),
                    'layer': meta.get('layer'),
                    'idx': meta.get('idx'),
                    'activation': out,                 # CPU tensor or None
                    'activation_norms': out_norm,     # CPU tensor or None (e.g. [batch, seq_len])
                    'shape': shape,
                }
                result[node_name] = node_info
        finally:
            # always remove hooks
            self.remove_hooks()

        return result

    def get_partition_activations(self, partition_idx: int, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Convenience: get activations+meta only for nodes belonging to a partition.
        """
        all_acts = self.extract_activations(input_ids, attention_mask)
        partition_nodes = {n for n, d in self.comp_graph.nodes(data=True) if d.get(self.partition_attr) == partition_idx}
        return {n: v for n, v in all_acts.items() if n in partition_nodes}  
    
    def get_circuit_output_nodes(self) -> Dict[int, Set[str]]:
        """
        Returns output nodes for each modular circuit using graph structure.
        Output nodes are those that have edges to nodes outside their partition
        or have no outgoing edges.
        """
        circuit_outputs = {}

        # Get all unique partition indices (excluding 0)
        partitions = set([int(val.item()) for val in nx.get_node_attributes(self.comp_graph, self.partition_attr).values()])

        for partition_idx in partitions:
            circuit_outputs[partition_idx] = set()

            # Get all nodes in this partition
            partition_nodes = {
                node for node, data in self.comp_graph.nodes(data=True)
                if data.get(self.partition_attr) == partition_idx
            }

            # For each node in the partition
            for node in partition_nodes:
                successors = set(self.comp_graph.successors(node))

                # Node is an output if either:
                # 1. It has no successors (final output)
                # 2. All its successors are outside the partition or in partition 0
                is_output = (
                    len(successors) == 0 or
                    all(
                        self.comp_graph.nodes[succ].get(self.partition_attr, 0) != partition_idx
                        for succ in successors
                    )
                )

                if is_output:
                    circuit_outputs[partition_idx].add(node)

        # circuit_outputs = dict(list(circuit_outputs.items())[:5])

        print("Number of circuits : ",len(circuit_outputs))
        for partition_idx, outputs in circuit_outputs.items():
            print(partition_idx)
            if not outputs:   # skip partitions that are not real circuits
                continue

            # Nodes before cutting = all nodes in this partition
            partition_nodes = {
                node for node, data in self.comp_graph.nodes(data=True)
                if data.get(self.partition_attr) == partition_idx
            }

            # --- Before cutting ---
            subgraph_before = self.comp_graph.subgraph(partition_nodes).copy()
            with open(f"./results/circuits_before_new/circuit_{partition_idx}.pkl", "wb") as f:
                pickle.dump(subgraph_before, f)
            # visualize_circuit2(subgraph_before, out_file=f"./results/before_cutting_new/circuit_{partition_idx}.png",max_layers=self.model.config.num_hidden_layers,)

            print(f"Subgraph BEFORE partition {partition_idx}:")
            print("Nodes:", len(list(subgraph_before.nodes)))
            print("Edges:", len(list(subgraph_before.edges)))

            #--- After cutting ---  
            subgraph_after = self.comp_graph.subgraph(outputs).copy()
            with open(f"./results/circuits_after_new/circuit_{partition_idx}.pkl", "wb") as f:
                pickle.dump(subgraph_after, f)
            # visualize_circuit2(subgraph_after, out_file=f"./results/after_cutting_new/circuit_{partition_idx}.png",max_layers=self.model.config.num_hidden_layers,)
            # Print nodes and edges
            print(f"Subgraph AFTER partition {partition_idx}:")
            print("Nodes:", len(list(subgraph_after.nodes)))
            print("Edges:", len(list(subgraph_after.edges)))

        return circuit_outputs

    def _process_attention_head_activation(self, layer_idx: int, head_idx: int) -> torch.Tensor:
        """Process activation for a specific attention head by aggregating q, k, v projections. OUTPUT NORM!"""
        attn_name = f"self_attn,{layer_idx},{head_idx},attn_head"
        attn_cache = self.activation_cache.get(attn_name, {})

        if not attn_cache:
            return None
        else:
            return attn_cache["output"].cpu()

    def extract_activations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations for a single forward pass. This method now manages
        the registration and removal of hooks to ensure the model is left clean.
        """
        self.activation_cache.clear()
        self._register_hooks() # Register hooks right before they are needed

        processed_act_norms = {}

        try:
            # Run the forward pass with hooks active
            with torch.no_grad():
                self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )

            # Process the cached activations
            for node_name in self.comp_graph.nodes():
                cache = self.activation_cache.get(node_name)
                if cache:
                    processed_act_norms[node_name] = cache["output"].cpu()
        finally:
            # CRITICAL: Always remove the hooks, even if an error occurs
            self.remove_hooks()

        return processed_act_norms

    def cleanup(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activation_cache = {}
        self.activation_cache.clear()  # Use clear() instead of reassignment
        torch.cuda.empty_cache()

    def remove_hooks(self): 
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []