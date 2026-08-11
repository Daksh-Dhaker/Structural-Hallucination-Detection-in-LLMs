import torch
import networkx as nx
from typing import Dict, Set
from transformers import AutoModelForCausalLM
import matplotlib.pyplot as plt
from ..circuit_discovery import visualize_circuit, visualize_circuit2
import pickle


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

        # print("Leeeen2: ",len(circuit_outputs))
        # for partition_idx, outputs in circuit_outputs.items():
        #     print(partition_idx)
        #     if not outputs:   # skip partitions that are not real circuits
        #         continue

        #     # Nodes before cutting = all nodes in this partition
        #     partition_nodes = {
        #         node for node, data in self.comp_graph.nodes(data=True)
        #         if data.get(self.partition_attr) == partition_idx
        #     }

        #     # --- Before cutting ---
        #     subgraph_before = self.comp_graph.subgraph(partition_nodes).copy()
            # with open(f"./results/circuits_before_new/circuit_{partition_idx}.pkl", "wb") as f:
            #     pickle.dump(subgraph_before, f)
            # visualize_circuit2(subgraph_before, out_file=f"./results/before_cutting_new/circuit_{partition_idx}.png",max_layers=self.model.config.num_hidden_layers,)

            # --- After cutting ---
            # subgraph_after = self.comp_graph.subgraph(outputs).copy()
            # with open(f"./results/circuits_after_new/circuit_{partition_idx}.pkl", "wb") as f:
            #     pickle.dump(subgraph_after, f)
            # visualize_circuit2(subgraph_after, out_file=f"./results/after_cutting_new/circuit_{partition_idx}.png",max_layers=self.model.config.num_hidden_layers,)

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