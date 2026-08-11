import argparse
from datetime import datetime
import json
import os
import random
import time
import numpy as np
import google.generativeai as genai
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

import transformers
from models.neurocut.objectives import Objectives
from models.neurocut.reward import RewardCalculator
from models.neurocut.trainer import NeuroCUTTrainer
from models.circuit_discovery import get_random_circuit
from models.neurocut.gnn import GNN
from transformers import AutoTokenizer, AutoModelForCausalLM
from models.func_interp import ActivationExtractor, ImportanceAnalyzer
from models.circuit_discovery import (
    combine_circuits,
    find_node_importance,
    find_sig_nodes,
    get_activation_partitions,
    get_kmeans_partitions,
    get_neurons_and_heads,
    get_random_partitions,
    load_partitions,
)
from utils import get_dataset, get_save_paths
from evaluation import MCevaluator
import networkx as nx


from extract_all_node_features import extract_all_node_features



MODEL_MAP = {"medllama": "ProbeMedicalYonseiMAILab/medllama3-v20", "codellama":"meta-llama/CodeLlama-13b-Instruct-hf","llama2":"meta-llama/Meta-Llama-3-8B-Instruct"}
# DATASET_LIST = ["symptom2disease", "medal", "medmcqa", "medicalabstract"]
DATASET_LIST = ["hallucination_data"]
# EVAL_DATASET_LIST = ["medi_attr", "medi_status", "coreference", "pubmed_summ"]
EVAL_DATASET_LIST = ["hallucination_data"]

# HALoGen Dataset
# DATASET_LIST = ["numerical"]
# EVAL_DATASET_LIST = ["numerical_eval"]
ACTIVATION_MAP = {
    ("down_proj", "o_proj"): 1,
    ("attn_head",): 128 * 3,
    ("mlp_in",): 1 * 2,
}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    transformers.set_seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modular Circuit Vocabulary Discovery")
    parser.add_argument("--save_path", "-sp", type=str, default="./results", help="Directory where experiment outputs will be saved")
    parser.add_argument("--device", type=int, default=0, help="cuda device")
    parser.add_argument(
        "--num_exp", "-ne", type=int, default=5, help="number of experiments"
    )
    parser.add_argument("--epochs", "-e", type=int, default=10, help="number of epochs")
    #parser.add_argument("--model", "-m", type=str, default="llama2", choices=["medllama", "codellama","llama2"],  help="LLM model")

    parser.add_argument("--model", "-m", type=str, default="llama2", choices=MODEL_MAP.keys(),  help="LLM model")
    parser.add_argument("--seed", type=int, default=69, help="random seed")
    parser.add_argument(
        "--topk_nodes",
        "-tk",
        type=int,
        default=10,
        help="number of nodes to select when creating end-to-end circuit",
    )
    parser.add_argument(
        "--x_dim", "-xd", type=int, default=128, help="reduced dimension of activations"
    )
    parser.add_argument(
        "--hidden_dim", "-hd", type=int, default=64, help="hidden dimension for GNN"
    )
    parser.add_argument(
        "--emb_dim", "-ed", type=int, default=32, help="embedding dimension for GNN"
    )
    parser.add_argument(
        "--num_layers", "-nl", type=int, default=3, help="number of layers in GNN"
    )
    parser.add_argument(
        "--lr", "-lr", type=float, default=0.001, help="learning rate for trainning GNN"
    )
    parser.add_argument(
        "--alpha", "-a", type=float, default=0.5, help="composability coefficient"
    )
    parser.add_argument(
        "--beta", "-b", type=int, default=10, help="total unique circuits limit"
    )
    parser.add_argument(
        "--gamma", "-g", type=int, default=5, help="reduced dimension of activations"
    )
    parser.add_argument(
        "--infty_val",
        "-inf",
        type=float,
        default=1000000,
        help="infinity value for penalty",
    )

    parser.add_argument(
        "--max_partitions",
        "-mp",
        type=int,
        default=5,
        help="max number of partitions for NeuroCut",
    )
    parser.add_argument(
        "--num_graphs",
        "-ng",
        type=int,
        default=55,
        help="number of graphs for baselines",
    )
    parser.add_argument(
        "--exp_type",
        "-et",
        type=str,
        default="modcirc",
        choices=[
            "modcirc",
            "modcirc_param",
            "random",
            "freq_subgraph",
            "activation",
            "kmeans",
            "ablation_initparti",
            "ablation_neuroncut",
            "ablation_circ"
        ],
        help="experiment type",
    )

    
    return parser.parse_args()

def get_circuits(seed: int, device: str, topk_nodes: int, exp_type: str, nodes_path: str, dataset_map: dict, model, tokenizer):
    """
    Get circuits based on experiment type - either random computational subgraph or importance-based circuits.

    Args:
        exp_type: Experiment type ('ablation_circ' for random subgraph, otherwise importance-based)
        nodes_path: Path to save/load node information
        dataset_map: Dictionary mapping dataset names to datasets
        model: The neural network model
        tokenizer: The tokenizer for the model

    Returns:
        dict: Dictionary mapping dataset names to their circuits
    """
    # Load or compute node importance
    if not os.path.exists(nodes_path):
        set_seed(seed)
        print("Finding node importance")
        data_nodes = {}
        for dataset_name in dataset_map:
            data_nodes[dataset_name] = find_node_importance(
                tokenizer, model, dataset_map[dataset_name], device
            )
        print("saving...")
        os.makedirs(os.path.dirname(nodes_path), exist_ok=True)
        torch.save(data_nodes, nodes_path)
    
    print("Reached")
    data_nodes = torch.load(nodes_path)
    set_seed(seed)
    print("Getting circuits")

    circuits = {}
    for data in tqdm(data_nodes):
        nodes = get_neurons_and_heads(data_nodes[data])
        if exp_type == 'ablation_circ':
            # Random computational subgraph with 1280 nodes
            circuits[data] = get_random_circuit(
                nodes,
                topk_nodes,
                32
            )
        else:
            # Original importance-based circuit finding
            circuits[data] = find_sig_nodes(
                nodes,
                topk_nodes,
                32
            )
    print("circuilts length", len(circuits))
    return circuits

def main(args: argparse.Namespace):
    # Configure Google Generative AI API key
    # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    # Instantiate chat model (e.g., Gemini 1.5 Flash)
    # gen_model = genai.GenerativeModel('gemini-1.5-flash')

    tokenizer, model = (
        AutoTokenizer.from_pretrained(args.model_name),
        AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.float16 if args.device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
            # device_map="auto",
        ).to(args.device),
    )
    dataset_map = {name: get_dataset(name, tokenizer) for name in DATASET_LIST}
    eval_data_map = {name: get_dataset(name, tokenizer) for name in EVAL_DATASET_LIST}

    # Print lengths of each dataset in dataset_map
    for name, ds in dataset_map.items():
        try:
            print(f"{name}: {len(ds)} samples")
        except TypeError:
            print(f"{name}: length not available")

    for i in range(args.num_exp):
        start = time.time()
        (
            nodes_path,
            eval_nodes_path,
            circuit_dataset,
            model_path,
            result_path,
            imp_score_path,
            func_interp_path,
            partitions_path,
        ) = get_save_paths(args, i)
        with open(result_path, "a+") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S")+"\n")
        print("\n","starting to get circuits")
        circuits = get_circuits(args.seed, args.device, args.topk_nodes, args.exp_type, nodes_path, dataset_map, model, tokenizer)
        print("\n","Got them circuits")
        with open(result_path, "a+") as f:
            f.write(f"Time taken to get circuits: {time.time() - start}s\n")
        start = time.time()
        print("\n","starting to combine circuits")
        comp_graph = combine_circuits(circuits, model.config.num_hidden_layers)
        print("\n","combined circuits")
        with open(result_path, "a+") as f:
            f.write(f"Time taken to combine circuits: {time.time() - start}s\n")
        start = time.time()
        print()
        print("partitions_path: ",partitions_path)
        print()

        if not os.path.exists(partitions_path):
            print("not already have partitions : ")
            print()

            if "modcirc" in args.exp_type or "ablation" in args.exp_type:
                print("1")
                for i, n in enumerate(comp_graph.nodes):
                    print(n, comp_graph.nodes[n]["activation"].shape)
                    if i > 10:
                        break

                comp_graph = comp_graph.subgraph(
                    [
                        n
                        for n in comp_graph
                        if comp_graph.nodes[n]["activation"].shape[0] >= 2
                    ]
                ).copy()

                comp_graph.remove_nodes_from(list(nx.isolates(comp_graph)))

                if comp_graph.number_of_nodes() == 0 and comp_graph.number_of_edges() == 0:
                    print("Combined graph is empty")
                else:
                    print(f"Combined graph has {comp_graph.number_of_nodes()} nodes and {comp_graph.number_of_edges()} edges")
    
                print("after3")

                opt_partitions = train_neuroncut(args, comp_graph, result_path)
                nx.set_node_attributes(
                    comp_graph,
                    {n: p for n, p in zip(comp_graph.nodes, opt_partitions)},
                    "partition",
                )
            else:
                if args.exp_type == "random":
                    print("2")
                    comp_graph = comp_graph.subgraph(
                        [
                            n
                            for n in comp_graph
                            if comp_graph.nodes[n]["activation"].shape[0] == 1
                        ]
                    ).copy()
                    comp_graph.remove_nodes_from(list(nx.isolates(comp_graph)))
                    comp_graph = get_random_partitions(comp_graph, args.num_graphs)
                elif args.exp_type == "freq_subgraph":
                    print("3")
                    comp_graph = comp_graph.subgraph(
                        [
                            n
                            for n in comp_graph
                            if len(comp_graph.nodes[n]["activation"].shape) == 3
                        ]
                    ).copy()
                    comp_graph.remove_nodes_from(list(nx.isolates(comp_graph)))
                    comp_graph = get_random_partitions(comp_graph, args.num_graphs)
                elif args.exp_type == "kmeans":
                    print("4", "num_graphs: ",args.num_graphs)
                    comp_graph = get_kmeans_partitions(comp_graph, args.num_graphs)
                elif args.exp_type == "activation":
                    print("5", "num_graphs: ",args.num_graphs)
                    comp_graph = get_activation_partitions(comp_graph, args.num_graphs)
            print("6")
            torch.save(nx.get_node_attributes(comp_graph, "partition"), partitions_path)
            print("7")


        with open(result_path, "a+") as f:
            f.write(f"Time taken to partition circuits: {time.time() - start}s\n")
        start = time.time()
        comp_graph = load_partitions(
            torch.load(partitions_path), model.config.num_hidden_layers
        )
        activation_extractor = ActivationExtractor(model, comp_graph)
        circuit_outputs = activation_extractor.get_circuit_output_nodes()


        print("len: ",len(circuit_outputs))


    
        

        return 
    


        analyzer = ImportanceAnalyzer(tokenizer, activation_extractor, circuit_outputs)
        if not os.path.exists(imp_score_path):
            set_seed(args.seed)
            input_texts = [s["clean_text"] for d in dataset_map for s in dataset_map[d]]
            analyzer.compute_importance_scores(input_texts)
            analyzer.save_importance_db(imp_score_path)
        if not os.path.exists(func_interp_path):
            set_seed(args.seed)
            analyzer.load_importance_db(imp_score_path)
            print( "starting interp_from_imporatance ")
            analyzer.interp_from_importance(
                imp_score_path.replace("_imp_scores_", "_func_interp_"), model, args.device
            )

        
        activation_extractor.cleanup()
        func_interp = {
            int(k): v for k, v in json.load(open(func_interp_path, "r")).items()
        }
        with open(result_path, "a+") as f:
            f.write(f"Time taken to get functional interpretation: {time.time() - start}s\n")
        start = time.time()
        eval_circuits = get_circuits(
            args.seed, args.device, args.topk_nodes, "", eval_nodes_path, eval_data_map, model, tokenizer
        )
        with open(result_path, "a+") as f:
            f.write(f"Time taken to get evaluation circuits: {time.time() - start}s\n")
        start = time.time()
        dataset_list = list(eval_data_map.values())
        mc_evaluator = MCevaluator(
            result_path,
            tokenizer,
            model,
            eval_circuits,
            comp_graph,
            func_interp,
            dataset_list,
        )
        consistenty, reusability, composability = (
            mc_evaluator.evaluate_mod_circ(model, args.device)
        )  # circuit_outpus is a dictionary, where key is mc idx. the value is nodes that are in the mc.
        with open(result_path, "a+") as f:
            f.write(f"Time taken to evaluate circuits: {time.time() - start}s\n")
        start = time.time()
        # save the results
        with open(result_path, "a+") as f:
            f.write(f"Consistency: {consistenty} | Reusability: {reusability} | Composability: {composability}\n")

def train_neuroncut(args, comp_graph, result_path):
    gnn = GNN(
        node_activation_dims=ACTIVATION_MAP.values(),
        input_dim=args.x_dim,
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        num_layers=args.num_layers,
    )
    trainer = NeuroCUTTrainer(
        gnn=gnn,
        reward_calculator=RewardCalculator(),
        objective=Objectives(),
        device=args.device,
        results_path=result_path,
        comp_graph=comp_graph,
        lr=args.lr
    )
    if args.exp_type != "ablation_neuroncut":
        print("Train NeuroCut")
        trainer.train(args.exp_type, comp_graph, epochs=args.epochs)
    optimized_partitions = trainer.get_partitions(args.exp_type, comp_graph)
    del trainer, gnn
    torch.cuda.empty_cache()
    return optimized_partitions


if __name__ == "__main__":
    __spec__ = None
    args = parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    # args.device = "cpu"
    args.model_name = MODEL_MAP[args.model]
    main(args)
