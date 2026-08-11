# test_func_interp.py

import torch
import networkx as nx
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
from activation_extractor import ActivationExtractor
from importance_analyzer import ImportanceAnalyzer
from typing import Dict, List
import random
import math


def create_sample_graph() -> nx.DiGraph:
    """Create a sample computational graph for testing."""
    G = nx.DiGraph()

    # Add attention components for layer 0
    # First attention head circuit (partition 1)
    G.add_node("model.layers.0.self_attn.q_proj", partition=1)
    G.add_node("model.layers.0.self_attn.k_proj", partition=1)
    G.add_node("model.layers.0.self_attn.v_proj", partition=1)
    G.add_node("self_attn, 0, 0, attn_head", partition=1)
    G.add_node("self_attn, 0, 0, o_proj", partition=1)

    # Second attention head circuit (partition 2)
    G.add_node("self_attn, 0, 1, attn_head", partition=2)
    G.add_node("self_attn, 0, 1, o_proj", partition=2)

    # MLP components
    G.add_node("mlp, 0, 0, gate_proj", partition=1)
    G.add_node("mlp, 0, 0, up_proj", partition=1)
    G.add_node("mlp, 0, 1, down_proj", partition=2)

    # Add edges for first circuit (partition 1)
    G.add_edge("model.layers.0.self_attn.q_proj", "self_attn, 0, 0, attn_head")
    G.add_edge("model.layers.0.self_attn.k_proj", "self_attn, 0, 0, attn_head")
    G.add_edge("model.layers.0.self_attn.v_proj", "self_attn, 0, 0, attn_head")
    G.add_edge("self_attn, 0, 0, attn_head", "self_attn, 0, 0, o_proj")
    G.add_edge("self_attn, 0, 0, o_proj", "mlp, 0, 0, gate_proj")
    G.add_edge("mlp, 0, 0, gate_proj", "mlp, 0, 0, up_proj")

    # Add edges for second circuit (partition 2)
    G.add_edge("model.layers.0.self_attn.q_proj", "self_attn, 0, 1, attn_head")
    G.add_edge("model.layers.0.self_attn.k_proj", "self_attn, 0, 1, attn_head")
    G.add_edge("model.layers.0.self_attn.v_proj", "self_attn, 0, 1, attn_head")
    G.add_edge("self_attn, 0, 1, attn_head", "self_attn, 0, 1, o_proj")
    G.add_edge("self_attn, 0, 1, o_proj", "mlp, 0, 1, down_proj")

    return G

def get_sample_medical_questions(tokenizer) -> List[str]:
    """Generate some sample medical questions for testing."""
    prompt_list = []
    corpus = [
        "What are the symptoms of type 2 diabetes?",
        "How is pneumonia diagnosed?",
        "What are common treatments for hypertension?",
        "What are the risk factors for heart disease?",
        "How is rheumatoid arthritis different from osteoarthritis?"
    ]
    for sentence in corpus:
        prompt = tokenizer.apply_chat_template(
                [
                    {
                        "role": "system",
                        # "content": "You are a medical assistant. Choose the best answer",
                        "content": "You are a medical assistant.",
                    },
                    {"role": "user", "content": sentence},
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
        prompt_list.append(prompt)
    # tokens_nums = tokenizer(prompt, return_tensors="pt")
    return prompt_list

def main(args):
    # Set device
    device = args.device if torch.cuda.is_available() else "cpu"

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model_path = "ProbeMedicalYonseiMAILab/medllama3-v20"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)

    # Create sample computational graph
    print("Creating sample computational graph...")
    # comp_graph = torch.load("ModCirc/comp_graph.pt")

    try:
        # Initialize activation extractor
        print("Initializing activation extractor...")
        # activation_extractor = ActivationExtractor(model, comp_graph)
        activation_extractor = ActivationExtractor(model, nx.DiGraph())

        # Get circuit output nodes
        # print("Getting circuit output nodes...")
        # circuit_outputs = activation_extractor.get_circuit_output_nodes()
        # print("\nCircuit output nodes:")
        # for circuit_idx, nodes in circuit_outputs.items():
        #     print(f"Circuit {circuit_idx}: {nodes}")

        # Initialize importance analyzer
        print("\nInitializing importance analyzer...")
        # analyzer = ImportanceAnalyzer(tokenizer, activation_extractor, circuit_outputs)
        analyzer = ImportanceAnalyzer(tokenizer, activation_extractor, {})

        # Get sample questions
        # input_texts = get_sample_medical_questions(tokenizer)
        # print("\nSample medical questions:")
        # for text in input_texts:
        #     print(f"- {text}")

        # Compute importance scores with smaller batch size for testing
        # print("\nComputing importance scores...")
        # analyzer.compute_importance_scores(input_texts, batch_size=2)  # Reduced batch size for testing

        # Save results
        # output_path = "test_importance_scores.json"
        # print(f"\nSaving results to {output_path}...")
        # analyzer.save_importance_db(output_path)
        output_path = "ModCirc/saved_results/random/exp_0_imp_scores_num-graphs=55.json"
        
        # Load and verify results
        print("\nLoading and verifying results...")
        analyzer.load_importance_db(output_path)

        # Print sample results
        print("\nSample importance scores:")
        for circuit_idx, text_scores in list(analyzer.importance_db.items())[:2]:
            print(f"\nCircuit {circuit_idx}:")
            sample_text = list(text_scores.keys())[0]
            importance_vector = text_scores[sample_text]
            print(f"Text: {sample_text}")
            print(f"Importance vector (first 5 tokens): {importance_vector[:5]}")

        # put the saved results into LLM to generate modular circuits' functional interpretation
        analyzer.interp_from_importance(output_path.replace("_imp_scores_", "_func_interp_"))

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

    finally:
        # Cleanup
        print("\nCleaning up...")
        if 'activation_extractor' in locals():
            activation_extractor.cleanup()

    print("Test completed successfully!")

if __name__ == "__main__":
    # Example usage
    # model_path = "ProbeMedicalYonseiMAILab/medllama3-v20"
    # tokenizer, model = (
    #     AutoTokenizer.from_pretrained(model_path),
    #     AutoModelForCausalLM.from_pretrained(
    #         model_path,
    #         torch_dtype=torch.float32,
    #         low_cpu_mem_usage=True,
    #     ).to('cuda:0'),
    # )
    # sentence = "The quick brown fox jumps"
    # mapping = map_words_to_tokens(tokenizer, sentence)

    # Print the mapping
    # for word, tokens in mapping.items():
    #     print(f"Word: {word} -> Tokens: {tokens}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the model on")
    args = parser.parse_args()
    args.device = f"cuda:{args.device}"
    main(args)