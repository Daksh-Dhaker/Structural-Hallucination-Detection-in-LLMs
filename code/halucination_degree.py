import json
from sentence_transformers import SentenceTransformer, util
import torch
import csv # <--- 1. Import the csv library

def load_json_file(file_path):
    """Loads a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return None

def load_text_file(file_path):
    """Loads a text file with one item per line."""
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []

# --- 2. Create a new function to save the results ---
def save_results_to_csv(results, filename):
    """Saves the circuit hallucination scores to a CSV file."""
    try:
        with open(filename, 'w', newline='') as csvfile:
            # Create a CSV writer object
            writer = csv.writer(csvfile)
            # Write the header row
            writer.writerow(['circuit_id', 'hallucination_score'])
            # Write the data from the results dictionary
            for circuit_id, score in results.items():
                writer.writerow([circuit_id, score])
        print(f"\nResults successfully saved to {filename}")
    except IOError:
        print(f"Error: Could not write to the file {filename}.")


def calculate_degree_of_hallucination(interpretations_file, hallucinated_tokens_file):
    """
    Calculates the degree of hallucination for each circuit based on semantic similarity.

    Args:
        interpretations_file (str): The path to the JSON file with functional interpretations.
        hallucinated_tokens_file (str): The path to the text file with hallucinated tokens.

    Returns:
        dict: A dictionary with circuit IDs as keys and their degree of hallucination as values.
    """
    # Load a pre-trained model for generating embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load the input files
    interpretations = load_json_file(interpretations_file)
    hallucinated_tokens = load_text_file(hallucinated_tokens_file)

    if interpretations is None or not hallucinated_tokens:
        return {}

    # Separate circuit IDs and their descriptions
    circuit_ids = list(interpretations.keys())
    interpretation_texts = list(interpretations.values())

    # Generate embeddings for the interpretations and hallucinated tokens
    print("Generating embeddings for interpretations and tokens...")
    interpretation_embeddings = model.encode(interpretation_texts, convert_to_tensor=True)
    token_embeddings = model.encode(hallucinated_tokens, convert_to_tensor=True)

    # Calculate the cosine similarity between each interpretation and all hallucinated tokens
    cosine_scores = util.cos_sim(interpretation_embeddings, token_embeddings)

    # Calculate the degree of hallucination for each circuit
    hallucination_degrees = {}
    for i in range(len(circuit_ids)):
        # The degree of hallucination is the average similarity to the hallucinated tokens
        avg_similarity = torch.mean(cosine_scores[i]).item()
        hallucination_degrees[circuit_ids[i]] = avg_similarity

    return hallucination_degrees

if __name__ == '__main__':
    # Define the paths to your input files
    interpretations_json_file = 'exp_0_func_interp_x=128_hdim=64_edim=32_mp=5_a=0.5_b=10_g=5_inf=1000000_ep=1_nl=3_lr=0.001_tk=10.json'
    hallucinated_tokens_txt_file = 'hallucinated_tokens.txt'

    # Calculate the degree of hallucination
    results = calculate_degree_of_hallucination(interpretations_json_file, hallucinated_tokens_txt_file)

    if results:
        # --- 3. Call the new function to save the results ---
        output_csv_filename = 'hallucination_scores.csv'
        save_results_to_csv(results, output_csv_filename)

        # The rest of the script prints the results to the console as before
        print("\n--- Degree of Hallucination for each Circuit ---")
        # Sort the results by the degree of hallucination in descending order
        sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
        for circuit_id, degree in sorted_results:
            print(f"{circuit_id}: {degree:.4f}")