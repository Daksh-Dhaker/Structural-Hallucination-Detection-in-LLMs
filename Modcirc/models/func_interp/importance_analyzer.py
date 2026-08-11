import os
import sys
import json
import random
from typing import Dict, List, Set, Union
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
import torch
import traceback
#import google.generativeai as genai

MODEL_MAP = {"medllama": "ProbeMedicalYonseiMAILab/medllama3-v20", "codellama":"meta-llama/CodeLlama-13b-Instruct-hf","llama2":"meta-llama/Meta-Llama-3-8B-Instruct"}

# This conditional import seems specific to your project structure.
# It's kept as is.
if "models/func_interp" in sys.path[0]:
    from activation_extractor import ActivationExtractor
else:
    from models.func_interp.activation_extractor import ActivationExtractor


def map_words_to_tokens(tokenizer: AutoTokenizer, sentence: str, token_ids: torch.Tensor) -> Dict[str, List[int]]:
    """
    Maps words in a sentence to their corresponding token indices.
    This function's behavior is highly dependent on the specific tokenizer used.

    Args:
        tokenizer: The Hugging Face tokenizer instance.
        sentence: The input sentence string.
        token_ids: The tensor of token IDs for the sentence.

    Returns:
        A dictionary mapping each word to a list of its token indices.
    """
    tokens = [tokenizer.decode([token_id]) for token_id in token_ids.tolist()]
    # Ensure special tokens are treated as separate words
    sentence = sentence.replace("<|eot_id|>", " <|eot_id|>")
    words = sentence.split()
    word_to_tokens: Dict[str, List[int]] = defaultdict(list)
    current_token_idx = 0

    for word in words:
        current_word_tokens: List[str] = []
        current_idx_tokens: List[int] = []
        while current_token_idx < len(tokens):
            current_token = tokens[current_token_idx]
            current_idx_tokens.append(current_token_idx)
            current_word_tokens.append(current_token)
            current_token_idx += 1
            # Check if the concatenated tokens form the current word
            if word in ''.join(t.replace(' ', '') for t in current_word_tokens):
                break
        word_to_tokens[word].extend(current_idx_tokens)
    return dict(word_to_tokens)

class ImportanceAnalyzer:
    """Analyzes the importance of modular circuits for input sequences."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        activation_extractor: ActivationExtractor,
        circuit_output_nodes: Dict[int, Set[str]],
    ):
        self.tokenizer = tokenizer
        self.activation_extractor = activation_extractor
        self.circuit_output_nodes = circuit_output_nodes
        self.importance_db: Dict[int, Dict[str, List[float]]] = defaultdict(dict)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def compute_importance_scores(
        self,
        input_data: List[str],
        batch_size: int = 2,
    ):
        """
        Compute importance scores for all circuits on all input sequences.

        Args:
            input_data: List of input text sequences.
            batch_size: Batch size for processing.
        """
        for i in tqdm(range(0, len(input_data), batch_size), desc="Computing Importance Scores"):
            batch_texts = input_data[i:i + batch_size]

            encoded = self.tokenizer(
                batch_texts, padding=True, return_tensors="pt"
            ).to(self.activation_extractor.model.device)

            activations = self.activation_extractor.extract_activations(
                encoded.input_ids, encoded.attention_mask
            )
            # === NEW: Generate responses for each input text ===
            with torch.no_grad():
                outputs = self.activation_extractor.model.generate(
                    input_ids=encoded.input_ids,
                    attention_mask=encoded.attention_mask,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True
                )

            responses = [self.tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

            for batch_idx, text in enumerate(batch_texts):
                active_tokens = encoded['input_ids'][batch_idx][encoded['attention_mask'][batch_idx] == 1]
                word_pos_map = map_words_to_tokens(self.tokenizer, text, active_tokens)
                
                for circuit_idx, output_nodes in self.circuit_output_nodes.items():
                    # Initialize as 0; it will be converted to a tensor on the first addition.
                    # This sums the activation vectors of all output nodes for the circuit.
                    importance_vector: Union[int, torch.Tensor] = 0
                    for node in output_nodes:
                        if node in activations:
                            act = activations[node]
                            if isinstance(act, tuple):
                                act = act[0]
                            importance_vector += act[batch_idx].cpu()
                            del act
                    
                    word_importance_list = []
                    if isinstance(importance_vector, torch.Tensor):
                        for _, loc_list in word_pos_map.items():
                            importance = 0.0
                            if len(loc_list) > 0:
                                importance = importance_vector[loc_list].mean().item()
                            word_importance_list.append(importance)
                    
                    self.importance_db[circuit_idx][text] = {
                        "response": responses[batch_idx],
                        "importance_scores": word_importance_list
                    }
            del activations, encoded
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def save_importance_db(self, output_path: str):
        """Save importance database to a JSON file."""
        # The importance scores are already floats, so we just need to ensure
        # the dictionary keys are standard types for JSON serialization.
        save_dict = {int(k): v for k, v in self.importance_db.items()}
        try:
            with open(output_path, 'w') as f:
                json.dump(save_dict, f, indent=2)
            print(f"Successfully saved importance database to {output_path}")
        except IOError as e:
            print(f"Error saving importance database to {output_path}: {e}")

    def load_importance_db(self, input_path: str):
        """Load importance database from a JSON file."""
        try:
            with open(input_path, 'r') as f:
                loaded_dict = json.load(f)
            # Convert string keys from JSON back to integers
            for k, v in loaded_dict.items():
                self.importance_db[int(k)] = v
            print(f"Successfully loaded importance database from {input_path}")
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading importance database from {input_path}: {e}")

    # def interp_from_importance(
    #     self,
    #     save_path: str,
    #     model_name: str = 'gemini-1.5-flash'
    # ) -> Dict[int, str]:
    #     """
    #     Generate functional interpretation of modular circuits from importance scores.

    #     Args:
    #         save_path: Path to save interpretations JSON.
    #         model_name: Name of the Google Generative AI model to use.

    #     Returns:
    #         A mapping from circuit indices to their functional interpretations.
    #     """
    #     functional_interpretations: Dict[int, str] = {}
        
    #     try:
    #         genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    #     except Exception as e:
    #         print(f"Fatal Error: Could not configure Google API key. Check GOOGLE_API_KEY environment variable. Details: {e}")
    #         return {"error": "Failed to configure Google API key."}

    #     model = genai.GenerativeModel(model_name)

    #     for circuit_idx, importance_data in tqdm(self.importance_db.items(), desc="Generating Interpretations"):
    #         if not importance_data:
    #             functional_interpretations[circuit_idx] = "No importance data available to generate interpretation."
    #             continue

    #         num_samples = min(5, len(importance_data))
    #         sample_keys = random.sample(list(importance_data.keys()), num_samples)
            
    #         examples = []
    #         for text_key in sample_keys:
    #             scores = importance_data[text_key]
    #             formatted_scores = [f"{s:.4f}" for s in scores]
    #             examples.append(f"Text: {text_key}\nImportance Scores: {', '.join(formatted_scores)}")
            
    #         example_str = "\n\n".join(examples)
    #         prompt = (
    #             "You are an expert assistant interpreting neural circuits based on importance scores. "
    #             f"Given the following examples for modular circuit {circuit_idx}, "
    #             "provide a concise, one-sentence functional interpretation focusing on patterns in the high-importance scores.\n\n"
    #             "--- EXAMPLES ---\n"
    #             f"{example_str}\n"
    #             "--- INTERPRETATION ---\n"
    #         )

    #         try:
    #             response = model.generate_content(
    #                 prompt,
    #                 generation_config=genai.types.GenerationConfig(
    #                     temperature=0.3,
    #                     max_output_tokens=150,
    #                     candidate_count=1,
    #                 )
    #             )
    #             interpretation = response.text.strip()
    #         except Exception as e:
    #             interpretation = f"Error generating interpretation: {e}"
            
    #         functional_interpretations[circuit_idx] = interpretation
        
    #     if save_path:
    #         try:
    #             with open(save_path, 'w') as f:
    #                 json.dump(functional_interpretations, f, indent=2)
    #             print(f"Successfully saved interpretations to {save_path}")
    #         except IOError as e:
    #             print(f"Error saving interpretations to {save_path}: {e}")

    #     return functional_interpretations

    # def interp_from_importance(
    #     self,
    #     save_path: str,
    #     model,
    #     device,
    #     model_name: str = "llama2",
    # ) -> Dict[int, str]:
    #     """
    #     Generate functional interpretation of modular circuits from importance scores
    #     using LLaMA2 instead of Gemini.

    #     Args:
    #         save_path (str): Path to save interpretations JSON.
    #         model_name (str): Key for model in MODEL_MAP (e.g., "llama2").

    #     Returns:
    #         Dict[int, str]: Mapping from circuit indices to interpretations.
    #     """
    #     functional_interpretations: Dict[int, str] = {}

    #     # Pick the model id from MODEL_MAP
    #     # if model_name not in MODEL_MAP:
    #     #     raise ValueError(f"Unknown model_name {model_name}, must be one of {list(MODEL_MAP.keys())}")
    #     # model_id = MODEL_MAP[model_name]

    #     tokenizer = self.tokenizer
    #     if tokenizer.pad_token is None:
    #         tokenizer.pad_token = tokenizer.eos_token
    #     tokenizer.padding_side = "left" 
        

    #     for circuit_idx, importance_data in tqdm(self.importance_db.items(), desc="Generating Interpretations"):
    #         if not importance_data:
    #             functional_interpretations[circuit_idx] = "No importance data available to generate interpretation."
    #             continue

    #         # sample some examples
    #         num_samples = min(5, len(importance_data))
    #         sample_keys = random.sample(list(importance_data.keys()), num_samples)

    #         examples = []
    #         for text_key in sample_keys:
    #             scores = importance_data[text_key]
    #             formatted_scores = [f"{s:.4f}" for s in scores]
    #             examples.append(f"Text: {text_key}\nImportance Scores: {', '.join(formatted_scores)}")

    #         example_str = "\n\n".join(examples)


    #         user_prompt = (
    #             f"Given the following examples for modular circuit {circuit_idx}, "
    #             "provide a concise, one-sentence functional interpretation focusing on "
    #             "patterns in the high-importance scores.\n\n"
    #             "--- EXAMPLES ---\n"
    #             f"{example_str}\n"
    #             "--- INTERPRETATION ---\n"
    #         )

    #         # wrap as user-only chat input
    #         prompt = tokenizer.apply_chat_template(
    #             [{"role": "user", "content": user_prompt}],
    #             tokenize=False,
    #         )
    #         inputs = tokenizer(prompt, return_tensors="pt").to(device)
    #         input_ids = inputs["input_ids"]
    #         attention_mask = inputs["attention_mask"]

            

    #         try:
    #             outputs = model.generate(
    #                 input_ids=input_ids,
    #                 attention_mask=attention_mask,
    #                 max_new_tokens=1000,
    #                 temperature=0.3,
    #                 do_sample=True,
    #                 use_cache= False,
    #                 past_key_values=None
    #             )

    #             decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    #             # post-process: keep only the part after "--- INTERPRETATION ---"
    #             if "--- INTERPRETATION ---" in decoded:
    #                 interpretation = decoded.split("--- INTERPRETATION ---")[-1].strip()
    #             else:
    #                 interpretation = decoded.strip()

    #         except Exception as e:
    #             interpretation = f"Error generating interpretation: {e}"

    #         functional_interpretations[circuit_idx] = interpretation

    #     # save to file
    #     if save_path:
    #         try:
    #             with open(save_path, "w") as f:
    #                 json.dump(functional_interpretations, f, indent=2)
    #             print(f"Successfully saved interpretations to {save_path}")
    #         except IOError as e:
    #             print(f"Error saving interpretations to {save_path}: {e}")

    #     return functional_interpretations



    def interp_from_importance(
        self,
        save_path: str,
        model,
        device,
    ) -> Dict[int, str]:
        """
        A debug version of the function with extensive print statements to diagnose the issue.
        """
        functional_interpretations: Dict[int, str] = {}
        tokenizer = self.tokenizer

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # Use a standard for loop instead of tqdm for cleaner print output during debugging
        for circuit_idx, importance_data in self.importance_db.items():
            # --- Start of Loop Diagnostic ---
            print(f"\n{'='*20} Processing circuit_idx: {circuit_idx} {'='*20}")

            if not importance_data:
                print("No importance data. Skipping.")
                functional_interpretations[circuit_idx] = "No importance data available to generate interpretation."
                continue

            # --- Prompt Creation (same as before) ---
            num_samples = min(5, len(importance_data))
            sample_keys = random.sample(list(importance_data.keys()), num_samples)
            #examples = [f"Text: {k}\nImportance Scores: {', '.join(f'{s:.4f}' for s in importance_data[k])}" for k in sample_keys]
            examples = []
            for k in sample_keys:
                entry = importance_data[k]
                scores = entry["importance_scores"]
                formatted_scores = [f"{s:.4f}" for s in scores]
                response = entry.get("response", "")
                examples.append(
                    f"Text: {k}\nResponse: {response}\nImportance Scores: {', '.join(formatted_scores)}"
                )

            
            example_str = "\n\n".join(examples)
            user_prompt = (
                f"Given the following examples for modular circuit {circuit_idx}, "
                "provide a concise, one-sentence functional interpretation focusing on "
                "patterns in the high-importance scores.\n\n"
                "--- EXAMPLES ---\n"
                f"{example_str}\n"
                "--- INTERPRETATION ---\n"
            )
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}], tokenize=False,
            )
            
            # --- Pre-Generation Diagnostics ---
            print(f"Length of prompt string (characters): {len(prompt)}")
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            print(f"Shape of input_ids tensor: {inputs.input_ids.shape}")
            print(f"Shape of attention_mask tensor: {inputs.attention_mask.shape}")
            
            try:
                print("Calling model.generate()...")
                # --- The generate call with all our fixes ---
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.3,
                    do_sample=True,
                    use_cache=False,
                    past_key_values=None
                )
                print(f"SUCCESS: Generation for circuit_idx {circuit_idx} completed.")
                
                output_tokens = outputs[0][inputs.input_ids.shape[1]:]
                interpretation = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
                print(f"Generated interpretation: '{interpretation[:70]}...'")

            except Exception as e:
                print(f"\n!!!!!!!!! ERROR on circuit_idx {circuit_idx} !!!!!!!!!")
                # Using traceback will give us the full error, which might be more helpful
                print(traceback.format_exc())
                interpretation = f"Error generating interpretation: {e}"
                # After the first error, we stop the loop to analyze the output
                functional_interpretations[circuit_idx] = interpretation
                break 

            functional_interpretations[circuit_idx] = interpretation

        if save_path:
            try:
                with open(save_path, "w") as f:
                    json.dump(functional_interpretations, f, indent=2)
                print(f"Successfully saved interpretations to {save_path}")
            except IOError as e:
                print(f"Error saving interpretations to {save_path}: {e}")

        return functional_interpretations