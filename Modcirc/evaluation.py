import networkx as nx
from typing import List, Set, Tuple
import torch
from models.func_interp.activation_extractor import ActivationExtractor
from models.func_interp.importance_analyzer import ImportanceAnalyzer
import json

import os
#import google.generativeai as genai
import logging
import re


class MCevaluator:
    def __init__(self, result_path, tokenizer, model, eval_circuits, merged_mc_graph, func_interp, dataset_list):
        """
        Initialize MCevaluator

        Args:
            eval_circuits (Dict): Dictionary mapping dataset names to their circuits
            mc_set (set/dict): Set of modular circuits
            merged_mc_graph (nx.DiGraph): Merged modular circuit graph
        """
        self.importance_path = result_path.replace("_results_", "_eval_importance_scores_").replace(".txt", ".pt")
        self.func_interp_path = result_path.replace("_results_", "_eval_func_interp_").replace(".txt", ".json")
        self.consistency_path = result_path.replace("_results_", "_consistency_scores_").replace(".txt", ".json")
        self.reusability_path = result_path.replace("_results_", "_reusability_scores_").replace(".txt", ".json")
        self.composability_path = result_path.replace("_results_", "_composability_scores_").replace(".txt", ".json")
        self.model = model
        self.dataset_list = dataset_list
        self.tokenizer = tokenizer
        self.eval_circuits = [eval_circuits[key] for key in eval_circuits]
        self.merged_mc_graph = merged_mc_graph
        self.func_interp = func_interp
        self.circuit_order = self.get_circuit_order(merged_mc_graph)
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def get_circuit_order(self, mc_graph:nx.DiGraph):
        partitions = nx.get_node_attributes(mc_graph, "partition")
        partition_layers = {}
        for n in partitions:
            if partitions[n].item() not in partition_layers:
                partition_layers[partitions[n].item()] = []
            partition_layers[partitions[n].item()].append(int(n.split(",")[1]))
        partition_layers = {k: sorted(v) for k,v in partition_layers.items()}
        order = sorted(list(partition_layers.items()), key= lambda x: (x[1][0], x[1][-1]))
        return [o[0] for o in order]

    def evaluate_mod_circ(self, model, device) -> Tuple[float, float, float]:
        """
        Evaluate modular circuits based on consistency, reusability, and composability

        Returns:
            Tuple[float, float, float]: Consistency, reusability, and composability scores
        """
        print( "Started evaluate_mod_circ ","\n")

        mc_set_intersected, mc_idx_for_dataset = self.get_intersected_mcs()

        print("Done intersection")
        eval_func_interp = self._get_mc_interpretation(mc_set_intersected) # a list of dictionaries.
        print("Done interpretation")
        consistency = self.cal_consistency(model, device, mc_set_intersected, mc_idx_for_dataset, eval_func_interp)
        print("Done cal_consistency")
        reusability = self.cal_reusability(mc_set_intersected, mc_idx_for_dataset)
        print("Done cal_reusability")
        composability = self.cal_composability(model, device, eval_func_interp)
        print("Done cal_composability")
        return consistency, reusability, composability

    def get_intersected_mcs(self) -> List[Set]:
        """
        Calculate the intersected modular circuit sets of each evaluation dataset.

        Returns:
            List[Set]: List of intersected modular circuit sets
        """
        all_circuits = self.eval_circuits
        # Find intersections between all pairs of circuits
        intersected_mcs = []
        mc_idx_for_dataset = []
        for i in range(len(all_circuits)):
            # calculate intersection of all_circuits[i] with mc_set
            intersection = nx.intersection(all_circuits[i], self.merged_mc_graph)
            nx.set_node_attributes(intersection, {n: self.merged_mc_graph.nodes[n]['partition'].long() for n in intersection.nodes()}, name='partition')
            intersected_mcs.append(intersection)
            # get al "partition" of the nodes in intersection and take the idx set
            mc_idx_for_dataset.append({intersection.nodes[node]["partition"].long().item() for node in intersection.nodes()})
        return intersected_mcs, mc_idx_for_dataset

    def cal_consistency(self, model, device, mc_set_intersected: List[nx.DiGraph], mc_idx_for_dataset, eval_func_interp: List[dict]) -> float:
        """
        Calculate the proportion of semantically consistent modular circuits

        Args:
            mc_set_intersected (List[Set]): List of intersected modular circuit sets

        Returns:
            float: Consistency score
        """
        if not os.path.exists(self.consistency_path):
            total_mcs = len(mc_set_intersected)
            if total_mcs == 0:
                return 0.0
            consistency_list = []
            for i in range(len(mc_idx_for_dataset)): # iterate through all eval circuits.
                consistency_score = 0.0
                counter = 0
                for idx in mc_idx_for_dataset[i]:
                    print(i," ",idx)
                    # print(eval_func_interp[i][idx])
                    # print(self.func_interp[idx])
                    if i >= len(eval_func_interp):
                        continue
                    if idx not in (eval_func_interp[i]):
                        continue
                    if idx not in (self.func_interp):
                        continue
                    counter += 1
                    consistency_score += self._check_semantic_consistency(eval_func_interp[i][idx],self.func_interp[idx],model, device)

                # consistency_score /= len(mc_idx_for_dataset[i])
                print("counter-",counter)
                if (counter > 0):
                    consistency_score /= counter
                consistency_list.append(consistency_score)
            with open(self.consistency_path, "w") as f:
                json.dump(consistency_list, f)
        else:
            with open(self.consistency_path, "r") as f:
                consistency_list = json.load(f)
        print("consistency list -> ", consistency_list)
        return consistency_list

    # def _check_semantic_consistency(
    #     self,
    #     sen_1: str,
    #     sen_2: str,
    #     prompt_model: str = 'gemini-1.5-flash'
    # ) -> float:
    #     """
    #     Check the semantic consistency between two sentences describing LLM circuit functionalities.

    #     Args:
    #         sen_1 (str): First sentence describing an LLM circuit functionality
    #         sen_2 (str): Second sentence describing an LLM circuit functionality
    #         prompt_model (str): The model to use for the check.

    #     Returns:
    #         float: Normalized similarity score between 0 and 1
    #             0 indicates completely different functionalities
    #             1 indicates identical/highly similar functionalities
    #             -1.0 indicates an error occurred.
    #     """

    #     # Configure the API key from environment variables
    #     # It's good practice to do this once when your application starts
    #     try:
    #         api_key = os.getenv("GOOGLE_API_KEY")
    #         if not api_key:
    #             raise ValueError("GOOGLE_API_KEY environment variable not set.")
    #         genai.configure(api_key=api_key)
    #     except Exception as e:
    #         logging.error(f"Failed to configure Google API: {e}")
    #         return -1.0


    #     # Construct the prompt for the Gemini model
    #     # The system role is combined into the main prompt for this library's simple generation.
    #     prompt = f"""You are an expert in machine learning and neural network architectures. Your task is to rate the semantic similarity between two sentences describing functionalities of LLM circuit components.

    #     Sentence 1: "{sen_1}"
    #     Sentence 2: "{sen_2}"

    #     Rate the similarity on a scale of 1-5 where:
    #     1 = Completely different functionalities
    #     2 = Slightly related but distinct functionalities
    #     3 = Moderately similar functionalities
    #     4 = Very similar functionalities
    #     5 = Identical or nearly identical functionalities

    #     Provide only the single numerical rating as your response, without any explanation or additional text.
    #     """

    #     try:
    #         # Initialize the generative model
    #         model = genai.GenerativeModel(prompt_model)

    #         # Generate content
    #         response = model.generate_content(
    #             prompt,
    #             generation_config=genai.types.GenerationConfig(
    #                 # candidate_count is not directly set like this, the API returns candidates by default
    #                 temperature=0.3,
    #                 max_output_tokens=5 # A single number is expected, so a small token limit is fine
    #             )
    #         )

    #         # Extract the text, remove potential whitespace, and convert to integer
    #         rating_text = response.text.strip()
    #         rating = int(rating_text)
            
    #         # Validate the rating is within the expected range
    #         if not 1 <= rating <= 5:
    #             raise ValueError(f"Model returned an invalid rating outside the 1-5 range: {rating}")
                
    #         # Normalize the rating to a 0-1 range
    #         return (rating - 1) / 4.0

    #     except ValueError as ve: # Catch specific errors for better logging
    #         logging.error(f"Error processing model response. Raw response: '{response.text if 'response' in locals() else 'N/A'}'. Details: {ve}")
    #         return -1.0
    #     except Exception as e:
    #         logging.error(f"An unexpected error occurred in semantic consistency check: {e}")
    #         # You might want to inspect the response object for more details if it exists
    #         if 'response' in locals() and response.prompt_feedback:
    #             logging.error(f"Prompt feedback: {response.prompt_feedback}")
    #         return -1.0

    def _check_semantic_consistency(
        self,
        sen_1: str,
        sen_2: str,
        model,
        device,
    ) -> float:
        
        print( " hello ")
        """
        A debug version of the function with extensive print statements to diagnose the issue.
        """
        tokenizer = self.tokenizer

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        # Build the prompt for LLaMA2
        user_prompt = f"""You are an expert in machine learning and neural network architectures. Your task is to rate the semantic similarity between two sentences describing functionalities of LLM circuit components.

        Sentence 1: "{sen_1}"
        Sentence 2: "{sen_2}"

        Rate the similarity on a scale of 1-5 where:
        1 = Completely different functionalities
        2 = Slightly related but distinct functionalities
        3 = Moderately similar functionalities
        4 = Very similar functionalities
        5 = Identical or nearly identical functionalities

        Provide only the single numerical rating as your response, without any explanation or additional text.
        """

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
        )

        try:
            # Encode inputs
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # Generate response
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                use_cache=False,
                past_key_values=None
            )

            # Decode only the new tokens
            output_tokens = outputs[0][inputs.input_ids.shape[1]:]
            rating_text = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

            # Extract integer rating
            match = re.search(r"[1-5]", rating_text)
            if not match:
                raise ValueError(f"Model did not return a valid rating. Got: {rating_text}")
            
            rating = int(match.group(0))
            # rating = int(rating_text)
            print("rating->",rating)
            # Validate the rating is within the expected range
            if not 1 <= rating <= 5:
                raise ValueError(f"Model returned an invalid rating outside the 1-5 range: {rating}")
                
            # Normalize the rating to a 0-1 range
            print("ans",(rating-1)/4.0)

            return (rating - 1) / 4.0

        except ValueError as ve: # Catch specific errors for better logging
            logging.error(f"Error processing model response. Raw response: '{outputs.text if 'response' in locals() else 'N/A'}'. Details: {ve}")
            print("ans",-1)
            return -1.0
        except Exception as e:
            logging.error(f"An unexpected error occurred in semantic consistency check: {e}")
            # You might want to inspect the response object for more details if it exists
            if 'response' in locals() and outputs.prompt_feedback:
                logging.error(f"Prompt feedback: {outputs.prompt_feedback}")
            print("ans",-1)
            return -1.0


    def cal_reusability(self, mc_set_intersected: List[Set], mc_idx_for_dataset: List[Set]) -> float:
        """
        Calculate reusability score using F1 score

        Args:
            mc_set_intersected (List[Set]): List of intersected modular circuit sets

        Returns:
            float: Reusability score (F1)
        """
        if not os.path.exists(self.reusability_path):
            # Calculate precision: average percentage of circuit portion in MCs
            precision = self._calculate_precision(mc_set_intersected, mc_idx_for_dataset) # list of 4 each entry is precision of one eval dataset
            # Calculate recall: percentage of circuit covered by MC vocabulary
            recall = self._calculate_recall(mc_set_intersected, mc_idx_for_dataset) # list of 4 each entry is recall of one eval dataset
            # Calculate F1 score
            if precision + recall == 0:
                return 0.0
            f1 = 2 * (torch.tensor(precision) * torch.tensor(recall)) / (torch.tensor(precision) + torch.tensor(recall))
            with open(self.reusability_path, "w") as f:
                json.dump(f1.tolist(), f)
        else:
            with open(self.reusability_path, "r") as f:
                f1 = json.load(f)
        return f1

    def _calculate_precision(self, mc_set_intersected: List[Set], mc_idx_for_dataset: List[Set]) -> float:
        """
        Calculate precision for reusability, i.e., average circuit portion of MCs.

        Args:
            mc_set_intersected (List[Set]): List of intersected modular circuit sets

        Returns:
            float: Precision score
        """
        if not mc_set_intersected:
            return 0.0
        precision_for_datasets = []
        for i in range(len(mc_set_intersected)):
            intersected_size = 0
            for idx in mc_idx_for_dataset[i]:
                # get trained modular circuits of index idx in mc_set_intersected[i]
                # find all nodes with partition idx in mc_set_intersected[i]
                intersected_size += len({node for node in mc_set_intersected[i].nodes()
                                   if mc_set_intersected[i].nodes[node]['partition'] == idx})
            precision = intersected_size / len(self.merged_mc_graph)
            precision_for_datasets.append(precision)
        return precision_for_datasets


    def _calculate_recall(self, mc_set_intersected: List[Set], mc_idx_for_dataset: List[Set]) -> float:
        """
        Calculate recall for reusability: percentage of circuit covered by MC vocabulary

        Args:
            mc_set_intersected (List[Set]): List of intersected modular circuit sets

        Returns:
            float: Recall score
        """
        # Combine all nodes in evaluation circuits
        recall_list = []
        for i in range(len(mc_set_intersected)):
            recall = len(mc_set_intersected[i]) / len(self.eval_circuits[i])
            recall_list.append(recall)
        return recall_list

    def cal_composability(self, model, device, eval_func_interp) -> float:
        """
        Calculate composability using LLM interpretations

        Args:
            mc_set_intersected (List[Set]): List of intersected modular circuit sets

        Returns:
            float: Composability score
        """
        if not os.path.exists(self.composability_path):
            # data sequence: ["medi_attr", "medi_status", "coreference", "pubmed_summ"]
            task_semantics = [
            "This task is designed to identify specific entities such as medications, symptoms, or key terms within a clinical scenario.",
            "The task focuses on determining the status of medications (active, discontinued, or neither) based on the provided clinical context.",
            "The task addresses the resolution of pronouns or specific entities within clinical text to understand references correctly.",
            "The task evaluates models on summarization and comprehension tasks by identifying categories or contexts of biomedical abstracts."
            ]
            score_list = []
            for i in range(len(eval_func_interp)):
                # make all items in eval_func_interp[i] to be "key:value"
                mc_circuit_str=','.join(f"{idx}:{eval_func_interp[i][cir_idx].replace(str(cir_idx), str(idx))}" for idx, cir_idx in enumerate(self.circuit_order) if cir_idx in eval_func_interp[i])
                score = self._prompt_for_composability(task_semantics[i], mc_circuit_str, model,self.tokenizer, device)
                score_list.append(score)
            with open(self.composability_path, "w") as f:
                json.dump(score_list, f)
        else:
            with open(self.composability_path, "r") as f:
                score_list = json.load(f)
        return score_list

    # def _prompt_for_composability(self, task_semantics: str, mc_circuit_str: str, prompt_model='gpt-4o-mini') -> float:
    #     """
    #     Evaluate if a set of modular circuits can be semantically composed to accomplish a task.

    #     Args:
    #         task_semantics (str): Description of the target task to be accomplished
    #         mc_circuit_str (str): String representing the functional interpretations of modular circuits,
    #                             typically in format "circuit1:circuit2:circuit3"

    #     Returns:
    #         float: Normalized composability score between 0 and 1
    #             0 indicates circuits cannot be composed for the task
    #             1 indicates perfect composability for the task
    #     """

    #     # Construct the prompt for GPT-4
    #     prompt = f"""
    #     Evaluate whether the following set of modular circuit components can be meaningfully composed to accomplish the specified task.

    #     Task to accomplish: "{task_semantics}"

    #     Available circuit components and their functionalities:
    #     {mc_circuit_str}

    #     Rate the composability on a scale of 1-5 where:
    #     1 = Components cannot be meaningfully composed to accomplish the task
    #     2 = Components could partially support the task but major functional gaps exist
    #     3 = Components could accomplish the task with significant adaptations
    #     4 = Components can be well-composed to accomplish the task with minor adaptations
    #     5 = Components can be perfectly composed to accomplish the task

    #     Consider:
    #     - Functional completeness (do the components cover all required capabilities?)
    #     - Semantic compatibility (can the components work together coherently?)
    #     - Interface compatibility (can the components be connected meaningfully?)

    #     Provide only the numerical rating without any explanation.
    #     """

    #     try:
    #         # Configure Google Generative AI API key
    #         genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    #         # Instantiate GenerativeModel
    #         gen_model = genai.GenerativeModel(prompt_model)
    #         # Call chat predict
    #         response = gen_model.predict(
    #             model=prompt_model,
    #             messages=[
    #                 {"author": "system", "content": "You are an expert in machine learning and neural network architectures."},
    #                 {"author": "user", "content": prompt}
    #             ],
    #             temperature=0.3,
    #             max_output_tokens=150,
    #             candidate_count=1,
    #         )
    #         # Extract the numerical rating from the response
    #         rating = int(response.candidates[0].content)
    #         if not 1 <= rating <= 5:
    #             raise ValueError(f"Google API returned invalid rating: {rating}")
    #         return (rating - 1) / 4.0
    #     except Exception as e:
    #         logging.error(f"Error in composability check: {e}")
    #         return -1.0

    def _prompt_for_composability(
        self,
        task_semantics: str,
        mc_circuit_str: str,
        model,
        tokenizer,
        device: str = "cuda"
    ) -> float:
        """
        Evaluate if a set of modular circuits can be semantically composed to accomplish a task,
        using a local Llama 2 model.

        Args:
            task_semantics (str): Description of the target task to be accomplished.
            mc_circuit_str (str): String representing the functional interpretations of modular circuits.
            model: The loaded Hugging Face model.
            tokenizer: The loaded Hugging Face tokenizer.
            device: The device to run the model on (e.g., "cuda").

        Returns:
            float: Normalized composability score between 0 and 1, or -1.0 on error.
        """

        # 1. The prompt is excellent and can be reused directly.
        prompt_text = f"""
        Evaluate whether the following set of modular circuit components can be meaningfully composed to accomplish the specified task.

        Task to accomplish: "{task_semantics}"

        Available circuit components and their functionalities:
        {mc_circuit_str}

        Rate the composability on a scale of 1-5 where:
        1 = Components cannot be meaningfully composed to accomplish the task
        2 = Components could partially support the task but major functional gaps exist
        3 = Components could accomplish the task with significant adaptations
        4 = Components can be well-composed to accomplish the task with minor adaptations
        5 = Components can be perfectly composed to accomplish the task

        Consider:
        - Functional completeness (do the components cover all required capabilities?)
        - Semantic compatibility (can the components work together coherently?)
        - Interface compatibility (can the components be connected meaningfully?)

        Provide only the numerical rating without any explanation.
        """
        
        #Apply the chat template appropriate for Llama instruct models
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are an expert in machine learning and neural network architectures."},
                {"role": "user", "content": prompt_text}
            ],
            tokenize=False,
        )
        prompt = prompt_text

        try:
            # 2. Tokenize the prompt and prepare for the model
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            # 3. Call model.generate with the robust settings we established
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,  # We only need a single digit and maybe a period.
                temperature=0.1,   # Low temperature for a deterministic rating
                do_sample=True,
                use_cache=False,
                past_key_values=None
            )

            # Decode only the newly generated part of the output
            response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

            # 4. Parse the response to find the numerical rating
            # Use a regular expression to find the first digit in the response.
            # This is more robust than `int()` as it handles responses like "The rating is 4."
            match = re.search(r'\d+', response_text)
            if not match:
                raise ValueError(f"Could not find a numerical rating in the model's response: '{response_text}'")

            rating = int(match.group(0))
            if not 1 <= rating <= 5:
                raise ValueError(f"Model returned an invalid rating: {rating}")

            # 5. Normalize the score to be between 0.0 and 1.0
            return (rating - 1) / 4.0

        except Exception as e:
            logging.error(f"Error in composability check with Llama: {e}")
            return -1.0


    def _get_mc_interpretation(self, mc_set_intersected):
        """
        Get functional interpretation for a modular circuit

        Args:
            mc_set (Set): Set of nodes in the modular circuit

        Returns:
            str: Functional interpretation of the MC
        """
        # Combine interpretations of individual nodes
        if not os.path.exists(self.func_interp_path):
            eval_func_interp = []
            for i in range(len(mc_set_intersected)):
                activation_extractor = ActivationExtractor(self.model, mc_set_intersected[i])
                circuit_outputs = activation_extractor.get_circuit_output_nodes()
                input_texts = [s["clean_text"] for s in self.dataset_list[i]]
                analyzer = ImportanceAnalyzer(self.tokenizer, activation_extractor, circuit_outputs)
                analyzer.compute_importance_scores(input_texts)
                print("Input ",input_texts)

                analyzer.save_importance_db(self.importance_path)
                analyzer.load_importance_db(self.importance_path)
                eval_func_interp.append(analyzer.interp_from_importance(None, self.model, self.model.device))

                # Clean up before processing next dataset
                activation_extractor.cleanup()
                del analyzer
                torch.cuda.empty_cache()
            with open(self.func_interp_path, "w") as f:
                json.dump(eval_func_interp, f)
        else:
            with open(self.func_interp_path, "r") as f:
                eval_func_interp = [{int(k): v for k,v in interp.items()} for interp in json.load(f)]
            # make sure eval_func_interp is a list of dictionaries
        return eval_func_interp

    def _calculate_semantic_similarity(self, interp1: str, interp2: str) -> float:
        """
        Calculate semantic similarity between two interpretations

        Args:
            interp1 (str): First interpretation
            interp2 (str): Second interpretation

        Returns:
            float: Similarity score between 0 and 1
        """
        # Use the model to calculate semantic similarity
        inputs = self.tokenizer(
            f"Compare the similarity between these two interpretations: 1) {interp1} 2) {interp2}",
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Process logits to get similarity score
            # This is a simplified approach - you might want to use a more sophisticated method
            similarity = torch.sigmoid(logits.mean()).item()

        return similarity