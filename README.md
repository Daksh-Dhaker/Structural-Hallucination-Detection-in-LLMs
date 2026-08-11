
# Mechanistic Interpretability of LLM Hallucinations
## Usage

### Phase 1: Response Generation & Classification (`/ModCirc/FinalDataset`)

The first step is to generate responses from the LLM and classify them into **Hallucination** or **Non-hallucination** categories. This is handled by the automated shell script `run_all_response.sh`.

**What this script does:**
1. Dynamically patches `ResponseGen.py` to route outputs to `*_umang` folders.
2. **Activates the `modcirc_env` Conda environment. 
3. Iterates through all `.json` and `.jsonl` files in the input directory.
4. Generates responses using the specified LLM.

**How to run:**

1. Navigate to the dataset directory:
   ```bash
   cd ModCirc/FinalDataset
2. Run with default settings (uses meta-llama/Meta-Llama-3-8B-Instruct):
   ```bash
    run_all_response.sh
3. (Optional) Run with a custom model: You can pass a HuggingFace model path as an argument:
    ```bash

    run_all_response.sh "meta-llama/Llama-2-7b-chat-hf"

4. Output Directories: After the script finishes, you will find the classified data in:

Non_hallucination_umang/ (Correct responses)

hallucination_umang/ (Hallucinated responses)

Note: The script includes a setup for network proxy (proxyiit.py).

### Phase 2: Circuit Extraction & Feature Processing (`/ModCirc`)

Once the response datasets are generated, use the `run.sh` script to extract computational circuits and their features. This script automates the entire pipeline: batching data, running the GNN extractor, and organizing results.

**How to run:**
    ```bash
    
    cd ModCirc
    bash run.sh

What this script does (run.sh workflow):

1. Environment Setup: Activates the modcirc_env conda environment.

2. Batch Creation: - Splits large JSONL files from FinalDataset/ into manageable batches using create_batches.py.

 - Default: Uses 100% of the dataset (DEFAULT_PCT=100).

3. Circuit Discovery (main.py):

- Runs the circuit discovery algorithm (K-Means based) on each batch.

- Model used: meta-llama/Meta-Llama-3-8B-Instruct.

4. Feature Extraction (batch_extract_circuit_features.py):

- Extracts node features (activations) and edge attributes from the discovered circuits.

5. Organization:

- Moves processed circuits into:

   - graphs/hallucinating_circuits/

   - graphs/Non_hallucinating_circuits/

- Automatically cleans the intermediate results/ folder between batches to prevent data leakage.

## Project Structure

divided into two main components: **Code** (Analysis & GNN Classifiers) and **ModCirc** (Circuit Extraction & Data Generation).

### 1. Analysis & Classification (`/code`)

This directory contains the Graph Neural Network implementations used to classify the extracted circuits.

<details>
<summary><b>code/</b> <i>(Click to expand file list)</i></summary>
<br>

| File / Folder | Description |
| :--- | :--- |
| **GNN Classifiers** | |
| `gat_classifier.py` |Graph Attention Network (GAT) implementation for circuit classification. |
| `gcn_classifier.py` |Graph Convolutional Network (GCN) implementation. |
| `gin_classifier.py` |Graph Isomorphism Network (GIN) implementation. |
| `gsage_classifier.py` |GraphSAGE implementation for scalable inductive representation learning. |
| `xgboost_classifier.py` |Baseline XGBoost classifier using tabular features rather than graph structures. |
| `pairwise_classifier.py` |Pairwise classifier implementation with contrastive pairing. |
| **Execution** | |
| `final_gnn.py` | **Main Entry Point.** The primary script for training and evaluating the selected GNN architecture. |
| `run_gnn.sh` | Shell script to automate the training pipeline across different hyperparameters. |
| **Data & Analysis** | |
| `hallucinating_circuit/` |Contains processed subgraph datasets (Medical, Code, Numerical) corresponding to instances where the LLM <b>hallucinated</b>. |
| `Non_hallucinating_circuit/` |Contains processed subgraph datasets corresponding to correct (factual) LLM responses. |
| `hallucinating_circuits/` | Similar to the hallucination_circuit, its just for the bootstrapped dataset |
| `Non_hallucinating_circuits/` | Similar to the Non_hallucination_circuit, its just for the bootstrapped dataset |
| `analysis.py` | Scripts for statistical analysis of the dataset (node distribution, edge density). Not usefull as such |
| `plot.py` | Generates visualizations found in the `/images` folder.  Not usefull as such |

</details>

---

### 2. Circuit Extraction & Processing (`/ModCirc`)

This directory handles : generating LLM responses, identifying circuits, and extracting features for the GNNs.

<details>
<summary><b>ModCirc/</b> <i>(Click to expand file list)</i></summary>
<br>

| File / Folder | Description |
| :--- | :--- |
| **Core Extraction** | |
| `batch_extract_circuit_features.py` |Processes circuits in batches to extract node and edge attributes required for GNN training. |
| `extract_all_node_features.py` | Extracts hidden state activations and other metadata from specific nodes in the computational graph. |
| **Data Generation** | |
| `FinalDataset/` |Contains the raw JSONL datasets (e.g., `medal_1500_mc.jsonl`, `medical_abstract_1500.jsonl`) and the `ResponseGen.py` script used to query the LLM. |

</details>
