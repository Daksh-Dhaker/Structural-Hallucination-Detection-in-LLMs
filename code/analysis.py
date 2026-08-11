import os
import pickle
import pandas as pd
import networkx as nx
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import statistics

# ======================
# 1️⃣  Load Graph Data
# ======================
def load_graphs_from_folder(folder_path):
    graphs = []
    graph_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
    print(f"📂 Loading {len(graph_files)} graphs from {folder_path}")

    for graph_file in graph_files:
        try:
            circuit_id = int(os.path.splitext(graph_file)[0].split('_')[-1])
        except (IndexError, ValueError):
            continue
        try:
            with open(os.path.join(folder_path, graph_file), 'rb') as f:
                G = pickle.load(f)
            if G.number_of_nodes() == 0:
                continue
            graphs.append((circuit_id, G))
        except Exception as e:
            print(f"⚠️ Failed to load {graph_file}: {e}")
    return graphs

# ======================
# 2️⃣  Folder Paths
# ======================
# HALLU_FOLDERS = [
#     os.path.join("hallucinating_circuit", 'code_dataset/circuits_before_new'),
#     os.path.join("hallucinating_circuit", 'medal_1500/circuits_before_new'),
#     os.path.join("hallucinating_circuit", 'medical_abstract/circuits_before_new'),
#     os.path.join("hallucinating_circuit", 'numerical_dataset/circuits_before_new'),
# ]

# NON_HALLU_FOLDERS = [
#     os.path.join("Non_hallucinating_circuit", 'code_dataset/circuits_before_new'),
#     os.path.join("Non_hallucinating_circuit", 'medal_1500/circuits_before_new'),
#     os.path.join("Non_hallucinating_circuit", 'medical_abstract/circuits_before_new'),
#     os.path.join("Non_hallucinating_circuit", 'numerical_dataset/circuits_before_new'),
# ]

HALLU_FOLDERS = [
    os.path.join("hallucinating_circuit", 'code_dataset/circuits_before_new'),
    os.path.join("hallucinating_circuit", 'medal_1500/circuits_before_new'),
    os.path.join("hallucinating_circuit", 'medical_abstract/circuits_before_new'),
    # os.path.join("hallucinating_circuit", 'numerical_dataset/circuits_before_new'),
    os.path.join("hallucinating_circuit", 'rationalization_binary/circuits_before_new'),
]

NON_HALLU_FOLDERS = [
    os.path.join("Non_hallucinating_circuit", 'code_dataset/circuits_before_new'),
    os.path.join("Non_hallucinating_circuit", 'medal_1500/circuits_before_new'),
    os.path.join("Non_hallucinating_circuit", 'medical_abstract/circuits_before_new'),
    # os.path.join("Non_hallucinating_circuit", 'numerical_dataset/circuits_before_new'),
    os.path.join("Non_hallucinating_circuit", 'rationalization_binary/circuits_before_new'),
]

def collect_all_graphs(folders):
    all_graphs = []
    for folder in folders:
        all_graphs.extend(load_graphs_from_folder(folder))
    return all_graphs

hallu_graphs = collect_all_graphs(HALLU_FOLDERS)
non_hallu_graphs = collect_all_graphs(NON_HALLU_FOLDERS)

print(f"\n✅ Loaded {len(hallu_graphs)} hallucination graphs")
print(f"✅ Loaded {len(non_hallu_graphs)} non-hallucination graphs")


# ======================
# 3️⃣  Count Node Occurrences
# ======================
def count_node_frequencies(graphs):
    node_to_circuits = defaultdict(set)
    degree_stats = []
    for circuit_id, G in graphs:
        for node, degree in G.degree():
            node_to_circuits[node].add(circuit_id)
            degree_stats.append(degree)
    freq_counter = {node: len(circuits) for node, circuits in node_to_circuits.items()}
    return freq_counter, degree_stats

hallu_freq, hallu_degrees = count_node_frequencies(hallu_graphs)
non_hallu_freq, non_hallu_degrees = count_node_frequencies(non_hallu_graphs)


# ======================
# 4️⃣  Cross-Frequency Analysis
# ======================
# Count how often hallucination nodes appear in non-hallucination graphs, and vice versa
hallu_nodes = set(hallu_freq.keys())
non_hallu_nodes = set(non_hallu_freq.keys())

hallu_in_non = {n: non_hallu_freq.get(n, 0) for n in hallu_nodes}
non_in_hallu = {n: hallu_freq.get(n, 0) for n in non_hallu_nodes}


# ======================
# 5️⃣  Statistics + Helper Functions
# ======================
def summarize_distribution(values, label, outfile):
    if not values:
        outfile.write(f"\n⚠️ No values for {label}\n")
        return
    mean_val = np.mean(values)
    median_val = np.median(values)
    max_val = np.max(values)
    min_val = np.min(values)
    std_val = np.std(values)
    outfile.write(f"\n📊 {label} Statistics:\n")
    outfile.write(f"  Count: {len(values)}\n")
    outfile.write(f"  Mean: {mean_val:.2f}, Median: {median_val}, Std: {std_val:.2f}\n")
    outfile.write(f"  Min: {min_val}, Max: {max_val}\n")

def top_nodes(freq_dict, label, outfile, top_k=10):
    top = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    outfile.write(f"\n🏆 Top {top_k} Most Frequent Nodes in {label}:\n")
    for i, (node, count) in enumerate(top, 1):
        outfile.write(f"  {i:2d}. {node} — appears in {count} circuits\n")


# ======================
# 6️⃣  Save Stats to Text File
# ======================
with open("node_frequency_report.txt", "w") as f:
    f.write("======== NODE FREQUENCY ANALYSIS REPORT ========\n\n")

    # Own category frequencies
    summarize_distribution(list(hallu_freq.values()), "Hallucination Node Frequency", f)
    summarize_distribution(list(non_hallu_freq.values()), "Non-Hallucination Node Frequency", f)
    summarize_distribution(list(hallu_in_non.values()), "Hallucination Nodes in Non-Hallucination Graphs", f)
    summarize_distribution(list(non_in_hallu.values()), "Non-Hallucination Nodes in Hallucination Graphs", f)

    # Degree stats
    summarize_distribution(hallu_degrees, "Hallucination Node Degree", f)
    summarize_distribution(non_hallu_degrees, "Non-Hallucination Node Degree", f)

    # Top 10 nodes
    top_nodes(hallu_freq, "Hallucination", f)
    top_nodes(non_hallu_freq, "Non-Hallucination", f)
    top_nodes(hallu_in_non, "Hallucination Nodes in Non-Hallucination Graphs", f)
    top_nodes(non_in_hallu, "Non-Hallucination Nodes in Hallucination Graphs", f)

print("\n✅ Report saved to 'node_frequency_report.txt'")


# ======================
# 7️⃣  Plot Distributions
# ======================
def plot_distribution(values, label, color):
    plt.figure(figsize=(7,4))
    plt.hist(values, bins=30, color=color, alpha=0.7)
    plt.title(f"{label} Distribution")
    plt.xlabel("Frequency")
    plt.ylabel("Count of Nodes")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{label.replace(' ','_').lower()}_hist.png")
    plt.close()

plot_distribution(list(hallu_freq.values()), "Hallucination Node Frequency", "tomato")
plot_distribution(list(non_hallu_freq.values()), "Non-Hallucination Node Frequency", "royalblue")
plot_distribution(list(hallu_in_non.values()), "Hallucination Nodes in Non-Hallucination Graphs", "orange")
plot_distribution(list(non_in_hallu.values()), "Non-Hallucination Nodes in Hallucination Graphs", "green")

plot_distribution(hallu_degrees, "Hallucination Node Degree", "tomato")
plot_distribution(non_hallu_degrees, "Non-Hallucination Node Degree", "royalblue")

print("✅ Histograms saved as PNGs for all distributions.")
