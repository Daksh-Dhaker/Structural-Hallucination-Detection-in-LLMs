import os
import pickle
import networkx as nx
import matplotlib.pyplot as plt

def plot_subgraphs(pkl_path, output_folder):
    """Load subgraphs+freqs from pickle and save each as PNG."""
    os.makedirs(output_folder, exist_ok=True)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    subgraphs = data["graphs"]
    freqs = data["freqs"]

    print(f"Loaded {len(subgraphs)} subgraphs from {pkl_path}")

    for i, (sig, G) in enumerate(subgraphs.items()):
        freq = freqs.get(sig, 0)
        plt.figure(figsize=(3, 3))
        pos = nx.spring_layout(G, seed=42)
        
        # Extract simple node labels (e.g. only 'mlp')
        labels = {n: n.split(",")[0] for n in G.nodes()}
        nx.draw(
            G, pos,
            with_labels=True,
            labels=labels,
            node_size=800,
            node_color="lightblue",
            font_size=8,
            font_weight="bold",
            edgecolors="black"
        )
        
        plt.title(f"Frequency: {freq}", fontsize=10)
        fname = os.path.join(output_folder, f"subgraph_{i+1}_freq{freq}.png")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
    print(f"✅ Saved {len(subgraphs)} subgraph plots to '{output_folder}'")

if __name__ == "__main__":
    plot_subgraphs("hallucination_subgraphs_struct.pkl", "plots_hallu")
    plot_subgraphs("non_hallucination_subgraphs_struct.pkl", "plots_nonhallu")
