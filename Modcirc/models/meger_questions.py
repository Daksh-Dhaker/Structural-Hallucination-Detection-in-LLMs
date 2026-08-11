import os
import json
import random
from itertools import product

def load_jsonl(path):
    """Load JSONL file as a list of objects."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except:
                    pass
    return items


def save_jsonl(path, data):
    """Save list of dicts to JSONL."""
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_pair(q1, q2):
    """Merge two questions using your required format."""

    cop = q1["cop"]
    # Assuming your dataset always has same COP for q1 and q2

    merged = {
        "question": f"q1 : {q1['question']}, q2 : {q2['question']}",
        "term": f"q1 : {q1.get('term','')}, q2 : {q2.get('term','')}",

        "opa": f"ans-1: {q1['opa']}, ans-2: {q2['opa']}",
        "opb": f"ans-1: {q1['opb']}, ans-2: {q2['opb']}",
        "opc": f"ans-1: {q1['opc']}, ans-2: {q2['opc']}",

        "cop": cop  # 0-based index
    }

    return merged


def process_folder(folder):
    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(folder, filename)
        print(f"Processing {filepath} ...")

        questions = load_jsonl(filepath)
        n = len(questions)

        if n < 2:
            print(f"Skipping {filename}, not enough questions.")
            continue

        # All ordered pairs (i,j)
        all_pairs = list(product(range(n), repeat=2))

        # Sample 10k pairs
        sampled_pairs = random.sample(all_pairs, 10000)

        # Merge pairs
        merged = []
        for i, j in sampled_pairs:
            merged.append(merge_pair(questions[i], questions[j]))

        # Split into 10 chunks
        base = filename.replace(".json", "")
        out_dir = os.path.join(folder, f"{base}_generated")
        os.makedirs(out_dir, exist_ok=True)

        for k in range(10):
            chunk = merged[k*1000:(k+1)*1000]
            out_path = os.path.join(out_dir, f"{base}_set_{k+1}.jsonl")
            save_jsonl(out_path, chunk)

        print(f"Generated 10 files in: {out_dir}")


if __name__ == "__main__":
    folder = input("Enter folder containing .json files: ").strip()
    process_folder(folder)
