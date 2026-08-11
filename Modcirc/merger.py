import os
import json
import random
from itertools import combinations, product

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

def get_option_text(q, idx):
    """Return option text from question dict by index (0->opa,1->opb,2->opc)."""
    if idx == 0:
        return q.get("opa", "")
    elif idx == 1:
        return q.get("opb", "")
    elif idx == 2:
        return q.get("opc", "")
    else:
        return ""

def build_merged_options(q1, q2):
    """
    Build 3 merged options such that:
    - one of them is the correct pair (q1.cop, q2.cop)
    - two others are random distinct distractor pairs
    Returns: (options_list, cop_index)
    options_list: list of strings formatted: "ans-1: <Q1.opt>, ans-2: <Q2.opt>"
    cop_index: index in options_list which is the correct combined option (0-based)
    """
    correct_pair = (int(q1["cop"]), int(q2["cop"]))
    # all possible pairs (0..2 x 0..2)
    all_pairs = [(i, j) for i in range(3) for j in range(3)]
    # remove correct from distractor candidates
    distractor_candidates = [p for p in all_pairs if p != correct_pair]

    # sample two distractors (or fewer if not available)
    num_distractors_needed = min(2, len(distractor_candidates))
    distractors = random.sample(distractor_candidates, num_distractors_needed)

    # final list of pairs (first is correct, rest distractors)
    chosen_pairs = [correct_pair] + distractors

    # Create merged option strings
    option_texts = []
    for (i1, i2) in chosen_pairs:
        t1 = get_option_text(q1, i1)
        t2 = get_option_text(q2, i2)
        merged = f"ans-1: {t1}, ans-2: {t2}"
        option_texts.append(merged)

    # If less than 3 options (edge cases with missing option fields), we may duplicate distractors
    while len(option_texts) < 3:
        # duplicate a random existing option to make length 3
        option_texts.append(random.choice(option_texts))

    # Shuffle the three options but track where the correct one went
    indexed = list(enumerate(option_texts))  # (orig_idx, text)
    random.shuffle(indexed)
    shuffled_texts = [t for (_, t) in indexed]
    # find where original correct (orig_idx==0) landed
    cop_index = next(idx for idx, (orig_i, _) in enumerate(indexed) if orig_i == 0)

    return shuffled_texts, cop_index

def merge_pair(q1, q2):
    """Merge two questions into the required format, using build_merged_options."""
    options, cop = build_merged_options(q1, q2)
    merged = {
        "question": f"q1 : {q1['question']}, q2 : {q2['question']}",
        "term": f"q1 : {q1.get('term','')}, q2 : {q2.get('term','')}",
        "opa": options[0],
        "opb": options[1],
        "opc": options[2],
        "cop": cop  # 0-based index into opa/opb/opc
    }
    return merged

def process_folder(folder):
    for filename in os.listdir(folder):
        if not filename.endswith(".jsonl") and not filename.endswith(".json"):
            continue

        filepath = os.path.join(folder, filename)
        print(f"\nProcessing {filepath} ...")

        questions = load_jsonl(filepath)
        n = len(questions)

        if n < 2:
            print(f"Only {n} question(s) in file → cannot form any pair. Skipping.")
            continue

        # Use unordered distinct pairs (combinations) so each pair appears only once
        all_pairs = list(combinations(range(n), 2))  # pairs (i,j) with i<j

        # sample up to 10k distinct unordered pairs
        num_pairs = min(10000, len(all_pairs))
        sampled_pairs = random.sample(all_pairs, num_pairs)

        # Merge pairs
        merged = []
        for i, j in sampled_pairs:
            merged.append(merge_pair(questions[i], questions[j]))

        # Output directory
        base = filename.replace(".jsonl", "").replace(".json", "")
        out_dir = os.path.join(folder, f"{base}_generated")
        os.makedirs(out_dir, exist_ok=True)

        # Number of output files needed (<=1000 items per file)
        num_files = (len(merged) + 999) // 1000

        print(f"Total merged question-pairs: {len(merged)}")
        print(f"Creating {num_files} output file(s).")

        # Save chunks (≤1000 each)
        for k in range(num_files):
            chunk = merged[k*1000:(k+1)*1000]
            out_path = os.path.join(out_dir, f"{base}_set_{k+1}.jsonl")
            save_jsonl(out_path, chunk)

        print(f"Done. Files saved in: {out_dir}")

if __name__ == "__main__":
    folder = input("Enter folder containing .json/.jsonl files: ").strip()
    process_folder(folder)
