#!/usr/bin/env python3
"""
create_batches.py

Read a JSONL file (one JSON object per line) and create random overlapping
batches. Each batch contains approximately pct% of the dataset (at least 1
line). At least min_batches batches are created (default 10). The script
ensures the union of batches covers the entire input file by adding extra
batches that include any missing lines.

Outputs files into: <outdir>/<input_basename>_batch_<idx>.jsonl
Also creates a manifest: batches_manifest.json (list of filenames)

Usage:
  python3 create_batches.py --input data.jsonl --outdir ./batches --pct 30 --min-batches 10 --seed 123
"""

import argparse
import json
import math
import os
import random
from pathlib import Path

def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip() != ""]
    return lines

def write_batch(lines, outpath):
    with open(outpath, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

def make_batches(lines, pct, min_batches, outdir, basename, seed=None):
    if seed is not None:
        random.seed(seed)

    n = len(lines)
    if n == 0:
        raise ValueError("Input file has zero lines")

    batch_size = max(1, int(round(pct / 100.0 * n)))


    num_batches = max(1, min_batches)


    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    all_indices = set(range(n))
    created_files = []

    # Create initial batches
    for i in range(num_batches):
        # sample without replacement within a batch to avoid duplicates in same batch
        batch_idx = sorted(random.sample(range(n), k=min(batch_size, n)))
        batch_lines = [lines[idx] for idx in batch_idx]
        batch_name = f"{basename}_batch_{i+1}.jsonl"
        outpath = outdir_path / batch_name
        write_batch(batch_lines, outpath)
        created_files.append(str(outpath))

    # Check coverage and generate additional batches until coverage == all_indices
    covered = set()
    for fname in created_files:
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                # find original line index by matching line string (may be duplicate lines)
                # but to be robust we use indices from created batches if possible. Here we will
                # compute indices by scanning original list.
                pass

    # Instead of matching by content (duplicates could exist), we'll compute covered indices
    # by reading created batches and mapping to original indices deterministically:
    # Build a dict mapping line->list of indices; then when a line from batch appears, pop one index.
    index_map = {}
    for idx, line in enumerate(lines):
        index_map.setdefault(line, []).append(idx)

    covered_indices = set()
    for fname in created_files:
        with open(fname, "r", encoding="utf-8") as f:
            for line in f:
                lst = index_map.get(line)
                if lst:
                    # pop one index from the list to avoid double-counting identical lines
                    covered_indices.add(lst.pop(0))

    missing = all_indices - covered_indices
    extra_count = 0
    while missing:
        extra_count += 1
        # include as many missing as possible into a single batch (up to batch_size)
        take_missing = min(len(missing), batch_size)
        missing_list = list(missing)
        selected_missing = random.sample(missing_list, k=take_missing)

        # fill rest of the batch with random indices (could overlap)
        remaining_slots = batch_size - take_missing
        fill = []
        if remaining_slots > 0:
            fill = random.sample(range(n), k=min(remaining_slots, n))

        batch_indices = selected_missing + fill
        # dedupe indices within this batch
        batch_indices = sorted(set(batch_indices))

        batch_lines = [lines[idx] for idx in batch_indices]
        batch_name = f"{basename}_batch_extra_{extra_count}.jsonl"
        outpath = outdir_path / batch_name
        write_batch(batch_lines, outpath)
        created_files.append(str(outpath))

        # update covered & missing
        for idx in batch_indices:
            if idx in missing:
                missing.remove(idx)

    # Final manifest
    manifest = {
        "input": basename,
        "total_lines": n,
        "pct": pct,
        "min_batches_requested": min_batches,
        "batch_size_estimate": batch_size,
        "created_batches": [os.path.basename(p) for p in created_files],
    }
    manifest_path = outdir_path / f"{basename}_batches_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    return created_files, str(manifest_path)

def main():
    parser = argparse.ArgumentParser(description="Create random overlapping JSONL batches.")
    parser.add_argument("--input", "-i", required=True, help="Input .jsonl file")
    parser.add_argument("--outdir", "-o", required=True, help="Output directory for batches")
    parser.add_argument("--pct", type=float, default=30.0, help="Percent of dataset per batch (default 30)")
    parser.add_argument("--min-batches", type=int, default=10, help="Minimum number of batches (default 10)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = parser.parse_args()

    inpath = Path(args.input)
    if not inpath.exists():
        raise FileNotFoundError(f"Input file not found: {inpath}")

    lines = read_lines(inpath)
    basename = inpath.stem
    created_files, manifest_path = make_batches(lines, args.pct, args.min_batches, args.outdir, basename, seed=args.seed)

    print(f"Created {len(created_files)} batches in {args.outdir}")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
