#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transform_skyworkmath.py  (v2, add --sample_size)

Convert raw Skywork-Math JSON/JSONL to target schema → Parquet,
support (i) pre-sampling a fixed total size   (ii) multiple test splits.

Usage example
-------------
python transform_skyworkmath.py \
    --input raw.jsonl \
    --output_dir ./out \
    --sample_size 8000 \
    --test_sizes 100,500 \
    --seed 42
"""

import argparse
import json
import pathlib
import random
from typing import List

import pandas as pd


# --------------------------------------------------------------------------- #
def convert_sample(sample: dict) -> dict:
    """Map one raw record to the desired target-schema dictionary."""
    return {
        "data_source": f"skyworkmath_{sample['data_source']}",
        "prompt": [
            {
                "content": (
                    f"{sample['problem']}\n"
                    r"Please put your final answer within \boxed{}."
                ),
                "role": "user",
            }
        ],
        "ability": "math",
        "reward_model": {
            "ground_truth": sample["ground_truth"],
            "style": "rule",
        },
        "extra_info": {
            "problem": sample["problem"],
            "index": sample["index"],
            "model_difficulty": sample.get("model_difficulty", {}),
        },
    }


# --------------------------------------------------------------------------- #
def load_raw(path: pathlib.Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


# --------------------------------------------------------------------------- #
def split_and_dump(
    records: List[dict],
    test_size: int,
    out_dir: pathlib.Path,
    seed: int,
    total_tag: int,
):
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)

    test_records = shuffled[:test_size]
    train_records = shuffled[test_size:]

    tag = f"{total_tag}_{test_size}"
    for split, data in [("train", train_records), ("test", test_records)]:
        if not data:
            print(f"[WARN] skip empty split {split}_{tag}")
            continue
        df = pd.DataFrame(data)
        out_file = out_dir / f"{split}_{tag}.parquet"
        df.to_parquet(out_file, index=False)
        print(f"[OK] {split:5}  {len(df):6} rows → {out_file}")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser("SkyworkMath converter (Parquet)")
    p.add_argument("--input", required=True, type=pathlib.Path)
    p.add_argument("--output_dir", required=True, type=pathlib.Path)
    p.add_argument("--test_sizes", required=True,
                   help="comma-separated list, e.g. 100,500")
    p.add_argument("--sample_size", default=0, type=int,
                   help="pre-sample N rows before split; 0=all")
    p.add_argument("--seed", default=42, type=int)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(args.input)
    print(f"[LOAD] {len(raw)} raw samples")

    rng = random.Random(args.seed)
    if args.sample_size and args.sample_size < len(raw):
        raw = rng.sample(raw, args.sample_size)
        print(f"[SAMPLE] keep {len(raw)} randomly selected samples")
    total_tag = len(raw)

    converted = [convert_sample(r) for r in raw]

    for ts in args.test_sizes.split(","):
        ts_int = int(ts)
        if not 0 <= ts_int <= len(converted):
            raise ValueError(f"Illegal test size {ts_int}")
        split_and_dump(converted, ts_int, args.output_dir,
                       args.seed, total_tag)


if __name__ == "__main__":
    main()
