#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
clean_pt_in_steps.py

Scan a root directory, find all "global_step_<n>" directories, and recursively
delete all *.pt files under each step directory.

Usage:
  python clean_pt_in_steps.py --root /path/to/checkpoints --dry-run
  python clean_pt_in_steps.py --root /path/to/checkpoints --yes
"""

from __future__ import annotations
import argparse
import os
import re
from pathlib import Path
from typing import List, Tuple

STEP_RE = re.compile(r"^global_step_(\d+)$")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Delete *.pt files under global_step_<n> directories recursively.")
    p.add_argument("--root", type=Path, required=True,
                   help="Root directory to scan (will walk recursively).")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview what would be deleted, without actually deleting.")
    p.add_argument("--yes", action="store_true",
                   help="Skip confirmation prompt (dangerous).")
    p.add_argument("--pattern", default="*.pt",
                   help="Glob pattern to delete (default: *.pt).")
    p.add_argument("--max-print", type=int, default=50,
                   help="Max paths to list per step in logs (default: 50).")
    return p.parse_args()

def find_step_dirs(root: Path) -> List[Path]:
    steps: List[Path] = []
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if STEP_RE.match(d):
                steps.append(Path(dirpath) / d)
    return steps

def delete_pattern_under(step_dir: Path, pattern: str, dry_run: bool, max_print: int) -> Tuple[int, int]:
    files = [p for p in step_dir.rglob(pattern) if p.is_file()]
    total = len(files)
    shown = 0
    if total:
        print(f"[step] {step_dir} | candidates: {total}")
        for p in files:
            if shown < max_print:
                print(("DRY-RUN " if dry_run else "") + f"[delete] {p}")
                shown += 1
        if total > max_print:
            print(f"... and {total - max_print} more")
    deleted = 0
    if not dry_run:
        for p in files:
            try:
                p.unlink(missing_ok=True)
                deleted += 1
            except Exception as e:
                print(f"[WARN] failed to delete {p}: {e}")
    return total, deleted

def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        print(f"[FATAL] root not found: {root}")
        return 2

    steps = find_step_dirs(root)
    if not steps:
        print(f"[INFO] No global_step_* found under {root}")
        return 0

    # Summary & optional confirmation
    print(f"[INFO] Found {len(steps)} step dirs under {root}")
    if not args.dry_run and not args.yes:
        ans = input("Proceed to delete files? Type 'yes' to continue: ").strip().lower()
        if ans != "yes":
            print("Aborted.")
            return 1

    grand_total = grand_deleted = 0
    for step in sorted(steps):
        total, deleted = delete_pattern_under(step, args.pattern, args.dry_run, args.max_print)
        grand_total += total
        grand_deleted += deleted

    print("\n===== SUMMARY =====")
    print(f"step dirs scanned : {len(steps)}")
    print(f"files matched     : {grand_total}")
    if args.dry_run:
        print("deleted           : 0 (dry-run)")
    else:
        print(f"deleted           : {grand_deleted}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
    # python /fs-computility/llm_fudan/lvhuijie/my_git_repo/verl/lhj_scripts/model_merge_transfer/delete_extra_ptdata.py --root /fs-computility/llm_fudan/lvhuijie/my_git_repo/verl/checkpoints_verl --yes