#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Periodic mover for VERL checkpoints (global_step_<n>) to TOS via tosutil,
with format conversion using `verl.model_merger merge` on every run.

Per run directory (the parent of global_step_*):
  - If any *.log ends with 'Success' (last non-empty line), move ALL checkpoints.
  - Else keep newest TWO locally; move the rest (only when there are ≥ 2 newer ones).

Workflow per checkpoint to move:
  1) For each immediate subdir under global_step_<n>, run:
       python -m <merge_module> merge --backend <backend> --local_dir <subdir> --target_dir <subdir>/<hf_subdir>
     (ALWAYS run merge, even if <subdir>/<hf_subdir> exists & non-empty)
  2) tosutil cp <ckpt_dir> <tos://bucket/prefix/...> -r
  3) Verify by comparing recursive byte sums through a mounted path (--mount-base).
  4) Delete local checkpoint on success and mark done in state file.

Runs forever; scans every --interval seconds.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

STEP_RE = re.compile(r"^global_step_(\d+)$")

# --------------------------- CLI ---------------------------

def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser("Periodic checkpoint mover to TOS with format conversion (always-merge, no backoff)")
    # mover args
    p.add_argument("--source-root", type=Path, required=True,
                   help="Root to scan for global_step_* directories.")
    p.add_argument("--interval", type=int, default=600,
                   help="Seconds between scans (default: 600)")
    p.add_argument("--log-success-text", default="FINISHED",
                   help="The exact last non-empty line in *.log that means 'finished successfully'.")
    p.add_argument("--state-file", type=Path,
                   help="Path to state JSON (default: <source-root>/.ckpt_merge_state.json)")
    p.add_argument("--dry-run", action="store_true", help="Print actions without copying/deleting.")
    # merge args
    p.add_argument("--merge-backend", default="fsdp", help="merge backend for verl.model_merger (default: fsdp)")
    p.add_argument("--merge-module", default="verl.model_merger", help="Python module to run merge (default: verl.model_merger)")
    p.add_argument("--merge-hf-subdir", default="huggingface", help="subdir name under each shard to place merged HF model")
    # clean
    p.add_argument("--pattern", default="*.pt",
                   help="Glob pattern to delete (default: *.pt).")
    p.add_argument("--max-print", type=int, default=50,
                   help="Max paths to list per step in logs (default: 50).")
    return p.parse_args()

# --------------------------- Utils ---------------------------
def iter_global_steps(root: Path) -> List[Tuple[Path, int]]:
    """Find all global_step_<n> directories (return (path, n))."""
    out: List[Tuple[Path, int]] = []
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            m = STEP_RE.match(d)
            if m:
                out.append((Path(dirpath) / d, int(m.group(1))))
    return out

def group_by_run(steps: List[Tuple[Path, int]]):
    """
    Group by parent directory (run directory) and sort steps desc.
    这个函数把一组 (路径, 步数) 元组按父目录（run 目录）分组，并且在每个分组内按步数从大到小排序，然后逐组产出结果。
    """
    runs: Dict[Path, List[Tuple[Path, int]]] = {}
    for p, s in steps:
        runs.setdefault(p.parent, []).append((p, s))
    for run_dir, lst in runs.items():
        lst.sort(key=lambda t: t[1], reverse=True)  # newest first
        yield run_dir, lst

def run_has_success_log(run_dir: Path, success_text: str) -> bool:
    """
    Return True if any *.log in run_dir ends with a line exactly equal to success_text.
    判断一个实验是不是完全结束（通过该实验路径下的日志判断）"""
    logs = list(run_dir.glob("*.log"))
    if not logs:
        return False
    logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for log in logs:
        try:
            with log.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 65536))
                data = f.read().decode(errors="ignore")
            lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
            if lines and lines[-1] == success_text:
                return True
        except Exception:
            continue
    return False

def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

def iter_immediate_subdirs(d: Path):
    for entry in d.iterdir():
        if entry.is_dir():
            yield entry

# --------------------------- State (dedup only) ---------------------------

def load_state(state_path: Path) -> Dict[str, Any]:
    '''
    已经处理过的会信息会保存，避免重复处理
    '''
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text())
    except Exception:
        return {}

def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    tmp.replace(state_path)

def mark_done(state: Dict[str, Any], key: str) -> None:
    state[key] = {"status": "done", "ts": int(time.time())}

def should_skip(state: Dict[str, Any], key: str) -> bool:
    rec = state.get(key)
    return bool(rec and rec.get("status") == "done")

# ---------------------------  ---------------------------
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

# --------------------------- Merge ops (ALWAYS run) ---------------------------

def run_merge_on_subdir(local_dir: Path, target_dir: Path, backend: str, module: str, dry: bool) -> bool:
    """Always run: python -m <module> merge --backend ... --local_dir <local_dir> --target_dir <target_dir>."""
    cmd = [
        sys.executable, "-m", module, "merge",
        "--backend", backend,
        "--local_dir", str(local_dir),
        "--target_dir", str(target_dir),
    ]
    print(f"[MERGE] local_dir='{local_dir}' -> target_dir='{target_dir}'")
    print(f"[CMD]   {' '.join(cmd)}")

    if dry:
        return True

    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
        if completed.stdout:
            print(completed.stdout.rstrip())
        if completed.stderr:
            print(completed.stderr.rstrip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Merge failed for local_dir='{local_dir}'\n"
              f"Exit code: {e.returncode}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        return False

def merge_entire_checkpoint(src_ckpt: Path, hf_subdir: str, backend: str, module: str, dry: bool) -> bool:
    """Run merge for each immediate subdir under the checkpoint dir (ALWAYS)."""
    ok_all = True
    for subdir in iter_immediate_subdirs(src_ckpt):
        local_dir = subdir
        target_dir = subdir / hf_subdir
        ok = run_merge_on_subdir(local_dir, target_dir, backend, module, dry)
        if not ok:
            ok_all = False
    return ok_all

# --------------------------- Core ---------------------------
def process_run(run_dir: Path, lst: List[Tuple[Path, int]], args: argparse.Namespace,
                state: Dict[str, Any]) -> Tuple[int, int]:
    """Return (merged, failed) for this run."""
    merged = failed = 0
    # 判断一个实验是不是已经结束，如果已经结束的话，就需要把所有的 step 的都 merge
    merge_all = run_has_success_log(run_dir, args.log_success_text)
    steps_str = ", ".join(str(s) for _, s in lst[:8]) + (" ..." if len(lst) > 8 else "")
    print(f"[run ] {run_dir} | steps(newest→oldest): {steps_str} | success-log={merge_all}")

    # choose candidates
    if merge_all:
        candidates = lst[:]  # move all
    else:
        candidates = lst[2:] if len(lst) > 2 else []

    if not candidates:
        print("[info] nothing to move.")
        return merged, failed

    for src_ckpt, step in candidates:
        key = str(src_ckpt.resolve())
        if should_skip(state, key):
            print(f"[skip] already done: {src_ckpt}")
            continue

        # 1) ALWAYS merge before copy (so TOS gets converted artifacts too)
        print(f"[prep] merging checkpoint: {src_ckpt}")
        merged_ok = merge_entire_checkpoint(
            src_ckpt=src_ckpt,
            hf_subdir=args.merge_hf_subdir,
            backend=args.merge_backend,
            module=args.merge_module,
            dry=args.dry_run,
        )
        if not merged_ok:
            print(f"[warn] merge failed for {src_ckpt}; will retry next scan")
            failed += 1
            continue

        # 2) delete pt
        # delete_tree(src_ckpt, args.dry_run)
        delete_pattern_under(src_ckpt, args.pattern, args.dry_run, args.max_print)
        mark_done(state, key)
        merged += 1

    return merged, failed

def scan_once(args: argparse.Namespace, state: Dict[str, Any]) -> Tuple[int, int, int]:
    # 找到所有的steps
    steps = iter_global_steps(args.source_root.resolve())
    if not steps:
        print("[INFO] No global_step_* found under", args.source_root)
        return 0, 0, 0

    total_runs = merged = failed = 0
    for run_dir, lst in group_by_run(steps):
        # 按照不同的实验进行分组，一个实验内部会有不同步数的 checkpoint
        total_runs += 1
        # 结合state，对 该实验下，扫描到的所有步数的 checkpoint 进行处理
        m, f = process_run(run_dir, lst, args, state)
        merged += m
        failed += f
    return total_runs, merged, failed

def main():
    args = args_parser()
    source_root = args.source_root.resolve()
    if not source_root.exists():
        print("[FATAL] source-root not found:", source_root)
        return 2
    state_path = args.state_file or (source_root / ".ckpt_merge_state.json")

    while True:
        print("\n========== SCAN START ==========")
        state = load_state(state_path)
        # 根据跟路径和已经处理过的信息，继续进行扫描
        total_runs, merged, failed = scan_once(args, state)
        save_state(state_path, state)
        print("========== SCAN DONE ==========")
        print(f"runs scanned     : {total_runs}")
        print(f"checkpoints merged: {merged}")
        print(f"failed           : {failed}")
        print(f"Sleeping {args.interval} seconds...\n")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("Interrupted by user, exiting.")
            break

if __name__ == "__main__":
    raise SystemExit(main())

    # python /mnt/shared-storage-user/lvhuijie/my_git_repo/verl/lhj_scripts/model_merge_transfer/checkpoint_monitor_merge_cleanpt.py \
    # --source-root /mnt/ailab-llmfudan/lvhuijie/checkpoints_verl \
    # --interval 1200 \
    # --dry-run
    # python lhj_scripts/model_merge_transfer/checkpoint_monitor_merge_cleanpt.py --source-root /mnt/ailab-llmfudan/lvhuijie/checkpoints_verl --interval 1200 --state-file /mnt/shared-storage-user/lvhuijie/my_git_repo/verl/lhj_scripts/model_merge_transfer/ckpt_merge_state.json --dry-run