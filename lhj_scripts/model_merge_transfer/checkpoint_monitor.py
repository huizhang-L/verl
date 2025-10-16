#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Periodic mover for VERL checkpoints (global_step_<n>) to TOS via tosutil.

Per run directory (the parent of global_step_*):
  - If any *.log ends with 'Success' (last non-empty line), move ALL checkpoints.
  - Else keep newest TWO locally; move the rest (only when there are ≥ 2 newer ones).

Copy & verify:
  - Copy: tosutil cp <src_dir> <tos://bucket/prefix/...> -r
  - Verify: map tos://... to local mounted path (--mount-base), compare recursive byte sums.

Dedup (no backoff):
  - State file marks a step dir as "done" once successfully copied+verified+deleted.
  - No retry delay; failures will be retried every scan.
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

STEP_RE = re.compile(r"^global_step_(\d+)$")

# --------------------------- CLI ---------------------------

def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser("Periodic checkpoint mover to TOS with dedup (no backoff)")
    p.add_argument("--source-root", type=Path, required=True,
                   help="Root to scan for global_step_* directories.")
    p.add_argument("--dest-root", type=str, required=True,
                   help="tos://bucket/prefix ... e.g. tos://tos-bjml-llm-fudan/lvhuijie/checkpoints_verl")
    p.add_argument("--tosutil", default="/fs-computility/llm_fudan/lvhuijie/tosutil",
                   help="Path to tosutil binary.")
    p.add_argument("--mount-base", default="/",
                   help="Local mount root for TOS buckets (for verification), e.g. '/' -> /<bucket>/<key>")
    p.add_argument("--interval", type=int, default=600,
                   help="Seconds between scans (default: 600)")
    p.add_argument("--log-success-text", default="Success",
                   help="The exact last non-empty line in *.log that means 'finished successfully'.")
    p.add_argument("--state-file", type=Path,
                   help="Path to state JSON (default: <source-root>/.ckpt_mover_state.json)")
    p.add_argument("--dry-run", action="store_true", help="Print actions without copying/deleting.")
    return p.parse_args()

# --------------------------- Utils ---------------------------

def tos_url_to_local(url: str, mount_base: str) -> Path:
    """Map tos://bucket/key -> <mount_base>/bucket/key"""
    pr = urlparse(url)
    if pr.scheme != "tos":
        raise ValueError(f"Not a tos:// URL: {url}")
    return Path(mount_base) / pr.netloc / pr.path.lstrip("/")

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
    """Group by parent directory (run directory) and sort steps desc."""
    runs: Dict[Path, List[Tuple[Path, int]]] = {}
    for p, s in steps:
        runs.setdefault(p.parent, []).append((p, s))
    for run_dir, lst in runs.items():
        lst.sort(key=lambda t: t[1], reverse=True)  # newest first
        yield run_dir, lst

def run_has_success_log(run_dir: Path, success_text: str) -> bool:
    """Return True if any *.log in run_dir ends with a line exactly equal to success_text."""
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

# --------------------------- State (dedup only) ---------------------------

def load_state(state_path: Path) -> Dict[str, Any]:
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

# --------------------------- tosutil ops ---------------------------

def copy_ckpt(tosutil: str, src: Path, dest_parent: str, dry: bool) -> bool:
    cmd = [tosutil, "cp", str(src), dest_parent, "-r"]
    print("[copy]", " ".join(cmd))
    if dry:
        return True
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print("[ERROR] tosutil failed:", res.stderr.decode()[:300])
    return res.returncode == 0

def verify_ckpt(src: Path, dest_parent: str, mount_base: str) -> bool:
    """Verify by comparing recursive byte sums: src vs mounted dest/<src.name>."""
    dest_local = tos_url_to_local(f"{dest_parent.rstrip('/')}/{src.name}", mount_base)
    if not dest_local.exists():
        return False
    try:
        s = dir_bytes(src)
        d = dir_bytes(dest_local)
        ok = (s == d)
        print(f"[verify] {src} -> {dest_local}  bytes src={s} dst={d} -> {'OK' if ok else 'FAIL'}")
        return ok
    except Exception as e:
        print("[ERROR] verify failed:", e)
        return False

def delete_tree(path: Path, dry: bool):
    print("[delete]", path)
    if dry:
        return
    shutil.rmtree(path, ignore_errors=True)

# --------------------------- Core ---------------------------

def dest_parent_for(src_ckpt: Path, source_root: Path, dest_root_url: str) -> str:
    """Return tos://.../<relative parent> (dest parent for tosutil cp)."""
    rel_parent = src_ckpt.parent.relative_to(source_root)
    return f"{dest_root_url.rstrip('/')}/{rel_parent.as_posix()}"

def process_run(run_dir: Path, lst: List[Tuple[Path, int]], args: argparse.Namespace,
                state: Dict[str, Any]) -> Tuple[int, int]:
    """Return (moved, failed) for this run."""
    moved = failed = 0
    move_all = run_has_success_log(run_dir, args.log_success_text)
    steps_str = ", ".join(str(s) for _, s in lst[:8]) + (" ..." if len(lst) > 8 else "")
    print(f"[run ] {run_dir} | steps(newest→oldest): {steps_str} | success-log={move_all}")

    if move_all:
        candidates = lst[:]  # move all
    else:
        candidates = lst[2:] if len(lst) > 2 else []

    if not candidates:
        print("[info] nothing to move.")
        return moved, failed

    src_root_resolved = args.source_root.resolve()

    for src_ckpt, step in candidates:
        key = str(src_ckpt.resolve())
        if should_skip(state, key):
            print(f"[skip] already done: {src_ckpt}")
            continue

        dest_parent = dest_parent_for(src_ckpt, src_root_resolved, args.dest_root)

        # If destination already present & verified OK, just delete source & mark done.
        if verify_ckpt(src_ckpt, dest_parent, args.mount_base):
            delete_tree(src_ckpt, args.dry_run)
            mark_done(state, key)
            moved += 1
            continue

        ok_cp = copy_ckpt(args.tosutil, src_ckpt, dest_parent, args.dry_run)
        if not ok_cp:
            print(f"[warn] copy failed: {src_ckpt}")
            failed += 1
            continue

        ok_v = True if args.dry_run else verify_ckpt(src_ckpt, dest_parent, args.mount_base)
        if not ok_v:
            print(f"[warn] verify failed: {src_ckpt}")
            failed += 1
            continue

        delete_tree(src_ckpt, args.dry_run)
        mark_done(state, key)
        moved += 1

    return moved, failed

def scan_once(args: argparse.Namespace, state: Dict[str, Any]) -> Tuple[int, int, int]:
    steps = iter_global_steps(args.source_root.resolve())
    if not steps:
        print("[INFO] No global_step_* found under", args.source_root)
        return 0, 0, 0

    total_runs = moved = failed = 0
    for run_dir, lst in group_by_run(steps):
        total_runs += 1
        m, f = process_run(run_dir, lst, args, state)
        moved += m
        failed += f
    return total_runs, moved, failed

def main():
    args = args_parser()
    source_root = args.source_root.resolve()
    if not source_root.exists():
        print("[FATAL] source-root not found:", source_root)
        return 2
    state_path = args.state_file or (source_root / ".ckpt_mover_state.json")

    while True:
        print("\n========== SCAN START ==========")
        state = load_state(state_path)
        total_runs, moved, failed = scan_once(args, state)
        save_state(state_path, state)
        print("========== SCAN DONE ==========")
        print(f"runs scanned     : {total_runs}")
        print(f"checkpoints moved: {moved}")
        print(f"failed           : {failed}")
        print(f"Sleeping {args.interval} seconds...\n")
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("Interrupted by user, exiting.")
            break

if __name__ == "__main__":
    raise SystemExit(main())


    # python /fs-computility/llm_fudan/lvhuijie/my_git_repo/verl/lhj_scripts/model_merge_transfer/checkpoint_monitor.py \
    # --source-root /fs-computility/llm_fudan/lvhuijie/my_git_repo/verl/checkpoints_verl \
    # --dest-root   tos://tos-bjml-llm-fudan/lvhuijie/checkpoints_verl \
    # --tosutil     /fs-computility/llm_fudan/lvhuijie/tosutil \
    # --mount-base  / \
    # --interval    1200
