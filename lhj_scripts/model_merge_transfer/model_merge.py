#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import subprocess
from pathlib import Path
from shutil import which

def find_global_step_dirs(root: Path, pattern: str = "global_step_*"):
    """递归找到所有名为 global_step_* 的目录"""
    # 只匹配目录
    for p in root.rglob(pattern):
        if p.is_dir():
            yield p

def iter_immediate_subdirs(d: Path):
    """遍历目录 d 的直接子目录"""
    for entry in d.iterdir():
        if entry.is_dir():
            yield entry

def run_merge(local_dir: Path, target_dir: Path, backend: str = "fsdp", module: str = "verl.model_merger", dry_run: bool = False):
    """执行 python -m verl.model_merger merge ..."""
    cmd = [
        sys.executable, "-m", module, "merge",
        "--backend", backend,
        "--local_dir", str(local_dir),
        "--target_dir", str(target_dir),
    ]
    print(f"[MERGE] local_dir='{local_dir}'  ->  target_dir='{target_dir}'")
    print(f"[CMD]   {' '.join(cmd)}")

    if dry_run:
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        # 捕获输出便于调试；如日志太多可改为 None
        completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
        if completed.stdout:
            print(completed.stdout.rstrip())
        if completed.stderr:
            # 某些库会把info写到stderr，这里不直接当错误处理
            print(completed.stderr.rstrip())
        return completed.returncode
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Merge failed for local_dir='{local_dir}'\n"
              f"Exit code: {e.returncode}\n"
              f"STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        return e.returncode

def main():
    parser = argparse.ArgumentParser(description="批量对 global_step_* 目录下的子目录执行 verl.model_merger merge")
    parser.add_argument("root", type=Path, help="根目录（递归搜索 global_step_*）")
    parser.add_argument("--pattern", default="global_step_*", help="匹配 global_step 目录的模式（默认：global_step_*）")
    parser.add_argument("--backend", default="fsdp", help="merge 使用的 backend（默认：fsdp）")
    parser.add_argument("--module", default="verl.model_merger", help="merge 的 Python 模块（默认：verl.model_merger）")
    parser.add_argument("--hf-subdir", default="huggingface", help="目标目录子路径名称（默认：huggingface）")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令不执行")
    args = parser.parse_args()

    # 基础检查
    if which(sys.executable) is None:
        print(f"[FATAL] Python 不可用：{sys.executable}")
        sys.exit(2)

    root = args.root.resolve()
    if not root.exists():
        print(f"[FATAL] 根目录不存在：{root}")
        sys.exit(2)

    print(f"[INFO] 搜索根目录：{root}")
    print(f"[INFO] 匹配模式：{args.pattern}")

    total_steps = 0
    total_subdirs = 0
    failed = 0

    for gs_dir in find_global_step_dirs(root, args.pattern):
        total_steps += 1
        print(f"\n[STEP] 发现 global step 目录：{gs_dir}")

        for subdir in iter_immediate_subdirs(gs_dir):
            total_subdirs += 1
            local_dir = subdir
            target_dir = subdir / args.hf_subdir
            rc = run_merge(local_dir, target_dir, backend=args.backend, module=args.module, dry_run=args.dry_run)
            if rc != 0:
                failed += 1

    print("\n========== 总结 ==========")
    print(f"global_step_* 目录数量：{total_steps}")
    print(f"处理的子目录数量：{total_subdirs}")
    print(f"失败数量：{failed}")
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()
