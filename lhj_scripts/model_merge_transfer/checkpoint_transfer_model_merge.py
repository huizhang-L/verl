#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
convert_and_push_ckpt.py

对【单个】checkpoint 目录 (global_step_<n>) 执行：
  1) 对其下每个一级子目录运行 ALWAYS merge:
       python -m <merge_module> merge --backend <backend> --local_dir <subdir> --target_dir <subdir>/<hf_subdir>
  2) 复制整个 checkpoint 目录到目标父路径：
       - 若 dest-parent 以 tos:// 开头：用 tosutil cp -r
       - 否则：本地/网络盘路径下用 shutil.copytree 到 <dest-parent>/<ckpt.name>
  3) 递归字节和校验（tos 用挂载校验，本地用直接对比）
  4) 校验通过后（可选）删除本地源目录
"""

from __future__ import annotations
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

STEP_RE = re.compile(r"^global_step_(\d+)$")

def args_parser() -> argparse.Namespace:
    p = argparse.ArgumentParser("One-shot checkpoint converter & mover (always-merge)")
    p.add_argument("--ckpt-dir", type=Path, required=True,
                   help="目标 checkpoint 目录（形如 .../global_step_<n>）")
    p.add_argument("--dest-parent", type=str, required=True,
                   help="目标父路径：本地/共享盘路径 或 tos://bucket/prefix")
    p.add_argument("--merge-backend", default="fsdp", help="verl.model_merger 的 backend (默认: fsdp)")
    p.add_argument("--merge-module", default="verl.model_merger",
                   help="merge 所用的 Python 模块 (默认: verl.model_merger)")
    p.add_argument("--merge-hf-subdir", default="huggingface",
                   help="HF 输出子目录名，写入到每个 shard 目录下 (默认: huggingface)")
    p.add_argument("--tosutil", default="/fs-computility/llm_fudan/lvhuijie/tosutil",
                   help="tosutil 可执行文件路径（当 dest-parent 为 tos://* 时需要）")
    p.add_argument("--mount-base", default="/",
                   help="TOS 挂载根，用于校验字节和，比如 '/' -> /<bucket>/<key>")
    p.add_argument("--delete-local", action="store_true",
                   help="校验成功后删除本地源 checkpoint 目录")
    p.add_argument("--dry-run", action="store_true", help="仅打印动作，不实际执行")
    return p.parse_args()

# --------------------- 基础工具 ---------------------

def is_tos_url(url: str) -> bool:
    return urlparse(url).scheme == "tos"

def tos_url_to_local(url: str, mount_base: str) -> Path:
    pr = urlparse(url)
    if pr.scheme != "tos":
        raise ValueError(f"Not a tos:// URL: {url}")
    return Path(mount_base) / pr.netloc / pr.path.lstrip("/")

def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

def iter_immediate_subdirs(d: Path):
    for x in d.iterdir():
        if x.is_dir():
            yield x

# --------------------- Merge（总是执行） ---------------------

def run_merge_on_subdir(local_dir: Path, target_dir: Path, backend: str, module: str, dry: bool) -> bool:
    cmd = [
        sys.executable, "-m", module, "merge",
        "--backend", backend,
        "--local_dir", str(local_dir),
        "--target_dir", str(target_dir),
    ]
    print(f"[MERGE] {local_dir}  ->  {target_dir}")
    print(f"[CMD]   {' '.join(cmd)}")
    if dry:
        return True
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        cp = subprocess.run(cmd, check=True, text=True, capture_output=True)
        if cp.stdout:
            print(cp.stdout.rstrip())
        if cp.stderr:
            print(cp.stderr.rstrip())
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Merge failed: {local_dir}\nExit={e.returncode}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        return False

def merge_entire_checkpoint(src_ckpt: Path, hf_subdir: str, backend: str, module: str, dry: bool) -> bool:
    ok_all = True
    for subdir in iter_immediate_subdirs(src_ckpt):
        target_dir = subdir / hf_subdir
        if not run_merge_on_subdir(subdir, target_dir, backend, module, dry):
            ok_all = False
    return ok_all

# --------------------- 复制 & 校验 ---------------------

def copy_to_dest(src_ckpt: Path, dest_parent: str, tosutil: str, dry: bool) -> bool:
    if is_tos_url(dest_parent):
        cmd = [tosutil, "cp", str(src_ckpt), dest_parent, "-r"]
        print("[COPY tos]", " ".join(cmd))
        if dry:
            return True
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            print("[ERROR] tosutil failed:", res.stderr.decode(errors="ignore")[:300])
        return res.returncode == 0
    else:
        # 本地 / 共享盘
        dst = Path(dest_parent) / src_ckpt.name
        print(f"[COPY fs] {src_ckpt}  ->  {dst}")
        if dry:
            return True
        if dst.exists():
            # 保守：若已存在则先删（避免旧文件残留影响校验）
            shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src_ckpt, dst)
        return True

def verify_copy(src_ckpt: Path, dest_parent: str, mount_base: str) -> bool:
    if is_tos_url(dest_parent):
        dest_local = tos_url_to_local(f"{dest_parent.rstrip('/')}/{src_ckpt.name}", mount_base)
    else:
        dest_local = Path(dest_parent) / src_ckpt.name
    if not dest_local.exists():
        print("[VERIFY] dest path not found:", dest_local)
        return False
    try:
        s = dir_bytes(src_ckpt)
        d = dir_bytes(dest_local)
        ok = (s == d)
        print(f"[VERIFY] bytes src={s}  dst={d} -> {'OK' if ok else 'FAIL'}")
        return ok
    except Exception as e:
        print("[ERROR] verify failed:", e)
        return False

def delete_tree(path: Path, dry: bool):
    print("[DELETE]", path)
    if not dry:
        shutil.rmtree(path, ignore_errors=True)

# --------------------- 主流程 ---------------------

def main():
    a = args_parser()
    ckpt = a.ckpt_dir.resolve()
    if not ckpt.exists() or not ckpt.is_dir():
        print("[FATAL] ckpt-dir 不存在或不是目录：", ckpt)
        return 2

    # 可选：提醒名称规范，但不强制
    if not STEP_RE.match(ckpt.name):
        print(f"[WARN] 目录名看起来不是 global_step_<n>：{ckpt.name}")

    # 1) ALWAYS merge
    print("\n==> STEP 1/3: Always-merge per shard")
    ok_merge = merge_entire_checkpoint(
        src_ckpt=ckpt,
        hf_subdir=a.merge_hf_subdir,
        backend=a.merge_backend,
        module=a.merge_module,
        dry=a.dry_run,
    )
    if not ok_merge:
        print("[ABORT] 部分分片 merge 失败，终止。")
        return 3

    # 2) 复制
    print("\n==> STEP 2/3: Copy to dest-parent")
    ok_cp = copy_to_dest(ckpt, a.dest_parent, a.tosutil, a.dry_run)
    if not ok_cp:
        print("[ABORT] 复制失败，终止。")
        return 4

    # 3) 校验
    print("\n==> STEP 3/3: Verify by recursive byte sum")
    ok_v = True if a.dry_run else verify_copy(ckpt, a.dest_parent, a.mount_base)
    if not ok_v:
        print("[ABORT] 校验失败，保留本地目录以便排查。")
        return 5

    # 4) 可选删除
    if a.delete_local:
        delete_tree(ckpt, a.dry_run)

    print("\n[OK] 转换/转存完成。")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

    # python /fs-computility/llm_fudan/lvhuijie/my_git_repo/verl/lhj_scripts/model_merge_transfer/checkpoint_transfer_model_merge.py \
    # --ckpt-dir /fs-computility/llm_fudan/lvhuijie/my_git_repo/verl/checkpoints_verl/process_adv_2/qwen2.5-7b-instruct/dapo_math/DAPO/dapo-8k-train_bsz8-ppo_bsz64-rollout_n8-critique-reflection-nostepadv-nosteptopn/global_step_20 \
    # --dest-parent tos://tos-bjml-llm-fudan/lvhuijie/checkpoints_verl/process_adv_2/qwen2.5-7b-instruct/dapo_math/DAPO/dapo-8k-train_bsz8-ppo_bsz64-rollout_n8-critique-reflection-nostepadv-nosteptopn \
    # --delete-local

