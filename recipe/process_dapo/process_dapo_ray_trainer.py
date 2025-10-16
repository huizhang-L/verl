# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
import re
from typing import Dict, Any, Tuple, List
from tensordict import TensorDict
import os
import numpy as np
import torch
from tqdm import tqdm
from bisect import bisect_left
import json

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics_process_dapo,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.utils.profiler import marked_timer


class RayProcessDAPOTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def _get_step_position_info(self, data: DataProto, pattern: str ="\n\n") -> DataProto:
        """
        Compute a per-token step id for each response based on '\n\n' separators.
        Writes data.batch['response_step_ids'] of shape [B, R_max].
        - step 从 1 开始；每遇到一次 '\n\n'，其后的 token step+1
        - padding 与（可选）尾部 eos 位置为 0
        """
        responses = data.batch["responses"]               # [B, R_max]
        response_mask = data.batch["response_mask"]       # [B, R_max] （已在调用点前计算）
        B, R_max = responses.shape
        device = responses.device

        # 更小的 dtype 以节省显存；若后面要做索引运算再转 long 为了支持 all-gather 设置为int32 类型
        step_matrix = torch.zeros((B, R_max), dtype=torch.int32, device=device)

        is_fast = bool(getattr(self.tokenizer, "is_fast", False))
        eos_id  = getattr(self.tokenizer, "eos_token_id", None)
        eos_tok = getattr(self.tokenizer, "eos_token", None)

        for i in range(B):
            resp_ids_i = responses[i]                                   # [R_max]
            R = int(response_mask[i].sum().item())                      # 有效响应长度
            if R <= 0:
                continue

            # 取有效 ids；转 CPU list 供 tokenizer 使用
            valid_resp_ids = resp_ids_i[:R].to("cpu").tolist()

            # 解码文本（与 re-tokenize 一致）
            response_str = self.tokenizer.decode(valid_resp_ids, skip_special_tokens=True)

            # 对齐：如果文本末尾还有 eos 文本，去掉它，并在 R_fill 上做对应处理
            R_fill = R
            if eos_id is not None and R > 0 and resp_ids_i[R - 1].item() == int(eos_id):
                # 模型实际最后一个 token 是 eos；decode(skip_special_tokens=True) 可能已去掉文本中的 eos
                # 为保持 offsets 与字符串一致，这里只对前 R-1 个 token 赋 step
                R_fill = R - 1
            if R_fill <= 0:
                continue

            # 找到所有 '\n\n' 的起始字符位置 这里的 starts 和 pos 都是字符串中的索引
            starts, pos = [], 0
            while True:
                j = response_str.find(pattern, pos)
                if j == -1:
                    break
                starts.append(j)
                pos = j + len(pattern)

            # 优先用 fast tokenizer offsets
            char_starts = None
            if is_fast:
                enc = self.tokenizer(response_str, add_special_tokens=False, return_offsets_mapping=True)
                # 尝试与前 R_fill 个 token 对齐（有些 tokenizer 的空格归一化会导致对不上）
                ids_ok = len(enc.input_ids) >= R_fill and enc.input_ids[:R_fill] == valid_resp_ids[:R_fill]
                if ids_ok:
                    char_starts = [s for (s, e) in enc.offset_mapping[:R_fill]]

            # 回退：前缀解码（O(R^2)），但正确
            if char_starts is None:
                char_starts = []
                # 为减少反复 decode 的开销，可以累计字符串长度而不是每次 decode 全前缀
                # 但最稳妥的是直接 decode 前缀（这里保守实现）
                for t in range(R_fill):
                    prefix = self.tokenizer.decode(valid_resp_ids[:t], skip_special_tokens=True)
                    char_starts.append(len(prefix))

            # 字符起点 → step id；用 bisect_left：分隔符之后才进入下一 step
            step_vals = [1 + bisect_left(starts, cs) for cs in char_starts]
            
            # 写入矩阵；保持原 dtype，避免隐式类型转换
            if R_fill > 0:
                step_matrix[i, :R_fill] = torch.as_tensor(step_vals, dtype=step_matrix.dtype, device=device)
        
        # 挂到 batch，此处返回的是经过 token-level 的分步骤信息，每一次遇到步骤 pattern 之后的部分才算下一个步骤，最终以 tensor 形式返回
        data.batch["response_step_ids"] = step_matrix
        return data


    def _compute_step_position_entropy_metrics(
        self,
        batch: "DataProto",
        top_n: int = 8,
        prefix: str = "actor/entropy_steppos",
    ) -> dict[str, float]:
        """
        计算“每个 response step 的第 k 个 token（k=1..N）的 entropy 在整个 batch 上的平均值”，返回 N 个监控指标。
        - 输入:
            batch.batch["response_step_ids"]: [B, R]，0=padding，1..S 为step id
            batch.batch["response_mask"]:     [B, R]，0/1
            batch.batch["old_policy_entropy"] 或 batch.batch["entropys"]: [B, R] 的逐token熵
        - 输出:
            { f"{prefix}/{k}": mean_entropy_of_kth_token_over_all_steps,  k=1..N }
        若某个 k 在整个 batch 中不存在（所有 step 的长度都 < k），则对应值为 NaN，且不会报错。
        """
        td = batch.batch
        
        # 取熵矩阵（优先使用你保留的完整矩阵）
        if "old_policy_entropy" in td.keys():
            ent = td["old_policy_entropy"]
        elif "entropys" in td.keys():
            ent = td["entropys"]
        else:
            raise KeyError("Missing entropy tensor: expected 'old_policy_entropy' or 'entropys' in batch.batch.")

        steps = td["response_step_ids"]                  # [B, R]
        resp_mask = td["response_mask"]                  # [B, R]
        prior_response_mask = td.get("prior_response_mask", None)
        if prior_response_mask is not None:
            # 仅在 prior_response_mask==1 且 response_mask==1 的 token 上参与计算
            combined_mask = (resp_mask > 0) & (prior_response_mask > 0)  # bool [bs, T]
            combined_mask_f = combined_mask.to(torch.long)
        else:
            combined_mask = (resp_mask > 0)  # bool [bs, T]
            combined_mask_f = combined_mask.to(torch.long)
        # combined_mask = (resp_mask > 0) & (prior_response_mask > 0)  # bool [bs, T]
        # combined_mask_f = combined_mask.to(torch.long)
        B, R = ent.shape
        device = ent.device

        # 转换到便于比较的 dtype
        ent = ent.float()
        steps_long = steps.long()
        valid_mask = (combined_mask_f > 0) & (steps_long > 0)  # 只在有效token且属于某个step上取值

        # 汇总容器：对每个 k=1..N，累加和与计数
        sums = torch.zeros(top_n, dtype=torch.float64, device=device)
        counts = torch.zeros(top_n, dtype=torch.long, device=device)

        # 逐样本：对每个出现的 step id，取该 step 的前 k 个 token 的位置
        for i in range(B):
            if not valid_mask[i].any():
                continue

            steps_i = steps_long[i]
            mask_i = valid_mask[i]
            ent_i  = ent[i]

            # 该样本中，实际出现的 step ids（>0）且有有效token
            present_steps = torch.unique(steps_i[mask_i])
            for s in present_steps.tolist():
                pos = torch.nonzero((steps_i == s) & mask_i, as_tuple=False).squeeze(-1)  # 该step的所有有效位置（升序）
                if pos.numel() == 0:
                    continue

                # 该 step 的前 K 个 token（K <= top_n）
                K = min(top_n, int(pos.numel()))
                if K <= 0:
                    continue

                selected = pos[:K]
                # 累加到各自的 k 槽位（k=1..K）
                # k 索引（0..K-1）对应 selected 的第 k 个位置
                vals = ent_i[selected].to(sums.dtype)  # [K]
                sums[:K] += vals
                counts[:K] += 1
        
        metrics = {}
        for k in range(top_n):
            name = f"{prefix}/{k+1}"
            if counts[k].item() > 0:
                mean_k = (sums[k] / counts[k].to(sums.dtype)).item()
                metrics[name] = float(mean_k)
            else:
                metrics[name] = float("nan")  # 该 k 在整个 batch 中不存在

        return metrics


    def _get_word_entropies(
        self,
        batch,                          # DataProto-like: .batch / .non_tensor_batch
        tokenizer,
        *,
        entropy_key: str = "old_policy_entropy",          # [B, R_e]  token级熵
        responses_key: str = "responses",       # [B, R_max] response token ids（含pad）
        response_mask_key: str = "response_mask",  # [B, R_max] 有效=1, 无效=0
        offset_mapping_info_key: str = "offset_mapping_info",
    ) -> None:
        """
        用 response_mask 裁切有效的 response token，计算每个样本“词 → 该词对应token的熵列表”，
        并存入 non_tensor_batch（np.array(..., dtype=object)）。
        - 一个词由多个token组成 → 列表里有多个熵
        - 一个token跨多个词（少见） → 该token的熵会出现在多个词的列表里
        """
        
        ent_mat: torch.Tensor = batch.batch[entropy_key]         # [B, R_e]
        responses: torch.Tensor = batch.batch[responses_key]     # [B, R_max]
        resp_mask: torch.Tensor = batch.batch[response_mask_key] # [B, R_max]

        B = responses.size(0)
        R_max = responses.size(1)
        R_e = ent_mat.size(1)

        is_fast = bool(getattr(tokenizer, "is_fast", False))

        # all_words: List[List[tuple]] = []
        all_offset_mapping_info: List[List[tuple]] = []

        for i in range(B):
            # 取该样本有效长度（通常 mask 是左对齐的 1...1 0...0）
            L_mask = int(resp_mask[i].sum().item())
            if L_mask <= 0:
                all_words.append([])
                all_word_ents.append([])
                continue

            # 与 entropy/ids 实际长度取 min，防御性处理不同长度
            L = min(L_mask, R_max, R_e) - 1

            resp_ids_i: torch.Tensor = responses[i, :L].detach().cpu()
            ent_vec_i: torch.Tensor  = ent_mat[i, :L].detach().cpu()

            # 解码有效的 response 字符串
            response_str: str = tokenizer.decode(resp_ids_i.tolist(), skip_special_tokens=True)

            # === 计算每个 token 在 response_str 中的字符起止位置 ===
            token_starts: List[int] = []
            token_ends:   List[int] = []

            if is_fast:
                enc = tokenizer(response_str, add_special_tokens=False, return_offsets_mapping=True)
                ids_ok = (len(enc.input_ids) >= L and enc.input_ids[:L] == resp_ids_i.tolist())
                if ids_ok:
                    for (s, e) in enc.offset_mapping[:L]:
                        token_starts.append(int(s)); token_ends.append(int(e))
                else:
                    # 回退：用前缀解码法（O(L^2) 但最稳）
                    token_starts = []
                    for t in range(L):
                        prefix = tokenizer.decode(resp_ids_i[:t].tolist(), skip_special_tokens=True)
                        token_starts.append(len(prefix))
                    token_ends = token_starts[1:] + [len(response_str)]
            else:
                token_starts = []
                for t in range(L):
                    prefix = tokenizer.decode(resp_ids_i[:t].tolist(), skip_special_tokens=True)
                    token_starts.append(len(prefix))
                token_ends = token_starts[1:] + [len(response_str)]
            
            assert len(token_starts) == len(token_ends), "starts/ends 长度不一致"
            offset_mapping_info = [(int(s), int(e)) for s, e in zip(token_starts, token_ends)]
            
            all_offset_mapping_info.append(offset_mapping_info)
            # # === 按 \S+ 切词（你可替换为更合适的分词规则）===
            # words: List[str] = []
            # word_spans: List[Tuple[int, int]] = []
            # for m in re.finditer(r"\S+", response_str):
            #     words.append(m.group(0))
            #     word_spans.append((m.start(), m.end()))

            # # === token 熵映射到词：区间相交则归属 ===
            # ent_list = ent_vec_i.tolist()
            # word_entropy_lists: List[List[float]] = []
            # for (ws, we) in word_spans:
            #     acc = []
            #     for t, (ts, te) in enumerate(zip(token_starts, token_ends)):
            #         if ts < we and te > ws:  # 有交集
            #             acc.append(ent_list[t])
            #     word_entropy_lists.append(acc)
            
            # all_words.append(words)
            # all_word_ents.append(word_entropy_lists)

        # 存入 non_tensor_batch（变长 -> dtype=object，避免 numpy 2.0 的 ragged 报错）
        # batch.non_tensor_batch[store_words_key] = np.array(all_words, dtype=object)
        batch.non_tensor_batch[offset_mapping_info_key] = np.array(all_offset_mapping_info, dtype=object)


    def _to_jsonable(self, x: Any):
        """将任意常见科学计算对象转换为 JSON 可序列化类型（递归）。
        规则：
        - torch.Tensor: 标量->item()；否则->cpu().tolist()
        - np.ndarray: tolist()
        - np.generic: item()
        - dict/list/tuple/set: 递归处理
        - 其它非常见类型: 转成 str(x) 兜底
        """
        # torch tensor
        if torch.is_tensor(x):
            if x.numel() == 1:
                return x.item()
            return x.detach().cpu().tolist()

        # numpy array / numpy scalar
        if isinstance(x, np.ndarray):
            # 注意 object 数组也能 tolist()，递归继续处理
            return [self._to_jsonable(e) for e in x.tolist()]
        if isinstance(x, np.generic):  # e.g. np.int64, np.float32
            return x.item()

        # 基本容器（递归）
        if isinstance(x, dict):
            return {k: self._to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [self._to_jsonable(e) for e in x]
        if isinstance(x, set):
            return [self._to_jsonable(e) for e in x]

        # bytes/bytearray 可按需定制；这里用 str 兜底
        if isinstance(x, (bytes, bytearray)):
            # 你也可以选择 base64.b64encode(x).decode('ascii')
            # 这里用可读性更好的 repr 形式
            try:
                return x.decode("utf-8")
            except Exception:
                return repr(x)

        # 基本类型（int/float/bool/str/None）原样返回
        if isinstance(x, (int, float, bool, str)) or x is None:
            return x

        # 其他非常见类型统一转成字符串避免崩溃
        return str(x)

    def _dump_generations(self, batch, inputs, outputs, scores, advantages, reward_extra_infos_dict, dump_path, entropies=None, offset_mapping_info=None):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "global_step": [self.global_steps] * n,
        }

        # 如果传入了 token 级熵，确保与样本数一致后加入
        if entropies is not None:
            assert len(entropies) == n, f"len(entropies)={len(entropies)} != n={n}"
            # 每个元素是一条样本对应的 list[float]
            base_data["token_entropies"] = entropies
        if advantages is not None:
            assert len(advantages) == n, f"len(advantages)={len(advantages)} != n={n}"
            # 每个元素是一条样本对应的 list[float]
            base_data["advantages"] = advantages
        
        if offset_mapping_info is not None:
            assert len(offset_mapping_info) == n, f"len(offset_mapping_info)={len(offset_mapping_info)} != n={n}"
            # 每个元素是一条样本对应的 list[float]
            base_data["offset_mapping_info"] = offset_mapping_info
        # if response_word_list is not None:
        #     # assert len(entropies) == n, f"len(entropies)={len(entropies)} != n={n}"
        #     # 每个元素是一条样本对应的 list[float]
        #     base_data["response_word_list"] = response_word_list
        # 合并额外信息（长度必须与 n 一致）
        for k, v in reward_extra_infos_dict.items():
            v = batch.non_tensor_batch[k]
            if len(v) == n:
                base_data[k] = v

        # 逐行写 JSONL
        lines = []
        for i in range(n):
            # 按原逻辑取第 i 条
            entry_raw = {k: v[i] for k, v in base_data.items()}

            # 先尝试序列化（快速路径）
            try:
                entry = self._to_jsonable(entry_raw)
                lines.append(json.dumps(entry, ensure_ascii=False))
                continue
            except Exception:
                # 如果仍失败，逐字段定位并兜底
                bad_keys = []
                for k, val in entry_raw.items():
                    try:
                        json.dumps(self._to_jsonable({k: val}), ensure_ascii=False)
                    except Exception:
                        bad_keys.append((k, type(val)))
                # 打印一下问题键，便于排查
                if bad_keys:
                    print(
                        "[WARN] JSON serialization failed at keys: "
                        + ", ".join(f"{k}({t.__name__})" for k, t in bad_keys)
                    )
                # 兜底：将整个 entry_raw 做转换再序列化
                entry = self._to_jsonable(entry_raw)
                lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _strip_left_pad_1d(self, ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        """去掉左侧 pad，返回有效 token（1D）。"""
        assert ids.ndim == 1
        if pad_id is None:
            return ids
        non_pad = (ids != pad_id).nonzero(as_tuple=False).squeeze(-1)
        if non_pad.numel() == 0:
            return ids.new_empty((0,), dtype=ids.dtype)
        first = int(non_pad[0].item())
        return ids[first:]

    def _encode_reflection(self, reflection: str, tokenizer) -> List[int]:
        txt = reflection or ""
        return tokenizer.encode(txt, add_special_tokens=False)

    def _build_pos_ids_from_mask(self, attn_mask: torch.Tensor) -> torch.Tensor:
        """简单 position_ids：对每行有效位（1）做累计（从 0 开始）。"""
        pos = torch.cumsum(attn_mask.to(torch.long), dim=-1) - 1
        return torch.clamp(pos, min=0)

    def _pad_to_max_length(
        self,
        ids_1d: torch.Tensor,
        attn_1d: torch.Tensor,
        max_length: int,
        pad_token_id: int,
        left_pad: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """把 1D 的 ids/attn 按指定方向 pad 到 max_length；超长保尾部。"""
        L = ids_1d.numel()
        if L >= max_length:
            return ids_1d[-max_length:], attn_1d[-max_length:]

        pad_len = max_length - L
        if left_pad:
            ids_new  = torch.cat([torch.full((pad_len,), pad_token_id, dtype=ids_1d.dtype, device=ids_1d.device), ids_1d], dim=-1)
            attn_new = torch.cat([torch.zeros(pad_len, dtype=attn_1d.dtype, device=attn_1d.device),                     attn_1d], dim=-1)
        else:
            ids_new  = torch.cat([ids_1d,  torch.full((pad_len,), pad_token_id, dtype=ids_1d.dtype, device=ids_1d.device)], dim=-1)
            attn_new = torch.cat([attn_1d, torch.zeros(pad_len, dtype=attn_1d.dtype, device=attn_1d.device)], dim=-1)
        return ids_new, attn_new

    def _insert_reflection_and_rebuild(
        self,
        err_batch,                       # DataProto：含 .batch (TensorDict) 与 .non_tensor_batch
        *,
        tokenizer,                       # HF tokenizer
        pad_token_id: int,
        response_length: int,
        prompt_length: int,
        left_pad: bool = True,
    ):
        """
        对 batch 中每条样本：
        1) 找到 process_step_critique 的首个 0（首个出错 step，对齐到 response_step_ids 的 step_id=1..K）；
        2) 仅保留该错误 step（含）之前的 response token，拼接到原 prompt 后，再插入 reflection；
        3) 以此新上下文重建 input_ids / attention_mask / position_ids（按 max_length pad/截断）；
        4) 计算新的 prior_response_mask（形状与 response_mask 相同）：
            - 将“错误 step（含）之前的 response token”置为 1，其余为 0。
            - 注意：reflection 是上下文一部分，不在 response_mask 序列内，因此不在 prior_response_mask 中置位。
        5) 将未 pad 的新上下文 token 序列保存到 non_tensor_batch["raw_prompt_ids"]（list[list[int]] / np.object）。

        返回：原地修改后的 err_batch
        """
        # 计算/补充 response_mask、response_step_ids 等（项目已有工具）
        err_batch.batch["response_mask"] = compute_response_mask(err_batch)
        err_batch = self._get_step_position_info(err_batch)

        batch = err_batch.batch
        meta  = err_batch.non_tensor_batch

        prompts: torch.Tensor        = batch["prompts"]            # [B, P] 左 pad
        responses: torch.Tensor      = batch["responses"]          # [B, R]
        resp_step_ids: torch.Tensor  = batch["response_step_ids"]  # [B, R]  0=非响应, 1..K=第k步
        response_mask: torch.Tensor  = batch["response_mask"]      # [B, R]  1=有效 response token

        B, P = prompts.shape
        _, R = responses.shape
        device = prompts.device
        dtype_ids = prompts.dtype

        # 从 meta 取 per-sample 字段的小工具
        def _get_meta_per_sample(meta_val, i: int, default=None):
            if meta_val is None:
                return default
            if isinstance(meta_val, np.ndarray):
                return meta_val[i]
            if isinstance(meta_val, (list, tuple)):
                if len(meta_val) == B:
                    return meta_val[i]
                if len(meta_val) == 1:
                    return meta_val[0]
            return meta_val

        # prior_response_mask 形状需与 response_mask 一致
        new_prior_response_mask = torch.ones_like(response_mask, dtype=torch.long, device=device)

        # raw_prompt_ids 批量收集
        err_raw_prompt_ids_list = []
        no_err_index_list = []
        err_index_list = []

        psc_all  = meta.get("process_step_critique", None)
        refl_all = meta.get("reflection", "")

        for i in range(B):
            # 1) 取第 i 个样本的 step 判定并规整为 list[int]
            psc_i = _get_meta_per_sample(psc_all, i)
            if isinstance(psc_i, np.ndarray):
                if psc_i.ndim == 0:
                    psc_i = [int(psc_i.item())]
                else:
                    psc_i = [int(v) for v in psc_i.tolist()]
            elif isinstance(psc_i, (list, tuple)):
                psc_i = [int(v) for v in psc_i]
            else:
                psc_i = [int(psc_i)]
            
            prompts_i       = prompts[i]         # [P]
            # 4) prompt 去左 pad
            prompt_clean_i = self._strip_left_pad_1d(prompts_i, pad_token_id)
            if any(v == 0 for v in psc_i):
                # assert any(v == 0 for v in psc_i), f"[sample {i}] process_step_critique 必须至少含一个 0"
                first_err_step_idx0 = next(j for j, v in enumerate(psc_i) if v == 0)
                target_sid = first_err_step_idx0 + 1  # 1-based，对齐 response_step_ids

                # 2) 当前样本张量
                
                responses_i     = responses[i]       # [R]
                resp_step_ids_i = resp_step_ids[i]   # [R]
                resp_mask_i     = (response_mask[i] > 0)  # [R] bool

                # 3) 仅考虑有效 response token（step_id>0 且 response_mask==1）
                valid_idx = ((resp_step_ids_i > 0) & resp_mask_i).nonzero(as_tuple=False).squeeze(-1)  # [Nv]
                kept_resp_i = responses_i.new_empty((0,), dtype=responses_i.dtype)
                kept_resp_indices = responses_i.new_empty((0,), dtype=torch.long)

                if valid_idx.numel() > 0:
                    sid_in_valid = resp_step_ids_i[valid_idx]  # [Nv]
                    hit = (sid_in_valid == target_sid).nonzero(as_tuple=False).squeeze(-1)

                    if hit.numel() > 0:
                        # 到该错误 step 的最后一个 token（含）
                        last_pos_in_valid = valid_idx[int(hit[-1].item())]  # 标量 idx（在 [0..R-1]）
                        kept_mask_in_all = (valid_idx <= last_pos_in_valid)  # [Nv] bool（按原索引阈值）
                        kept_resp_indices = valid_idx[kept_mask_in_all]      # [K]
                        kept_resp_i = responses_i[: int(last_pos_in_valid.item()) + 1]
                    else:
                        # 错误 step 没 token：保留到第一个 > target_sid 的 token 之前；若不存在则保留所有有效响应
                        gt = (sid_in_valid > target_sid).nonzero(as_tuple=False).squeeze(-1)
                        if gt.numel() > 0:
                            first_gt_pos_in_valid = valid_idx[int(gt[0].item())]   # 标量 idx
                            kept_mask_in_all = (valid_idx < first_gt_pos_in_valid)
                            kept_resp_indices = valid_idx[kept_mask_in_all]
                            kept_resp_i = responses_i[: max(0, int(first_gt_pos_in_valid.item()))]
                        else:
                            kept_resp_indices = valid_idx
                            kept_resp_i = responses_i[valid_idx]


                # 5) reflection -> ids
                reflection_i = _get_meta_per_sample(refl_all, i, default="") or ""
                ref_ids_i = self._encode_reflection(reflection_i, tokenizer)
                ref_ids_i = (torch.tensor(ref_ids_i, dtype=dtype_ids, device=device)
                            if len(ref_ids_i) > 0 else prompts_i.new_empty((0,), dtype=dtype_ids))

                # 6) 拼接新上下文并生成新三件套
                new_ctx_ids_i = torch.cat([prompt_clean_i, kept_resp_i, ref_ids_i], dim=-1)     # [ctx_len]
                # new_attn_i    = torch.ones_like(new_ctx_ids_i, dtype=torch.long, device=device) # [ctx_len]

                # ids_pad_i, attn_pad_i = self._pad_to_max_length(
                #     new_ctx_ids_i, new_attn_i, max_length=response_length, pad_token_id=pad_token_id, left_pad=left_pad
                # )
                # pos_i = self._build_pos_ids_from_mask(attn_pad_i)

                # new_input_ids[i]      = ids_pad_i
                # new_attention_mask[i] = attn_pad_i
                # new_position_ids[i]   = pos_i

                # 7) 生成 prior_response_mask（与 response_mask 同形），仅把“错误 step（含）之前”的 response token 置 1
                if kept_resp_indices.numel() > 0:
                    new_prior_response_mask[i, kept_resp_indices] = 0  # 其余保持 0

                # 8) 记录未 pad 的新上下文 token 序列（供 rollout 引擎作为 prompt_token_ids 使用）
                err_raw_prompt_ids_list.append(new_ctx_ids_i.detach().cpu().tolist())
                err_index_list.append(i)
            else:
                err_raw_prompt_ids_list.append(prompt_clean_i.detach().cpu().tolist())
                no_err_index_list.append(i)
                new_prior_response_mask[i, :] = 0

        err_batch.batch["prior_response_mask"] = new_prior_response_mask  # 精确版本（不再用 attention_mask 克隆）

        # 保存 raw_prompt_ids
        try:
            err_batch.non_tensor_batch["raw_prompt_ids"] = np.array(err_raw_prompt_ids_list, dtype=object)
        except Exception:
            err_batch.non_tensor_batch["raw_prompt_ids"] = err_raw_prompt_ids_list
        
        # new_err_batch = err_batch[err_index_list]
        new_err_batch = err_batch
        new_no_err_batch = err_batch[no_err_index_list]

        # 结果承载
        new_attention_mask: torch.Tensor  = new_err_batch.batch["attention_mask"]  # [B, R]  0=非响应, 1..K=第k步
        new_position_ids: torch.Tensor  = new_err_batch.batch["position_ids"]      # [B, R]  1=有效 response token
        new_prompts:torch.Tensor  = new_err_batch.batch["prompts"]
        new_err_batch.batch["input_ids"] = new_prompts[..., :prompt_length]
        new_err_batch.batch["attention_mask"] = new_attention_mask[..., :prompt_length]
        new_err_batch.batch["position_ids"] = new_position_ids[..., :prompt_length]

        return new_err_batch, new_no_err_batch, err_index_list


    def _prepare_err_gen_batch(
        self,
        next_batch,                         # DataProto：含 .batch (TensorDict) 与 .non_tensor_batch
        *,
        tokenizer,
        pad_token_id: int,
        response_length: int,
        prompt_length: int,
        left_pad: bool = True,
    ):
        """
        基于 _insert_reflection_and_rebuild 的结果，产出用于 rollout 的 err_gen_batch（DataProto）：
        - err_gen_batch.batch 仅包含：input_ids / attention_mask / position_ids（均为“插入 reflection 后”的新上下文）
        - err_gen_batch.non_tensor_batch 仅包含：raw_prompt_ids（未 pad 的新上下文 ids）
        同时：
        - 在 next_batch.batch 中设置：prior_response_mask（按“错误 step（含）之前”为 1，其余 0）
        - 从 next_batch.batch 中移除 prompts / responses（避免后续干扰）
        返回：(next_batch, err_gen_batch)
        """
        # 先重建（会在 next_batch 中就地写入 input_ids/attention_mask/position_ids/prior_response_mask/raw_prompt_ids）
        new_err_batch, new_no_err_batch, err_index_list = self._insert_reflection_and_rebuild(
            next_batch,
            tokenizer=tokenizer,
            pad_token_id=pad_token_id,
            response_length=response_length,
            prompt_length=prompt_length,
            left_pad=left_pad,
        )

        # 用 pop 构造“只包含三件套 + raw_prompt_ids”的 DataProto（满足 rollout 输入格式）
        err_gen_batch = new_err_batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids", "uid"],
        )

        # 从原 batch 中移除 prompts / responses（你后续还有其它计算，不再需要它们）
        td = new_no_err_batch.batch
        for k in ("response_mask", "response_step_ids"):
            if k in td.keys():
                td.pop(k)
        td_non = new_no_err_batch.non_tensor_batch
        for k in ("seq_outcome_reward", "raw_prompt_ids"):
            if k in td_non.keys():
                td_non.pop(k)
        sd = new_err_batch.batch
        for k in ("response_mask", "responses", "token_level_rewards", "token_level_scores", "response_step_ids"):
            if k in sd.keys():
                sd.pop(k)
        sd_non = new_err_batch.non_tensor_batch
        for k in ("overall_score", "outcome_score", "process_critique_raw", "process_critique_parsed", "process_step_critique", "next_step", "reflection", "seq_outcome_reward"):
            if k in sd_non.keys():
                sd_non.pop(k)

        prior_response_mask = new_err_batch.pop(
            batch_keys=["prior_response_mask"],
        )

        return prior_response_mask, err_gen_batch, new_err_batch, err_index_list, new_no_err_batch


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        next_batch = None
        next_completion_batch = None
        num_prompt_in_next_batch = 0
        num_traj_in_next_batch = 0
        num_gen_batches = 0
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                # with marked_timer("start_profile", timing_raw):
                if do_profile:
                    self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
                    if self.use_reference_policy:
                        self.ref_policy_wg.start_profile()
                    if self.use_critic:
                        self.critic_wg.start_profile()
                    if self.use_rm:
                        self.rm_wg.start_profile()
            
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if "multi_modal_data" in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                # breakpoint()
                is_last_step = self.gen_steps >= self.total_training_steps
                
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)
                          
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)
                    
                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        try:
                            reward_result = self.reward_fn(new_batch, return_dict=True)
                            # 这里的 tensor 实质上就是总分数赋予每一个 token
                            reward_tensor = reward_result["reward_tensor"]
                            # 单独的 outcome 和 process 的分数放在这里
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
                        except Exception as e:
                            print(f"Error in reward_fn: {e}")
                            reward_tensor = self.reward_fn(new_batch)
                            reward_extra_infos_dict = {}

                        # 综合考虑 outcome 和 process reward 之后的
                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v, dtype=object) for k, v in reward_extra_infos_dict.items()}
                            )
                        
                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            # 由于没有使用 kl 散度，因此 token_level_rewards 跟 token_level_scores 没有区别
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        assert self.config.custom_reward_function.reflection.enable == True, "只有引入 reflection 才支持积累batch"
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_outcome_reward":
                            new_batch.non_tensor_batch["seq_outcome_reward"] = (
                                new_batch.non_tensor_batch["next_step"]
                            )

                            responses = new_batch.batch["responses"]
                            response_length = responses.size(1)
                            attention_mask = new_batch.batch["attention_mask"]
                            new_batch.batch["prior_response_mask"] = torch.ones_like(attention_mask[:, -response_length:])                         
                        
                        err_batch_prior = None
                        if batch is None:
                            batch = next_completion_batch
                            next_completion_batch = None

                            if batch is not None and len(batch) > 0: 
                                batch.non_tensor_batch["seq_outcome_reward"] = (
                                    batch.non_tensor_batch["next_step"]
                                )
                                prompt_uid2metric_vals_prior = defaultdict(list)
                                for uid, metric_val in zip(
                                    batch.non_tensor_batch["uid"], batch.non_tensor_batch[metric_name], strict=True
                                ):
                                    prompt_uid2metric_vals_prior[uid].append(metric_val)
                                
                                err_prompt_uids_prior = [
                                    uid
                                    for uid, val in prompt_uid2metric_vals_prior.items()
                                    if len(val) > 0 and (sum(1 for v in val if v == 1) >= 0.3 * len(val)) # 只有错误超过一半的问题才会放到下个批次进行优化
                                ]
                                num_prompt_in_next_batch += len(err_prompt_uids_prior)

                                err_traj_idxs_prior = []
                                prior_masks_prior = batch.batch["prior_response_mask"]  # [B, T] 张量
                                for idx, (traj_from_prompt_uid, mask_i) in enumerate(zip(batch.non_tensor_batch["uid"], prior_masks_prior)):
                                    if traj_from_prompt_uid in err_prompt_uids_prior:
                                        # 兼容 bool / 整型 / 浮点型 mask
                                        if mask_i.dtype == torch.bool:
                                            all_one = mask_i.all().item()
                                        else:
                                            all_one = torch.all(mask_i == 1).item()
                                        if all_one:
                                            err_traj_idxs_prior.append(idx)
                                # breakpoint()
                                err_batch_prior = batch[err_traj_idxs_prior]

                        # 这个  gen_batch 的数据 合并到当前 优化 batch 中
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])
                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        # prompt_uid2metric_std = {}
                        # for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                        #     prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                        err_prompt_uids = [
                            uid
                            for uid, val in prompt_uid2metric_vals.items()
                            if len(val) > 0 and (sum(1 for v in val if v == 1) >= 0.1 * len(val)) # 只有错误超过一半的问题才会放到下个批次进行优化
                        ]
                        num_prompt_in_next_batch += len(err_prompt_uids)
                        
                        err_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in err_prompt_uids:
                                err_traj_idxs.append(idx)
                        
                        # for idx, traj_outcome_reward in enumerate(new_batch.non_tensor_batch[metric_name]):
                        #     if traj_outcome_reward == 1:
                        #         err_traj_idxs.append(idx)
                        num_traj_in_next_batch += len(err_traj_idxs)

                        err_batch = new_batch[err_traj_idxs]
                        if err_batch_prior is not None:
                            err_batch =  DataProto.concat([err_batch_prior, err_batch])
                        next_batch = err_batch if next_batch is None else DataProto.concat([next_batch, err_batch])

                        # 现在已经对当前 gen_batch 的数据做好分类了
                        traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                        if len(batch) < traj_bsz:
                            print(f"当前 batch 数{len(batch)} < {traj_bsz}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === Updating ===

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # breakpoint()
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # step 矩阵 batch 数据中每个 step 的位置信息，为后续基于 step 的分析做准备
                    batch = self._get_step_position_info(batch)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, "blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        # breakpoint()
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        prior_response_mask = batch.batch.get("prior_response_mask", None)
                        if prior_response_mask is not None:
                            # 仅在 prior_response_mask==1 且 response_mask==1 的 token 上参与计算
                            combined_mask = (response_masks > 0) & (prior_response_mask > 0)  # bool [bs, T]
                            combined_mask_f = combined_mask.to(torch.long)
                        else:
                            combined_mask = (response_masks > 0)  # bool [bs, T]
                            combined_mask_f = combined_mask.to(torch.long)
                        # combined_mask = (response_masks > 0) & (prior_response_mask > 0)  # bool [bs, T]
                        # combined_mask_f = combined_mask.to(torch.long)
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=combined_mask_f, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)

                        # 保存完整的 entropy 矩阵到 batch 中，用于后续基于 step 的分析
                        batch.batch["old_policy_entropy"] = entropys
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    # recompute old_log_probs
                    with marked_timer("log_old_log_prob", timing_raw, "blue"):
                        step_token_entropy_metrics = self._compute_step_position_entropy_metrics(batch)
                        self._get_word_entropies(batch, self.tokenizer)
                        metrics.update(step_token_entropy_metrics)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, "olive"):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            coef_bad_step_if_adv_pos=self.config.actor_rollout_ref.actor.process_grpo_adv.coef_bad_step_if_adv_pos,
                            coef_good_step_if_adv_neg=self.config.actor_rollout_ref.actor.process_grpo_adv.coef_good_step_if_adv_neg,
                            top_n=self.config.actor_rollout_ref.actor.process_grpo_adv.top_n,
                            topn_mode=self.config.actor_rollout_ref.actor.process_grpo_adv.topn_mode,
                            topn_scale=self.config.actor_rollout_ref.actor.process_grpo_adv.topn_scale
                        )
                    
                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            advantages = batch.batch["advantages"].detach().cpu().tolist()
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )
                            # === 构造每条样本对应的 token-level entropy 列表 ===
                            # 优先使用你之前保存在 batch 里的完整熵矩阵 'old_policy_entropy'；否则退回 'entropys'
                            if "old_policy_entropy" in batch.batch.keys():
                                ent = batch.batch["old_policy_entropy"]            # [B, R_max]
                            elif "entropys" in batch.batch.keys():
                                ent = batch.batch["entropys"]                      # [B, R_max]
                            else:
                                ent = None

                            token_entropies = None
                            if ent is not None:
                                responses     = batch.batch["responses"]           # [B, R_max]
                                response_mask = batch.batch["response_mask"]       # [B, R_max]
                                eos_id        = getattr(self.tokenizer, "eos_token_id", None)

                                B, R_max = responses.shape
                                token_entropies = []
                                for i in range(B):
                                    valid_len = int(response_mask[i].sum().item())
                                    if valid_len <= 0:
                                        token_entropies.append([])                 # 空输出对应空列表
                                        continue

                                    # 取有效 token 对应的熵；转为 Python 列表
                                    ent_i = ent[i, :valid_len].detach().cpu().tolist()

                                    # 为了与上面的 batch_decode(skip_special_tokens=True) 的文本对齐：
                                    # 若最后一个有效 token 是 eos，则去掉最后一个熵
                                    if eos_id is not None and responses[i, valid_len - 1].item() == int(eos_id):
                                        ent_i = ent_i[:-1]

                                    token_entropies.append(ent_i)
                            # ======= token 熵整合结束 ===========

                            # offset_mapping_info = batch.non_tensor_batch['offset_mapping_info']

                            self._dump_generations(
                                batch=batch,
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                advantages=advantages,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                                entropies=token_entropies,
                                # offset_mapping_info=offset_mapping_info,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, "green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, "green"):
                            self._save_checkpoint()
                    
                    if self.config.custom_reward_function.reflection.enable:
                        with marked_timer("err_step", timing_raw, "red"):
                            # === 错误样本引入反思后继续 rollout，用于下一个 batch 更新 Actor ===
                            if next_batch is not None and len(next_batch) > 0:
                                err_prior_response_mask, err_gen_batch, new_err_batch, err_index_list, no_err_batch = self._prepare_err_gen_batch(next_batch,
                                                                                        tokenizer=self.tokenizer,                 # HF tokenizer
                                                                                        pad_token_id=self.tokenizer.pad_token_id,
                                                                                        response_length=self.config.data.max_response_length,
                                                                                        prompt_length=self.config.data.max_prompt_length,)
                                # generate a error batch
                                with marked_timer("err_gen_with_reflection", timing_raw, "red"):
                                    err_gen_batch_output = self.actor_rollout_wg.generate_sequences_with_reflection(err_gen_batch)
                                    timing_raw.update(err_gen_batch_output.meta_info["timing"])
                                    err_gen_batch_output.meta_info.pop("timing", None)
                                new_err_batch = new_err_batch.union(err_gen_batch_output)
                                new_err_batch = new_err_batch[err_index_list]
                                with marked_timer("err_batch_reward", timing_raw, "yellow"):
                                    # compute scores. Support both model and function-based.
                                    # We first compute the scores using reward model. Then, we call reward_fn to combine
                                    # the results from reward model and rule-based results.
                                    if self.use_rm:
                                        # we first compute reward model score
                                        reward_tensor = self.rm_wg.compute_rm_score(err_gen_batch)
                                        err_gen_batch = err_gen_batch.union(reward_tensor)

                                    # we combine with rule-based rm
                                    err_reward_extra_infos_dict: dict[str, list]
                                    try:
                                        err_reward_result = self.reward_fn(new_err_batch, return_dict=True)
                                        err_reward_tensor = err_reward_result["reward_tensor"]
                                        err_reward_extra_infos_dict = err_reward_result.get("reward_extra_info", {})
                                    except Exception as e:
                                        print(f"Error in reward_fn: {e}")
                                        err_reward_tensor = self.reward_fn(new_err_batch)
                                        err_reward_extra_infos_dict = {}

                                    new_err_batch.batch["token_level_scores"] = err_reward_tensor

                                    if err_reward_extra_infos_dict:
                                        new_err_batch.non_tensor_batch.update(
                                            {k: np.array(v, dtype=object) for k, v in err_reward_extra_infos_dict.items()}
                                        )
                                    
                                    # compute rewards. apply_kl_penalty if available
                                    if self.config.algorithm.use_kl_in_reward:
                                        err_gen_batch, kl_metrics = apply_kl_penalty(
                                            err_gen_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                                        )
                                        metrics.update(
                                            kl_metrics
                                        )  # TODO: This will be cleared if we use multiple genenration batches
                                    else:
                                        new_err_batch.batch["token_level_rewards"] = new_err_batch.batch["token_level_scores"]

                                new_err_batch = new_err_batch.union(err_prior_response_mask[err_index_list])
                                # batch = new_batch if batch is None else DataProto.concat([batch, new_batch])
                                # breakpoint()
                                next_completion_batch = new_err_batch if len(no_err_batch) == 0 else DataProto.concat([new_err_batch, no_err_batch])
                                next_batch = None

                                            # collect metrics
                        
                metrics.update(compute_data_metrics_process_dapo(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
                
                if next_completion_batch is not None and len(next_completion_batch) == self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n:
                    next_metrics = {}
                    with marked_timer("step", timing_raw, "red"):
                        # === 当前 batch 的错误数据已经够一个更新 batch，直接 Updating ===

                        next_completion_batch.batch["response_mask"] = compute_response_mask(next_completion_batch)
                        # breakpoint()
                        # Balance the number of valid tokens across DP ranks.
                        # NOTE: This usually changes the order of data in the `batch`,
                        # which won't affect the advantage calculation (since it's based on uid),
                        # but might affect the loss calculation (due to the change of mini-batching).
                        # TODO: Decouple the DP balancing and mini-batching.
                        if self.config.trainer.balance_batch:
                            self._balance_batch(next_completion_batch, metrics=next_metrics)

                        # step 矩阵 batch 数据中每个 step 的位置信息，为后续基于 step 的分析做准备
                        next_completion_batch = self._get_step_position_info(next_completion_batch)

                        # compute global_valid tokens
                        next_completion_batch.meta_info["global_token_num"] = torch.sum(next_completion_batch.batch["attention_mask"], dim=-1).tolist()

                        # recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, "blue"):
                            next_old_log_prob = self.actor_rollout_wg.compute_log_prob(next_completion_batch)
                            # breakpoint()
                            next_entropys = next_old_log_prob.batch["entropys"]
                            next_response_masks = next_completion_batch.batch["response_mask"]
                            next_prior_response_mask = next_completion_batch.batch.get("prior_response_mask", None)
                            if next_prior_response_mask is not None:
                                # 仅在 prior_response_mask==1 且 response_mask==1 的 token 上参与计算
                                next_combined_mask = (next_response_masks > 0) & (next_prior_response_mask > 0)  # bool [bs, T]
                                next_combined_mask_f = next_combined_mask.to(torch.long)
                            else:
                                next_combined_mask = (next_response_masks > 0)  # bool [bs, T]
                                next_combined_mask_f = next_combined_mask.to(torch.long)
                            # next_combined_mask = (next_response_masks > 0) & (next_prior_response_mask > 0)  # bool [bs, T]
                            # next_combined_mask_f = next_combined_mask.to(torch.long)
                            next_loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            next_entropy_agg = agg_loss(loss_mat=next_entropys, loss_mask=next_combined_mask_f, loss_agg_mode=next_loss_agg_mode)
                            next_old_log_prob_metrics = {"actor/entropy": next_entropy_agg.detach().item()}
                            next_metrics.update(next_old_log_prob_metrics)

                            # 保存完整的 entropy 矩阵到 batch 中，用于后续基于 step 的分析
                            next_completion_batch.batch["old_policy_entropy"] = next_entropys
                            next_old_log_prob.batch.pop("entropys")
                            next_completion_batch = next_completion_batch.union(next_old_log_prob)

                        # recompute old_log_probs
                        with marked_timer("log_old_log_prob", timing_raw, "blue"):
                            next_step_token_entropy_metrics = self._compute_step_position_entropy_metrics(next_completion_batch)
                            self._get_word_entropies(next_completion_batch, self.tokenizer)
                            next_metrics.update(next_step_token_entropy_metrics)

                        if self.use_reference_policy:
                            # compute reference log_prob
                            with marked_timer("ref", timing_raw, "olive"):
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(next_completion_batch)
                                next_completion_batch = next_completion_batch.union(ref_log_prob)

                        # compute values
                        if self.use_critic:
                            with marked_timer("values", timing_raw, "cyan"):
                                values = self.critic_wg.compute_values(next_completion_batch)
                                next_completion_batch = next_completion_batch.union(values)

                        with marked_timer("adv", timing_raw, "brown"):
                            # compute advantages, executed on the driver process
                            norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                            next_completion_batch = compute_advantage(
                                next_completion_batch,
                                adv_estimator=self.config.algorithm.adv_estimator,
                                gamma=self.config.algorithm.gamma,
                                lam=self.config.algorithm.lam,
                                num_repeat=self.config.actor_rollout_ref.rollout.n,
                                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                coef_bad_step_if_adv_pos=self.config.actor_rollout_ref.actor.process_grpo_adv.coef_bad_step_if_adv_pos,
                                coef_good_step_if_adv_neg=self.config.actor_rollout_ref.actor.process_grpo_adv.coef_good_step_if_adv_neg,
                                top_n=self.config.actor_rollout_ref.actor.process_grpo_adv.top_n,
                                topn_mode=self.config.actor_rollout_ref.actor.process_grpo_adv.topn_mode,
                                topn_scale=self.config.actor_rollout_ref.actor.process_grpo_adv.topn_scale
                            )

                        # update critic
                        if self.use_critic:
                            with marked_timer("update_critic", timing_raw, "pink"):
                                critic_output = self.critic_wg.update_critic(next_completion_batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                        # implement critic warmup
                        if self.config.trainer.critic_warmup <= self.global_steps:
                            # update actor
                            with marked_timer("update_actor", timing_raw, "red"):
                                next_actor_output = self.actor_rollout_wg.update_actor(next_completion_batch)
                            next_actor_output_metrics = reduce_metrics(next_actor_output.meta_info["metrics"])
                            next_metrics.update(next_actor_output_metrics)

                        # Log rollout generations if enabled
                        next_rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                        if next_rollout_data_dir:
                            with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                                next_inputs = self.tokenizer.batch_decode(next_completion_batch.batch["prompts"], skip_special_tokens=True)
                                next_outputs = self.tokenizer.batch_decode(next_completion_batch.batch["responses"], skip_special_tokens=True)
                                next_scores = next_completion_batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                                next_advantages = next_completion_batch.batch["advantages"].detach().cpu().tolist()
                                if "request_id" in next_completion_batch.non_tensor_batch:
                                    err_reward_extra_infos_dict.setdefault(
                                        "request_id",
                                        next_completion_batch.non_tensor_batch["request_id"].tolist(),
                                    )
                                # === 构造每条样本对应的 token-level entropy 列表 ===
                                # 优先使用你之前保存在 batch 里的完整熵矩阵 'old_policy_entropy'；否则退回 'entropys'
                                if "old_policy_entropy" in next_completion_batch.batch.keys():
                                    next_ent = next_completion_batch.batch["old_policy_entropy"]            # [B, R_max]
                                elif "entropys" in next_completion_batch.batch.keys():
                                    next_ent = next_completion_batch.batch["entropys"]                      # [B, R_max]
                                else:
                                    next_ent = None

                                next_token_entropies = None
                                if next_ent is not None:
                                    next_responses     = next_completion_batch.batch["responses"]           # [B, R_max]
                                    next_response_mask = next_completion_batch.batch["response_mask"]       # [B, R_max]
                                    next_eos_id        = getattr(self.tokenizer, "eos_token_id", None)

                                    NEXT_B, NEXT_R_max = next_responses.shape
                                    next_token_entropies = []
                                    for i in range(NEXT_B):
                                        next_valid_len = int(next_response_mask[i].sum().item())
                                        if next_valid_len <= 0:
                                            next_token_entropies.append([])                 # 空输出对应空列表
                                            continue

                                        # 取有效 token 对应的熵；转为 Python 列表
                                        next_ent_i = next_ent[i, :next_valid_len].detach().cpu().tolist()

                                        # 为了与上面的 batch_decode(skip_special_tokens=True) 的文本对齐：
                                        # 若最后一个有效 token 是 eos，则去掉最后一个熵
                                        if next_eos_id is not None and next_responses[i, next_valid_len - 1].item() == int(next_eos_id):
                                            next_ent_i = next_ent_i[:-1]

                                        next_token_entropies.append(next_ent_i)
                                # ======= token 熵整合结束 ===========

                                # next_offset_mapping_info = next_completion_batch.non_tensor_batch['offset_mapping_info']

                                self._dump_generations(
                                    batch=next_completion_batch,
                                    inputs=next_inputs,
                                    outputs=next_outputs,
                                    scores=next_scores,
                                    advantages=next_advantages,
                                    reward_extra_infos_dict=err_reward_extra_infos_dict,
                                    dump_path=next_rollout_data_dir,
                                    entropies=next_token_entropies,
                                    # offset_mapping_info=next_offset_mapping_info,
                                )

                        # validate
                        if (
                            self.val_reward_fn is not None
                            and self.config.trainer.test_freq > 0
                            and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                        ):
                            with marked_timer("testing", timing_raw, "green"):
                                val_metrics: dict = self._validate()
                                if is_last_step:
                                    last_val_metrics = val_metrics
                            metrics.update(val_metrics)

                        if self.config.trainer.save_freq > 0 and (
                            is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                        ):
                            with marked_timer("save_checkpoint", timing_raw, "green"):
                                self._save_checkpoint()
                        
                    # with marked_timer("stop_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.stop_profile()
                        if self.use_reference_policy:
                            self.ref_policy_wg.stop_profile()
                        if self.use_critic:
                            self.critic_wg.stop_profile()
                        if self.use_rm:
                            self.rm_wg.stop_profile()

                    # collect metrics
                    next_metrics.update(compute_data_metrics_process_dapo(batch=next_completion_batch, use_critic=self.use_critic))
                    next_metrics.update(compute_timing_metrics(batch=next_completion_batch, timing_raw=timing_raw))
                    # TODO: implement actual tflpo and theoretical tflpo
                    next_n_gpus = self.resource_pool_manager.get_n_gpus()
                    next_metrics.update(compute_throughout_metrics(batch=next_completion_batch, timing_raw=timing_raw, n_gpus=next_n_gpus))
                    timing_raw = defaultdict(float)  # clear timing

                    next_metrics["train/num_gen_batches"] = num_gen_batches
                    next_completion_batch = None
                    num_gen_batches = 0

                    # TODO: make a canonical logger that supports various backend
                    logger.log(data=next_metrics, step=self.global_steps)

                    if is_last_step:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                        progress_bar.close()
                        return
                    progress_bar.update(1)
                    self.global_steps += 1
                    # self.gen_steps += 1
                # ----------------------split----------------------
                else:
                    # with marked_timer("stop_profile", timing_raw):
                    if do_profile:
                        self.actor_rollout_wg.stop_profile()
                        if self.use_reference_policy:
                            self.ref_policy_wg.stop_profile()
                        if self.use_critic:
                            self.critic_wg.stop_profile()
                        if self.use_rm:
                            self.rm_wg.stop_profile()

                    # # collect metrics
                    # metrics.update(compute_data_metrics_process_dapo(batch=batch, use_critic=self.use_critic))
                    # metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                    # # TODO: implement actual tflpo and theoretical tflpo
                    # n_gpus = self.resource_pool_manager.get_n_gpus()
                    # metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                    # timing_raw = defaultdict(float)  # clear timing

                    # metrics["train/num_gen_batches"] = num_gen_batches
                    # batch = None
                    # num_gen_batches = 0

                    # # TODO: make a canonical logger that supports various backend
                    # logger.log(data=metrics, step=self.global_steps)

                    # if is_last_step:
                    #     pprint(f"Final validation metrics: {last_val_metrics}")
                    #     progress_bar.close()
                    #     return

                    # progress_bar.update(1)
                    # self.global_steps += 1
                    # self.gen_steps += 1
