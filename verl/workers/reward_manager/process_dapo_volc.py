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

import os
from collections import defaultdict
from typing import List, Dict, Any, Optional, Union, Literal, Tuple
from openai import OpenAI, AsyncOpenAI
import torch
from math import sqrt
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


PROCESS_SCORE_TEMPLATE = """ROLE
You are RedundancyStepDetector. Your ONLY job is to determine whether a given math solution (full or fragment) contains ANY redundant step(s).


BACKGROUND
You will receive a raw solution text to a difficult math problem. The author attempted to separate steps with blank lines("\n\n"). We call a step “redundant” if it restates prior content without adding substance, or otherwise does not materially help solve the problem.


STEP BOUNDARIES
- Treat steps as contiguous blocks separated by exactly two newline characters ("\n\n").
- If multiple blank lines appear, treat them as a single boundary.
- Ignore empty blocks created by extra blank lines.


INPUT
- One raw solution text (full answer or a fragment) with steps separated as above.


DECISION TARGET (BINARY)
- Output 0 if there exists at least one redundant step.
- Output 1 if there are no redundant steps.


DEFINITION OF A REDUNDANT STEP
A step is redundant if it is ANY of the following:
1) Pure restatement: It repeats the problem or a previous step without adding new insight, information, or direction for what to do next.
2) Empty encouragement or meta-comment: e.g., “Good job!”, “This is hard/easy”, complaints or filler that do not move the solution forward.
3) Trivial or stalling progress: It adds only the tiniest amount of information that does not materially guide the solution (no real plan-setting, no decisive progress).
4) Repetition of facts/results already established, with no refinement, correction, or new use.
5) Irrelevant digression that does not contribute to solving the problem.
6) Duplicate verification of an already confirmed result: After a result/computation has been verified correct, performing additional checks of the same item—by the same or a different method—that do not introduce new information, correct an earlier mistake, or unlock a new step is redundant.


NON-REDUNDANT STEP (counterexamples)
A step is NOT redundant if it:
- Introduces a new approach, subgoal, or plan that guides subsequent work.
- Performs a computation or inference that is used later.
- Defines notation or conditions that are referenced later.
- Eliminates a candidate path based on valid reasoning (thus narrowing the search).
- Corrects a prior error in a way that materially affects the final solution.
- Performs a single verification of a critical result/computation/idea (at most once) that is then relied upon downstream.

Note: More than one verification of the same result/computation/idea is considered redundant unless the later check reveals a real error or changes the subsequent reasoning in a material way.


SCOPE
- Judge redundancy only; do NOT evaluate mathematical correctness of the entire solution.
- Minor paraphrases that provide no new content are redundant.
- If a step mixes small new content with large repetition, decide based on whether the NEW content materially advances the solution.


PROCESS
1) Split the input into steps using the boundary rules.
2) For each step, assess redundancy using the definitions above.
3) If ANY step is redundant → final label = 0. Otherwise → final label = 1.


OUTPUT FORMAT (STRICT, TWO-PART)
Produce your output in exactly two sections:


A) ANALYSIS
- Report the total number of steps.
- List each step in order as:
  Step i — [Redundant | Not Redundant] — one-sentence justification.
  (Optionally include a short snippet of ≤12 words from the step for identification; do not quote long text.)
- Include a line: Redundant step indices: [ ... ] (empty list if none).


B) FINAL
- On the VERY LAST LINE of your entire output, print ONLY a single digit: 0 or 1.
- Do not include anything after that digit. No extra spaces, punctuation, or text.


EXAMPLES OF FINAL LINE
0
(or)
1


BEGIN RAW SOLUTION
{response}
END RAW SOLUTION


Now produce the output.
"""

PROCESS_CRITIQUE_TEMPLATE = """BACKGROUND
Your task is to assess whether a given reasoning step in a solution to a math problem is logically and mathematically correct.

A reasoning step is correct if:
1. It logically follows from prior steps or established mathematical principles.
2. It is mathematically valid, adhering to the rules of arithmetic, algebra, geometry, etc.
3. The step provides a valid argument or computation that is part of the process leading to the final answer.
4. It does not contain any mistakes, miscalculations, or flawed assumptions.
5. If the reasoning step relies on a specific theorem or rule, it correctly applies it.

INPUT
1. A math problem presented clearly.
2. The final answer to the problem.
3. The reasoning step being assessed.

OUTPUT FORMAT
Provide your analysis directly, followed by your decision. Your analysis should explain:
- Whether the reasoning step logically follows from prior steps or is mathematically valid.
- Any relevant mathematical principles, rules, or assumptions being applied.
- A check for any miscalculations or contradictions.
- Conclude with a decision based on your analysis.
- Your output must be written in the same language as the original problem.

After your reasoning, print the decision on the final line, preceded by "####". The decision should be one of the following:
- 1: if the reasoning step is correct.
- 0: if the reasoning step is incorrect.

BEGIN INPUT

Problem:
{problem}

Final Answer:
{final_answer}

Reasoning Step {current_step} of {total_steps}:
{reasoning_step}

END INPUT
"""

REFLECTION_TEMPLATE = """Transform the critique into a concise “reflection” suitable for immediate insertion after the incorrect step in a math solution, guiding self-correction.

Input critique:
{critique}

Rewrite it so that it:

1. Pinpoints the exact mistake and cites the correct rule/formula.
2. Supplies the corrected expression or the minimal next action to proceed.
3. Is maximally concise and information-dense (≤2 sentences, ≤50 words).
4. Uses inline LaTeX for math where needed.
5. Includes no labels, quotes, or meta commentary.
6. Writes the reflection in the same language as the input critique (Chinese or English).

Output only the rewritten reflection text.
"""
ReducerName = Literal["mean", "sum", "first", "last", "l2norm", "dot"]
ModelType = Literal["chat", "embedding"]


def _reduce_embedding_to_reward(
    vec: List[float],
    method: ReducerName = "mean",
    dot_ref: Optional[List[float]] = None,
) -> float:
    if not vec:
        return 0.0
    if method == "mean":
        return sum(vec) / len(vec)
    if method == "sum":
        return float(sum(vec))
    if method == "first":
        return float(vec[0])
    if method == "last":
        return float(vec[-1])
    if method == "l2norm":
        return float(sqrt(sum(x * x for x in vec)))
    if method == "dot":
        if dot_ref is None or len(dot_ref) != len(vec):
            raise ValueError("dot reducer requires dot_ref with same length as embedding")
        return float(sum(a * b for a, b in zip(vec, dot_ref)))
    raise ValueError(f"Unknown reducer: {method}")


class UniversalAIClient:
    """
    一个统一的 OpenAI/SGLang 兼容客户端：
    - 支持 chat（文字对话）与 embedding（奖励/嵌入）两类模型
    - chat: 优先 Responses API，失败回退 Chat Completions
    - embedding: 使用 Embeddings API，可将向量约简为奖励标量（默认 mean）
    - 支持单条与并发批量；提供 auto(...) 自动调度入口
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 600.0,
        extra_headers: Optional[Dict[str, str]] = None,
        force_chat: bool = True,
        model_type: ModelType = "chat",
        reward_reducer: ReducerName = "mean",
        dot_ref: Optional[List[float]] = None,  # reducer='dot' 时需要
    ):
        # 统一保证 base_url 末尾包含 /v1
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"

        self.api_key = api_key
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout,
            default_headers=extra_headers or {},
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=timeout,
            default_headers=extra_headers or {},
        )
        self.force_chat = force_chat
        self.model_type: ModelType = model_type

        # 对 embedding/reward 的约简策略
        self.reward_reducer: ReducerName = reward_reducer
        self.dot_ref: Optional[List[float]] = dot_ref

    # ---------------------------
    # 工具：将 messages 线性化为纯文本（用于 Responses API 的 input 兜底）
    # ---------------------------
    @staticmethod
    def _messages_to_text(messages: List[Dict[str, str]]) -> str:
        lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    # ========== chat（文本对话） ==========
    def respond_text(
        self,
        text: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        if self.model_type != "chat":
            raise RuntimeError("respond_text only valid when model_type='chat'")

        if not self.force_chat:
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=text,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    **kwargs,
                )
                if hasattr(resp, "output_text"):
                    return resp.output_text
                return "".join(getattr(p, "text", "") for p in getattr(resp, "output", []))
            except Exception:
                pass

        chat = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return chat.choices[0].message.content

    def respond_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        if self.model_type != "chat":
            raise RuntimeError("respond_messages only valid when model_type='chat'")

        if not self.force_chat:
            text = self._messages_to_text(messages)
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=text,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    **kwargs,
                )
                return getattr(resp, "output_text", "")
            except Exception:
                pass
        
        chat = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return chat.choices[0].message.content

    async def a_respond_text(
        self,
        text: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        if self.model_type != "chat":
            raise RuntimeError("a_respond_text only valid when model_type='chat'")

        if not self.force_chat:
            try:
                resp = await self.async_client.responses.create(
                    model=self.model,
                    input=text,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    **kwargs,
                )
                return getattr(resp, "output_text", "")
            except Exception:
                pass

        chat = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return chat.choices[0].message.content

    async def a_respond_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        if self.model_type != "chat":
            raise RuntimeError("a_respond_messages only valid when model_type='chat'")

        if not self.force_chat:
            text = self._messages_to_text(messages)
            try:
                resp = await self.async_client.responses.create(
                    model=self.model,
                    input=text,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    **kwargs,
                )
                return getattr(resp, "output_text", "")
            except Exception:
                pass

        chat = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return chat.choices[0].message.content

    async def a_batch_messages(
        self,
        batch_messages: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        concurrency: int = 8,
        **kwargs: Any,
    ) -> List[str]:
        if self.model_type != "chat":
            raise RuntimeError("a_batch_messages only valid when model_type='chat'")

        sem = asyncio.Semaphore(concurrency)

        async def _one(msgs: List[Dict[str, str]]) -> str:
            async with sem:
                return await self.a_respond_messages(
                    msgs, temperature=temperature, max_tokens=max_tokens, **kwargs
                )

        tasks = [asyncio.create_task(_one(m)) for m in batch_messages]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def batch_messages(
        self,
        batch_messages: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        concurrency: int = 8,
        **kwargs: Any,
    ) -> List[str]:
        if self.model_type != "chat":
            raise RuntimeError("batch_messages only valid when model_type='chat'")

        if concurrency <= 1:
            return [
                self.respond_messages(msgs, temperature=temperature, max_tokens=max_tokens, **kwargs)
                for msgs in batch_messages
            ]

        outs = ["" for _ in range(len(batch_messages))]
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = {
                ex.submit(self.respond_messages, msgs, temperature=temperature, max_tokens=max_tokens, **kwargs): idx
                for idx, msgs in enumerate(batch_messages)
            }
            for fut in as_completed(futs):
                outs[futs[fut]] = fut.result()
        return outs

    def messages_auto(
        self,
        messages_or_batch: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        concurrency: int = 1,
        **kwargs: Any,
    ) -> Union[str, List[str]]:
        if self.model_type != "chat":
            raise RuntimeError("messages_auto only valid when model_type='chat'")

        is_single = bool(messages_or_batch) and isinstance(messages_or_batch, list) \
            and (len(messages_or_batch) == 0 or isinstance(messages_or_batch[0], dict))

        if is_single:
            return self.respond_messages(
                messages_or_batch, temperature=temperature, max_tokens=max_tokens, **kwargs
            )

        batch_messages = messages_or_batch  # type: ignore
        if concurrency <= 1:
            return [
                self.respond_messages(msgs, temperature=temperature, max_tokens=max_tokens, **kwargs)
                for msgs in batch_messages  # type: ignore
            ]
        else:
            return self.batch_messages(
                batch_messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                concurrency=concurrency,
                **kwargs,
            )

    # ========== embedding（奖励/嵌入） ==========
    def embed_one(
        self,
        text: str,
        as_reward: bool = True,
        reward_reducer: Optional[ReducerName] = None,
        dot_ref: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> Union[float, List[float]]:
        """同步：对单条文本做嵌入；默认约简为奖励标量返回。"""
        if self.model_type != "embedding":
            raise RuntimeError("embed_one only valid when model_type='embedding'")

        resp = self.client.embeddings.create(
            model=self.model,
            input=text,
            **kwargs,
        )
        vec = list(resp.data[0].embedding)
        # vec = resp.data[0].embedding
        # 2) 还原成 [T, 2]（logits）
        # arr = torch.tensor(vec, dtype=torch.float32)
        # if arr.numel() % 2 != 0:
        #     raise ValueError(f"Unexpected embedding length {arr.numel()} (not divisible by 2).")
        # T = arr.numel() // 2
        # logits = arr.view(T, 2)          # [T, 2]
        # logits = logits.unsqueeze(0)     # [1, T, 2] 方便后续与 mask 对齐
        if as_reward:
            reducer = reward_reducer or self.reward_reducer
            return _reduce_embedding_to_reward(vec, reducer, dot_ref or self.dot_ref)
        return vec

    def embed_batch(
        self,
        texts: List[str],
        as_reward: bool = True,
        reward_reducer: Optional[ReducerName] = None,
        dot_ref: Optional[List[float]] = None,
        concurrency: int = 8,
        **kwargs: Any,
    ) -> Union[List[float], List[List[float]]]:
        """同步批量：内部起事件循环并发调用。"""
        if self.model_type != "embedding":
            raise RuntimeError("embed_batch only valid when model_type='embedding'")

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.a_embed_batch(
                texts,
                as_reward=as_reward,
                reward_reducer=reward_reducer,
                dot_ref=dot_ref,
                concurrency=concurrency,
                **kwargs,
            ))
        else:
            raise RuntimeError("检测到正在运行的事件循环，请使用: await client.a_embed_batch(...)")

    async def a_embed_one(
        self,
        text: str,
        as_reward: bool = True,
        reward_reducer: Optional[ReducerName] = None,
        dot_ref: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> Union[float, List[float]]:
        """异步：单条嵌入；默认返回标量奖励。"""
        if self.model_type != "embedding":
            raise RuntimeError("a_embed_one only valid when model_type='embedding'")

        resp = await self.async_client.embeddings.create(
            model=self.model,
            input=text,
            **kwargs,
        )
        vec = list(resp.data[0].embedding)
        if as_reward:
            reducer = reward_reducer or self.reward_reducer
            return _reduce_embedding_to_reward(vec, reducer, dot_ref or self.dot_ref)
        return vec

    async def a_embed_batch(
        self,
        texts: List[str],
        as_reward: bool = True,
        reward_reducer: Optional[ReducerName] = None,
        dot_ref: Optional[List[float]] = None,
        concurrency: int = 8,
        **kwargs: Any,
    ) -> Union[List[float], List[List[float]]]:
        """异步批量：并发获取嵌入；默认返回标量奖励列表。"""
        if self.model_type != "embedding":
            raise RuntimeError("a_embed_batch only valid when model_type='embedding'")

        sem = asyncio.Semaphore(concurrency)

        async def _one(t: str):
            async with sem:
                return await self.a_embed_one(
                    t,
                    as_reward=as_reward,
                    reward_reducer=reward_reducer,
                    dot_ref=dot_ref,
                    **kwargs,
                )

        tasks = [asyncio.create_task(_one(t)) for t in texts]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def embeddings_auto(
        self,
        text_or_texts: Union[str, List[str]],
        as_reward: bool = True,
        reward_reducer: Optional[ReducerName] = None,
        dot_ref: Optional[List[float]] = None,
        concurrency: int = 1,
        **kwargs: Any,
    ) -> Union[float, List[float], List[List[float]]]:
        """自动：单条字符串 -> 标量/向量；列表 -> 批量（可并发）。"""
        if self.model_type != "embedding":
            raise RuntimeError("embeddings_auto only valid when model_type='embedding'")

        if isinstance(text_or_texts, str):
            return self.embed_one(
                text_or_texts,
                as_reward=as_reward,
                reward_reducer=reward_reducer,
                dot_ref=dot_ref,
                **kwargs,
            )

        texts = text_or_texts
        if concurrency <= 1:
            # 串行
            out = []
            for t in texts:
                out.append(self.embed_one(
                    t, as_reward=as_reward, reward_reducer=reward_reducer, dot_ref=dot_ref, **kwargs
                ))
            return out  # type: ignore
        else:
            # 并发
            return self.embed_batch(
                texts,
                as_reward=as_reward,
                reward_reducer=reward_reducer,
                dot_ref=dot_ref,
                concurrency=concurrency,
                **kwargs,
            )

    # ========== 统一自动入口 ==========
    def auto(
        self,
        payload: Union[
            str,
            List[str],
            List[Dict[str, str]],
            List[List[Dict[str, str]]],
        ],
        # chat-only
        temperature: float = 0.7,
        max_tokens: int = 1024,
        # embedding-only
        as_reward: bool = False,
        reward_reducer: Optional[ReducerName] = None,
        dot_ref: Optional[List[float]] = None,
        # shared
        concurrency: int = 1,
        **kwargs: Any,
    ):
        """
        统一自动调度：
        - model_type='chat':
            payload 为 messages（单条或批量）。单条 -> str；批量 -> List[str]
        - model_type='embedding':
            payload 为 str 或 List[str]。默认返回奖励标量（或向量，根据 as_reward）
        """
        if self.model_type == "chat":
            # 判定单条/批量 messages
            is_single_msgs = bool(payload) and isinstance(payload, list) \
                and (len(payload) == 0 or (isinstance(payload[0], dict) and "role" in payload[0]))

            if is_single_msgs:
                return self.respond_messages(
                    payload,  # type: ignore
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )

            # 批量 messages
            if not payload or not isinstance(payload, list) or not payload or not isinstance(payload[0], list):
                raise ValueError("For chat mode, payload must be messages or batch of messages")

            if concurrency <= 1:
                return [
                    self.respond_messages(msgs, temperature=temperature, max_tokens=max_tokens, **kwargs)
                    for msgs in payload
                ]
            else:
                return self.batch_messages(
                    payload,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    concurrency=concurrency,
                    **kwargs,
                )

        elif self.model_type == "embedding":
            # 单条或批量文本
            if isinstance(payload, str):
                return self.embed_one(
                    payload,
                    as_reward=as_reward,
                    reward_reducer=reward_reducer,
                    dot_ref=dot_ref,
                    **kwargs,
                )
            elif isinstance(payload, list) and (len(payload) == 0 or isinstance(payload[0], str)):
                if concurrency <= 1:
                    # 串行
                    out = []
                    for t in payload:  # type: ignore
                        out.append(self.embed_one(
                            t,
                            as_reward=as_reward,
                            reward_reducer=reward_reducer,
                            dot_ref=dot_ref,
                            **kwargs,
                        ))
                    return out
                else:
                    return self.embed_batch(
                        payload,  # type: ignore
                        as_reward=as_reward,
                        reward_reducer=reward_reducer,
                        dot_ref=dot_ref,
                        concurrency=concurrency,
                        **kwargs,
                    )
            else:
                raise ValueError("For embedding mode, payload must be a string or a list of strings")

        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")


def generate_proccess_reward_data(solution_str: str,
                                 split_step_num: int,
                                 template: str = PROCESS_SCORE_TEMPLATE) -> List[List[Dict[str, str]]]:
    """
    根据输入 solution_str 生成用于“过程评分”的 batch 数据：
    - 若包含 '</think>'，仅取其之前作为 CoT，并移除 '<think>' / '</think>' 标签；
    - 如果无 '</think>'，两种情况，cot 不完整，非think 模型，都直接将全部输出作为回答。
    - 以两个及以上换行符作为 step 分隔（忽略空步）；
    - 按 split_step_num 将步骤分段，最后一段可少于该数；
    - 每段用 template.format(response=..., reference=...) 构造 content；
      若 template 不含 {reference}，会自动退化为仅传入 response。

    返回：
        形如 [[{"role": "user", "content": ...}], ...] 的 batch。
    """
    if not isinstance(split_step_num, int) or split_step_num <= 0:
        raise ValueError("split_step_num must be a positive integer")

    # 统一换行
    text = (solution_str or "").replace("\r\n", "\n")

    # 提取/清洗 CoT
    if "</think>" in text:
        response = text.split("</think>", 1)[0]
        response = response.replace("<think>", "").replace("</think>", "")
    else:
        response = text.replace("<think>", "")
    response = response.strip()

    # 按两个及以上换行切分为步骤；忽略空块
    raw_steps = re.split(r"\n{2,}", response) if response else []
    steps = [s.strip() for s in raw_steps if s.strip()]

    # 若完全切不出步骤，则把整体当作一个步骤（非空时）
    if not steps and response:
        steps = [response]

    batch: List[List[Dict[str, str]]] = []
    for i in range(0, len(steps), split_step_num):
        # 组段并恢复为以两个换行符连接
        segment = "\n\n".join(steps[i:i + split_step_num])

        # 格式化模板：优先尝试带 reference；若模板无该占位符则回退
        content = template.format(response=str(segment))

        batch.append([{"role": "user", "content": content}])

    return batch


def parse_last_line_binary(strings: List[str]) -> List[int]:
    """
    对每个字符串，检查其“最后一个非空行”是否为 '0' 或 '1'。
    若是，则输出对应的 0 或 1；否则输出 0。
    返回与输入等长的 int 列表（元素仅为 0 或 1）。

    说明：
    - 忽略字符串末尾可能存在的空白行与空格（取最后一个非空行）。
    - 对非字符串或空字符串也返回 0。
    """
    result: List[int] = []
    for s in strings:
        if not isinstance(s, str):
            result.append(0)
            continue

        # 标准化换行
        text = s.replace("\r\n", "\n").replace("\r", "\n")
        lines = text.split("\n")

        # 取最后一个非空行（去除首尾空格）
        last_nonempty = ""
        for ln in reversed(lines):
            if ln.strip():
                last_nonempty = ln.strip()
                break

        if last_nonempty in ("0", "1"):
            result.append(int(last_nonempty))
        else:
            result.append(0)

    return result


def llm_process_scores_for_batch_items_with_universal_client(
    batch_items: List[Dict[str, Any]],
    *,
    client,                         # UniversalAIClient 实例（model 已在构造时指定）
    split_step_num: int = 4,
    keep_last_n: int = 2,           # 若每题段数>2，只保留最后两段
    temperature: float = 0.1,
    max_tokens: int = 512,
    concurrency: int = 8,
    request_extra: Optional[Dict[str, Any]] = None,  # 透传给 client.auto 的其他字段，如 stop/top_p 等
) -> None:
    """
    为每个样本构造段级判分消息，展平后用 UniversalAIClient 批量请求，
    然后按 (item_idx, seg_idx) 映射回填到 batch_items[i].

    回填字段：
      - item["process_scores_raw"]: List[str]   每段 judge 的原始输出
      - item["process_scores"]    : List[int]   每段解析后的 0/1
    """
    request_extra = request_extra or {}

    flat_messages: List[List[Dict[str, str]]] = []
    mapping: List[Tuple[int, int]] = []  # (item_idx, seg_idx)

    for i, item in enumerate(batch_items):
        solution_str = item["response"]

        # 用你现有的工具函数生成段消息
        seg_msgs = generate_proccess_reward_data(solution_str, split_step_num)

        # 只保留最后 N 段（满足你“长度>2 只保留最后两项”的需求）
        # if keep_last_n is not None and len(seg_msgs) > keep_last_n:
        #     seg_msgs = seg_msgs[-keep_last_n:]

        # 展平并建立映射
        for j, msgs in enumerate(seg_msgs):
            flat_messages.append(msgs)   # 每个 msgs 是 [{"role":"user","content":...}]
            mapping.append((i, j))
    if not flat_messages:
        # 没有任何段
        for item in batch_items:
            item["process_scores_raw"] = []
            item["process_scores"] = []
        return

    # 使用 UniversalAIClient 的批量接口
    # 注意：传入的是 List[List[Dict]]，这样 client.auto 会走 batch 分支并返回 List[str]
    req_kwargs = dict(temperature=temperature, max_tokens=max_tokens, concurrency=concurrency)
    req_kwargs.update(request_extra)

    judge_outputs = client.auto(flat_messages, **req_kwargs)
    if isinstance(judge_outputs, str):
        # 理论上不会发生（因为传的是 batch），但稳妥起见做个兜底
        judge_outputs = [judge_outputs]


    # 解析每段输出为 0/1
    judge_scores = parse_last_line_binary(judge_outputs)

    # 回填到每个样本
    per_item_raw: Dict[int, List[str]] = defaultdict(list)
    per_item_bin: Dict[int, List[int]] = defaultdict(list)
    for (i, _j), raw, sc in zip(mapping, judge_outputs, judge_scores):
        per_item_raw[i].append(raw)
        per_item_bin[i].append(sc)

    for i, item in enumerate(batch_items):
        item["process_scores_raw"] = per_item_raw.get(i, [])
        item["process_scores"]     = per_item_bin.get(i, [])

def generate_proccess_critique_data(prompt_str: str,
                                    response_str: str,
                                    ground_truth: str,
                                    split_step_num: int,
                                    template: str = PROCESS_CRITIQUE_TEMPLATE) -> List[List[Dict[str, str]]]:
    """
    根据输入 response_str 生成用于“过程评分”的 batch 数据：
    - 若包含 '</think>'，仅取其之前作为 CoT，并移除 '<think>' / '</think>' 标签；
    - 如果无 '</think>'，两种情况，cot 不完整，非think 模型，都直接将全部输出作为回答。
    - 以两个及以上换行符作为 step 分隔（忽略空步）；
    - 按 split_step_num 将步骤分段，最后一段可少于该数；
    - 每段用 template.format(response=..., reference=...) 构造 content；
      若 template 不含 {reference}，会自动退化为仅传入 response。

    返回：
        形如 [[{"role": "user", "content": ...}], ...] 的 batch。
    """
    if not isinstance(split_step_num, int) or split_step_num <= 0:
        raise ValueError("split_step_num must be a positive integer")

    # 统一换行
    text = (response_str or "").replace("\r\n", "\n")

    # 提取/清洗 CoT
    if "</think>" in text:
        response = text.split("</think>", 1)[0]
        response = response.replace("<think>", "").replace("</think>", "")
    else:
        response = text.replace("<think>", "")
    response = response.strip()

    # 按两个及以上换行切分为步骤；忽略空块
    raw_steps = re.split(r"\n{2,}", response) if response else []
    steps = [s.strip() for s in raw_steps if s.strip()]

    # 若完全切不出步骤，则把整体当作一个步骤（非空时）
    if not steps and response:
        steps = [response]

    batch: List[List[Dict[str, str]]] = []
    for i in range(0, len(steps), split_step_num):
        # 组段并恢复为以两个换行符连接
        segment = "\n\n".join(steps[i:i + split_step_num])

        # 格式化模板：优先尝试带 reference；若模板无该占位符则回退
        content = template.format(problem=str(prompt_str),reasoning_step=str(segment),final_answer=str(ground_truth),current_step=str(i+1),total_steps=str(len(steps)))

        batch.append([{"role": "user", "content": content}])

    return batch


def parse_last_line_int_bounded(strings: List[str]) -> List[int]:
    """
    对每个字符串，检查其“最后一个非空行”是否为整数：
    - 若是整数且等于 -1，输出 -1
    - 若是整数且不等于 -1：
        - 若该整数 <= num，输出该整数
        - 否则输出 -1
    - 若不是整数或输入不是字符串，输出 -1

    说明：
    - 忽略字符串末尾可能存在的空白行与空格（取最后一个非空行）
    - 兼容不同换行符（\r\n / \r / \n）
    """
    result: List[int] = []
    for s in strings:
        if not isinstance(s, str):
            result.append(0)
            continue

        # 标准化换行并切分
        text = s.replace("\r\n", "\n").replace("\r", "\n")
        lines = text.split("\n")

        # 取最后一个非空行（去除首尾空格）
        last_nonempty = ""
        for ln in reversed(lines):
            stripped = ln.strip()
            if stripped:
                last_nonempty = stripped
                break

        if last_nonempty == "":
            result.append(0)
            continue

        # 尝试解析为整数
        try:
            val = last_nonempty.strip()
        except ValueError:
            result.append(0)
            continue

        # 按规则输出
        if val == "#### 0":
            result.append(0)
        elif val == "#### 1":
            result.append(1)
        else:
            # result.append(val if val <= split_num else -1)
            result.append(0)

    return result


def _count_response_steps_by_separator(text: str, sep: str = "\n\n") -> int:
    """
    依据分隔符（默认两个换行）统计 response 的 step 数。
    - 会标准化换行，并对分段做 strip；空段不计入。
    """
    if not isinstance(text, str) or not text:
        return 0
    norm = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = [p.strip() for p in norm.split(sep)]
    return sum(1 for p in parts if p)


def llm_process_critique_steppos_for_batch_items_with_universal_client(
    batch_items: List[Dict[str, Any]],
    *,
    client,                         # UniversalAIClient 实例
    split_step_num: int = 4,        # 每个块的目标 step 数（最后一块可能不足）
    temperature: float = 0.1,
    max_tokens: int = 512,
    concurrency: int = 8,
    request_extra: Optional[Dict[str, Any]] = None,
    # 可选：分隔规则（若与你的切步规则一致，建议保持默认）
    step_separator: str = "\n\n",
) -> None:
    """
    以“块”为单位向 judge 模型请求，但将 judge 结果“还原”为原始 step 级别的 0/1 列表：
      - 若某块判定为正确（解析结果=0）：该块内的所有 step 置 1；
      - 若某块判定为错误（解析结果为 k，1<=k<=该块实际步数）：
            块内第 1..(k-1) 个 step 置 1，第 k 及其后的 step 置 0；
            且后续所有块的 step 全部置 0（不再依赖后续 judge 结果）；
      - 若解析结果为 -1（无效）：视为此块失败，块内所有 step 置 0，并将后续块全置 0。

    注意：
    - 这里**假设** judge 输出的“最后一行整数”含义为：0=块正确；1..M=第 k 个 step 出错；-1=无法解析/无效。
      （你的 `parse_last_line_int_bounded` 已经提供了此语义：>num 或非整数 -> -1）
    - 该实现会**保留**每个块的原始输出（item["process_scores_raw"]）以及解析后的整数（item["process_scores_parsed"]），
      并新增 step 级别列表（item["process_step_scores"]）。
    """
    request_extra = request_extra or {}

    # 展平请求与映射
    flat_messages: List[List[Dict[str, str]]] = []
    mapping: List[Tuple[int, int]] = []  # (item_idx, seg_idx)

    # 每个样本的段消息（用于回填）
    per_item_seg_msgs: Dict[int, List[List[Dict[str, str]]]] = defaultdict(list)

    # 统计每个样本的 step 总数 & 每段的实际长度（最后一段可能不足 split_step_num）
    per_item_seg_lens: Dict[int, List[int]] = {}
    
    for i, item in enumerate(batch_items):
        prompt_str = item.get("prompt")
        raw_question = item.get("raw_question")
        response_str = item.get("response")
        ground_truth = item.get("ground_truth")

        # 1) 生成用于 judge 的段消息（与现有函数一致）
        seg_msgs: List[List[Dict[str, str]]] = generate_proccess_critique_data(raw_question, response_str, ground_truth, split_step_num)
        per_item_seg_msgs[i] = seg_msgs

        # 2) 估算原始 step 总数（按分隔符）
        total_steps = _count_response_steps_by_separator(response_str, sep=step_separator)

        # 与段数对齐，计算每段实际步数：前面段固定为 split_step_num，最后一段为剩余
        S = len(seg_msgs)
        if S == 0 or total_steps == 0:
            per_item_seg_lens[i] = []
            continue

        seg_lens = [split_step_num] * S
        full_needed = split_step_num * (S - 1)
        last_len = max(0, total_steps - full_needed)
        # 若 last_len==0，说明总步数正好整除，最后段也取 split_step_num
        seg_lens[-1] = last_len if last_len > 0 else split_step_num
        per_item_seg_lens[i] = seg_lens

        # 展平请求
        for j, msgs in enumerate(seg_msgs):
            flat_messages.append(msgs)
            mapping.append((i, j))

    # 若没有任何段，直接回填空并返回
    if not flat_messages:
        for item in batch_items:
            item["process_critique_raw"] = []
            item["process_critique_parsed"] = []
            item["process_step_critique"] = []
        return

    # 批量请求
    req_kwargs = dict(temperature=temperature, max_tokens=max_tokens, concurrency=concurrency)
    req_kwargs.update(request_extra)
    judge_outputs = client.auto(flat_messages, **req_kwargs)
    if isinstance(judge_outputs, str):  # 理论上不会（因为 batch），但容错
        judge_outputs = [judge_outputs]
        # 将原始输出回填到每个样本
    per_item_raw: Dict[int, List[str]] = defaultdict(list)
    for (i, _j), raw in zip(mapping, judge_outputs):
        per_item_raw[i].append(raw)

    # 解析每段输出为整数（使用 parse_last_line_int_bounded，按各段实际步数限制）
    per_item_parsed: Dict[int, List[int]] = {}
    pos = 0
    for i, seg_msgs in per_item_seg_msgs.items():
        seg_cnt = len(seg_msgs)
        seg_lens = per_item_seg_lens.get(i, [])
        if seg_cnt == 0 or not seg_lens:
            per_item_parsed[i] = []
            continue

        raws_i = judge_outputs[pos:pos + seg_cnt]
        pos += seg_cnt

        parsed_i: List[int] = []
        for raw, seg_len in zip(raws_i, seg_lens):
            # 逐段解析，限制 num=seg_len
            v = parse_last_line_int_bounded([raw])[0]
            parsed_i.append(v)
        per_item_parsed[i] = parsed_i
     
    # 根据 per_item_parsed + seg_lens 还原到“step 级别”的 0/1 列表
    for i, item in enumerate(batch_items):
        seg_lens = per_item_seg_lens.get(i, [])
        parsed = per_item_parsed.get(i, [])
        raws = per_item_raw.get(i, [])

        # 回填原始与解析结果（按段）
        item["process_critique_raw"] = raws
        item["process_critique_parsed"] = parsed

        if not seg_lens:
            item["process_step_critique"] = []
            continue

        step_scores: List[int] = []
        # failed = False

        # 说明：我们采用**更合理的语义**（常用于“过程判分”）：块正确->全1；
        # 若块内第 k 个 step 出错->该块 1..(k-1)=1, k..end=0，并停止后续块（全0）。
        # 若解析为 -1（无效）->该块全0，并停止后续块（全0）。
        # （注：如果你的业务语义不同，可在这里调整。）

        for j, seg_len in enumerate(seg_lens):
            if seg_len <= 0:
                continue

            # if failed:
            #     # 后续块全部置 0
            #     step_scores.extend([0] * seg_len)
            #     continue

            v = parsed[j] if j < len(parsed) else -1  # 兜底：缺失解析则按无效处理

            if v == 1:
                # 块正确：该块全 1
                step_scores.extend([1] * seg_len)
            # elif 1 <= v <= seg_len:
            #     # 在第 v 个 step 出错：之前 1，之后 0
            #     step_scores.extend([1] * (v - 1))
            #     step_scores.extend([0] * (seg_len - (v - 1)))
                # failed = True
            else:
                # 无效（0 或越界）：视为整块失败
                step_scores.extend([0] * seg_len)
                # failed = True

        item["process_step_critique"] = step_scores

def get_reflection(client,
                   process_step_critique,
                   process_critique_raw,
                   temperature,
                   max_tokens,
                   concurrency,
                   request_extra: Optional[Dict[str, Any]] = None
):
    """
    逻辑：
      1) 断言 process_step_critique 至少包含一个 0，找到第一个 0 的索引 idx。
      2) 取 raw = process_critique_raw[idx]（字符串）。
      3) 若 raw 中包含 <\\think> 或 </think>，删除该标记及其之前所有内容（取最后一次出现更稳）。
      4) 删除最后一次出现的 "#### 0" 或 "#### 1"：
         - 若它位于末尾，连同其后的空白/换行一并移除；
         - 否则仅移除该标记本身。
      5) 返回清理并 strip 后的字符串。
    """
    request_extra = request_extra or {}
    # 1) 找到第一个 0 的位置
    try:
        idx = next(i for i, v in enumerate(process_step_critique) if v == 0)
    except StopIteration:
        raise AssertionError("process_step_critique must contain at least one 0")

    # 2) 取对应原始字符串
    raw = process_critique_raw[idx]
    if not isinstance(raw, str):
        raw = str(raw) if raw is not None else ""

    s = raw

    # 3) 去除 <\think> / </think> 以及之前内容（取最后一次出现，最符合“最终可见输出在闭合标记之后”的习惯）
    close_markers = ["</think>"]
    cut_pos = -1
    cut_len = 0
    for m in close_markers:
        p = s.rfind(m)
        if p > cut_pos:
            cut_pos = p
            cut_len = len(m)
    if cut_pos != -1:
        s = s[cut_pos + cut_len:]

    # 4) 删除最后一次出现的 "#### 0" 或 "#### 1"
    def _remove_last_score_marker(text: str) -> str:
        # 优先：若末尾是该标记（含尾随空白），整体裁掉
        m_end = re.search(r"(?:\s*|\r?\n)?####\s*[01]\s*$", text)
        if m_end:
            return text[:m_end.start()]

        # 否则：移除最后一次出现的标记本身
        matches = list(re.finditer(r"####\s*[01]", text))
        if matches:
            m = matches[-1]
            return text[:m.start()] + text[m.end():]
        return text

    s = _remove_last_score_marker(s)
    s.replace("<think>", "").replace("\n\n", "\n").strip()
    
    content = REFLECTION_TEMPLATE.format(critique=s)
    message = [{"role": "user", "content": content}]
    req_kwargs = dict(temperature=temperature, max_tokens=max_tokens, concurrency=concurrency)
    req_kwargs.update(request_extra)
    judge_outputs = client.auto(message, **req_kwargs)

    close_markers = ["</think>"]
    cut_pos = -1
    cut_len = 0
    for m in close_markers:
        p = judge_outputs.rfind(m)
        if p > cut_pos:
            cut_pos = p
            cut_len = len(m)
    if cut_pos != -1:
        judge_outputs = judge_outputs[cut_pos + cut_len:]
    if cut_pos == -1:
        judge_outputs = ""

    return judge_outputs.replace("\n\n", "\n").strip() + "\n\n"

@register("process_dapo_volc")
class ProcessDAPOVOLCRewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        llm_reward_cfg=None,
        llm_critique_cfg=None,
        reflection_cfg=None,
        prm_cfg=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len
        self.llm_reward_cfg = llm_reward_cfg
        self.llm_critique_cfg = llm_critique_cfg
        self.reflection_cfg = reflection_cfg
        self.prm_cfg = prm_cfg

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )

    def __call__(self, data: DataProto, return_dict: bool = False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        base_url = os.getenv("LLM_AS_A_JUDGE_SERVICE_URL")
        api_key = os.getenv("LLM_AS_A_JUDGE_SERVICE_API_KEY")
        model_name = os.getenv("LLM_AS_A_JUDGE_SERVICE_MODEL")
        # 1) SGLang 本地/私有化（OpenAI 兼容）
        client = UniversalAIClient(
            base_url=base_url,
            api_key=api_key,
            model=model_name,
            model_type="chat",
            force_chat=True,
        )

        # 先把这一批要用到的字段统一解析出来
        batch_items = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            raw_prompt = data_item.non_tensor_batch.get("raw_prompt", None)
            if raw_prompt:
                raw_prompt = raw_prompt[0]["content"].replace("\nPlease reason step by step, and put your final answer within \\boxed{}.", "").strip()

            batch_items.append({
                "index": i,
                "prompt": prompt_str,
                "raw_question": raw_prompt,
                "response": response_str,
                "ground_truth": ground_truth,
                "data_source": data_source,
                "valid_response_length": valid_response_length,
                "extra_info": extra_info,
            })

        if self.llm_reward_cfg.enable == True:
            llm_process_scores_for_batch_items_with_universal_client(
                batch_items,
                client=client,  # 你在 __call__ 开头 _ensure_client() 拿到的 UniversalAIClient 实例
                split_step_num=self.llm_reward_cfg.split_step_num,
                temperature=self.llm_reward_cfg.temperature,
                max_tokens=self.llm_reward_cfg.max_tokens,
                concurrency=self.llm_reward_cfg.concurrency,
            )
        
        if self.llm_critique_cfg.enable == True:
            extra_body={"chat_template_kwargs": {"enable_thinking": self.llm_critique_cfg.enable_think}, "presence_penalty": self.llm_critique_cfg.presence_penalty}
            request_extra = {"extra_body": extra_body}
            llm_process_critique_steppos_for_batch_items_with_universal_client(
                batch_items,
                client=client,  # 你在 __call__ 开头 _ensure_client() 拿到的 UniversalAIClient 实例
                split_step_num=self.llm_critique_cfg.split_step_num,
                temperature=self.llm_critique_cfg.temperature,
                max_tokens=self.llm_critique_cfg.max_tokens,
                concurrency=self.llm_critique_cfg.concurrency,
                request_extra=request_extra,
            )

        # 3) 再逐个样本 compute_score（把 process_scores 通过 extra_info 传入）
        for i in range(len(batch_items)):
            item = batch_items[i]
            valid_response_length = item.get("valid_response_length")
            data_source = item.get("data_source")
            prompt_str = item.get("prompt")
            response_str = item.get("response")
            ground_truth = item.get("ground_truth")

            # 合并额外信息，注入 process_scores（段级 0/1 列表）
            base_extra = item.get("extra_info") or {}

            # 想追加的键（举例：段级/步级判分、熵、uid、长度等）
            # extras_to_merge = {
            #     "process_scores": item.get("process_scores", []),
            #     "process_scores_raw": item.get("process_scores_raw", []),
            #     "process_critique_raw": item.get("process_critique_raw", []),
            #     "process_critique_parsed": item.get("process_critique_parsed", []),
            #     "process_step_critique": item.get("process_step_critique", []),
            # }
            # extras_to_merge = {k: v for k, v in extras_to_merge.items() if v is not None}
            # enhanced_extra = {**base_extra, **extras_to_merge}
            enhanced_extra = {**base_extra, "process_scores": item.get("process_scores", [])}
            enhanced_extra = {**base_extra, "process_critique": item.get("process_step_critique", [])}

            
            result = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=enhanced_extra,
                llm_reward_cfg=self.llm_reward_cfg,
                prm_cfg=self.prm_cfg,
                llm_critique_cfg=self.llm_critique_cfg,
            )

            score: float
            if isinstance(result, dict):
                score = result["overall_score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result
                # reward_extra_info["outcome_score"].append(score)
                # reward_extra_info['process_scores'].append(enhanced_extra['process_scores'])
            
            if self.llm_critique_cfg.enable:
                assert item.get("process_critique_raw", []) is not None, (f"process_critique_raw must be provided, but got None")
                reward_extra_info['process_critique_raw'].append(item.get("process_critique_raw", []))
                reward_extra_info['process_critique_parsed'].append(item.get("process_critique_parsed", []))
                reward_extra_info['process_step_critique'].append(item.get("process_step_critique", []))

            if self.llm_reward_cfg.enable:
                assert item.get("process_scores", []) is not None, (f"llm_process_scores must be provided, but got None")
                reward_extra_info['process_scores_raw'].append(item.get("process_scores_raw", []))
                reward_extra_info['process_scores'].append(item.get("process_scores", []))

            if self.reflection_cfg.enable:
                assert self.llm_critique_cfg.enable, "reflection 必须在 critique 的基础上才可以实现"
                if reward_extra_info["outcome_score"][i] == 0.0:
                    if self.llm_critique_cfg.enable and 0.0 in item.get("process_step_critique", []):
                        reward_extra_info["next_step"].append(1)
                        extra_body={"chat_template_kwargs": {"enable_thinking": self.reflection_cfg.enable_think},  "presence_penalty": self.reflection_cfg.presence_penalty}
                        request_extra = {"extra_body": extra_body}
                        reward_extra_info["reflection"].append(get_reflection(client,
                                                                              item.get("process_step_critique", []),
                                                                              item.get("process_critique_raw", []),
                                                                              self.reflection_cfg.temperature,
                                                                              self.reflection_cfg.max_tokens,
                                                                              self.reflection_cfg.concurrency,
                                                                              request_extra=request_extra))
                    else:
                        reward_extra_info["next_step"].append(0)
                        reward_extra_info["reflection"].append("")
                else:
                    reward_extra_info["next_step"].append(0)
                    reward_extra_info["reflection"].append("")

            reward = score
            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine: # 默认初始化 self.num_examine = 0 ，需要在 main_process_dapo.py 中设置
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
