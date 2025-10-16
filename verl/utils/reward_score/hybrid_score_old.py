# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py

import os
import asyncio
from typing import List, Dict, Any, Optional
from openai import OpenAI, AsyncOpenAI
import ast
import numpy as np

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

class UniversalAIClient:
    """
    一个统一的 OpenAI/SGLang 兼容客户端：
    - 优先使用 Responses API: client.responses.create(...)
    - 若 404/405/501/NotImplemented 则自动回退到 Chat Completions
    - 支持单条调用（文本或 messages）与并发批量调用
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        extra_headers: Optional[Dict[str, str]] = None,
        force_chat: bool = True,
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

    # ---------------------------
    # 同步：文本输入（优先 responses，失败 fallback chat.completions）
    # ---------------------------
    def respond_text(
        self,
        text: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        if not self.force_chat:
            # 1) 尝试 Responses API
            try:
                resp = self.client.responses.create(
                    model=self.model,
                    input=text,
                    temperature=temperature,
                    max_output_tokens=max_tokens,  # Responses API 的参数名
                    **kwargs,
                )
                # 新 SDK 有 output_text 便捷属性
                if hasattr(resp, "output_text"):
                    return resp.output_text
                # 兜底：拼接输出片段
                return "".join(getattr(p, "text", "") for p in getattr(resp, "output", []))
            except Exception as e:
                # 部分第三方实现没有 /v1/responses，回退到 chat.completions
                # 仅在常见“未实现/路由不存在”时回退，其它错误继续抛出也可以
                fallback_errors = ("NotFoundError", "BadRequestError", "APIError", "NotImplementedError")
                if any(err in type(e).__name__ for err in fallback_errors):
                    pass
                else:
                    # 某些实现直接抛通用 Exception，也允许回退
                    pass

        # 2) 回退到 Chat Completions
        chat = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
            temperature=temperature,
            max_tokens=max_tokens,  # Chat Completions 的参数名
            **kwargs,
        )
        return chat.choices[0].message.content

    # ---------------------------
    # 同步：多轮 messages（优先 responses，失败 fallback chat.completions）
    # ---------------------------
    def respond_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        if not self.force_chat:
            # 先尝试将 messages 线性化为文本交给 Responses API
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

        # 回退 chat.completions（SGLang/大多数第三方一定支持这个）
        chat = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return chat.choices[0].message.content

    # ---------------------------
    # 异步：文本输入（支持并发批量）
    # ---------------------------
    async def a_respond_text(
        self,
        text: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
        if not self.force_chat:
            # 1) 尝试 Responses API
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

        # 2) 回退 Chat Completions
        chat = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return chat.choices[0].message.content

    # ---------------------------
    # 异步：messages 输入（支持并发批量）
    # ---------------------------
    async def a_respond_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> str:
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

    # ---------------------------
    # 批量并发：输入为若干文本（建议优先用）
    # ---------------------------
    async def a_batch_texts(
        self,
        texts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        concurrency: int = 8,
        **kwargs: Any,
    ) -> List[str]:
        sem = asyncio.Semaphore(concurrency)

        async def _one(t: str) -> str:
            async with sem:
                return await self.a_respond_text(
                    t, temperature=temperature, max_tokens=max_tokens, **kwargs
                )

        tasks = [asyncio.create_task(_one(t)) for t in texts]
        return await asyncio.gather(*tasks, return_exceptions=False)

    # 批量并发：输入为若干 messages
    async def a_batch_messages(
        self,
        batch_messages: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        concurrency: int = 8,
        **kwargs: Any,
    ) -> List[str]:
        sem = asyncio.Semaphore(concurrency)

        async def _one(msgs: List[Dict[str, str]]) -> str:
            async with sem:
                return await self.a_respond_messages(
                    msgs, temperature=temperature, max_tokens=max_tokens, **kwargs
                )

        tasks = [asyncio.create_task(_one(m)) for m in batch_messages]
        return await asyncio.gather(*tasks, return_exceptions=False)

    # ---- 新增：同步包装（内部起事件循环），返回 List[str] ----
    def batch_messages(
        self,
        batch_messages: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        concurrency: int = 8,
        **kwargs: Any,
    ) -> List[str]:
        """在非异步环境下直接用同步方法完成并发批量调用。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # 没有事件循环，直接跑
            return asyncio.run(self.a_batch_messages(
                batch_messages, temperature=temperature, max_tokens=max_tokens,
                concurrency=concurrency, **kwargs
            ))
        else:
            # 已在事件循环（如 Jupyter），请改用 await client.a_batch_messages(...)
            raise RuntimeError("检测到正在运行的事件循环，请使用: await client.a_batch_messages(...)")

    # ---- 新增：自动调度入口（单条或批量，按 concurrency 决定并发）----
    def messages_auto(
        self,
        messages_or_batch,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        concurrency: int = 1,
        **kwargs: Any,
    ):
        """
        智能入口：
        - 如果传入的是【单条 messages】（形如 [{'role':..., 'content':...}, ...]）
          -> concurrency 无论多少，都返回 str（单条结果），内部走 respond_messages
        - 如果传入的是【批量 messages】（形如 [[{...}], [{...}], ...]）
          -> concurrency==1：串行跑，返回 List[str]
          -> concurrency>1：并发跑，返回 List[str]
        """
        # 判定是「单条」还是「批量」
        is_single = bool(messages_or_batch) and isinstance(messages_or_batch, list) \
            and (len(messages_or_batch) == 0 or isinstance(messages_or_batch[0], dict))

        if is_single:
            # 单条请求 -> 返回 str
            return self.respond_messages(
                messages_or_batch,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

        # 批量
        batch_messages = messages_or_batch  # List[List[Dict[str, str]]]
        if concurrency <= 1:
            # 串行跑，返回 List[str]
            return [
                self.respond_messages(msgs, temperature=temperature, max_tokens=max_tokens, **kwargs)
                for msgs in batch_messages
            ]
        else:
            # 并发跑，返回 List[str]
            return self.batch_messages(
                batch_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                concurrency=concurrency,
                **kwargs,
            )

def to_list_or_default(s: str) -> List[Any]:
    """
    判断字符串是否可转为 Python 列表：
    1) 直接尝试 ast.literal_eval，成功且为 list 则返回。
    2) 若失败：
       - 若原始字符串（去右侧空白）以 ']' 结尾，直接返回 [0]。
       - 否则移除末尾一个逗号（若有），追加一个 ']'，再尝试解析；
         成功则返回，否则返回 [0]。
    """
    if not isinstance(s, str):
        return [0]

    def _try_parse(txt: str):
        try:
            val = ast.literal_eval(txt)
            return val if isinstance(val, list) else None
        except Exception:
            return None

    # 1) 直接解析
    parsed = _try_parse(s)
    if parsed is not None:
        return parsed

    # 2) 失败后根据规则修复
    rs = s.rstrip()
    if rs.endswith(']'):
        return [0]

    # 去掉末尾一个逗号（如果有），然后补一个 ']'
    if rs.endswith(','):
        rs = rs[:-1]
    fixed = rs + ']'

    parsed_fixed = _try_parse(fixed)
    return parsed_fixed if parsed_fixed is not None else [0]


def get_process_score(solution_str: str,
                      ground_truth: str,
                      temperature: float = 0.1,
                      max_tokens: int = 512,
                      concurrency: int = 4,
                      template: str = PROCESS_SCORE_TEMPLATE):
    base_url = os.getenv("LLM_JUDGE_PROCESS_REWARD_SERVICE_URL")
    api_key = os.getenv("LLM_JUDGE_PROCESS_REWARD_SERVICE_API_KEY")
    model_name = os.getenv("LLM_JUDGE_PROCESS_REWARD_SERVICE_MODEL")
    # 1) SGLang 本地/私有化（OpenAI 兼容）
    client = UniversalAIClient(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
    )
    content = template.format(response=str(solution_str), reference=str(ground_truth))
    message = [{"role": "user", "content": content}]
    out = client.messages_auto(message, temperature=temperature, max_tokens=max_tokens, concurrency=concurrency)  # concurrency 任意

    # 讲输出的字符串解析为 python 列表
    processed_out = to_list_or_default(out)

    return processed_out


def get_prm_score(solution_str: str,
                      ground_truth: str,
                      temperature: float = 0.1,
                      max_tokens: int = 512,
                      concurrency: int = 4,
                      template: str = PROCESS_SCORE_TEMPLATE):
    base_url = os.getenv("LLM_JUDGE_PROCESS_REWARD_SERVICE_URL")
    api_key = os.getenv("LLM_JUDGE_PROCESS_REWARD_SERVICE_API_KEY")
    model_name = os.getenv("LLM_JUDGE_PROCESS_REWARD_SERVICE_MODEL")
    # 1) SGLang 本地/私有化（OpenAI 兼容）
    client = UniversalAIClient(
        base_url=base_url,
        api_key=api_key,
        model=model_name,
    )
    content = template.format(response=str(solution_str), reference=str(ground_truth))
    message = [{"role": "user", "content": content}]
    out = client.messages_auto(message, temperature=temperature, max_tokens=max_tokens, concurrency=concurrency)  # concurrency 任意

    # 讲输出的字符串解析为 python 列表
    processed_out = to_list_or_default(out)

    return processed_out


def compute_score(data_source, solution_str, ground_truth, extra_info, llm_judge_cfg, prm_cfg) -> float:
    retval = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                retval = 1.0
    except Exception as e:
        print(e)
    
    if llm_judge_cfg.enable == True:
        llm_judge_process_score_list = get_process_score(solution_str, ground_truth, llm_judge_cfg.temperature, llm_judge_cfg.max_tokens, llm_judge_cfg.concurrency)
        llm_judge_process_score = float(np.mean(llm_judge_process_score_list))
    if prm_cfg.enable == True:
        prm_score_list = get_prm_score(solution_str, ground_truth, llm_judge_cfg.temperature, llm_judge_cfg.max_tokens, llm_judge_cfg.concurrency)
        prm_score = float(np.mean(prm_score_list))

    return retval


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:  # noqa: E722
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
