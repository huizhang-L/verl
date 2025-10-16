import asyncio
from typing import List, Dict, Any, Optional
from openai import OpenAI, AsyncOpenAI
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F


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
        self.force_chat = force_chat # 是不是强制使用旧的 chat.completions 接口

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
# ---------------------------
# 使用示例
# ---------------------------
if __name__ == "__main__":
    # 1) SGLang 本地/私有化（OpenAI 兼容）
    client = UniversalAIClient(
        base_url="https://sd2fka8nq6e0b2blduijg.apigateway-cn-beijing.volceapi.com/v1",  # 你的服务地址
        api_key="caa6246b-afbe-4d9b-ab34-87bf9922032b",                      # 本地 SGLang 一般用 EMPTY
        model="Qwen2.5-72B-Instruct",         # 启服时的 --model-path 或目录名
    )

    # 单条：文本
    # print(client.respond_text("用两句话解释什么是 SGLang？"))

    # # 单条：messages
    # msgs = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user", "content": "你好！自我介绍一下。"},
    # ]
    # print(client.respond_messages(msgs, max_tokens=512))

    # # 批量并发：文本
    # async def demo_batch():
    #     texts = [f"第 {i} 个问题，给我一句励志语。" for i in range(5)]
    #     outs = await client.a_batch_texts(texts, concurrency=3, max_tokens=64)
    #     for i, o in enumerate(outs):
    #         print(f"[{i}] {o}")

    # asyncio.run(demo_batch())

    # 准备一批 messages（每个元素是一条完整对话）
    data = {
        "system": "Please reason step by step, and put your final answer within \\boxed{}.",
        "query": "Sue lives in a fun neighborhood.  One weekend, the neighbors decided to play a prank on Sue.  On Friday morning, the neighbors placed 18 pink plastic flamingos out on Sue's front yard.  On Saturday morning, the neighbors took back one third of the flamingos, painted them white, and put these newly painted white flamingos back out on Sue's front yard.  Then, on Sunday morning, they added another 18 pink plastic flamingos to the collection. At noon on Sunday, how many more pink plastic flamingos were out than white plastic flamingos?",
        "response": [
        "To find out how many more pink plastic flamingos were out than white plastic flamingos at noon on Sunday, we can break down the problem into steps. First, on Friday, the neighbors start with 18 pink plastic flamingos.",
        "On Saturday, they take back one third of the flamingos. Since there were 18 flamingos, (1/3 \\times 18 = 6) flamingos are taken back. So, they have (18 - 6 = 12) flamingos left in their possession. Then, they paint these 6 flamingos white and put them back out on Sue's front yard. Now, Sue has the original 12 pink flamingos plus the 6 new white ones. Thus, by the end of Saturday, Sue has (12 + 6 = 18) pink flamingos and 6 white flamingos.",
        "On Sunday, the neighbors add another 18 pink plastic flamingos to Sue's front yard. By the end of Sunday morning, Sue has (18 + 18 = 36) pink flamingos and still 6 white flamingos.",
        "To find the difference, subtract the number of white flamingos from the number of pink flamingos: (36 - 6 = 30). Therefore, at noon on Sunday, there were 30 more pink plastic flamingos out than white plastic flamingos. The answer is (\\boxed{30})."
        ]
    }

    messages = [
        [
            {"role": "system", "content": data['system']},
            {"role": "user", "content": data['query']},
            {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
        ],
        [
            {"role": "system", "content": data['system']},
            {"role": "user", "content": data['query']},
            {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
        ],
        [
            {"role": "system", "content": data['system']},
            {"role": "user", "content": data['query']},
            {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
        ],
        [
            {"role": "system", "content": data['system']},
            {"role": "user", "content": data['query']},
            {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
        ],
    ]
    out = client.messages_auto(messages, temperature=1.0, max_tokens=512, concurrency=4)  # concurrency 任意
    print(out)
    print(type(out))
    print(len(out))
    print(out[0])
