import asyncio
from typing import List, Dict, Any, Optional, Literal, Union, Callable
from math import sqrt

from openai import OpenAI, AsyncOpenAI
from transformers import AutoModel, AutoTokenizer
import torch

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
        timeout: float = 60.0,
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

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.a_batch_messages(
                batch_messages, temperature=temperature, max_tokens=max_tokens,
                concurrency=concurrency, **kwargs
            ))
        else:
            raise RuntimeError("检测到正在运行的事件循环，请使用: await client.a_batch_messages(...)")

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
                    for msgs in payload  # type: ignore
                ]
            else:
                return self.batch_messages(
                    payload,  # type: ignore
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


if __name__ == "__main__":
    # # 1) Chat / 文本对话
    # client_chat = UniversalAIClient(
    #     base_url="https://sd2fka8nq6e0b2blduijg.apigateway-cn-beijing.volceapi.com/v1/",
    #     api_key="caa6246b-afbe-4d9b-ab34-87bf9922032b",
    #     model="/fs-computility/llm_fudan/shared/models/Qwen2.5/Qwen2.5-72B-Instruct",
    #     model_type="chat",
    #     force_chat=True,  # 允许 Responses 优先
    # )
    # mags = [
    #     [{"role":"user","content":"你好，用一句话介绍你自己"}],
    #     [{"role":"user","content":"你好，用一句话介绍你自己"}],
    #     [{"role":"user","content":"你好，用一句话介绍你自己"}],
    #     [{"role":"user","content":"你好，用一句话介绍你自己"}],
    #     [{"role":"user","content":"你好，用一句话介绍你自己"}],
    #     [{"role":"user","content":"你好，用一句话介绍你自己"}],
    #     [{"role":"user","content":"你好，用一句话介绍你自己"}],
    #     [{"role":"user","content":"你好，用一句话介绍你自己"}],
    # ]
    # resp = client_chat.auto(mags)
    # print(resp)

    # 2) Embedding / 奖励（标量）
    client_reward = UniversalAIClient(
        base_url="https://sd2je6m25ni4n75n9n7t0.apigateway-cn-beijing.volceapi.com/v1/",
        api_key="caa6246b-afbe-4d9b-ab34-87bf9922032b",
        model="Qwen2.5-Math-PRM-72B",   # 你的“奖励/嵌入”模型
        model_type="embedding",
        reward_reducer="mean",          # 也可 "sum"/"first"/"last"/"l2norm"/"dot"
    )
    model_name = "/fs-computility/llm_fudan/shared/models/Qwen2.5-Math/Qwen2.5-Math-PRM-72B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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
        {"role": "system", "content": data['system']},
        {"role": "user", "content": data['query']},
        {"role": "assistant", "content": "<extra_0>".join(data['response']) + "<extra_0>"},
    ]
    conversation_str = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    # print(conversation_str)
    score = client_reward.auto(conversation_str)
    print(score)  # -> 默认返回标量奖励
    print(type(score))

    # 3) 批量并发（embedding -> 奖励列表）
    texts = ["step A ...", "step B ...", "step C ..."]
    scores = client_reward.auto(texts, concurrency=8)  # -> List[float]
    print(scores)
