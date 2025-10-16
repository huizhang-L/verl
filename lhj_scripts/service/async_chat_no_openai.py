import asyncio
from typing import List, Dict, Any, Optional, Union, Callable
import json
import httpx
import time
import math


class SGLangClient:
    """
    纯 HTTP(S) 客户端：不依赖 openai 包，只做 Chat 推理。
    - 单条 / 批量（并发）调用
    - 字段映射（标准 → 服务端字段）
    - 返回解析（多种可能结构自动兜底）
    - 简单重试与超时控制
    """

    def __init__(
        self,
        base_url: str,
        endpoint: str = "/generate",
        *,
        timeout: float = 60.0,
        headers: Optional[Dict[str, str]] = None,
        # 字段映射：把“通用字段名”映射为服务端真实字段名
        field_map: Optional[Dict[str, str]] = None,
        # 自定义 payload 构造器（可覆盖默认）
        payload_builder: Optional[
            Callable[[List[Dict[str, str]], float, int, Dict[str, Any]], Dict[str, Any]]
        ] = None,
        # 自定义返回解析器（可覆盖默认）
        response_parser: Optional[Callable[[httpx.Response], str]] = None,
        # 重试
        max_retries: int = 2,
        retry_backoff_base: float = 0.5,
    ):
        """
        base_url: 例如 "http://127.0.0.1:30000"
        endpoint: 例如 "/generate" 或 "/v1/chat/completions"
        field_map: 例如 {"messages":"messages","temperature":"temperature","max_tokens":"max_new_tokens"}
        """
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint if endpoint.startswith("/") else ("/" + endpoint)
        self.timeout = timeout
        self.headers = headers or {}
        self.field_map = field_map or {
            "messages": "messages",
            "temperature": "temperature",
            "max_tokens": "max_tokens",
            "stream": "stream",
            "stop": "stop",
            "top_p": "top_p",
            "top_k": "top_k",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
        }
        self.payload_builder = payload_builder
        self.response_parser = response_parser
        self.max_retries = max_retries
        self.retry_backoff_base = retry_backoff_base

    # ---------------------
    # 默认 payload 构造与解析
    # ---------------------
    def _default_payload_builder(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        把“通用字段”构造成服务端期望的 JSON。
        你可通过 field_map 改名，比如把 max_tokens → max_new_tokens。
        """
        payload = {
            self.field_map.get("messages", "messages"): messages,
            self.field_map.get("temperature", "temperature"): temperature,
            self.field_map.get("max_tokens", "max_tokens"): max_tokens,
        }
        # 其他可选字段透传（按映射改名）
        for key in ["stream", "stop", "top_p", "top_k", "presence_penalty", "frequency_penalty"]:
            if key in extra:
                payload[self.field_map.get(key, key)] = extra[key]
        # 其余未映射字段也允许直接塞进去（高级用法）
        for k, v in extra.items():
            if k not in ["stream", "stop", "top_p", "top_k", "presence_penalty", "frequency_penalty"]:
                payload.setdefault(k, v)
        return payload

    def _default_response_parser(self, resp: httpx.Response) -> str:
        """
        兼容多种常见返回结构，按“优先级”尽量提取文本：
        - OpenAI Chat: {"choices":[{"message":{"content":"..."}}]}
        - OpenAI Completions: {"choices":[{"text":"..."}]}
        - Responses 风格: {"output_text":"..."} 或 {"output":[{"text":"..."}]}
        - 通用: {"text":"..."} / {"data":{"text":"..."}}
        """
        resp.raise_for_status()
        data = resp.json()
        # 1) OpenAI Chat
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            pass
        # 2) OpenAI Text
        try:
            return data["choices"][0]["text"]
        except Exception:
            pass
        # 3) Responses 风格
        if isinstance(data, dict):
            if "output_text" in data and isinstance(data["output_text"], str):
                return data["output_text"]
            if "output" in data and isinstance(data["output"], list):
                texts = []
                for p in data["output"]:
                    t = p.get("text")
                    if t:
                        texts.append(t)
                if texts:
                    return "".join(texts)
        # 4) 常见兜底
        for key in ("text",):
            if key in data and isinstance(data[key], str):
                return data[key]
        if "data" in data and isinstance(data["data"], dict) and isinstance(data["data"].get("text"), str):
            return data["data"]["text"]
        # 5) 最后兜底：返回 json 序列化
        return json.dumps(data, ensure_ascii=False)

    # ---------------------
    # 基础请求（带重试）
    # ---------------------
    def _full_url(self) -> str:
        return self.base_url + self.endpoint

    def _should_retry(self, status: Optional[int], exc: Optional[BaseException]) -> bool:
        if exc is not None:
            return True  # 网络异常等
        if status is None:
            return False
        # 常见可重试状态
        return status in (408, 429, 500, 502, 503, 504)

    def _backoff_sleep(self, attempt: int):
        time.sleep(self.retry_backoff_base * (2 ** attempt))

    def _build_payload(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.payload_builder:
            return self.payload_builder(messages, temperature, max_tokens, extra)
        return self._default_payload_builder(messages, temperature, max_tokens, extra)

    def _parse_response(self, resp: httpx.Response) -> str:
        if self.response_parser:
            return self.response_parser(resp)
        return self._default_response_parser(resp)

    # ---------------------
    # 同步：单条
    # ---------------------
    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **extra: Any,
    ) -> str:
        payload = self._build_payload(messages, temperature, max_tokens, extra)
        url = self._full_url()
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout, headers=self.headers) as client:
                    resp = client.post(url, json=payload)
                if not resp.is_success and self._should_retry(resp.status_code, None) and attempt < self.max_retries:
                    self._backoff_sleep(attempt)
                    continue
                resp.raise_for_status()
                return self._parse_response(resp)
            except Exception as e:
                last_exc = e
                if attempt < self.max_retries and self._should_retry(None, e):
                    self._backoff_sleep(attempt)
                    continue
                raise

    # ---------------------
    # 异步：单条
    # ---------------------
    async def a_generate(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **extra: Any,
    ) -> str:
        payload = self._build_payload(messages, temperature, max_tokens, extra)
        url = self._full_url()
        last_exc = None
        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    resp = await client.post(url, json=payload)
                    if not resp.is_success and self._should_retry(resp.status_code, None) and attempt < self.max_retries:
                        self._backoff_sleep(attempt)
                        continue
                    resp.raise_for_status()
                    return self._parse_response(resp)
                except Exception as e:
                    last_exc = e
                    if attempt < self.max_retries and self._should_retry(None, e):
                        self._backoff_sleep(attempt)
                        continue
                    raise

    # ---------------------
    # 批量：并发（异步内部实现）
    # ---------------------
    async def a_generate_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        concurrency: int = 8,
        **extra: Any,
    ) -> List[str]:
        sem = asyncio.Semaphore(concurrency)

        async def _one(msgs: List[Dict[str, str]]) -> str:
            async with sem:
                return await self.a_generate(msgs, temperature=temperature, max_tokens=max_tokens, **extra)

        tasks = [asyncio.create_task(_one(m)) for m in batch_messages]
        return await asyncio.gather(*tasks)

    def generate_batch(
        self,
        batch_messages: List[List[Dict[str, str]]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        concurrency: int = 8,
        **extra: Any,
    ) -> List[str]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.a_generate_batch(
                    batch_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    concurrency=concurrency,
                    **extra,
                )
            )
        else:
            raise RuntimeError("检测到正在运行的事件循环，请改用: await client.a_generate_batch(...)")

if __name__ == "__main__":

    client = SGLangClient(
        base_url="https://sd2fka8nq6e0b2blduijg.apigateway-cn-beijing.volceapi.com/v1/",
        endpoint="/chat/completions",
        headers={"Authorization": "Bearer caa6246b-afbe-4d9b-ab34-87bf9922032b"},
        field_map={
            "messages": "messages",
            "temperature": "temperature",
            "max_tokens": "max_tokens",
            # 其他需要映射的字段也可加上
        }
    )


    batch = [
        [{"role":"user","content":"解释 PPO"}],
        [{"role":"user","content":"解释 A2C"}],
        [{"role":"user","content":"解释 DDPG"}],
    ]
    outs = client.generate_batch(batch, temperature=0.3, max_tokens=256, concurrency=8, model="/fs-computility/llm_fudan/shared/models/Qwen2.5/Qwen2.5-72B-Instruct")
    for o in outs:
        print(o)

