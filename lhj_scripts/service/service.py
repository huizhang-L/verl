import requests

def call_chat_completion(base_url, messages, api_key="EMPTY", model="LLM",
                         temperature=0.6, max_tokens=8192):
    """
    调用 SGLang / OpenAI 兼容的 Chat Completions 接口

    :param base_url: 服务访问地址，例如 "http://127.0.0.1:8000"
    :param messages: 消息列表，例如 [{"role":"user","content":"你是谁"}]
    :param api_key: API Key（本地测试可用 "EMPTY"）
    :param model: 模型名称，需与部署时一致
    :param temperature: 采样温度
    :param max_tokens: 生成最大 token 数
    :return: 模型返回的文本
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()

    return data["choices"][0]["message"]["content"]

# 示例调用
if __name__ == "__main__":
    base_url = "https://sd2fka8nq6e0b2blduijg.apigateway-cn-beijing.volceapi.com"
    msgs = [
        {"role": "system", "content": "You are a helpful AI assistant"},
        {"role": "user", "content": "你最长可以输出多长的内容，多少 token"}
    ]
    reply = call_chat_completion(base_url, msgs, api_key="caa6246b-afbe-4d9b-ab34-87bf9922032b", model="Qwen2.5-72B-Instruct")
    print("模型回复:", reply)
