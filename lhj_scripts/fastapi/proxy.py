from openai import OpenAI, AsyncOpenAI
from fastapi import FastAPI

app = FastAPI()

class Proxy:
    def __init__(self):
        self.clients = []

    def create(self, args):
        self.clients.append(OpenAI(**args["args"]))
        return len(self.clients) - 1

    def responses_create(self, args):
        return self.clients[args["idx"]].responses.create(**args["args"]).output_text

    def chat_completions_create(self, args):
        return self.clients[args["idx"]].chat.completions.create(**args["args"]).choices[0].message.content

    def embeddings_create(self, args):
        return list(self.clients[args["idx"]].embeddings.create(**args["args"]).data[0].embedding)

proxy = Proxy()

@app.post("/create")
async def create(args: dict):
    return proxy.create(args)

@app.post("/responses_create")
def responses_create(args: dict):
    return proxy.responses_create(args)

@app.post("/chat_completions_create")
def chat_completions_create(args: dict):
    return proxy.chat_completions_create(args)

@app.post("/embeddings_create")
def embeddings_create(args: dict):
    return proxy.embeddings_create(args)

class AsyncProxy:
    def __init__(self):
        self.clients = []

    def create(self, args):
        self.clients.append(AsyncOpenAI(**args["args"]))
        return len(self.clients) - 1

    def responses_create(self, args):
        return self.clients[args["idx"]].responses.create(**args["args"]).output_text

    def chat_completions_create(self, args):
        return self.clients[args["idx"]].chat.completions.create(**args["args"]).choices[0].message.content

    def embeddings_create(self, args):
        return list(self.clients[args["idx"]].embeddings.create(**args["args"]).data[0].embedding)

async_proxy = AsyncProxy()

@app.post("/async_create")
async def async_create(args: dict):
    return async_proxy.create(args)

@app.post("/async_responses_create")
async def async_responses_create(args: dict):
    return async_proxy.responses_create(args)

@app.post("/async_chat_completions_create")
async def async_chat_completions_create(args: dict):
    return async_proxy.chat_completions_create(args)

@app.post("/async_embeddings_create")
async def async_embeddings_create(args: dict):
    return async_proxy.embeddings_create(args)