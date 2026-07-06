import socket
import threading
import time
import urllib.request
import json

import pytest

openai = pytest.importorskip("openai")
import uvicorn  # noqa: E402

from drift.openai_api import GenerationResult, create_app  # noqa: E402


class SDKBackend:
    model_id = "drift-test"
    default_max_tokens = 8
    context_length = 128
    supports_sampling = True
    supports_embeddings = True

    def generate(self, prompt, max_tokens: int, session_id: str,
                 options: dict | None = None) -> GenerationResult:
        return GenerationResult(text="sdk answer", token_ids=[1, 2])

    def stream(self, prompt, max_tokens: int, session_id: str,
               options: dict | None = None):
        yield "sdk"
        yield " answer"

    def count_tokens(self, prompt) -> int:
        if isinstance(prompt, list):
            prompt = " ".join(str(m.get("content") or "") for m in prompt)
        return len(str(prompt).split())

    def decode_tokens(self, token_ids: list[int]) -> str:
        return " ".join(str(x) for x in token_ids)

    def encode_tokens(self, prompt) -> list[int]:
        if isinstance(prompt, list):
            prompt = " ".join(str(m.get("content") or "") for m in prompt)
        return [len(x) for x in str(prompt).split()]

    def embed(self, prompt, session_id: str) -> list[float]:
        return [0.1, 0.2]


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture()
def openai_server():
    port = free_port()
    app = create_app(SDKBackend(), api_keys=["secret"])
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(url + "/health", timeout=0.2).read()
            break
        except Exception:
            time.sleep(0.05)
    else:
        server.should_exit = True
        raise RuntimeError("uvicorn test server did not start")
    try:
        yield url
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_openai_python_sdk_smoke(openai_server):
    client = openai.OpenAI(base_url=openai_server + "/v1", api_key="secret")

    models = client.models.list()
    chat = client.chat.completions.create(
        model="drift-test",
        messages=[{"role": "user", "content": "hello"}],
    )
    chunks = list(client.chat.completions.create(
        model="drift-test",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    ))
    completion = client.completions.create(model="drift-test", prompt="hello")
    embedding = client.embeddings.create(model="drift-test", input="hello")
    json_chat = client.chat.completions.create(
        model="drift-test",
        messages=[{"role": "user", "content": "json please"}],
        response_format={"type": "json_object"},
    )
    tool_chat = client.chat.completions.create(
        model="drift-test",
        messages=[{"role": "user", "content": "use a tool"}],
        tools=[{
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }],
        tool_choice="required",
    )

    assert models.data[0].id == "drift-test"
    assert chat.choices[0].message.content == "sdk answer"
    assert any((chunk.choices and chunk.choices[0].delta.content) for chunk in chunks)
    assert completion.choices[0].text == "sdk answer"
    assert embedding.data[0].embedding == pytest.approx([0.1, 0.2])
    assert json.loads(json_chat.choices[0].message.content) == {"response": "sdk answer"}
    assert tool_chat.choices[0].finish_reason == "tool_calls"
    assert tool_chat.choices[0].message.tool_calls[0].function.name == "lookup"
