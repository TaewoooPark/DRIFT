import os
import shutil
import socket
import subprocess
import tempfile
import textwrap
import threading
import time
import urllib.request

import pytest

import uvicorn

from drift.openai_api import GenerationResult, create_app


class JSSDKBackend:
    model_id = "drift-test"
    default_max_tokens = 8
    context_length = 128
    supports_sampling = True
    supports_embeddings = True

    def generate(self, prompt, max_tokens: int, session_id: str,
                 options: dict | None = None) -> GenerationResult:
        return GenerationResult(text="js answer", token_ids=[1, 2])

    def stream(self, prompt, max_tokens: int, session_id: str,
               options: dict | None = None):
        yield "js"
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
    app = create_app(JSSDKBackend(), api_keys=["secret"])
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


@pytest.mark.skipif(
    os.environ.get("DRIFT_RUN_JS_SDK_SMOKE") != "1",
    reason="set DRIFT_RUN_JS_SDK_SMOKE=1 to install/run the OpenAI JS SDK smoke",
)
def test_openai_js_sdk_smoke(openai_server):
    if not shutil.which("node") or not shutil.which("npm"):
        pytest.skip("node and npm are required for the OpenAI JS SDK smoke")

    with tempfile.TemporaryDirectory(prefix="drift-openai-js-sdk-") as tmp:
        subprocess.run(["npm", "init", "-y"], cwd=tmp, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        subprocess.run(["npm", "install", "openai@latest"], cwd=tmp, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        script = textwrap.dedent(f"""
            import OpenAI from "openai";

            const client = new OpenAI({{
              baseURL: "{openai_server}/v1",
              apiKey: "secret",
            }});

            const models = await client.models.list();
            const chat = await client.chat.completions.create({{
              model: "drift-test",
              messages: [{{ role: "user", content: "hello" }}],
            }});
            const chatChunks = [];
            for await (const chunk of await client.chat.completions.create({{
              model: "drift-test",
              messages: [{{ role: "user", content: "hello" }}],
              stream: true,
            }})) {{
              chatChunks.push(chunk);
            }}
            const completion = await client.completions.create({{
              model: "drift-test",
              prompt: "hello",
            }});
            const embedding = await client.embeddings.create({{
              model: "drift-test",
              input: "hello",
            }});
            const response = await client.responses.create({{
              model: "drift-test",
              input: "hello",
            }});
            const responseEvents = [];
            for await (const event of await client.responses.create({{
              model: "drift-test",
              input: "hello",
              stream: true,
            }})) {{
              responseEvents.push(event);
            }}

            if (models.data[0].id !== "drift-test") throw new Error("models failed");
            if (chat.choices[0].message.content !== "js answer") throw new Error("chat failed");
            if (!chatChunks.some((chunk) => chunk.choices?.[0]?.delta?.content)) {{
              throw new Error("chat stream failed");
            }}
            if (completion.choices[0].text !== "js answer") throw new Error("completion failed");
            if (embedding.data[0].embedding.length !== 2) throw new Error("embedding failed");
            if (response.output_text !== "js answer") throw new Error("response failed");
            if (!responseEvents.some((event) => event.type === "response.output_text.delta")) {{
              throw new Error("responses stream failed");
            }}
        """)
        script_path = os.path.join(tmp, "smoke.mjs")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        subprocess.run(["node", script_path], cwd=tmp, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
