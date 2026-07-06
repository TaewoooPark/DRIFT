#!/usr/bin/env python3
"""Smoke-test a running `drift serve` OpenAI-compatible endpoint.

Usage:
  python scripts/openai_compat_smoke.py --base-url http://127.0.0.1:8000/v1 \
      --model Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def post(base_url: str, path: str, payload: dict, api_key: str | None):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    if api_key:
        req.add_header("authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=120) as res:
        return json.loads(res.read().decode("utf-8"))


def stream_post(base_url: str, path: str, payload: dict, api_key: str | None):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    if api_key:
        req.add_header("authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=120) as res:
        text = res.read().decode("utf-8")
        if "text/event-stream" not in res.headers.get("content-type", ""):
            raise RuntimeError(f"expected text/event-stream, got {res.headers.get('content-type')}")
        return parse_sse(text)


def get(base_url: str, path: str, api_key: str | None):
    req = urllib.request.Request(base_url.rstrip("/") + path, method="GET")
    if api_key:
        req.add_header("authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.loads(res.read().decode("utf-8"))


def parse_sse(text: str) -> list[tuple[str | None, object]]:
    events = []
    for raw in text.strip().split("\n\n"):
        name = None
        data_lines = []
        for line in raw.splitlines():
            if line.startswith("event: "):
                name = line.removeprefix("event: ")
            elif line.startswith("data: "):
                data_lines.append(line.removeprefix("data: "))
        if not data_lines:
            continue
        data = "\n".join(data_lines)
        events.append((name, "[DONE]" if data == "[DONE]" else json.loads(data)))
    return events


def require(condition: bool, message: str):
    if not condition:
        raise RuntimeError(message)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key")
    args = ap.parse_args(argv)

    def models():
        body = get(args.base_url, "/models", args.api_key)
        require(body.get("object") == "list", "models object mismatch")
        return body

    def chat():
        body = post(args.base_url, "/chat/completions", {
            "model": args.model,
            "messages": [{"role": "user", "content": "Say hello in five words."}],
            "max_tokens": 8,
            "temperature": 0,
        }, args.api_key)
        require(body.get("object") == "chat.completion", "chat object mismatch")
        return body

    def chat_stream():
        events = stream_post(args.base_url, "/chat/completions", {
            "model": args.model,
            "messages": [{"role": "user", "content": "Stream a greeting."}],
            "max_tokens": 8,
            "stream": True,
            "stream_options": {"include_usage": True},
        }, args.api_key)
        require(events[-1][1] == "[DONE]", "chat stream did not finish with [DONE]")
        require(any(
            isinstance(data, dict) and data.get("choices")
            and data["choices"][0].get("delta", {}).get("content") is not None
            for _, data in events
        ), "chat stream had no delta content")
        return {"object": "chat.completion.chunk", "events": len(events)}

    def completion():
        body = post(args.base_url, "/completions", {
            "model": args.model,
            "prompt": "Hello",
            "max_tokens": 8,
        }, args.api_key)
        require(body.get("object") == "text_completion", "completion object mismatch")
        return body

    def completion_stream():
        events = stream_post(args.base_url, "/completions", {
            "model": args.model,
            "prompt": "Hello",
            "max_tokens": 8,
            "stream": True,
            "stream_options": {"include_usage": True},
        }, args.api_key)
        require(events[-1][1] == "[DONE]", "completion stream did not finish with [DONE]")
        return {"object": "text_completion", "events": len(events)}

    def responses():
        body = post(args.base_url, "/responses", {
            "model": args.model,
            "input": "Say hello.",
            "max_output_tokens": 8,
        }, args.api_key)
        require(body.get("object") == "response", "responses object mismatch")
        return body

    def responses_stream():
        events = stream_post(args.base_url, "/responses", {
            "model": args.model,
            "input": "Say hello.",
            "max_output_tokens": 8,
            "stream": True,
        }, args.api_key)
        names = [name for name, _ in events]
        require("response.created" in names, "responses stream missing response.created")
        require("response.output_text.delta" in names, "responses stream missing text deltas")
        require("response.completed" in names, "responses stream missing response.completed")
        return {"object": "response.stream", "events": len(events)}

    def json_mode():
        body = post(args.base_url, "/chat/completions", {
            "model": args.model,
            "messages": [{"role": "user", "content": "Return JSON."}],
            "response_format": {"type": "json_object"},
            "max_tokens": 8,
        }, args.api_key)
        json.loads(body["choices"][0]["message"]["content"])
        return body

    def tool_choice():
        body = post(args.base_url, "/chat/completions", {
            "model": args.model,
            "messages": [{"role": "user", "content": "Use a tool."}],
            "tools": [{
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
            "tool_choice": "required",
            "max_tokens": 8,
        }, args.api_key)
        require(body["choices"][0]["finish_reason"] == "tool_calls",
                "tool_choice did not return tool_calls")
        return body

    def input_tokens():
        return post(args.base_url, "/chat/completions/input_tokens", {
            "model": args.model,
            "messages": [{"role": "user", "content": "hello world"}],
        }, args.api_key)

    def tokenize():
        return post(args.base_url.removesuffix("/v1"), "/tokenize", {
            "content": "hello world",
        }, args.api_key)

    def detokenize():
        toks = post(args.base_url.removesuffix("/v1"), "/tokenize", {
            "content": "hello world",
        }, args.api_key)["tokens"]
        return post(args.base_url.removesuffix("/v1"), "/detokenize", {
            "tokens": toks,
        }, args.api_key)

    def embedding():
        try:
            return post(args.base_url, "/embeddings", {
                "model": args.model,
                "input": "hello",
            }, args.api_key)
        except urllib.error.HTTPError as e:
            body = json.loads(e.read().decode("utf-8", errors="replace"))
            if body.get("error", {}).get("code") == "unsupported_embeddings":
                print("[ok] embedding: unsupported_embeddings capability error")
                return body
            raise

    checks = [
        ("models", models),
        ("chat", chat),
        ("chat_stream", chat_stream),
        ("completion", completion),
        ("completion_stream", completion_stream),
        ("responses", responses),
        ("responses_stream", responses_stream),
        ("json_mode", json_mode),
        ("tool_choice", tool_choice),
        ("input_tokens", input_tokens),
        ("tokenize", tokenize),
        ("detokenize", detokenize),
        ("embedding", embedding),
    ]
    ok = True
    for name, fn in checks:
        try:
            body = fn()
            print(f"[ok] {name}: {body.get('object', 'response')}")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            print(f"[fail] {name}: HTTP {e.code} {body}", file=sys.stderr)
            ok = False
        except Exception as e:
            print(f"[fail] {name}: {type(e).__name__}: {e}", file=sys.stderr)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
