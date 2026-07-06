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


def get(base_url: str, path: str, api_key: str | None):
    req = urllib.request.Request(base_url.rstrip("/") + path, method="GET")
    if api_key:
        req.add_header("authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=30) as res:
        return json.loads(res.read().decode("utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key")
    args = ap.parse_args(argv)

    checks = [
        ("models", lambda: get(args.base_url, "/models", args.api_key)),
        ("chat", lambda: post(args.base_url, "/chat/completions", {
            "model": args.model,
            "messages": [{"role": "user", "content": "Say hello in five words."}],
            "max_tokens": 8,
            "temperature": 0,
        }, args.api_key)),
        ("completion", lambda: post(args.base_url, "/completions", {
            "model": args.model,
            "prompt": "Hello",
            "max_tokens": 8,
        }, args.api_key)),
        ("embedding", lambda: post(args.base_url, "/embeddings", {
            "model": args.model,
            "input": "hello",
        }, args.api_key)),
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
