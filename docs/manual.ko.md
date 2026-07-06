# DRIFT — 운영 매뉴얼

**DRIFT를 실전으로 돌리는 법 — 처음부터 끝까지.** 언어: [English](manual.md) ·
**한국어** · [中文](manual.zh.md) · [日本語](manual.ja.md)

앞의 절반이 곧 전부다: 설치하고, 시도해 보고, 여러 머신에 걸쳐 하나의 모델을 실행한 다음,
OpenAI 호환 로컬 endpoint로 노출한다.
뒤의 절반 — **커스터마이즈 & 파인튜닝** — 은 기본값만으로 충분하지 않을 때에만 필요하다.
벤치마크 방법론과 수치는 [`benchmarks.md`](benchmarks.md)를 참고하라.

---

## 목차

**실행하기**
1. [설치](#1--설치)
2. [한 대의 머신에서 실행하기](#2--한-대의-머신에서-실행하기)
3. [여러 대에 걸쳐 실행 — 실전 예제](#3--여러-대에-걸쳐-실행--실전-예제)
4. [OpenAI 호환 API로 서빙하기](#4--openai-호환-api로-서빙하기)

**커스터마이즈 & 파인튜닝**
5. [`config.yaml` 레퍼런스](#5--configyaml-레퍼런스)
6. [분할 지점 선택](#6--분할-지점-선택)
7. [모델](#7--모델)
8. [디바이스 & dtype](#8--디바이스--dtype)
9. [생성이 동작하는 방식](#9--생성이-동작하는-방식)
10. [샤드를 손수 구동하기](#10--샤드를-손수-구동하기)
11. [CLI 레퍼런스](#11--cli-레퍼런스)
12. [와이어 & 세션](#12--와이어--세션)
13. [메모리](#13--메모리)
14. [트러블슈팅](#14--트러블슈팅)
15. [탈중앙화 — chain, encryption, failover, gossip, ledger, int8](#15--탈중앙화-v10)

---

## 1 · 설치

**Python 3.12**와 [`uv`](https://github.com/astral-sh/uv)가 필요하다. 번들된 두 모델 모두
**게이트가 없어서** Hugging Face 로그인이 필요하지 않다. **모든 머신에서** 다음을 실행하라:

```bash
git clone https://github.com/TaewoooPark/DRIFT && cd DRIFT
bash scripts/install.sh          # macOS / Linux
# Windows (NVIDIA):  powershell -ExecutionPolicy Bypass -File scripts\install.ps1
drift doctor                     # checks Python, torch, device, config, ports
```

인스톨러는 3.12 venv를 만들고 DRIFT(`drift` CLI)를 설치한다. 플랫폼에 맞는 torch 휠이 GPU
백엔드를 자동으로 고른다 — Apple에서는 MPS, Linux에서는 CUDA이며, Windows에서는 스크립트가 CUDA
빌드를 가져온다. `drift doctor`는 당신의 디바이스(`mps` 또는 `cuda`)를 표시해야 한다.

---

## 2 · 한 대의 머신에서 실행하기

```bash
drift up 2                        # spawn 2 local nodes, auto-split the model, open a chat
drift up 2 --prompt "hello world" # …or a one-shot answer
```

`drift up N`은 이 머신에서 N개의 워커 노드를 띄우고, 모델의 레이어 수를 읽어, 이를 고르게 분할하고,
각 노드에 범위를 배정한 뒤 생성한다. 레이어 범위도, 포트도, 디바이스 플래그도 없다. 동작을 확인하는
가장 빠른 방법이며, 다음 절에서는 노드들을 *서로 다른* 머신에 올린다.

---

## 3 · 여러 대에 걸쳐 실행 — 실전 예제

**목표:** **Mac**에서 `hello world`를 입력하고, Mac(Apple/MPS)**과** Windows PC(NVIDIA/CUDA)를
**둘 다** 써서 답을 계산한다.

**역할.** **헤드(head)**는 프롬프트를 입력하고 `embed` + `lm_head`를 쥐며, 디코더 레이어는
**노드들** 위에 놓인다. 그러니 레이어에 *두* GPU를 모두 쓰려면 Mac이 **노드** 하나와 헤드를 함께
돌리고, PC가 **노드** 하나를 돌린다:

```bash
# ── on the Windows PC (NVIDIA) ───────────────────  one terminal
drift node --port 52601           # auto device = cuda, announced on the LAN
#   (allow python through Windows Defender Firewall on Private networks,
#    and turn on Network Discovery so the Mac can find it)

# ── on the Mac (Apple) ───────────────────────────  terminal 1: a worker
export PYTORCH_ENABLE_MPS_FALLBACK=1
drift node --port 52600           # auto device = mps

# ── on the Mac ───────────────────────────────────  terminal 2: the head
drift run --prompt "hello world"  # finds both nodes, splits 28 layers, streams
```

**보게 될 화면** — 헤드가 두 노드를 발견하고, 모델을 분할하고, 스트리밍한다:

```
[run] discovering nodes on the LAN …
[run] found 192.168.0.22:52601(cuda), 127.0.0.1:52600(mps)

  model : Qwen/Qwen2.5-1.5B-Instruct
  head  : embed + norm + lm_head  · device=mps
  node  : 127.0.0.1:52600     layers [0:14)   · device=mps      ← Mac이 이것들을 계산한다
  node  : 192.168.0.22:52601  layers [14:28)  · device=cuda     ← PC가 이것들을 계산한다

Hello! How can I help you today?
```

**헤드가 PC를 찾지 못한다면**(게스트/기업 Wi-Fi에서는 mDNS가 막히는 경우가 흔하다), 노드를 명시적으로
이름 지정하라 — 위에서 포트를 고정해 둔 이유가 바로 이것이다:

```bash
drift run --nodes 192.168.0.22:52601,127.0.0.1:52600 --prompt "hello world"
```

(Windows 박스는 LAN IP로, Mac 자신의 노드는 `127.0.0.1`로.) 먼저
`drift doctor --nodes 192.168.0.22:52601`로 도달성을 확인하라.

**같은 명령, 어떤 조합이든.** 두 대의 Mac이나 두 대의 Windows PC도 똑같이 동작한다 — `drift node`가
각 디바이스를 자동 감지하고, `drift run`이 찾아서 분할한다. Mac + Windows 혼합에만 해당하는 것은 딱
두 가지다:

- **벤더 간 부동소수점 드리프트.** MPS와 CUDA는 fp16을 조금씩 다르게 반올림하므로, 긴 greedy 답변은
  *뒤쪽* 토큰에서 단일 머신과 발산할 수 있다. 이는 예상된 동작이며 버그가 아니다(초기 토큰은 일치하고,
  텍스트는 일관성을 유지한다). 같은 벤더의 두 노드는 단일 머신을 **비트 단위로** 재현한다.
- **두 OS.** Mac에서는 `install.sh`로, PC에서는 `install.ps1`로 설치하라. 그 이후는 모든 것이 동일하다.

---

## 4 · OpenAI 호환 API로 서빙하기

OpenAI Python/JS SDK, `curl`, LangChain, LiteLLM, LlamaIndex, 에디터 플러그인처럼
OpenAI 스타일의 로컬 백엔드를 기대하는 클라이언트가 있다면 `drift serve`를 사용하라. 모델은 여전히
DRIFT의 노드 간 프로토콜 위에서 분산 실행되고, 클라이언트와 맞닿는 표면만 HTTP/SSE가 된다.

**1) 먼저 워커 노드를 띄운다.** `drift serve`는 워커를 직접 스폰하지 않는다. LAN 디스커버리를 써도
되고, 포트를 고정한 뒤 명시적으로 넘겨도 된다:

```bash
# terminal 1
drift node --port 52600

# terminal 2
drift node --port 52601
```

**2) OpenAI 호환 서버를 시작한다.** 기본 바인딩은 로컬 전용인 `127.0.0.1:8000`이다. 다른 머신에서
HTTP로 호출해야 할 때만 `--host 0.0.0.0`을 사용하라.

```bash
export DRIFT_API_KEY=local-dev
drift serve \
  --nodes 127.0.0.1:52600,127.0.0.1:52601 \
  --port 8000
```

클라이언트의 base URL은 `http://127.0.0.1:8000/v1`이다. HTTP 서버를 자기 머신 밖으로 노출한다면
`--api-key` 또는 `DRIFT_API_KEY`를 켜 두어라. 공개 endpoint나 터널링된 노드 트래픽에는 §15의
DRIFT network key도 함께 사용하라.

**3) `curl`로 호출한다.**

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer local-dev" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in five words."}],
    "max_tokens": 32,
    "temperature": 0
  }'
```

**4) OpenAI Python SDK로도 그대로 붙일 수 있다.**

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="local-dev",
)

reply = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Explain DRIFT in one sentence."}],
)
print(reply.choices[0].message.content)
```

**지원 표면.** 주요 route는 `GET /v1/models`, `POST /v1/chat/completions`,
`POST /v1/completions`, `POST /v1/responses`, 선택한 모드가 hidden state를 노출할 수 있을 때의
`POST /v1/embeddings`, `POST /v1/chat/completions/input_tokens`, `/tokenize`,
`/detokenize`, `/health`, `/ready`, `/metrics`다. Chat/Completions streaming은 `[DONE]`으로
끝나는 SSE를 쓰고, Responses streaming은 `response.created`, `response.output_text.delta`,
`response.completed` 같은 semantic event를 낸다.

**호환성 범위.** 어댑터는 `n`, `temperature`, `top_p`, `top_k`, `min_p`, penalty류, `seed`,
`stop`, logprobs, tool-call shape, JSON response format 같은 OpenAI/vLLM 스타일의 텍스트 생성
파라미터를 받는다. 단, DRIFT가 tool을 대신 실행하지는 않고, 엄격한 schema-constrained decoding을
보장하지도 않는다. Multimodal/audio 요청이나 thin-head sampling/embedding처럼 지원하지 않는 기능은
조용히 무시하지 않고 OpenAI 형태의 명시적 오류를 반환한다. 전체 지원표는
[`openai-compatibility.md`](openai-compatibility.md), 체크리스트별 검증 기록은
[`openai-compatibility-audit.md`](openai-compatibility-audit.md)에 있다.

**자주 쓰는 serve 플래그.**

| 플래그 | 의미 |
|---|---|
| `--nodes host:port,...` | 이미 실행 중인 `drift node` 워커를 사용한다. |
| `--no-discover` / `--discover-timeout` | `--nodes`를 생략했을 때의 LAN 디스커버리를 제어한다. |
| `--model` | `config.yaml`의 `model_id`를 덮어쓴다. |
| `--served-model-name` | `/v1/models`에 노출할 모델 이름을 바꾼다. 클라이언트도 이 이름을 보내야 한다. |
| `--max-new-tokens` | 요청에 토큰 제한이 없을 때 사용할 기본 출력 예산. |
| `--chain`, `--thin`, `--int8`, `--expand` | `drift run`과 같은 DRIFT 라우팅 모드를 사용한다(§15). |
| `--api-key` / `DRIFT_API_KEY` | `Authorization: Bearer ...` 또는 `x-api-key`를 요구한다. 여러 키는 반복 지정하거나 쉼표로 구분한다. |
| `--cors-origin` / `DRIFT_CORS_ORIGINS` | 브라우저 클라이언트의 허용 origin을 지정한다. |
| `--max-concurrent-requests` | HTTP 요청 동시성 제한. 단, 한 backend의 생성은 여전히 직렬화된다. |

**실행 중인 서버 smoke test.**

```bash
python scripts/openai_compat_smoke.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --api-key local-dev
```

---

**커스터마이즈 & 파인튜닝** — 아래 내용은 모두 선택 사항으로, 위의 한 명령 흐름만으로 충분하지 않을
때를 위한 것이다(다른 모델, 고르지 않은 분할, 정확한 포트, 조각들을 손수 구동하기).

---

## 5 · `config.yaml` 레퍼런스

`config.yaml`은 모델, 정밀도, 그리고 (손수 하는 흐름에서의) 샤드 테이블의 유일한 진실의 원천이다.
`drift up` / `drift run` / `drift serve`는 이 파일에서 `model_id`, `dtype`, `generation`을 읽고,
분할은 스스로 계산한다.

```yaml
model_id: "Qwen/Qwen2.5-1.5B-Instruct"   # any HF causal-LM id
dtype: "float16"                          # float16 | float32  (see §8)
device: "mps"                             # default device: mps | cuda | cpu
port: 52600                               # default port for a shard that sets none

shards:                                   # only used by the by-hand flow (§10) / `drift run` fallback
  - { name: "mac",     host: "127.0.0.1", port: 52600, start_layer: 0,  end_layer: 14, device: "mps" }
  - { name: "windows", host: "127.0.0.1", port: 52601, start_layer: 14, end_layer: 28, device: "mps" }

generation:
  max_new_tokens: 50
  prompt: "Give me a short introduction to large language models."
```

| 키 | 의미 |
|---|---|
| `model_id` | Hugging Face 모델 id. 로컬 HF 캐시에 한 번 다운로드된다. |
| `dtype` | 연산 dtype **이면서** 와이어 dtype. `float16`(기본값, 무손실 CPU 왕복) 또는 `float32`. `bfloat16`은 와이어에서 **유효하지 않다** — §8. |
| `device` | 헤드의 기본 디바이스이자, 자기 디바이스를 생략한 샤드의 기본 디바이스. `mps` / `cuda` / `cpu`. |
| `port` | `port`도 `DRIFT_PORT`도 없는 샤드의 폴백 포트. |
| `shards[]` | 손수 하는 흐름(§10)과, 디스커버리가 아무것도 못 찾았을 때의 `drift run` 폴백에 쓰이는 순서 있는 샤드 테이블. |
| `shards[].host` / `port` | 오케스트레이터가 이 샤드에 연결하는 곳. 로컬이면 `127.0.0.1`, 원격이면 LAN IP. |
| `shards[].start_layer` / `end_layer` | 반열림(half-open) 디코더 레이어 범위 `[start, end)`. |
| `shards[].device` | 이 샤드의 디바이스. |
| `generation.max_new_tokens` | 기본 토큰 예산(`--max-new-tokens`로 덮어씀). |
| `generation.prompt` | `--prompt`가 생략됐을 때의 기본 프롬프트. |

---

## 6 · 분할 지점 선택

`drift run`과 `drift serve`는 노드 수로 고르게 분할한다. 이것을 신경 쓸 일은 손수 하는 흐름(§10)이나 고르지 않은
분할일 때뿐이다. 범위는 디코더 레이어를 **타일링(tile)**해야 한다: 연속적이고, 순서대로이며, 빈틈도
겹침도 없이 `[0, num_hidden_layers)`를 전부 덮어야 한다. 헤드는 `embed_tokens`, 최종 norm,
`lm_head`를 소유하며 — 결코 샤드 범위의 일부가 아니다.

```
model: 28 decoder layers (Qwen2.5-1.5B)
        └── [0, 14)  /  [14, 28)                 ✅ tiles 0..28 (the even 2-way split)
        └── [0, 10) / [10, 20) / [20, 28)        ✅ three shards, also valid
        └── [0, 14) / [16, 28)                   ❌ gap at 14–15
        └── [0, 16) / [14, 28)                   ❌ overlap at 14–15
```

어디서 자르든 정확성 면에서는 아무 비용이 없다(어떤 타일링이든 한 디바이스에서는 비트 단위로 정확하다).
다만 각 노드가 얼마만큼의 연산과 가중치 메모리를 짊어지는지만 달라진다. 머신 성능이 다르다면 더 빠른
머신 쪽으로 치우치게 나누라. 레이어 수: Qwen2.5-1.5B = 28, Gemma-4-E2B = 35.

---

## 7 · 모델

엔진은 아키텍처를 하드코딩하는 대신 로드된 모델을 **인트로스펙션(introspect)**한다(디코더 레이어 클래스,
`rotary_emb`, 캐시 타입, 레이어별 어텐션) — 그래서 새 계열은 id만으로 그대로 끼워 넣을 수 있다.
`config.yaml`에 `model_id`를 설정하거나(또는 `drift run --model <id>`), 그 외에는 아무것도 필요 없다.

| 모델 | 레이어 | 고른 분할 | 비고 |
|---|---:|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` (기본) | 28 | `0–14 / 14–28` | 평범한 디코더; 패리티 기준선 |
| `google/gemma-4-E2B-it` | 35 | `0–18 / 18–35` | 레이어별 임베딩(Per-Layer Embeddings; 노드가 `input_ids`로부터 재구성), 이중 RoPE θ, 하이브리드 어텐션, `HybridCache`; `transformers ≥ 5.5` 필요 |

더 큰 모델은 단지 나눌 레이어가 더 많을 뿐이다 — 그 레이어 수에 맞춰 범위를 연속적으로 유지하라.

---

## 8 · 디바이스 & dtype

**디바이스** — `mps`(Apple GPU), `cuda`(NVIDIA GPU), `cpu`(이식성 있음, 느림). 각 노드의
디바이스는 서로 독립적이며, 그 독립성이 바로 핵심이다. `drift node`가 이를 자동 감지하고,
`--device`로 덮어쓴다.

**dtype** — `float16`(기본값) 또는 `float32`. 와이어는 텐서를 이 dtype의 원시 바이트로 직렬화하며,
fp16 CPU 왕복은 비트 손실이 없으므로 직렬화가 결과를 교란하는 일은 없다. `bfloat16`은 와이어에서
**지원되지 않는다** — `float16`을 사용하라.

**같은 벤더 vs 혼합** — 같은 디바이스 계열의 두 샤드는 단일 머신을 **비트 단위로** 재현한다. `mps`와
`cuda`를 섞으면 비트 수준 fp16 반올림 차이가 생기므로 greedy 디코딩이 뒤쪽 토큰에서 발산할 수 있다
(예상된 동작 — §3).

---

## 9 · 생성이 동작하는 방식

- **Greedy.** 레퍼런스 오라클과 오케스트레이터 모두 매 스텝에서 `argmax`를 고른다. CLI에는 아직
  temperature/top-p 샘플링이 없다. OpenAI 호환 HTTP API는 non-thin 모드에서 일반적인 sampling
  control을 지원한다. 패리티 테스트는 결정론을 위해 greedy를 강제한다.
- **EOS.** `drift run` / `drift up`은 모델의 end-of-sequence id(들)에서 멈춘다(모든 특수 토큰이
  아니라 좁은 집합). 패리티/레퍼런스 경로는 정지 없이 고정된 `max_new_tokens`만큼 실행한다.
- **prefill 다음 decode.** 프롬프트 전체가 한 번 처리되고(위치 `0…S-1`), 그다음 한 번에 토큰 하나씩
  처리된다. 각 노드는 스텝을 가로질러 자기 KV 캐시를 유지한다.

---

## 10 · 샤드를 손수 구동하기

`drift node` / `drift run` 흐름이 쉬운 길이다. 더 낮은 수준의 명령들은 정확한 제어(고정된
포트/범위, 디스커버리 없음)를 주며, 패리티 게이트와 벤치마크가 쓰는 것이 바로 이것이다.

**1) `config.yaml`이 각 머신을 가리키게 한다** — 각 샤드의 `host`/`device`를 설정하고 포트를 연다:

```yaml
shards:
  - { name: "mac",     host: "192.168.0.11", port: 52600, start_layer: 0,  end_layer: 14, device: "mps"  }
  - { name: "windows", host: "192.168.0.22", port: 52601, start_layer: 14, end_layer: 28, device: "cuda" }
```

**2) 각 박스에서 미리 배정된 샤드 서버를 시작한다**(원격 피어를 받으려면 `0.0.0.0`에 바인딩):

```bash
# on the Mac
DRIFT_PORT=52600 python -m drift.shard_server --name mac     --start 0  --end 14 --device mps  --host 0.0.0.0 --preload
# on the Windows PC
set DRIFT_PORT=52601
python -m drift.shard_server --name windows --start 14 --end 28 --device cuda --host 0.0.0.0 --preload
```

**3) 헤드에서 구동한다** — `config.yaml`에서 호스트/포트를 읽는다:

```bash
python -m drift.orchestrator --ping                                  # both shards reply
python -m drift.orchestrator --prompt "Explain pipeline parallelism." # generate over the wire
```

각 방화벽에서 포트를 열어라. 오케스트레이터 노드도 (embed/norm/head를 위해) 모델을 로드하므로, 편한
아무 박스에서든 실행하면 된다.

---

## 11 · CLI 레퍼런스

모든 명령은 `--config`(기본값 `config.yaml`)를 받는다.

### `drift` — 상위 수준 명령

| 명령 | 하는 일 |
|---|---|
| `drift doctor` | 사전 점검: Python/torch/디바이스, 의존성, `config.yaml` 타일링, 포트 도달성(`--nodes`), 방화벽 힌트 |
| `drift up N` | localhost: N개 노드 스폰, 자동 분할, 채팅(또는 원샷이면 `--prompt`) |
| `drift node` | 이 머신을 워커로 실행: 자동 디바이스, `--port`, LAN에 알림, 헤드를 기다림 |
| `drift run` | 헤드: 노드 발견(또는 `--nodes host:port,…`), 자동 분할, 구성, 스트리밍/채팅 |
| `drift serve` | DRIFT 오케스트레이터 위의 OpenAI 호환 HTTP/SSE API(`/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`) |
| `drift keygen` | 네트워크 키와 노드 신원을 만들거나 출력한다(§15). |
| `drift ledger` | receipt journal에서 노드별 기여도를 집계한다 — `--verify` · `--csv`(§15). |

`up`, `node`, `run`은 `--max-new-tokens`를 받고, `run`/`up`은 `--chain`, `--thin`,
`--int8`도 받는다(§15). `run`은 추가로 `--model`, `--nodes`, `--no-discover`, `--expand`를
받는다. `node`는 `--tunnel`, `--join`, `--no-advertise`를 받는다. `serve`는 `--api-key`,
`--cors-origin`, `--served-model-name`, `--host`, `--port`, `--max-concurrent-requests`와
`run`과 같은 노드/모델 라우팅 플래그를 받는다. CLI 채팅은 `--prompt`를 생략하면 된다.

### 하위 수준 모듈

| 모듈 | 주요 플래그 |
|---|---|
| `python -m drift.shard_server` | `--name --start --end --device --host --port --preload --tamper` (+ `DRIFT_PORT`) |
| `python -m drift.orchestrator` | `--ping` · `--prompt` · `--max-new-tokens` · `--ports` |
| `python -m drift.reference` | `--device --out` — 단일 머신 오라클 |
| `python -m drift.parity_test` | `--mode inprocess\|socket` · `--ports` · `--selftest` · `--prefix-match K` |
| `python -m drift.itest` | 실제 노드 게이트: `--nodes N` · `--chain --secure --thin --int8` · `--kill K --tamper K --expand N --ledger` |
| `python -m drift.verify` | trustless recompute spot-check: `--nodes host:port,… --tol` |
| `python -m drift.ledger` | `<journal.jsonl> --verify --csv` |
| `python -m drift.bench` | `--quick --no-socket --json` ([`benchmarks.md`](benchmarks.md) 참조) |

---

## 12 · 와이어 & 세션

- **계약(`drift/protocol.py`, 고정):** 모든 메시지는 4바이트 빅엔디언 길이 프리픽스 + msgpack
  딕셔너리다(키가 설정되면 ChaCha20-Poly1305 frame 하나). 이 프레이밍을 구현하는 런타임이라면 무엇이든
  노드가 될 수 있다 — 와이어 위에 PyTorch는 없다. 메시지 타입: `ping` / `configure` / `prefill` /
  `decode` / `reset` / `peers_get` / `peer_announce`.
- **무엇이 넘어가는가:** 오직 `hidden_states`(fp16, 또는 `--int8`이면 int8) + `position_ids` +
  `input_ids`뿐이다. chain mode에서는 선택 필드 `route`, `collect`가 downstream 경로를 들고,
  각 hop이 서명된 `receipt`를 붙인다. **KV 캐시는 결코 넘어가지 않고** — 각 노드가 자기 것을 유지한다.
  토큰당 트래픽은 fp16에서 `hidden_size × 2` 바이트, int8에서 대략 `hidden_size × 1` 바이트에 정수
  몇 개가 더해질 뿐이다.
- **교체 가능한 노드.** `drift node`는 미배정 상태로 시작하고, 헤드가 `configure`(모델 + 레이어
  범위)를 보내므로 범위를 손으로 적을 일이 없다. 미리 배정된 서버(§10)는 이를 건너뛴다.
- **세션.** 하나의 생성은 하나의 `session_id`다. 각 노드는 세션별 KV 캐시를 보관하고, 생성이 끝나면
  헤드가 `reset`을 보낸다. 노드는 **한 번에 하나의 연결만** 처리한다 — 두 헤드를 한 노드에 붙이지 마라.

---

## 13 · 메모리

오늘로서는 **모든 노드가 전체 모델을 RAM/VRAM에 올린다**고 계획하라: 각 노드는 체크포인트 전체를 로드한
뒤 자기 레이어 슬라이스만 사용하며, 헤드도 (embed/norm/head를 위해) 모델을 로드한다. 노드당 *액티브*
파라미터는 더 작다(기본 2-way 분할에서 가장 무거운 노드의 자기 레이어는 모델의 약 42% —
[`benchmarks.md`](benchmarks.md) 참조). 하지만 로드를 슬라이스만으로 줄이는 것은 향후 과제다. 그때까지는:
각 노드에 맞는 모델을 쓰거나, **더 많은** 노드로 나누어 각 노드의 액티브 몫을 줄여라.

---

## 14 · 트러블슈팅

| 증상 | 가능한 원인 → 해결 |
|---|---|
| `drift run`이 노드를 못 찾음 | mDNS 막힘(게스트/기업 Wi-Fi) → 이름 지정: `drift run --nodes host:port,…`. 각 `drift node`가 자기 주소를 출력했는지 확인하라. |
| `ConnectionRefusedError` | 노드가 안 떴거나, host/port가 틀렸다. 노드를 먼저 시작하라; 포트를 확인하라; `drift doctor --nodes host:port`. |
| localhost에서는 되는데 머신 간에는 안 됨 | 노드가 `127.0.0.1`에 바인딩됨. `drift node`는 기본적으로 `0.0.0.0`에 바인딩한다; 손수 하는 서버라면 `--host 0.0.0.0`을 넘겨라. 방화벽 포트를 열어라. |
| Windows: 피어가 이 노드에 도달 못 함 | Defender Firewall(Private)에서 `python.exe`를 허용하고, Network Discovery를 켜라. |
| 출력이 **뒤쪽** 토큰에서만 드리프트(MPS↔CUDA) | 예상된 벤더 fp16 반올림(§3, §8). 버그가 아니다. |
| **토큰 1–2에서 패리티 FAIL** | float 노이즈가 아니라 진짜 버그(mask/KV/RoPE)다. |
| 로드 시 메모리 부족 | 각 프로세스가 전체 체크포인트를 로드한다(§13). 더 작은 모델, 더 많은 노드를 쓰거나, 벤치에는 `--no-socket`을 쓰라. |
| `unsupported wire dtype` | compute `dtype`은 `float16` 또는 `float32`여야 한다(§8). `int8`은 compute dtype이 아니라 `--int8` 와이어 옵션이다(§15). |
| `drift serve`가 노드 0개를 찾았다고 함 | 먼저 `drift node`를 띄우고, `--nodes host:port,...`를 넘기거나 `--discover-timeout`을 늘려라. |
| OpenAI 클라이언트가 `401`을 받음 | 서버가 `--api-key`/`DRIFT_API_KEY`로 시작됐다. `Authorization: Bearer <key>` 또는 `x-api-key: <key>`를 보내라. |
| OpenAI 클라이언트가 `model ... is not served`를 받음 | `GET /v1/models`에 나온 id를 쓰거나, `--served-model-name`을 설정한 뒤 그 이름을 정확히 보내라. |
| `--thin` 모드에서 `/v1/embeddings`나 sampling이 실패 | thin mode에서는 logits/hidden states가 head에 없으므로 embeddings와 sampling에는 non-thin 모드를 사용하라. |
| `refusing --tunnel without a network key` | 공개 endpoint가 열린 compute가 되는 것을 막기 위해서다. 먼저 `drift keygen` 후 `DRIFT_NETWORK_KEY`를 export하라(§15). |
| 노드가 SUSPECT로 표시됨 | receipt verifier가 불일치를 잡았다(§15). 노드 버전/상태를 확인하라. 실제 tamper일 수도 있다. |
| Mac에서 드문 MPS op 오류 | 프로세스를 띄운 셸에 `export PYTORCH_ENABLE_MPS_FALLBACK=1`이 설정됐는지 확인하라. |

---

## 15 · 탈중앙화 (v1.0)

분할 추론의 핵심은 그대로이고, 아래 기능들은 그 위에 얹는 opt-in layer다. 모두 bitwise gate를 통과하도록
검증되어 있으며, `--int8`만 relaxed gate를 사용한다. 확인은 `python -m drift.itest`로 한다.

### Peer-to-peer chain — `--chain`
기본값은 모든 hop이 head를 거치는 star 구조다. `--chain`은 hidden state를
node→node→…→tail→head로 흘려 보낸다. 토큰당 tensor crossing은 `2N`에서 `N+1`로 줄고,
head bandwidth는 노드 수에 대해 O(1)이 된다.

```bash
drift up 3 --chain
drift run --chain --nodes a:52600,b:52601,c:52602 --prompt "…"
```

### Weightless head — `--thin` (`--chain`을 포함)
`embed_tokens`는 첫 노드로, `norm`+`lm_head`+`argmax`는 마지막 노드로 이동한다. head는 tokenizer만
들고 token id만 주고받으며, tensor를 들지 않는다.

```bash
drift up 2 --thin
```

### 암호화 + 인증된 와이어 — `drift keygen`
네트워크는 하나의 pre-shared key를 공유한다. 각 연결은 X25519 ECDH → HKDF(PSK mix) →
ChaCha20-Poly1305를 사용하며, 키가 없는 dialer는 연결이 끊긴다.

```bash
drift keygen                       # ~/.config/drift/network.key + identity를 쓰고 key를 출력
export DRIFT_NETWORK_KEY=<hex>     # 모든 머신(head + nodes)에 설정하면 암호화됨
drift keygen --print               # 공유할 key를 다시 출력
```

키가 없으면 plaintext다. 자신이 소유한 LAN에서는 괜찮지만, `drift node --tunnel`은 키 없이 실행을
거부한다. 공개 endpoint가 열린 compute가 되면 안 되기 때문이다.

### 어디서나 참여 — `drift node --join` / `drift run --expand`
노드는 seed 하나를 통해 gossip으로 네트워크에 참여하고, head는 membership 전체를 발견해 그 위로
분할한다.

```bash
drift node --join seed-host:52600
drift run --expand --nodes seed-host:52600
```

### Failover
생성 도중 노드가 죽으면 head가 생존 노드와 spare 위로 다시 분할하고, 지금까지의 sequence를 replay한 뒤
계속 생성한다. 중단되지 않은 실행과 bitwise-identical해야 한다. 따로 설정할 것은 없으며, 살아남은 노드가
없으면 깔끔한 `NodeUnavailable`을 드러낸다.

### 검증 & 기여 ledger
각 hop은 Ed25519 receipt에 서명하고, head는 live traffic에서 이를 검증한다. journal을 지정하면 기록하고,
나중에 집계할 수 있다.

```bash
export DRIFT_JOURNAL=~/drift.jsonl && drift run --chain --prompt "…"
drift ledger ~/drift.jsonl --verify --csv out.csv
```

`drift verify --nodes host:port,…`는 내가 소유하지 않은 노드에 대한 recompute spot-check다.

### Half-size wire — `--int8`
hidden state를 group-wise int8로 보낸다(대략 0.51× bytes). 손실 압축이므로 bitwise가 아니라 relaxed
gate로 평가한다. `drift itest --int8`로 측정하라.

```bash
drift run --chain --int8 --prompt "…"
```

### Env vars

| Var | 효과 |
|---|---|
| `DRIFT_NETWORK_KEY` | hex/base64 PSK — 와이어를 암호화 + 인증한다. |
| `DRIFT_NETWORK_KEY_FILE` | key 파일 경로(기본 `~/.config/drift/network.key`). |
| `DRIFT_IDENTITY_FILE` | 이 노드의 Ed25519 identity(기본 `~/.config/drift/identity.key`). |
| `DRIFT_ADVERTISE_HOST` | peer가 이 노드에 도달할 때 사용할 주소(기본 LAN ip). |
| `DRIFT_JOURNAL` | `drift ledger`용 verified receipt append 경로. |

발행된 수치는 `python -m drift.bench`로 재현하라. 방법론은
[`benchmarks.md`](benchmarks.md)에 있다.
