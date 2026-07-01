# DRIFT — 운영 매뉴얼

**DRIFT를 실전으로 돌리는 법 — 처음부터 끝까지.** 언어: [English](manual.md) ·
**한국어** · [中文](manual.zh.md) · [日本語](manual.ja.md)

앞의 절반이 곧 전부다: 설치하고, 시도해 보고, 여러 머신에 걸쳐 하나의 모델을 실행한다.
뒤의 절반 — **커스터마이즈 & 파인튜닝** — 은 기본값만으로 충분하지 않을 때에만 필요하다.
벤치마크 방법론과 수치는 [`benchmarks.md`](benchmarks.md)를 참고하라.

---

## 목차

**실행하기**
1. [설치](#1--설치)
2. [한 대의 머신에서 실행하기](#2--한-대의-머신에서-실행하기)
3. [여러 대에 걸쳐 실행 — 실전 예제](#3--여러-대에-걸쳐-실행--실전-예제)

**커스터마이즈 & 파인튜닝**
4. [`config.yaml` 레퍼런스](#4--configyaml-레퍼런스)
5. [분할 지점 선택](#5--분할-지점-선택)
6. [모델](#6--모델)
7. [디바이스 & dtype](#7--디바이스--dtype)
8. [생성이 동작하는 방식](#8--생성이-동작하는-방식)
9. [샤드를 손수 구동하기](#9--샤드를-손수-구동하기)
10. [CLI 레퍼런스](#10--cli-레퍼런스)
11. [와이어 & 세션](#11--와이어--세션)
12. [메모리](#12--메모리)
13. [트러블슈팅](#13--트러블슈팅)

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

**커스터마이즈 & 파인튜닝** — 아래 내용은 모두 선택 사항으로, 위의 한 명령 흐름만으로 충분하지 않을
때를 위한 것이다(다른 모델, 고르지 않은 분할, 정확한 포트, 조각들을 손수 구동하기).

---

## 4 · `config.yaml` 레퍼런스

`config.yaml`은 모델, 정밀도, 그리고 (손수 하는 흐름에서의) 샤드 테이블의 유일한 진실의 원천이다.
`drift up` / `drift run`은 이 파일에서 `model_id`, `dtype`, `generation`을 읽고, 분할은 스스로
계산한다.

```yaml
model_id: "Qwen/Qwen2.5-1.5B-Instruct"   # any HF causal-LM id
dtype: "float16"                          # float16 | float32  (see §7)
device: "mps"                             # default device: mps | cuda | cpu
port: 52600                               # default port for a shard that sets none

shards:                                   # only used by the by-hand flow (§9) / `drift run` fallback
  - { name: "mac",     host: "127.0.0.1", port: 52600, start_layer: 0,  end_layer: 14, device: "mps" }
  - { name: "windows", host: "127.0.0.1", port: 52601, start_layer: 14, end_layer: 28, device: "mps" }

generation:
  max_new_tokens: 50
  prompt: "Give me a short introduction to large language models."
```

| 키 | 의미 |
|---|---|
| `model_id` | Hugging Face 모델 id. 로컬 HF 캐시에 한 번 다운로드된다. |
| `dtype` | 연산 dtype **이면서** 와이어 dtype. `float16`(기본값, 무손실 CPU 왕복) 또는 `float32`. `bfloat16`은 와이어에서 **유효하지 않다** — §7. |
| `device` | 헤드의 기본 디바이스이자, 자기 디바이스를 생략한 샤드의 기본 디바이스. `mps` / `cuda` / `cpu`. |
| `port` | `port`도 `DRIFT_PORT`도 없는 샤드의 폴백 포트. |
| `shards[]` | 손수 하는 흐름(§9)과, 디스커버리가 아무것도 못 찾았을 때의 `drift run` 폴백에 쓰이는 순서 있는 샤드 테이블. |
| `shards[].host` / `port` | 오케스트레이터가 이 샤드에 연결하는 곳. 로컬이면 `127.0.0.1`, 원격이면 LAN IP. |
| `shards[].start_layer` / `end_layer` | 반열림(half-open) 디코더 레이어 범위 `[start, end)`. |
| `shards[].device` | 이 샤드의 디바이스. |
| `generation.max_new_tokens` | 기본 토큰 예산(`--max-new-tokens`로 덮어씀). |
| `generation.prompt` | `--prompt`가 생략됐을 때의 기본 프롬프트. |

---

## 5 · 분할 지점 선택

`drift run`은 노드 수로 고르게 분할한다. 이것을 신경 쓸 일은 손수 하는 흐름(§9)이나 고르지 않은
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

## 6 · 모델

엔진은 아키텍처를 하드코딩하는 대신 로드된 모델을 **인트로스펙션(introspect)**한다(디코더 레이어 클래스,
`rotary_emb`, 캐시 타입, 레이어별 어텐션) — 그래서 새 계열은 id만으로 그대로 끼워 넣을 수 있다.
`config.yaml`에 `model_id`를 설정하거나(또는 `drift run --model <id>`), 그 외에는 아무것도 필요 없다.

| 모델 | 레이어 | 고른 분할 | 비고 |
|---|---:|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` (기본) | 28 | `0–14 / 14–28` | 평범한 디코더; 패리티 기준선 |
| `google/gemma-4-E2B-it` | 35 | `0–18 / 18–35` | 레이어별 임베딩(Per-Layer Embeddings; 노드가 `input_ids`로부터 재구성), 이중 RoPE θ, 하이브리드 어텐션, `HybridCache`; `transformers ≥ 5.5` 필요 |

더 큰 모델은 단지 나눌 레이어가 더 많을 뿐이다 — 그 레이어 수에 맞춰 범위를 연속적으로 유지하라.

---

## 7 · 디바이스 & dtype

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

## 8 · 생성이 동작하는 방식

- **Greedy.** 레퍼런스 오라클과 오케스트레이터 모두 매 스텝에서 `argmax`를 고른다. CLI에는 아직
  temperature/top-p 샘플링이 없다. 패리티 테스트는 결정론을 위해 greedy를 강제한다.
- **EOS.** `drift run` / `drift up`은 모델의 end-of-sequence id(들)에서 멈춘다(모든 특수 토큰이
  아니라 좁은 집합). 패리티/레퍼런스 경로는 정지 없이 고정된 `max_new_tokens`만큼 실행한다.
- **prefill 다음 decode.** 프롬프트 전체가 한 번 처리되고(위치 `0…S-1`), 그다음 한 번에 토큰 하나씩
  처리된다. 각 노드는 스텝을 가로질러 자기 KV 캐시를 유지한다.

---

## 9 · 샤드를 손수 구동하기

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

## 10 · CLI 레퍼런스

모든 명령은 `--config`(기본값 `config.yaml`)를 받는다.

### `drift` — 상위 수준 명령

| 명령 | 하는 일 |
|---|---|
| `drift doctor` | 사전 점검: Python/torch/디바이스, 의존성, `config.yaml` 타일링, 포트 도달성(`--nodes`), 방화벽 힌트 |
| `drift up N` | localhost: N개 노드 스폰, 자동 분할, 채팅(또는 원샷이면 `--prompt`) |
| `drift node` | 이 머신을 워커로 실행: 자동 디바이스, `--port`, LAN에 알림, 헤드를 기다림 |
| `drift run` | 헤드: 노드 발견(또는 `--nodes host:port,…`), 자동 분할, 구성, 스트리밍/채팅 |

`up`, `node`, `run`은 `--max-new-tokens`를 받고, `run`은 추가로 `--model`, `--nodes`,
`--no-discover`를 받는다. `run`/`up`에서 `--prompt`를 생략하면 인터랙티브 채팅이 된다.

### 하위 수준 모듈

| 모듈 | 주요 플래그 |
|---|---|
| `python -m drift.shard_server` | `--name --start --end --device --host --port --preload` (+ `DRIFT_PORT`) |
| `python -m drift.orchestrator` | `--ping` · `--prompt` · `--max-new-tokens` · `--ports` |
| `python -m drift.reference` | `--device --out` — 단일 머신 오라클 |
| `python -m drift.parity_test` | `--mode inprocess\|socket` · `--ports` · `--selftest` |
| `python -m drift.bench` | `--quick --no-socket --json` ([`benchmarks.md`](benchmarks.md) 참조) |

---

## 11 · 와이어 & 세션

- **계약(`drift/protocol.py`, 고정):** 모든 메시지는 4바이트 빅엔디언 길이 프리픽스 + msgpack
  딕셔너리다. 이 프레이밍을 구현하는 런타임이라면 무엇이든 노드가 될 수 있다 — 와이어 위에 PyTorch는
  없다. 메시지 타입: `ping` / `configure` / `prefill` / `decode` / `reset`.
- **무엇이 넘어가는가:** 오직 `hidden_states`(fp16) + `position_ids` + `input_ids`뿐이다. **KV
  캐시는 결코 넘어가지 않고** — 각 노드가 자기 것을 유지한다. 토큰당 트래픽은 `hidden_size × 2`
  바이트에 정수 몇 개(수 KB)가 더해질 뿐이며, 파라미터 수와 무관하다.
- **교체 가능한 노드.** `drift node`는 미배정 상태로 시작하고, 헤드가 `configure`(모델 + 레이어
  범위)를 보내므로 범위를 손으로 적을 일이 없다. 미리 배정된 서버(§9)는 이를 건너뛴다.
- **세션.** 하나의 생성은 하나의 `session_id`다. 각 노드는 세션별 KV 캐시를 보관하고, 생성이 끝나면
  헤드가 `reset`을 보낸다. 노드는 **한 번에 하나의 연결만** 처리한다 — 두 헤드를 한 노드에 붙이지 마라.

---

## 12 · 메모리

오늘로서는 **모든 노드가 전체 모델을 RAM/VRAM에 올린다**고 계획하라: 각 노드는 체크포인트 전체를 로드한
뒤 자기 레이어 슬라이스만 사용하며, 헤드도 (embed/norm/head를 위해) 모델을 로드한다. 노드당 *액티브*
파라미터는 더 작다(기본 2-way 분할에서 가장 무거운 노드의 자기 레이어는 모델의 약 42% —
[`benchmarks.md`](benchmarks.md) 참조). 하지만 로드를 슬라이스만으로 줄이는 것은 향후 과제다. 그때까지는:
각 노드에 맞는 모델을 쓰거나, **더 많은** 노드로 나누어 각 노드의 액티브 몫을 줄여라.

---

## 13 · 트러블슈팅

| 증상 | 가능한 원인 → 해결 |
|---|---|
| `drift run`이 노드를 못 찾음 | mDNS 막힘(게스트/기업 Wi-Fi) → 이름 지정: `drift run --nodes host:port,…`. 각 `drift node`가 자기 주소를 출력했는지 확인하라. |
| `ConnectionRefusedError` | 노드가 안 떴거나, host/port가 틀렸다. 노드를 먼저 시작하라; 포트를 확인하라; `drift doctor --nodes host:port`. |
| localhost에서는 되는데 머신 간에는 안 됨 | 노드가 `127.0.0.1`에 바인딩됨. `drift node`는 기본적으로 `0.0.0.0`에 바인딩한다; 손수 하는 서버라면 `--host 0.0.0.0`을 넘겨라. 방화벽 포트를 열어라. |
| Windows: 피어가 이 노드에 도달 못 함 | Defender Firewall(Private)에서 `python.exe`를 허용하고, Network Discovery를 켜라. |
| 출력이 **뒤쪽** 토큰에서만 드리프트(MPS↔CUDA) | 예상된 벤더 fp16 반올림(§3, §7). 버그가 아니다. |
| **토큰 1–2에서 패리티 FAIL** | float 노이즈가 아니라 진짜 버그(mask/KV/RoPE)다. |
| 로드 시 메모리 부족 | 각 프로세스가 전체 체크포인트를 로드한다(§12). 더 작은 모델, 더 많은 노드를 쓰거나, 벤치에는 `--no-socket`을 쓰라. |
| `unsupported wire dtype` | `dtype`은 `float16` 또는 `float32`여야 한다(§7). |
| Mac에서 드문 MPS op 오류 | 프로세스를 띄운 셸에 `export PYTORCH_ENABLE_MPS_FALLBACK=1`이 설정됐는지 확인하라. |

---

발행된 수치는 `python -m drift.bench`로 재현하라. 방법론은
[`benchmarks.md`](benchmarks.md)에 있다.
