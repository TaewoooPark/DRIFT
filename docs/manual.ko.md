# DRIFT — 운영 매뉴얼

**DRIFT를 실전으로 돌릴 때 제어할 수 있는 모든 것.** 언어: [English](manual.md) ·
**한국어** · [中文](manual.zh.md) · [日本語](manual.ja.md)

벤치마크 방법론과 실측 수치는
[`benchmarks.md`](benchmarks.md)를 참고하라. 이 문서는 시스템을 *운영*하는 방법을 다룬다.

---

## 목차

1. [설치](#1--설치)
2. [60초 실행](#2--60초-실행)
3. [`config.yaml` 레퍼런스](#3--configyaml-레퍼런스)
4. [분할 지점 선택](#4--분할-지점-선택)
5. [두 머신에 걸쳐 실행하기 (Mac + Windows)](#5--두-머신에-걸쳐-실행하기-mac--windows)
6. [CLI 레퍼런스](#6--cli-레퍼런스)
7. [모델](#7--모델)
8. [디바이스 & dtype](#8--디바이스--dtype)
9. [생성이 동작하는 방식](#9--생성이-동작하는-방식)
10. [와이어 & 세션](#10--와이어--세션)
11. [메모리](#11--메모리)
12. [트러블슈팅](#12--트러블슈팅)

---

## 1 · 설치

**Python 3.12**(PyTorch는 아직 3.14 휠을 내놓지 않았다)과
[`uv`](https://github.com/astral-sh/uv)가 필요하다. 번들된 두 모델 모두 **게이트가 없어서** Hugging
Face 로그인이 필요하지 않다.

```bash
git clone https://github.com/TaewoooPark/DRIFT && cd DRIFT
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install "torch" "transformers>=5.5" safetensors msgpack numpy huggingface_hub accelerate pyyaml
export PYTORCH_ENABLE_MPS_FALLBACK=1        # lets rare unimplemented MPS ops fall back to CPU
```

Windows/NVIDIA에서는 기본 휠 대신 사용 중인 툴킷에 맞는 CUDA 빌드의 PyTorch를 설치하라. 그 외에는
전부 동일하다. `PYTORCH_ENABLE_MPS_FALLBACK`은 Mac 전용이며 다른 환경에서는 무해하다.

---

## 2 · 60초 실행

**한 대의** 머신(localhost)에서 두 샤드를 띄운 뒤, TCP 위로 실제 생성을 수행한다:

```bash
# terminal A — front half, layers [0,14)
DRIFT_PORT=52600 python -m drift.shard_server --name mac     --start 0  --end 14 --device mps --preload
# terminal B — back half, layers [14,28)
DRIFT_PORT=52601 python -m drift.shard_server --name windows --start 14 --end 28 --device mps --preload
# terminal C — health check, then generate
python -m drift.orchestrator --ping   --ports 52600,52601
python -m drift.orchestrator --prompt "Explain pipeline parallelism in two sentences." --ports 52600,52601
```

마지막 명령이 **바로** 이 제품이다. 오케스트레이터는 프롬프트를 임베딩하고, hidden state를 샤드 A를
거쳐 샤드 B로 라우팅한 뒤, 디코딩된 답을 출력한다 — 각 샤드는 오직 자기 레이어만 실행한다. 아래
내용은 모두 그 실행이 하는 일을 바꾸는 방법이다.

---

## 3 · `config.yaml` 레퍼런스

`config.yaml`이 유일한 진실의 원천이다. 오케스트레이터, 샤드 서버, 레퍼런스 오라클, 벤치마크가 모두
이 파일을 읽는다.

```yaml
model_id: "Qwen/Qwen2.5-1.5B-Instruct"   # any HF causal-LM id
dtype: "float16"                          # float16 | float32  (see §8)
device: "mps"                             # default device: mps | cuda | cpu
port: 52600                               # default single-port (overridden per shard below)

shards:
  - { name: "mac",     host: "127.0.0.1", port: 52600, start_layer: 0,  end_layer: 14, device: "mps" }
  - { name: "windows", host: "127.0.0.1", port: 52601, start_layer: 14, end_layer: 28, device: "mps" }

generation:
  max_new_tokens: 50
  prompt: "Give me a short introduction to large language models."
```

| 키 | 의미 |
|---|---|
| `model_id` | Hugging Face 모델 id. 로컬 HF 캐시에 한 번 다운로드된다. |
| `dtype` | 연산 dtype **이면서** 와이어 dtype. `float16`(기본값, 무손실 CPU 왕복) 또는 `float32`. `bfloat16`은 와이어에서 **유효하지 않다** — §8 참조. |
| `device` | 샤드가 자기 디바이스를 생략했을 때의 기본 디바이스. `mps` / `cuda` / `cpu`. |
| `port` | `port`를 생략하고 `DRIFT_PORT`도 설정하지 않은 샤드의 폴백 포트. |
| `shards[]` | 순서가 있는 샤드 목록. 오케스트레이터는 **이 순서대로** 라우팅한다. |
| `shards[].name` | 논리적 이름. `--ports`/라우팅에 쓰이고 `--ping`에 표시된다. |
| `shards[].host` | 오케스트레이터가 이 샤드에 연결할 주소. 로컬이면 `127.0.0.1`, 원격이면 LAN IP(§5). |
| `shards[].port` | 샤드가 리슨하는 / 오케스트레이터가 연결하는 TCP 포트. |
| `shards[].start_layer` / `end_layer` | 이 샤드가 소유하는 반열림(half-open) 디코더 레이어 범위 `[start, end)`. |
| `shards[].device` | 이 샤드의 디바이스(Mac에서는 `mps`, PC에서는 `cuda`…). |
| `generation.max_new_tokens` | `reference`와 오케스트레이터 데모의 기본 토큰 개수. |
| `generation.prompt` | `--prompt`가 생략됐을 때 `reference`와 오케스트레이터의 기본 프롬프트. |

---

## 4 · 분할 지점 선택

`shards[]` 범위는 모델의 디코더 레이어를 **타일링(tile)**해야 한다: 연속적이고, 순서대로이며, 빈틈도
겹침도 없이 `[0, num_hidden_layers)`를 전부 덮어야 한다. 오케스트레이터 자신은 `embed_tokens`, 최종
norm, `lm_head`를 소유하는데 — 이들은 어떤 샤드 범위에도 **속하지 않는다**.

```
model: 28 decoder layers (Qwen2.5-1.5B)
        └── shard A: [0, 14)   ── shard B: [14, 28)      ✅ tiles 0..28
        └── [0, 10) / [10, 20) / [20, 28)                ✅ three shards, also valid
        └── [0, 14) / [16, 28)                           ❌ gap at 14–15
        └── [0, 16) / [14, 28)                           ❌ overlap at 14–15
```

- **샤드 개수**는 그저 `shards[]`의 길이일 뿐이다 — 2개는 데모이고 더 많아도 무방하다. 오케스트레이터는
  목록 순서대로 모두를 거쳐 라우팅한다.
- **어디서 자를지**는 정확성 면에서는 아무런 트레이드오프가 없다(어떤 타일링이든 한 디바이스에서는 비트
  단위로 정확하다). 다만 *얼마만큼의 연산과 가중치 메모리*가 각 노드에 실리는지만 달라진다. 레이어를 고르게
  나누는 것이 기본이며, 한쪽 머신이 더 빠르면 치우치게 나누라.
- **레이어 수:** Qwen2.5-1.5B = 28, Gemma-4-E2B = 35. 아무 실행의 시작 로그(`reference`가 `layers=…`를
  출력한다)나 모델 config에서 읽을 수 있다.

---

## 5 · 두 머신에 걸쳐 실행하기 (Mac + Windows)

§2의 localhost 실행은 세 가지만 바꾸면 실제 클러스터가 된다.

**1) config가 각 머신을 가리키게 한다.** 오케스트레이터 노드에서 각 샤드의 `host`를 해당 머신의 LAN
IP로 설정하고 열린 포트를 고른다:

```yaml
shards:
  - { name: "mac",     host: "192.168.0.11", port: 52600, start_layer: 0,  end_layer: 14, device: "mps"  }
  - { name: "windows", host: "192.168.0.22", port: 52601, start_layer: 14, end_layer: 28, device: "cuda" }
```

**2) 각 샤드 서버를 도달 가능한 주소에 바인딩한다.** 서버는 기본적으로 `127.0.0.1`(로컬 전용)로
바인딩된다. 원격 연결을 받으려면 `--host 0.0.0.0`으로 시작하라:

```bash
# on the Mac (192.168.0.11)
DRIFT_PORT=52600 python -m drift.shard_server --name mac     --start 0  --end 14 --device mps  --host 0.0.0.0 --preload
# on the Windows PC (192.168.0.22)
set DRIFT_PORT=52601
python -m drift.shard_server --name windows --start 14 --end 28 --device cuda --host 0.0.0.0 --preload
```

**3) 헤드를 가진 노드에서 오케스트레이터를 실행한다.** 오케스트레이터는 `host`/`port`를
`config.yaml`에서 곧바로 읽으므로, **`--ports`는 생략하라**(이것은 호스트가 아니라 포트만 덮어쓴다):

```bash
python -m drift.orchestrator --ping                                  # both shards should reply
python -m drift.orchestrator --prompt "Write a haiku about winter."  # front half on Apple, back half on NVIDIA
```

참고:
- 각 머신의 방화벽에서 선택한 포트를 열어라.
- 오케스트레이터 노드도 (embed/norm/head를 위해) 모델을 로드하므로, 편한 아무 박스에서든 실행하면 된다 —
  보통은 샤드 A와 같은 박스다.
- **서로 다른 GPU 벤더** 사이(MPS ↔ CUDA)에서는 fp16이 각각 약간씩 다르게 반올림되므로, greedy 출력이
  *뒤쪽* 토큰에서 발산할 수 있다. 초기 토큰은 일치하고 텍스트는 일관성을 유지한다. 이는 예상된
  동작이다. **같은** 디바이스 계열에서는 분할이 비트 단위로 정확하다(
  [`benchmarks.md`](benchmarks.md) 참조).

---

## 6 · CLI 레퍼런스

모든 엔트리 포인트는 `--config`(기본값 `config.yaml`)를 받는다.

### `drift.shard_server` — 샤드 하나 실행

```bash
DRIFT_PORT=<port> python -m drift.shard_server [flags]
```

| 플래그 | 기본값 | 의미 |
|---|---|---|
| `--name` | `shard` | 논리적 샤드 이름. |
| `--start` / `--end` | config shard[0]에서 | 디코더 레이어 범위 `[start, end)`. |
| `--device` | config `device` | `mps` / `cuda` / `cpu`. |
| `--host` | `127.0.0.1` | 바인딩 주소. 원격 노드를 받으려면 `0.0.0.0`을 사용하라. |
| `--port` | `$DRIFT_PORT` 또는 config `port` | 리슨 포트. |
| `--preload` | off | 리슨하기 **전에** 가중치를 로드한다(권장; 첫 요청의 콜드 스타트를 피한다). |

### `drift.orchestrator` — 헬스 체크 & 생성

```bash
python -m drift.orchestrator [--ping] [--prompt "…"] [--max-new-tokens N] [--ports P1,P2]
```

| 플래그 | 의미 |
|---|---|
| `--ping` | 모든 샤드를 TCP로 ping한 뒤 종료한다(이것이 "M0" 도달성 체크다). |
| `--prompt` | 생성에 쓸 프롬프트. 없으면 `generation.prompt`로 폴백한다. |
| `--max-new-tokens` | 토큰 예산. 없으면 `generation.max_new_tokens`로 폴백한다. |
| `--ports` | 각 샤드의 config 포트를 덮어쓰는 쉼표 구분 포트(호스트는 그대로 — 로컬 용도). |

여기서의 생성은 greedy이며 **EOS에서 멈춘다**.

### `drift.reference` — 단일 머신 오라클

```bash
python -m drift.reference [--device DEV] [--out reference_out.npz]
```

전체 모델을 한 디바이스에 로드하고 `generation.prompt`로부터 `generation.max_new_tokens`만큼 greedy로
생성하며, 토큰 id + 첫 스텝 logit을 저장한다. 이것이 분할 경로를 검증하는 기준이 되는 정답이다.

### `drift.parity_test` — 정확성 게이트

```bash
python -m drift.parity_test --mode inprocess               # split in one process, no sockets
python -m drift.parity_test --mode socket --ports 52600,52601   # split over TCP (servers must be up)
python -m drift.parity_test --selftest                     # 6 prompts (EN/code/Korean, n=1…180)
```

| 플래그 | 의미 |
|---|---|
| `--mode` | `inprocess`(직접 호출) 또는 `socket`(와이어 위로). |
| `--ports` | socket 모드용 포트. |
| `--ref` | 비교 대상 레퍼런스 파일(기본값 `reference_out.npz`). |
| `--selftest` | 새 레퍼런스를 다시 도출해 여러 프롬프트/길이에 걸쳐 비교한다. npz가 필요 없다. |

### `drift.bench` — 벤치마크

```bash
python -m drift.bench [--quick] [--no-socket] [--json out.json]
```

[`benchmarks.md`](benchmarks.md)를 참고하라. `--no-socket`은 RAM이 적은 머신에서 서버 스폰 오버헤드
측정을 건너뛴다.

---

## 7 · 모델

엔진은 아키텍처를 하드코딩하는 대신 로드된 모델을 **인트로스펙션(introspect)**한다(디코더 레이어 클래스,
`rotary_emb`, 캐시 타입, 레이어별 어텐션). 그래서 새 계열은 id만으로 그대로 끼워 넣을 수 있다.

| 모델 | 레이어 | 분할 예시 | 비고 |
|---|---:|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` (기본) | 28 | `0–14 / 14–28` | 평범한 디코더; 패리티 기준선 |
| `google/gemma-4-E2B-it` | 35 | `0–18 / 18–35` | 레이어별 임베딩(Per-Layer Embeddings; 샤드가 `input_ids`로부터 재구성), 이중 RoPE θ, 하이브리드 어텐션, `HybridCache`; `transformers ≥ 5.5` 필요 |

모델을 바꾸려면 `config.yaml`에서 `model_id`와 유효한 타일링을 설정하면 된다 — 그 외에는 아무것도
필요 없다. 더 큰 모델은 단지 나눌 레이어가 더 많을 뿐이다. 그 레이어 수에 맞춰 범위를 연속적으로 유지하라.

---

## 8 · 디바이스 & dtype

**디바이스** — `mps`(Apple GPU), `cuda`(NVIDIA GPU), `cpu`(이식성 있음, 느림). 각 샤드의 `device`는
다른 샤드와 독립적이며, 그 독립성이 바로 핵심이다.

**dtype** — `float16`(기본값) 또는 `float32`. 와이어는 텐서를 이 dtype의 원시 바이트로 직렬화하며, fp16
CPU 왕복은 비트 손실이 없으므로 직렬화가 결과를 교란하는 일은 없다. `bfloat16`은 와이어에서 **지원되지
않는다** — bf16 연산이 필요하다면 그것은 아직 연결되어 있지 않으니 `float16`을 사용하라.

**같은 디바이스 vs 혼합 벤더:** 같은 디바이스 계열의 두 샤드는 단일 머신을 **비트 단위로** 재현한다.
`mps`와 `cuda`를 섞으면 벤더 간 비트 수준 fp16 반올림 차이가 생기므로 greedy 디코딩이 뒤쪽 토큰에서
발산할 수 있다(예상된 동작이며 버그가 아니다 — §5).

---

## 9 · 생성이 동작하는 방식

- **Greedy 전용.** 레퍼런스 오라클과 오케스트레이터 모두 매 스텝에서 `argmax`를 고른다. CLI에는
  temperature/top-p 샘플링이 노출되지 않는다. 패리티 테스트는 출력이 결정론적이고 비교 가능하도록 greedy를
  강제한다.
- **EOS.** 오케스트레이터의 `--prompt` 경로는 모델의 end-of-sequence id(들)에서 멈춘다(진짜 EOS만 포함하는
  좁은 집합이며, 모든 특수 토큰을 포함하지 않는다). 패리티/레퍼런스 경로는 정확한 비교를 위해 조기 종료 없이
  고정된 `max_new_tokens`만큼 실행한다.
- **prefill 다음 decode.** 프롬프트 전체가 한 번 처리되고(prefill, 위치 `0…S-1`), 그다음 한 번에 토큰
  하나씩 처리된다(decode, `S, S+1, …`). 각 샤드는 스텝을 가로질러 자기 KV 캐시를 유지한다.

---

## 10 · 와이어 & 세션

- **계약(`drift/protocol.py`, 고정):** 모든 메시지는 4바이트 빅엔디언 길이 프리픽스 + msgpack 딕셔너리다.
  이 프레이밍을 구현하는 런타임이라면 무엇이든 노드가 될 수 있다 — 와이어 위에 PyTorch는 없다.
- **무엇이 넘어가는가:** 오직 `hidden_states`(fp16) + `position_ids` + `input_ids`뿐이다. **KV 캐시는
  결코 넘어가지 않고** — 각 샤드가 자기 것을 유지한다. 토큰당 트래픽은 `hidden_size × 2` 바이트에 정수 몇
  개(수 KB)가 더해질 뿐이며, 파라미터 수와 무관하다.
- **세션.** 하나의 생성은 하나의 `session_id`(기본값 `s0`)다. 각 샤드는 세션별 KV 캐시를 보관하고,
  오케스트레이터는 생성이 끝나면 `reset`을 보낸다. 샤드 서버는 **한 번에 하나의 연결만** 처리한다(순차적;
  동시성은 향후 과제) — 그러니 두 오케스트레이터를 한 샤드에 동시에 붙이지 마라.
- **TCP 튜닝.** 연결은 `TCP_NODELAY`를 설정하고, 서버는 `SO_REUSEADDR`를 설정한다.

---

## 11 · 메모리

오늘로서는 **모든 노드가 전체 모델을 RAM/VRAM에 올린다**고 계획하라: 각 샤드 서버는 현재 체크포인트 전체를
로드한 뒤 자기 레이어 슬라이스만 사용하며, 오케스트레이터도 (embed/norm/head를 위해) 모델을 로드한다. 노드당
*액티브* 파라미터는 더 작다 — 가장 무거운 노드의 자기 레이어는 기본 2-way 분할에서 모델의 약 42%다(
[`benchmarks.md`](benchmarks.md) 참조) — 하지만 디스크에서 로드하는 양을 슬라이스만으로 줄이는 것은 향후
과제다. 그때까지는:

- 각 노드의 메모리에 맞는 모델을 쓰거나, **더 많은** 노드로 나누어 각 노드의 *액티브* 몫을 줄여라.
- 메모리가 빠듯한 Mac에서는 `python -m drift.bench --no-socket`이 추가로 전체 모델 서버 프로세스를 스폰하는
  것을 건너뛴다.

---

## 12 · 트러블슈팅

| 증상 | 가능한 원인 → 해결 |
|---|---|
| 오케스트레이터에서 `ConnectionRefusedError` | 샤드가 아직 안 떴거나, `host`/`port`가 틀렸다. 서버를 먼저 시작하라. `listening on …`이 출력됐는지 확인하고, 포트가 일치하는지 체크하라. |
| localhost에서는 되는데 머신 간에는 안 됨 | 서버가 `127.0.0.1`에 바인딩됨. `--host 0.0.0.0`으로 재시작하고 방화벽 포트를 열어라. |
| 한 샤드에서 `--ping`이 실패 | 그 샤드 프로세스가 죽었거나 포트/호스트가 틀렸다. 그 샤드의 `--start/--end/--device`와 모델 로드 여부를 다시 확인하라. |
| **토큰 1–2에서 패리티 FAIL** | float 노이즈가 아니라 진짜 버그(mask/KV/RoPE)다 — 분할 로직이 발산했다. |
| greedy 출력이 **뒤쪽** 토큰에서만 드리프트, MPS↔CUDA | 예상된 벤더 fp16 반올림(§5). 버그가 아니다. |
| 로드 시 메모리 부족 | 각 프로세스가 전체 체크포인트를 로드한다(§11). 더 작은 모델, 더 많은 샤드를 쓰거나, 벤치에는 `--no-socket`을 쓰라. |
| `unsupported wire dtype` | `dtype`은 `float16` 또는 `float32`여야 한다(§8). |
| Mac에서 드문 MPS op 오류 | 프로세스를 띄운 셸에 `export PYTORCH_ENABLE_MPS_FALLBACK=1`이 설정됐는지 확인하라. |
| 헬스 체크 후 샤드가 멈춘 듯 보임 | 한 번에 하나만 처리하는 서버에 두 번째 연결이 잘못 붙었다. 오케스트레이터 하나만 쓰고 그 연결을 재사용하라(§10). |

---

발행된 수치는 `python -m drift.bench`로 재현하라. 방법론은
[`benchmarks.md`](benchmarks.md)에 있다.
