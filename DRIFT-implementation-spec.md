# DRIFT — Decentralized Routed Inference For Tokens
### 이종 기기 분할 추론 — 구현 명세서 (Claude Code 작업 지시서)

*채택안: PyTorch 단일 프레임워크, Mac=MPS / Windows=CUDA, 프레임워크 중립 activation 경계*
*이 문서는 Claude Code가 그대로 실행하도록 작성됨. 대화 이력 없이도 자급되도록 컨텍스트를 포함함.*

---

## DRIFT — 프로젝트 맥락 (README.md 시드)

**DRIFT**는 *Decentralized Routed Inference For Tokens*의 약자다. 서로 다른 개인 기기들(Mac, Windows PC, …)이 하나의 모델을 layer 단위로 나눠 함께 추론하는, 중앙 서버 없는 P2P 추론 네트워크다. 빅테크의 데이터센터를 거치지 않고 *내 기계와 남의 기계가* 모여 하나의 AI를 돌린다.

이름이 곧 시스템이다:
- **D — Decentralized:** 단일 통제자·단일 장애점 없음. 이종(heterogeneous) 기기가 대등한 P2P 노드로 참여한다.
- **R — Routed:** 오케스트레이터가 hidden state를 노드들을 *거쳐 라우팅*하며 추론을 진행한다(pipeline routing).
- **I — Inference:** 워크로드는 LLM 추론(이후 학습으로 확장 가능).
- **For T — For Tokens:** "토큰"의 이중 의미. (1) AI 추론 **토큰** = 기계 사고/출력의 최소 단위, (2) 가치의 **토큰** = 기여로 벌고 추론에 쓰는 경제 단위. DRIFT의 비전은 *사고의 단위와 가치의 단위를 하나로* 보는 것 — 사고를 만드는 토큰과 그 값을 치르는 토큰이 같아질 때 지능은 누구의 소유도 아니게 된다.

(이름의 결: *drift*는 흩어진 기계들 위로 연산이 *흘러 다니는* 이미지이기도 하다 — 한 곳에 고이지 않는 사고.)

**이 저장소의 범위:** DRIFT 비전 전체(토큰 경제·신뢰 없는 검증·글로벌 P2P)가 아니라, **그 첫 조각 — D·R·I, 즉 이종 기기 분할 추론 — 의 동작하는 데모**를 구현한다. "For Tokens"의 경제 계층은 비전이자 *향후 작업*이며 이 데모 범위 밖이다(§1 제약 5).

**차별점 한 줄:** Exo는 노드 간 통신이 MLX(`mx.distributed`)에 묶여 *애플 실리콘끼리만* 가능하다(Windows는 공식 로드맵에서 'Longer term'). DRIFT는 노드 간 통신을 **프레임워크 중립 프로토콜**로 빼내어 *서로 다른 런타임·서로 다른 GPU 벤더*가 한 모델을 함께 돌리게 한다 — 데이터 평면이 어떤 프레임워크에도 묶이지 않는 것이 핵심 기여다.

> Claude Code: 이 절을 다듬어 `README.md`의 기반으로 사용하라(영문 README가 필요하면 번역). 나머지 §0–§14는 빌드 명세다.

---

## 0. 이 문서의 사용법 (Claude Code에게)

- 프로젝트가 무엇인지·이름의 의미는 위 **"DRIFT — 프로젝트 맥락"** 절을 보라.
- 이 문서는 **실행 가능한 명세서**다. §9의 마일스톤을 **순서대로** 진행하고, 각 마일스톤의 **인수 기준(acceptance test)을 통과한 뒤에만** 다음으로 넘어간다.
- §1의 **하드 제약(가드레일)을 절대 위반하지 말 것.** 특히 cross-node 통신에 `torch.distributed`/NCCL/gloo를 쓰지 말 것 — 이유는 §1에 있음.
- 막히면 §13(디버깅 가이드)을 먼저 보라. 대부분의 함정이 거기 있다.
- 하드웨어별 값(IP, 모델, split 지점 등)은 §5의 config로 분리되어 있다. 사용자가 채워야 하는 값은 명시돼 있다.

---

## 1. 목표와 하드 제약

**목표 (한 문단):** Mac 한 대(Apple GPU, PyTorch `mps` 백엔드)와 Windows PC 한 대(NVIDIA GPU, PyTorch `cuda` 백엔드)가 **하나의 LLM을 layer 단위로 분할(pipeline parallelism)해 함께 추론**한다. 두 노드는 **프레임워크 중립적인 바이트 프로토콜(TCP)** 로 hidden state를 주고받는다. 같은 LAN을 가정한다.

**왜 "중립 경계"인가 (이 프로젝트의 존재 이유):** 노드 내부 추론 엔진은 **교체 가능**해야 한다(오늘 PyTorch-MPS, 내일 MLX). 경계가 어떤 프레임워크에도 묶이지 않아야 이종 런타임이 협력할 수 있다. 이것이 기존 솔루션(Exo는 노드 간 통신이 `mx.distributed`에 묶여 외래 런타임 합류 불가)과의 핵심 차별점이다.

**하드 제약 (위반 금지):**
1. **cross-node 통신에 `torch.distributed` / NCCL / gloo / RPC를 쓰지 말 것.** NCCL은 MPS↔CUDA를 잇지 못하고, 이들 모두 데이터 평면을 특정 백엔드에 *결합*시킨다. 데이터 평면은 §6의 **중립 바이트 프로토콜**이어야 한다.
2. **§6 경계 계약(wire contract)을 한번 정하면 바꾸지 말 것.** 노드 내부 구현이 바뀌어도 경계는 불변.
3. **정확성 우선(correctness-first).** 네트워크가 낀 모든 단계는 단일 머신 레퍼런스 출력을 **재현**해야 한다(§9의 parity gate). 성능 최적화는 정확성 입증 *후에*.
4. **노드 내부 엔진을 인터페이스 뒤로 격리**하라(§7). 교체 가능성이 곧 차별성이다.
5. 토큰 경제(For Tokens의 경제 계층)·검증·P2P 디스커버리 등 상위 레이어는 **이 데모 범위 밖**이다. 여기서는 *이종 분할 추론(D·R·I)이 동작함*만 입증한다.

---

## 2. 아키텍처 개요

```
            [Orchestrator — Mac 또는 별도 프로세스]
            · tokenizer / embed_tokens / rotary
            · final norm + lm_head + sampler
            · decode 루프 구동
                    │  hidden_states[B,S,D] + position_ids   (TCP, 중립 바이트)
                    ▼
   ┌─────────────────────────┐        ┌─────────────────────────┐
   │ ShardServer A (Mac)     │  hidden│ ShardServer B (Windows) │
   │ device = mps            │───────▶│ device = cuda           │
   │ layers [0, k)           │        │ layers [k, N)           │
   │ KV cache (자기 layer)   │        │ KV cache (자기 layer)   │
   └─────────────────────────┘        └─────────────────────────┘
```

- **제어 평면:** orchestrator가 순서대로 shard를 호출(설정된 순서). 별도 디스커버리 불필요(config의 주소 목록).
- **데이터 평면:** 스테이지 경계를 건너는 것은 `hidden_states`(floats)와 `position_ids`(ints)뿐. 프레임워크 무관, 토큰당 수 KB.
- **KV cache:** 각 ShardServer가 *자기 layer 구간의* KV를 세션별로 로컬 보유. 경계로 KV를 보내지 않는다.

---

## 3. 기술 스택 & 환경

**머신별 설치(각 머신은 자기 백엔드용 torch를 설치 — 단일 휠에 MPS+CUDA가 같이 들어있지 않음):**

- **공통:** Python 3.11+ (`pyenv`/`uv` 권장). 의존성: `torch`, `transformers`, `safetensors`, `msgpack`, `numpy`. **설치 후 `pip freeze`로 버전을 고정(requirements.lock)** 하고 두 머신을 동일 `transformers` 버전으로 맞출 것(파리티에 중요).
- **Mac:** macOS arm64용 torch(MPS 지원). 실행 시 `PYTORCH_ENABLE_MPS_FALLBACK=1` 환경변수 설정(누락 op CPU 폴백). device 문자열 `"mps"`.
- **Windows:** CUDA 빌드 torch(`pip install torch --index-url <pytorch CUDA index>`). device 문자열 `"cuda"`. (CUDA 버전은 설치된 드라이버에 맞춰 선택.)

> Claude Code 주의: 정확한 torch/transformers 버전은 설치 시점 호환 조합으로 고정하라. 이 문서에 하드코딩된 API/버전을 *맹신하지 말고*, §7의 introspection 단계와 §9의 parity test로 검증하라.

---

## 4. 레포 구조

```
drift/
  config.yaml                 # §5, 사용자가 채움
  protocol.py                 # §6, 경계 프로토콜 (송수신 프레이밍 + 메시지 스키마)
  engine_base.py              # §7, ShardEngine 인터페이스 (교체 가능성의 핵심)
  engine_torch.py             # §7, PyTorch(MPS|CUDA) 구현
  shard_server.py             # §7, TCP 서버: 메시지 수신 → engine.forward → 응답
  orchestrator.py             # §8, embed/route/sample/decode 루프 + 스트리밍
  reference.py                # §9 M1, 단일 머신 full-model 레퍼런스 (parity 기준)
  parity_test.py              # §9 M2/M3, split 출력 == 레퍼런스 검증
  display.py                  # §10, 부스용 노드/오케스트레이터 디스플레이
  fallback_llamacpp.md        # §11, 부스 폴백(안 3) 셋업 메모
  README.md                   # 위 "DRIFT — 프로젝트 맥락" 절 기반으로 생성
```

---

## 5. 설정 (사용자가 채울 값)

`config.yaml`:
```yaml
model_id: "meta-llama/Llama-3.2-1B-Instruct"   # 권장 시작점(소형, 표준 Llama). 대안: Qwen2.5-1.5B-Instruct
dtype: "float16"                                # 경계 텐서 dtype도 이 값으로 통일
port: 52600
shards:
  - { name: "mac",     host: "<MAC_LAN_IP>",     start_layer: 0,  end_layer: 8,  device: "mps"  }
  - { name: "windows", host: "<WINDOWS_LAN_IP>", start_layer: 8,  end_layer: 16, device: "cuda" }
generation:
  max_new_tokens: 200
  # 데모는 인터랙티브 sampling, parity test는 greedy(아래 §9)로 강제
```

> 사용자 제공 필수: `<MAC_LAN_IP>`, `<WINDOWS_LAN_IP>`(같은 LAN), 그리고 모델의 총 layer 수에 맞춘 split 지점(Llama-3.2-1B = 16 layers → 0–8 / 8–16). 메모리에 따라 memory-weighted로 조정 가능.

---

## 6. 경계 프로토콜 명세 (THE CONTRACT — 불변)

**전송:** TCP. 각 메시지 = `4-byte big-endian unsigned length prefix` + `msgpack로 인코딩된 dict payload`. (언어 중립 — 미래의 MLX/기타 노드도 이 프레이밍만 구현하면 합류 가능.)

**요청 메시지 스키마:**
```
{
  "type":       "prefill" | "decode" | "reset" | "ping",
  "session_id": str,                 # 세션(=하나의 생성 시퀀스) 식별자
  "seq_id":     int,                 # 단조 증가, 순서 보장/디버깅용
  "shape":      [B, S, D],           # hidden_states 모양 (decode 시 S=1)
  "dtype":      "float16",           # config.dtype과 동일
  "position_ids": [int, ...],        # 길이 S, 이 청크의 절대 위치 (RoPE용)
  "tensor":     <bytes>              # row-major hidden_states 원시 바이트 (np.frombuffer로 복원)
}
```
- `reset`: 해당 session_id의 KV cache 폐기(생성 종료 시).
- `ping`: 헬스체크, `{"ok": true, "name", "start_layer", "end_layer", "device"}` 반환.

**응답 메시지 스키마:**
```
{ "ok": bool, "shape": [B,S,D], "dtype": "float16", "tensor": <bytes>, "error": str|null }
```

**구현 메모(protocol.py):**
```python
import struct, msgpack, numpy as np
def send_msg(sock, obj: dict) -> None:
    body = msgpack.packb(obj, use_bin_type=True)
    sock.sendall(struct.pack(">I", len(body)) + body)
def recv_msg(sock) -> dict:
    (n,) = struct.unpack(">I", _recvn(sock, 4))
    return msgpack.unpackb(_recvn(sock, n), raw=False)
def _recvn(sock, n):  # n바이트를 채울 때까지 읽기
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk: raise ConnectionError("peer closed")
        buf += chunk
    return bytes(buf)
# 텐서 직렬화: tensor.detach().to("cpu", torch.float16).contiguous().numpy().tobytes()
# 역직렬화: torch.from_numpy(np.frombuffer(b, dtype=np.float16).reshape(shape).copy()).to(device)
```
> fp16의 CPU 왕복은 비트 무손실이다. 따라서 직렬화는 정확성을 해치지 않는다(§9 파리티의 전제).

---

## 7. ShardEngine 인터페이스 & PyTorch 구현

**engine_base.py (교체 가능성의 핵심 — 이 인터페이스 뒤에 백엔드를 숨긴다):**
```python
from abc import ABC, abstractmethod
import torch

class ShardEngine(ABC):
    @abstractmethod
    def load(self, model_id: str, start_layer: int, end_layer: int,
             device: str, dtype: str) -> None: ...
    @abstractmethod
    def forward(self, session_id: str, hidden: torch.Tensor,
                position_ids: torch.Tensor, mode: str) -> torch.Tensor:
        """layers [start,end)를 실행하고 hidden을 반환. mode='prefill'|'decode'.
        세션별 KV cache를 내부에서 갱신."""
    @abstractmethod
    def reset(self, session_id: str) -> None: ...
```

**engine_torch.py (PyTorch / MPS|CUDA 구현):**

구현 지침 (정확한 HF 내부 API는 introspection으로 확정):
1. `from transformers import AutoModelForCausalLM`로 모델 로드(`torch_dtype=float16`). **정확성 우선 경로:** 소형 모델은 full 모델을 로드한 뒤 `model.model.layers[start:end]`만 보유/실행한다(다른 layer는 무시). **메모리 경로(대형/필수 split용, 나중에):** `init_empty_weights()` + safetensors에서 해당 shard의 weight만 `load_state_dict`. v1은 정확성 우선 경로.
2. **HF decoder layer 시그니처를 introspect 하라.** 설치된 `transformers` 버전에서 `LlamaDecoderLayer.forward`가 받는 인자(`hidden_states`, `attention_mask`, `position_ids` 또는 `position_embeddings(cos,sin)`, `past_key_value` 등)를 직접 확인하고 그에 맞춰 호출하라. 버전마다 다르므로 **하드코딩 금지**.
3. **RoPE:** 최신 HF는 rotary cos/sin을 모델 레벨에서 한 번 계산해 각 layer에 전달한다. 각 ShardServer가 `position_ids`로부터 자기 rotary 모듈(`model.model.rotary_emb` 등)을 통해 cos/sin을 *자체 계산*하게 하라(경계로는 `position_ids`만 보냄). 이렇게 하면 경계 텐서가 작고 노드가 자족적이다.
4. **KV cache:** `session_id`별로 각 layer의 past_key_value를 dict에 보관. `prefill`은 전체 prompt를 처리하며 KV를 채우고, `decode`는 S=1 토큰을 처리하며 KV에 append. HF의 `Cache`/`DynamicCache`를 세션별로 보유하는 방식 권장. `reset`에서 폐기.
5. `attention_mask`/causal: prefill은 causal full, decode는 KV 길이에 맞춘 마스크. HF 유틸 사용.

**shard_server.py:** TCP listen → `recv_msg` → (`ping`/`reset` 처리 or) 텐서 역직렬화 → `engine.forward(...)` → 결과 직렬화 → `send_msg`. 단일 세션/순차 처리로 시작(동시성은 나중). 시작 시 자기 `name/layers/device`를 로그+표시(§10).

---

## 8. Orchestrator 명세 (orchestrator.py)

보유: tokenizer, `embed_tokens`, (필요시 rotary), **final `norm` + `lm_head`**, sampler, shard 소켓 연결들.

> 주의: `embed_tokens`와 `norm`+`lm_head`는 orchestrator(또는 첫/마지막 shard)가 가진다. v1은 단순화를 위해 orchestrator가 embed와 head를 모두 보유(full 모델을 로드해 해당 텐서만 사용). shard는 *디코더 레이어 구간만* 담당.
> 대형/필수 split(메모리 경로)에서는 `embed_tokens`를 첫 shard로, `norm`+`lm_head`를 마지막 shard로 옮겨 orchestrator가 full 모델을 들지 않게 한다 — v1은 단순화 우선.

**prefill:**
```
tokens = tokenizer(prompt)
hidden = embed_tokens(tokens)                       # [1, S, D]
pos    = arange(S)
for shard in shards_in_order:                       # mac → windows
    hidden = rpc_forward(shard, session_id, hidden, pos, mode="prefill")
hidden = final_norm(hidden)
logits = lm_head(hidden[:, -1:, :])
first_token = sample(logits)                        # parity test에서는 argmax
```

**decode 루프:**
```
cur = first_token; generated = [first_token]; p = S
while not stop(generated):
    hidden = embed_tokens(cur)                      # [1, 1, D]
    for shard in shards_in_order:
        hidden = rpc_forward(shard, session_id, hidden, [p], mode="decode")
    logits = lm_head(final_norm(hidden))
    cur = sample(logits); generated.append(cur); p += 1
    stream(tokenizer.decode(cur))                   # 토큰 스트리밍 출력
for shard in shards: rpc_reset(shard, session_id)
```

스트리밍: 토큰을 생성 즉시 stdout/웹소켓으로 흘릴 것(데모 체감).

---

## 9. 마일스톤 & 인수 기준 (실행 계획 — 순서 엄수)

**M0 — 환경.** 두 머신에 의존성 설치, 버전 고정, `ping`으로 상호 연결 확인. *통과:* orchestrator가 두 shard의 `ping`에 정상 응답받음.

**M1 — 단일 머신 레퍼런스 (parity 기준).** `reference.py`: 한 머신에서 full 모델을 평범하게 로드해, 고정 프롬프트 + **greedy(`do_sample=False`)** 로 50토큰 생성. 생성된 **token id 시퀀스**와 첫 스텝 logits를 파일로 저장. *통과:* 결정론적 출력 저장됨.

**M2 — in-process 2-shard 파리티 (네트워크 없음).** 같은 머신·같은 device에서 모델을 `[0,k)`/`[k,N)` 두 ShardEngine 객체로 쪼개, orchestrator 로직(함수 호출, 소켓 없이)으로 greedy 50토큰 생성. *통과(엄격):* token id 시퀀스가 **M1과 완전히 동일.** ← 샤딩/RoPE/KV 로직의 정확성을 네트워크와 분리해 검증하는 핵심 게이트.

**M3 — localhost 2-프로세스 파리티 (TCP, 같은 머신).** §6 프로토콜로 두 프로세스를 띄워 동일 테스트. *통과(엄격):* token id 시퀀스가 **M1과 완전히 동일.** ← 직렬화/프레이밍 정확성 검증.

**M4 — cross-machine (Mac MPS + Windows CUDA).** 실제 두 머신에서 동일 생성. *통과(완화):* 출력이 **일관되고(coherent) 초반 토큰이 레퍼런스와 일치.** 단, MPS와 CUDA 커널의 미세 float 차이로 후반부에 token이 갈라질 수 있음 — 이는 **버그가 아니라 정상**(§13 참조). greedy로 초반 ~10토큰 일치 + 의미 있는 문장 생성이면 통과.

**M5 — 부스 디스플레이 + 인터랙티브.** §10. 사용자가 프롬프트 입력 → 스트리밍 응답, 각 노드가 자기 layer/device 표시. *통과:* 관객이 프롬프트를 넣고 두 머신을 가로지른 응답을 실시간으로 봄.

**M6 (선택) — 부드러운 kill-node 회복력.** decode 중 한 shard 연결이 끊기면 orchestrator가 감지 → 사용자에게 알림 → (가능하면 남은 노드로 재구성 불가 시) graceful 재시작. *주의:* 무중단 failover는 범위 밖(복제 필요). "감지+재구성+재실행"의 부드러운 버전만.

---

## 10. 부스 디스플레이 (M5 상세)

- **각 ShardServer:** 시작 시·토큰 처리 시 자기 정보를 크게 표시 — `layer 0–7 · MacBook(MPS)` / `layer 8–15 · Windows(CUDA)` + 처리 카운터/활동 표시. 터미널이면 `rich`, 큰 화면이면 간단한 로컬 웹페이지.
- **Orchestrator:** 프롬프트 입력창 + 스트리밍 출력 + "이 답의 앞 절반은 Apple GPU가, 뒤 절반은 NVIDIA가 생각함" 라우트 표시.
- 한 줄 후크(전시용): *"Exo는 당신의 Mac들이 필요합니다. DRIFT는 Apple GPU와 NVIDIA GPU가 한 모델을 함께 돌립니다."*

---

## 11. 부스 폴백 (안 3, llama.cpp RPC) — `fallback_llamacpp.md`에 별도

커스텀 스택이 무대에서 말썽이면 즉시 전환할 안전망. **주력 코드와 엮지 말 것.**
- 각 머신에서 llama.cpp를 자기 백엔드로 빌드: Mac=Metal, Windows=CUDA. 같은 GGUF 모델.
- 각 워커에서 `rpc-server` 실행, 메인에서 `--rpc <mac_ip>:<port>,<win_ip>:<port>`로 layer 오프로드.
- 정확한 빌드 플래그·옵션은 현재 llama.cpp README로 확인. Metal+CUDA 혼합은 *실제 하드웨어에서 사전 검증* 필수.
- 위치: 데모의 **대조 베이스라인**으로도 활용 가능("기성 도구로도 이종이 되지만, DRIFT의 데이터 평면은 엔진 무관·경량").

---

## 12. v2 업그레이드 경로 (안 2, Mac=MLX)

아키텍처가 이미 이를 허용한다 — **노드 내부 엔진 스왑일 뿐, 재작성 아님.** (DRIFT의 핵심 기여가 코드 구조로 증명되는 지점.)
- `engine_mlx.py`로 `ShardEngine`을 MLX(mlx-lm)로 구현해 Mac ShardServer에 주입. 경계 프로토콜(§6)은 **불변** — Windows는 그대로 PyTorch/CUDA.
- **수치 일치 검증 필수:** MLX-LM Llama와 PyTorch 경로의 layer 출력이 일치하는지 §9의 M1/M2 게이트로 확인(MLX 레퍼런스 대비). RoPE 적용 방식·RMSNorm eps·weight layout 차이를 layer 단위 bisect로 추적(§13). 이게 v2의 핵심 리스크이므로, v1(PyTorch 통일)로 아키텍처를 먼저 입증한 뒤 진행.

---

## 13. 정직한 리스크 & 디버깅 가이드

- **MPS 누락 op:** `PYTORCH_ENABLE_MPS_FALLBACK=1`. 그래도 에러 나면 해당 op를 CPU로 우회하거나 모델 선택 변경.
- **파리티 불일치(M2/M3):** layer 단위로 bisect하라 — split 지점을 1 layer씩 옮기며 어느 경계에서 hidden state가 레퍼런스와 갈리는지 추적. 주범 후보: (a) RoPE/position_ids 전달 오류, (b) KV cache가 prefill→decode에서 위치를 잘못 누적, (c) attention mask 길이 오류, (d) `embed_tokens`/`norm`/`lm_head`를 shard가 중복 적용. 텐서를 `float32`로 비교해 max-abs-diff를 보라.
- **M4 후반부 token 갈라짐:** MPS와 CUDA는 동일 연산도 비트 단위로 다를 수 있어 greedy argmax가 가끔 뒤집히고 그 후 발산한다. **정상이다.** 초반 토큰 일치 + 일관된 문장이면 통과. 이걸 버그로 오인해 시간 낭비하지 말 것.
- **dtype 혼선:** 경계는 항상 `config.dtype`(fp16). device로 올릴 때만 변환. CPU 왕복 fp16은 무손실.
- **KV cache 정확성**이 디코드 정확성의 핵심이다. prefill에서 채운 KV에 decode가 정확한 절대 위치로 append하는지 단위 테스트하라.
- **순차 처리부터.** 동시 세션·배칭·speculative decoding은 데모 동작 입증 후. 정확성 > 속도(이 데모의 가치는 속도가 아니라 *이종 분할이 됨*이다).

---

## 14. Definition of Done (데모 기준)

M0–M5 통과 = DRIFT v1 데모 준비 완료. 즉: 관객이 프롬프트를 입력하면, 응답이 **Mac(Apple GPU)과 Windows(NVIDIA) 두 이종 머신을 가로질러** 생성되어 실시간 스트리밍되고, 각 노드가 자기가 맡은 layer 구간을 표시한다. 그리고 그 데이터 평면은 `torch.distributed`가 아니라 **DRIFT의 중립 프로토콜**이며, 노드 내부 엔진은 교체 가능하다 — 즉 *Decentralized Routed Inference*가 실제로 동작하고, Exo가 'Longer term'으로 미뤄둔 이종(Mac+Windows) 분할 추론을 손에 쥔 상태.
