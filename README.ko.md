<h1 align="center">DRIFT</h1>

<p align="center"><b>Decentralized Routed Inference For Tokens. 하나의 모델을 당신의 여러 머신에 쪼개어, 데이터센터 없이.</b></p>

<p align="center">
  <a href="./README.md">English</a> ·
  <b>한국어</b> ·
  <a href="./README.zh.md">中文</a> ·
  <a href="./README.ja.md">日本語</a>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/TaewoooPark/DRIFT?style=flat-square&logo=github&logoColor=white&labelColor=000000&color=333333" alt="GitHub stars">
  <img src="https://img.shields.io/github/v/release/TaewoooPark/DRIFT?style=flat-square&labelColor=000000&color=333333" alt="Release">
  <img src="https://img.shields.io/github/last-commit/TaewoooPark/DRIFT?style=flat-square&labelColor=000000&color=333333" alt="Last commit">
  <img src="https://img.shields.io/badge/License-MIT-000000?style=flat-square&labelColor=000000&color=333333" alt="License MIT">
  &nbsp;
  <img src="https://img.shields.io/badge/Python-3.12-000000?style=flat-square&logo=python&logoColor=white&labelColor=000000" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.12-000000?style=flat-square&logo=pytorch&logoColor=white&labelColor=000000" alt="PyTorch">
  <img src="https://img.shields.io/badge/Apple%20MPS-000000?style=flat-square&logo=apple&logoColor=white&labelColor=000000" alt="Apple MPS">
  <img src="https://img.shields.io/badge/CUDA-000000?style=flat-square&logo=nvidia&logoColor=white&labelColor=000000" alt="CUDA">
  &nbsp;
  <img src="https://img.shields.io/badge/Peer--to--peer%20chain-000000?style=flat-square&labelColor=000000" alt="Peer to peer chain">
  <img src="https://img.shields.io/badge/Encrypted%20wire-000000?style=flat-square&labelColor=000000" alt="Encrypted wire">
  <img src="https://img.shields.io/badge/Signed%20receipts-000000?style=flat-square&labelColor=000000" alt="Signed receipts">
  <img src="https://img.shields.io/badge/Bitwise%20failover-000000?style=flat-square&labelColor=000000" alt="Bitwise failover">
  <img src="https://img.shields.io/badge/Bitwise%20parity-000000?style=flat-square&labelColor=000000" alt="Bitwise parity">
  <img src="https://img.shields.io/badge/MPS%20%E2%86%94%20CUDA-000000?style=flat-square&labelColor=000000" alt="MPS to CUDA">
</p>

<p align="center">
  <img src="docs/img/hero.png" alt="하나의 모델을 지구 반대편에 쪼개다, 뉴욕과 서울이 한 모델을 함께, 데이터센터 없이" width="900">
</p>

<p align="center"><sub>뉴욕의 친구가 잠든 사이 노드를 켜 두고, 당신은 서울에 있다. DRIFT는 <b>하나의</b> 모델을 두 머신에 쪼갠다. 친구의 GPU가 앞쪽 레이어를, 당신의 GPU가 뒤쪽을 계산하고, hidden state는 <b>암호화된</b> 와이어 위로 <b>노드에서 노드로</b> 흐르며, 매 홉이 <b>영수증에 서명한다.</b> 그리하여 어느 한 대도 홀로 담지 못할 모델을 둘이 함께 돌리되, 그 답이 단일 머신과 같음이 증명된다.</sub></p>

<p align="center">
  <img src="docs/img/drift-demo.gif" alt="DRIFT 데모: 하나의 모델이 여러 노드에 걸쳐 실행되는 모습" width="900">
</p>

**DRIFT**는 **하나의** 대규모 언어 모델을 **이종(heterogeneous) 개인 머신들**, 곧 Mac(Apple GPU, PyTorch **MPS**)과 Windows/Linux PC(NVIDIA GPU, PyTorch **CUDA**)에 걸쳐 실행하되, 모델을 **레이어 단위로** 분할(파이프라인 병렬화, pipeline parallelism)하고 노드 사이로는 오직 **hidden state**만을 **프레임워크 중립 바이트 프로토콜**(TCP + msgpack) 위로 흘려보내는 방식을 취한다. 데이터센터도, `torch.distributed`도, NCCL도, 벤더 종속도 개입하지 않는바, 데이터 플레인(data plane)이 *어떤* 프레임워크에도 묶여 있지 않기에 서로 결코 대화할 수 없던 런타임, 곧 Apple Metal 그래프와 NVIDIA CUDA 그래프가 비로소 하나의 모델을 함께 돌릴 수 있으며, 그 출력은 전체 모델을 단일 머신에서 실행한 결과와 **비트 하나까지 동일하다**.

그 정확한 핵심 위에서 DRIFT는 진짜 **탈중앙 계층(decentralization layer)**을 길러 냈다. 이제 hidden state는 **피어투피어(peer-to-peer)**로 흐르고(헤드는 더 이상 대역폭 허브가 아니다), 와이어는 **암호화되고 멤버십이 인증되며**, 떨어져 나간 노드는 **비트 단위로** 복구되고, 헤드는 **무게 없는(weightless)** 것이 될 수 있으며, 매 홉은 헤드가 실시간 트래픽 위에서 검증하는 **영수증에 서명하고**, 노드들은 서로를 **가십(gossip)으로 발견하며**, 그 기여는 **원장(ledger)**에 집계된다.

**한 줄로 요약한 차별점:** [Exo](https://github.com/exo-explore/exo)는 노드 간 통신을 MLX(`mx.distributed`)에 결박해 둔 탓에 *Apple 실리콘 대 Apple 실리콘 전용*에 머문다. 그러나 DRIFT는 그 경계를 **중립적이고 암호화된 와이어 프로토콜**로 끌어올림으로써(*서로 다른 런타임, 서로 다른 GPU 벤더, 하나의 모델*) 분할이 정확함을 **비트 단위 패리티 게이트(bitwise parity gate)**로 입증하고, 홉마다 서명된 영수증으로 그것을 **자기 검증(self-verifying) 가능하게** 만든다. 어떤 프레임워크에도 묶이지 않고, 정확함이 증명되며, 노드를 신뢰하지 않고도 검사할 수 있는 데이터 플레인, 바로 그것이 핵심 기여다.

**확장.** 디코더 레이어 하나당 노드 하나, 곧 기본 Qwen 기준 최대 **28대**(Gemma는 **35대**)에 걸쳐 하나의 모델을 쪼개고 그 모두를 가로질러 스트리밍한다. 현재의 스위트 스팟은 **2~4대**다.

> *"트랜스크립트는 모델의 출력일 뿐이다. 흥미로운 지점은 그 연산이 실제로 **어디서** 돌았는가, 그리고 그것이 비트 하나까지 맞아떨어졌고, 와이어가 암호화되어 있었으며, 매 홉이 자기 몫의 일에 서명했다는 사실이다."*

[**taewoopark.com** 저자 사이트](https://taewoopark.com)

---

## 목차

- [무엇이 다른가](#무엇이-다른가): 엔지니어들이 찾던 바로 그 비교 표
- [DRIFT란 무엇인가](#drift란-무엇인가): 이름, 비전, 그리고 범위
- [다섯 개의 플레인](#다섯-개의-플레인): 제어 / 데이터 / KV / 보안 / 신뢰
- [와이어 계약](#와이어-계약-경계를-실제로-넘는-것): 스키마 + 토큰당 바이트
- [세 가지 정확성 문제](#올바른-분할이-풀어야-할-세-가지-문제): KV 재인덱싱, RoPE, 마스크
- [피어투피어, 무게 없는 헤드](#피어투피어-그리고-무게-없는-헤드): 체인 + 얇은 헤드
- [노드를 신뢰하지 않고 신뢰하기](#노드를-신뢰하지-않고-신뢰하기): 암호화, 서명된 영수증, 페일오버
- [정확성 & 패리티](#정확성-패리티-게이트): 비트 단위 게이트 + 측정 결과
- [벤치마크](#벤치마크): fidelity 100% · int8에서 와이어 절반 · O(1) 헤드 대역폭
- [인트로스펙션을 통한 모델 독립성](#인트로스펙션을-통한-모델-독립성): Qwen, Gemma, 하드코딩 없음
- [설계 근거 (why-not)](#설계-근거-왜-아닌가): 그 결정들과 이유
- [마일스톤](#마일스톤) · [빠른 시작](#빠른-시작) · [저장소 지도](#저장소-지도-어디를-봐야-하는가) · [FAQ](#faq) · [무엇이 아직 비전인가](#무엇이-구현됐고-무엇이-아직-비전인가)

---

## 무엇이 다른가

DRIFT의 핵심은 전적으로 **노드 사이의 경계**에 있다. 그 경계를 선행 기술과 비교하면 다음과 같다:

| | **DRIFT** | Exo | Petals | llama.cpp RPC | vLLM / Megatron PP |
|---|---|---|---|---|---|
| **분할 단위** | 디코더 레이어 | 레이어 | 트랜스포머 블록 | 레이어 / 텐서 | 레이어(스테이지) |
| **노드↔노드 전송** | **TCP + msgpack** | MLX `mx.distributed` | gRPC(torch 텐서) | 커스텀 RPC(ggml) | `torch.distributed` + NCCL |
| **프레임워크 중립 와이어** | **✅ 예** | ❌ MLX 종속 | ❌ torch 종속 | ggml 종속 | ❌ torch/NCCL 종속 |
| **이종 GPU 벤더** | **✅ MPS + CUDA 동시** | ❌ Apple 전용 | 부분적 | ✅ (ggml 백엔드) | ❌ NCCL로는 연결 불가 |
| **데이터 플레인 토폴로지** | **✅ 피어투피어 체인** | 액티베이션 | 액티베이션 | 액티베이션 | 액티베이션 |
| **와이어 암호화 + 노드 인증** | **✅ X25519 + ChaCha20 + PSK** | ❌ | ❌ | ❌ | ❌ |
| **자기 검증(홉마다 서명)** | **✅ Ed25519 영수증, 실시간** | ❌ | ❌ | ❌ | ❌ |
| **비트 단위 정확 페일오버** | **✅ 재분할 + 리플레이** | ❌ | ~ (재라우팅) | ❌ | ❌ |
| **토큰당 넘어가는 것** | **~1.5~3 KB (hidden만)** | 액티베이션 | 액티베이션 | 액티베이션 | 액티베이션 |
| **정확성 계약** | **단일 머신 대비 비트 단위 패리티** | 없음 | 없음 | 없음 | 없음 |

표를 위에서 아래로 훑어 내려가면 논지는 저절로 드러난다: **액티베이션(activation)을 전달한다는 점은 모두가 같으나, 그 전달을 프레임워크 중립적이고, 암호화되고, 피어투피어이며, *동시에* 비트 단위로 정확함까지 입증되게 만드는 것은 오직 DRIFT뿐이고, 나아가 모델을 다시 돌리지 않고도 노드가 거짓말하지 않는지 검사하게 해 준다.** NCCL은 Apple GPU와 NVIDIA GPU를 하나의 프로세스 그룹에 담을 수 없으며, MLX 또한 Apple 생태계를 벗어나지 못한다. 이에 대한 DRIFT의 답은 와이어가 *바이트 외에는 아무것도*, 곧 torch 객체도, MLX 배열도, CUDA 핸들도 실어 나르지 않게 하는 것인바, 그리하여 두 세계가 서로 구현 가능한 단 하나의 계약 위에서 만나고, 이어 그 계약을 더욱 단단하게 벼려 낸다.

---

## DRIFT란 무엇인가

서버가 존재하지 않는 피어투피어(peer-to-peer) 추론 네트워크로서, 이종 개인 기기들이 **하나의** 모델을 레이어 단위로 분할하여 **함께** 실행한다. 하이퍼스케일러(hyperscaler)의 데이터센터를 경유하는 대신, *당신의 머신과 다른 누군가의 머신*이 한자리에 모여 단 하나의 AI를 돌리는 것이다.

이름이 곧 시스템이다:

| 글자 | 의미 |
|---|---|
| **D**: Decentralized(탈중앙) | 데이터센터가 없다. hidden state는 노드에서 노드로 **피어투피어**로 흐르고, 와이어는 암호화 + 멤버십 인증되며, 떨어져 나간 노드는 복구된다. 여전히 오케스트레이터가 실행을 시작하고 헤드는 무게 없이 만들 수 있으나, 완전한 리더 없는 합의는 아직 비전이다([무엇이 아직 비전인가](#무엇이-구현됐고-무엇이-아직-비전인가) 참조). |
| **R**: Routed(라우팅) | 오케스트레이터가 노드들을 거쳐 hidden state를 *라우팅*하여 추론을 앞으로 밀고 나간다 |
| **I**: Inference(추론) | 워크로드는 LLM 추론이다(학습으로 확장 가능) |
| **For T**: For Tokens(토큰을 위하여) | "토큰"의 이중 의미: **추론** 토큰(기계 사고의 원자) **그리고** **가치** 토큰(기여로 벌고, 추론에 쓴다). 이제 매 홉이 영수증에 서명하고 `drift ledger`가 기여를 집계하는바, 이는 지급(payout) 계층이 소비하는 입력이다. DRIFT의 비전은 사고의 단위와 가치의 단위를 하나로 만드는 것이다. |

> **본 저장소의 범위.** 어려운 기술적 핵심, 곧 *Mac과 Windows 박스에 걸쳐 분할한 모델이 올바른 답을 내놓는가?* 는 완성됐고 **비트 단위로** 입증됐다. 그 위에서 **"For Tokens"** substrate는 더 이상 다이어그램에 그치지 않는다: **피어투피어 암호화 데이터 플레인**, **비트 단위 페일오버**, **무게 없는 헤드**, **실시간 트래픽 위에서의 서명 영수증 검증**, **가십 멤버십**, 그리고 **기여 원장**이 모두 구현되고 게이트를 통과했다. 완전한 토큰 이코노미, 온체인 정산, 그리고 리더 없는 합의는 여전히 비전으로 남아 있다.

---

## 다섯 개의 플레인

<p align="center"><img src="docs/img/arch.png" alt="DRIFT architecture, orchestrator head, per-layer shards, neutral wire" width="900"></p>

DRIFT는 여러 플레인(plane)으로 정연하게 분리된다:

- **제어 플레인(control plane)**: 오케스트레이터가 각 노드에 레이어 범위를 배정하고(`configure`) 디코드 루프를 구동한다. 노드는 네 가지 방식으로 발견된다. 무설정 LAN 디스커버리(mDNS), 명시적 `--nodes host:port` 목록, NAT 뒤 노드가 `drift node --tunnel`로 여는 공개 `bore.pub` 터널, 또는 **가십(gossip)**이다. 가십에서는 노드가 하나의 시드에 `--join`하면 네트워크가 스스로의 멤버십을 학습하고, 이어 `drift run --expand`가 그것을 가로질러 분할한다.
- **데이터 플레인(data plane)**: 스테이지 경계를 넘는 것은 오직 `hidden_states`(부동소수점)와 `position_ids` + `input_ids`(정수)뿐이다. 프레임워크에 독립적이며, 결정적으로, 그 크기가 파라미터 수가 아니라 `hidden_size`에 좌우된다. **이제 이것은 피어투피어로 흐른다**(`--chain`): head → n0 → n1 → … → tail → head, 하여 토큰당 텐서 횡단이 2N에서 **N+1**로 줄고, 헤드의 대역폭은 노드 수에 대해 O(N)이 아니라 **O(1)**이 된다. 선택적으로 **int8**(`--int8`)이 바이트를 절반으로 줄인다.
- **KV 캐시 플레인(KV cache plane)**: 각 샤드는 세션별로 *자기 자신의* 레이어 범위에 해당하는 KV를 자기 디바이스에 보관한다. 캐시는 결코 와이어를 넘지 않는데(그랬다간 토큰당 메가바이트 단위가 되어 설계 전체를 무너뜨린다), 오직 residual stream만 오간다.
- **보안 플레인(security plane)**: 하나의 네트워크는 단 하나의 사전 공유 키(pre-shared key)를 공유한다(`drift keygen`). 그러면 모든 연결이 X25519 ECDH → HKDF(PSK 혼합) → ChaCha20-Poly1305 채널을 거치는바, 스트림은 기밀이 되고 키 없는 발신자는 끊긴다. `drift node --tunnel`은 키 없이는 실행을 거부하며(개방된 공개 연산은 없다), 길이 프리픽스에는 상한이 걸려 있다(할당 DoS 방지).
- **신뢰 플레인(trust plane)**: 매 홉은 `(in_hash, out_hash, 레이어 범위)`에 대해 **Ed25519 영수증**에 서명한다. 헤드는 실제 트래픽의 **모든 토큰**에서(별도의 챌린지가 아니라) 서명 + 인접성 + 양 끝 앵커를 검증하는바, 하여 와이어 손상, 누락되거나 위조된 홉, 그리고 자신이 계산한 것과 보낸 것을 두고 거짓말하는 노드가 실시간으로 적발된다. 떨어져 나간 노드는 생존 노드들에 걸쳐 재분할하고 리플레이함으로써 **비트 단위로** 복구된다.

**분할은 두 대를 넘어 확장된다.** 디코더 레이어 하나당 노드 하나로 최대 28대(Gemma는 35대)까지 늘어나되, 헤드와 와이어는 그대로다:

<p align="center"><img src="docs/img/scale.png" alt="DRIFT scales one model across 2 to 28 nodes, one decoder layer per node" width="900"></p>

---

## 와이어 계약 (경계를 실제로 넘는 것)

본 계약(`drift/protocol.py`)은 **고정(frozen)**되어 있으니, 모든 메시지는 예외 없이 **4바이트 빅엔디언 길이 프리픽스 + msgpack 딕셔너리**로 이루어진다(네트워크 키가 설정되면 하나의 ChaCha20-Poly1305 프레임으로 암호화된다). 미래의 어떤 런타임, 곧 MLX, ggml, JAX, Rust 노드이든 합류하고자 한다면 오직 이 프레이밍만 구현하면 충분하다. 와이어 위에 PyTorch는 존재하지 않는다.

```jsonc
// 요청  (orchestrator → shard, 또는 체인 모드에서는 shard → shard)
{
  "type":         "prefill" | "decode" | "reset" | "ping" | "configure",
  "session_id":   "s0",               // 하나의 생성 시퀀스
  "seq_id":       42,                 // 단조 증가, 순서 정렬 / 디버그용
  "shape":        [1, 1, 1536],       // hidden_states 형태 (decode: S=1)
  "dtype":        "float16" | "int8",  // int8 → 와이어 절반 (손실 있음)
  "scale":        "<per-group fp16>",  // int8 역양자화 스케일 (fp16에는 없음)
  "position_ids": [37],               // 절대 위치  → RoPE, 샤드에서 계산
  "input_ids":    [785],              // 토큰 id → 레이어별 임베딩(PLE) / 얇은 헤드 임베드
  "tensor":       "<raw bytes>",       // 행 우선(row-major) hidden_states
  "route":        [["10.0.0.2", 52601]], // 체인 모드: 다운스트림 노드들
  "collect":      ["10.0.0.9", 6000]     // 체인 모드: 헤드의 싱크
}

// 응답 (shard → 다음 홉 / head)
{ "ok": true, "shape": [1,1,1536], "dtype": "float16", "tensor": "<bytes>",
  "receipt": { "node": "<pubkey>", "in_hash", "out_hash", "start", "end", "sig" },
  "token":  785 }   // 얇은 헤드의 tail은 텐서 대신 토큰 id를 반환한다
```

`route` / `collect`는 **가산적이며 선택적**이어서, 이들이 없는 노드는 고전적인 스타(star)와 정확히 똑같이 동작한다. `configure`는 **대체 가능한(fungible)** 노드에 레이어 범위(그리고 얇은 헤드의 가장자리 임무)를 배정하는바, 하여 사용자가 범위를 손으로 적을 일이 결코 없다.

**토큰당 바이트.** 디코드 중 액티베이션은 `[1, 1, hidden]`이다. Qwen의 `hidden = 1536`이면 fp16으로 **3,072 바이트**, int8로는 **1,560 바이트**다(H int8 + 그룹별 fp16 스케일 ≈ 0.51×). 체인은 토큰당 이러한 횡단을 `N+1`회, 스타는 `2N`회 수행한다. LAN에서는 연산에 견주면 하찮은 양이다.

**왜 와이어에서 fp16이 안전한가(비트 단위).** 직렬화는 CPU fp16 왕복이다. 연산 dtype이 fp16이라면 이 왕복에는 **비트 손실이 없으니**, 바로 이것이 분할 경로로 하여금 단일 머신을 근사가 아니라 *정확하게* 재현하게 하는 전제다. int8은 무손실이 *아니며* 선택적으로만 켜진다. 이것은 완화된 게이트 아래에서 돌 뿐, 결코 비트 단위 게이트에서 돌지 않는다.

---

## 올바른 분할이 풀어야 할 세 가지 문제

레이어를 여러 프로세스에 쪼개는 일은, 그 출력을 쪼개지 않은 모델과 *동일하게* 만들려 시도하기 전까지는 지극히 사소해 보인다. 그러나 발목을 무는 문제가 세 가지 있으니, DRIFT는 그 각각을 명시적으로 처리한다.

### 1 · KV 캐시 인덱싱: 미묘한 문제

Hugging Face의 `DynamicCache`는 "과거 길이(past length)"를 **레이어 0의** 슬롯에서 읽어 보고한다. 전역 레이어 `[14, 28)`을 보관하는 샤드가 그 전역 인덱스를 그대로 재사용하면 캐시 슬롯 0이 **비어 있게** 되는바, 그 결과 디코드 중 causal mask가 마치 *과거가 없는* 양 만들어지고, 패리티는 바로 첫 토큰 이후 소리 없이 깨지고 만다.

<p align="center"><img src="docs/img/kv-reindex.png" alt="KV-cache local re-indexing, the fix that keeps decode parity" width="900"></p>

DRIFT는 로드 시점에 각 샤드가 보관하는 레이어를 **로컬 0 기반** 캐시 슬롯으로 재인덱싱하고, 세션별 `DynamicCache`의 크기를 그 샤드의 로컬 레이어 수에 맞춘다.

### 2 · RoPE 자체 계산: 와이어를 작게 유지하기

로터리 위치 임베딩(RoPE)은 오직 `position_ids`에만 의존한다. 그러므로 각 샤드는 모델 자신의 `rotary_emb`를 통해 **절대** 위치로부터 자기 `cos/sin`을 계산한다. 경계는 완전한 `cos/sin` 텐서 대신 정수 몇 개만 실어 나르기에, 모든 노드가 자족적으로 유지된다.

### 3 · 스테이지별 어텐션 마스크

prefill에서 마스크는 causal-full이고, decode에서는 KV 길이를 인식한다. DRIFT는 설치된 Transformers의 마스킹 유틸리티로 각 샤드에서 마스크를 다시 빚어내되, 마스크는 그 레이어 자신의 어텐션 타입에 따라 **레이어별로** 선택된다(Gemma는 로컬/글로벌을 번갈아 쓴다). 하드코딩된 것은 아무것도 없다.

---

## 피어투피어, 그리고 무게 없는 헤드

**체인 스트리밍(`--chain`).** 매 홉을 헤드로 되돌리는 스타 라우팅 대신, hidden state가 경로를 따라 노드에서 노드로 흐르고 tail이 최종 상태를 헤드의 collect 싱크로 전달한다. 두 가지 이득이 있다: 토큰당 텐서 횡단이 **2N에서 N+1로** 줄고, 그리고 이것이 핵심인바, 헤드의 데이터 플레인 대역폭이 노드 수에 대해 O(N)이 아니라 **O(1)**이 된다. 헤드는 더 이상 모든 액티베이션이 지나가는 허브가 아니게 된다.

**얇은 헤드(`--thin`).** 헤드는 **모델 가중치를 하나도** 쥐지 않을 수 있다: `embed_tokens`는 첫 노드의 임무로, `norm` + `lm_head` + `argmax`는 마지막 노드의 임무로 옮겨 간다. 체인과 결합하면 헤드는 파이프라인에 **정수 토큰 id 하나**를 보내고 **정수 토큰 id 하나**를 돌려받을 뿐, 텐서 연산을 전혀 하지 않으며 파라미터를 하나도 실체화하지 않는다. 패리티가 유지되는 까닭은 `norm`+`lm_head`+`argmax`가 비트 단위로 동일한 hidden state 위에서 같은(묶인) 가중치로 같은 디바이스에서 돌기 때문인바, argmax는 그것을 헤드가 계산하든 tail이 계산하든 불변이다.

디코드 루프는 주입 가능한 전송 계층 위에서 **단 한 번** 작성되며, 교체되는 것은 오직 전송 계층(인프로세스 / 스타 / 체인)뿐이다. 하여 마일스톤 사이의 유일한 변수는 네트워크뿐이며, 어떤 회귀가 발생하든 그것은 *증명 가능하게* 전송 계층 버그일 뿐 결코 로직 버그가 아니다.

<p align="center"><img src="docs/img/decode-loop.png" alt="The decode loop over an injectable transport (in-process / TCP / chain)" width="900"></p>

---

## 노드를 신뢰하지 않고 신뢰하기

**암호화되고 인증된 와이어(`drift keygen`).** 하나의 네트워크는 32바이트 사전 공유 키 하나를 공유한다. 키가 걸린 연결은 X25519 ECDH(임시 키 → 순방향 비밀성) → PSK를 섞은 HKDF-SHA256 → 방향별 카운터 논스를 쓰는 ChaCha20-Poly1305를 거친다. PSK를 KDF에 섞는 것이 곧 멤버십 검사다: 그것이 없는 피어는 다른 키를 도출하며 그 첫 프레임의 복호화가 실패한다. 키가 없으면 로컬 개발을 위해 평문으로 남고, 키를 거는 것은 네트워크 전역에 걸친다.

**실시간 트래픽 위의 서명 영수증.** 매 홉은 `(session, seq, mode, 레이어 범위, in_hash, out_hash)`에 대해 Ed25519 영수증에 서명한다. **매 토큰**마다 헤드는 서명, 인접성(홉 *i*의 `out_hash` == 홉 *i+1*의 `in_hash`), 그리고 양 끝 앵커(첫 홉의 입력이 헤드가 보낸 것과 일치하고, 마지막 홉의 출력이 헤드가 받은 것과 일치)를 확인한다. 조작하는 노드는 평범한 생성 도중에 적발되고(정직해야 할 별도의 챌린지가 없다) 로컬 평판 테이블에 SUSPECT로 표시된다. *이것이 잡아내는 것:* 와이어 손상, 누락·재정렬·위조된 홉, 자신이 계산한 것과 보낸 것을 두고 거짓말하는 노드다. *잡아내지 못하는 것*(일관되게 잘못 계산하고 그 결과에 서명하는 노드)은 재계산 감사(`drift verify`)나 중복 N-of-M 실행(향후)의 몫이다.

<p align="center"><img src="docs/img/parity-gate.png" alt="The parity gate, strict bitwise on one device, relaxed across GPU vendors" width="900"></p>

**비트 단위 페일오버.** 생성 도중 노드가 죽어도 더 이상 세션이 끝장나지 않는다. 오케스트레이터는 생존 노드들(그리고 여분이 있다면 그것)에 걸쳐 모델을 재분할하고, 지금까지의 시퀀스를 다시 prefill하여 모든 노드의 KV를 재구축한 뒤 재개한다. 그리디 디코딩은 고정된 프리픽스 위에서 결정론적이므로, 재개된 이어짓기는 애초에 노드가 떨어진 적 없는 경우와 **비트 단위로 동일하다.** 이는 디코드 도중 노드를 죽이고 완성된 시퀀스를 중단 없는 레퍼런스와 대조하여 검증된다.

**기여 원장.** 헤드는 검증된 모든 영수증을 저널에 기록하고, `drift ledger`는 이를 노드별 집계(운반한 토큰, 처리한 레이어-토큰, 세션 수)로 접어 넣되 `--verify`는 모든 서명을 다시 확인하고 `--csv`는 내보내기를 제공한다. 이것이 정산 계층의 입력 substrate다.

---

## 정확성: 패리티 게이트

DRIFT는 **정확성 우선**을 원칙으로 삼는다: 모든 네트워크 단계는 어떤 성능 작업이나 탈중앙화 작업에 앞서 단일 머신 레퍼런스를 **비트 단위로** 재현해야 한다. 속도는 요점이 아니다. *이종 분할 추론이 정확하다는 것*이 요점이며, 위의 모든 기능은 바로 그것에 대해 게이트된다.

**측정 결과**(Qwen2.5-1.5B-Instruct, Apple MPS, fp16):

| 게이트 | 무엇을 격리하는가 | 결과 |
|---|---|---|
| **인프로세스** 2-샤드 | 샤딩 · RoPE · KV · 마스크 | ✅ **프롬프트 6 / 6 비트 단위** (`n = 1…180`) |
| **TCP 스타** 2-프로세스 | 직렬화 / 프레이밍 | ✅ **비트 단위 == 레퍼런스** |
| **체인** 2 & 3 노드 | 피어투피어 릴레이 | ✅ **비트 단위 == 레퍼런스** |
| **체인 + 암호화** | AEAD 채널 투명성 | ✅ **비트 단위** (암호화가 토큰을 교란하지 않는다) |
| **얇은 헤드** 2 & 3 노드 | 무게 없는 헤드, 가장자리 embed/lm_head | ✅ **비트 단위** |
| **디코드 도중 kill** (체인 / 스타, 진입 / 중간 / tail) | 페일오버 리플레이 | ✅ **48 / 48 비트 단위**, 복구 발동 |
| **노드 조작** | 실시간 영수증 검증 | ✅ **실시간 트래픽에서 적발**, 정직 실행 시 의심 0 |
| **MPS ↔ CUDA (M4)** | 크로스벤더 fp16 반올림 | ✅ 3개 프롬프트에서 **130 / 130 토큰** 일치 |

**MPS ↔ CUDA (M4).** 앞 절반을 Mac(Apple MPS)에서, 뒤 절반을 Colab NVIDIA T4(CUDA)에서 돌린 결과, 분할 경로가 단일 머신을 **정확히 재현했다(130/130 토큰).** 두 벤더의 fp16 커널이 첫 스텝 logit 격차를 ~2×10⁻²까지 벌렸음에도(같은 device는 ~8×10⁻³) 여기서 argmax를 뒤집기엔 부족했다. 더 큰 규모에서는 이 격차가 뒤쪽 토큰을 뒤집을 수 있는바, 그때를 위해 **완화된 게이트** `python -m drift.parity_test --prefix-match K`가 있다.

<p align="center"><img src="docs/img/m4-result.png" alt="M4 measured, Mac Apple MPS + Colab NVIDIA T4 CUDA, 130/130 token match vs one machine" width="900"></p>

---

## 벤치마크

*단일 머신 수치는 `python -m drift.bench`로, 통합 게이트는 `python -m drift.itest …`로 재현된다.*

**Fidelity: 분할이 출력을 바꾸는가?** *(분할 경로 vs 단일 기기 오라클, greedy)*

| 지표 | 결과 |
|---|---|
| 토큰 정확 일치(프롬프트 6개, `n = 1…180`) | **411 / 411 = 100.00 %** |
| 첫 스텝 logit 최대 절대차 (fp32) | 7.81 × 10⁻³ *(fp16 ULP)* |
| KL 발산 (nats) | ≤ 2.82 × 10⁻¹⁰ |

**Footprint: 어느 단일 노드도 모델 전체를 들지 않는다**(가장 무거운 노드 = 가중치의 **42 %**, 계산 책임 비율이 아니라 장치에 실제로 측정된 값). 각 노드는 자기 조각만 실체화하므로(`init_empty_weights` + 선택적 safetensors 읽기), 전체 모델이 어느 한 대에도 상주하지 않는다.

**와이어는 얇고, 피어투피어이며, 선택적으로 절반 크기다**

| 지표 | 값 |
|---|---|
| 토큰당·홉당 와이어 (fp16) | **3.10 KB**, hidden state뿐 |
| 토큰당·홉당 와이어 (**int8**) | **1.52 KB**, fp16의 51 % (실측 fidelity ~67 %, 완화) |
| 토큰당 텐서 횡단: 스타 → **체인** | 2N → **N+1** |
| 헤드 데이터 플레인 대역폭: 스타 → **체인** | O(N) → **O(1)** |
| 프로토콜 오버헤드 (localhost, fp16 스타) | 홉당 ~1.2 ms, ~41 ms/token 연산에 압도됨 |

> 체리피킹한 승리가 아니라, 절대적이고 재현 가능한 숫자다. Apple 전용 클러스터에서는 Exo의 네이티브 MLX 경로가 원시 처리량에서 앞서지만, DRIFT의 축은 *이종이고, 정확하며, 검증 가능한* 것, 곧 어떤 경쟁자도 아예 돌지 못하는 지점이다.

---

## 인트로스펙션을 통한 모델 독립성

엔진은 모델 아키텍처를 결코 하드코딩하지 않는다. 로드 시점에 로드된 모델을 **인트로스펙션(introspect)**하여 적응하는바, 진실의 원천은 고정된 클래스가 아니라 로드된 모델이다. 서로 판이한 두 계열이 *동일한* 엔진에 그대로 들어맞는다:

| 모델 | 레이어 → 분할 | DRIFT가 처리하는 특이점 (인트로스펙션, 결코 하드코딩 아님) |
|---|---|---|
| **Qwen/Qwen2.5-1.5B-Instruct** *(주력)* | 28 → `0–14 / 14–28` | 평범한 디코더, 단일 RoPE θ, `DynamicCache`, 묶인(tied) `lm_head`, 정확성 기준선 |
| **google/gemma-4-E2B-it** *(보조)* | 35 → `0–18 / 18–35` | **레이어별 임베딩(Per-Layer Embeddings)** · sqrt(hidden) 임베딩 스케일링 · **이중 RoPE θ** · **하이브리드** 슬라이딩/글로벌 어텐션 · `HybridCache` + KV 공유 그룹; `transformers ≥ 5.5` 필요 |

모든 특이점은 하나의 플레인에 정연히 매핑되고 로드 시점에 `config`/시그니처로부터 발견되기에, Qwen을 돌리던 코드가 Gemma를 수정 없이 그대로 돌린다: *관찰할 수 있는 것에 의존하라, 아무것도 하드코딩하지 말라.*

**이 둘 너머로:** 위 표는 패리티 스위트로 비트 단위 증명을 마친 *게이트된* 계열이다. 같은 인트로스펙션이 DRIFT의 제약 안에 있는 **모든 decoder-only Hugging Face causal LM**을 감당하도록 설계되어 있다(설치된 `transformers`가 지원하는 아키텍처, 노드 합산 메모리에 들어가는 fp16 가중치). `drift run --model <hf-id>`(또는 `config.yaml`의 `model_id`)로 가리키면 분할·와이어 크기·레이어 플랜이 스스로 다시 유도되며, 새 모델도 `python -m drift.parity_test`로 같은 기준을 통과시키면 된다. 자세한 내용은 [운영 매뉴얼 §6](docs/manual.ko.md).

---

## 설계 근거 (왜 아닌가)

- **왜 노드 간에 `torch.distributed` / NCCL을 쓰지 않는가?** NCCL은 Apple Metal 디바이스와 NVIDIA CUDA 디바이스를 하나의 프로세스 그룹에 담을 수 없으니, 그것으로 끝이다. 게다가 그것은 데이터 플레인을 특정 백엔드에 결박시키는바, 바로 그것이 DRIFT가 거부하는 지점이다.
- **왜 스타가 아니라 피어투피어 체인인가?** 스타는 헤드를 O(N) 대역폭 허브, 곧 모든 액티베이션이 지나가는 단일 지점으로 만든다. 체인은 그것을 O(1)로 떨어뜨리며, 특권을 내려놓은 헤드의 전제 조건이다.
- **왜 스팟체크가 아니라 매 홉마다 영수증에 서명하는가?** 고정된 챌린지는 챌린지에서만 정직한 노드에게 빠져나갈 구멍을 준다. 검증을 실제 트래픽에 묶어 두면 선택적으로 정직할 대상 자체가 없어진다.
- **왜 페일오버 시 KV를 복제하지 않고 다시 prefill하는가?** 복제는 설계가 거부하는 대역폭이다. 다시 prefill하는 것은 한 번의 O(시퀀스)이며, 그리디가 결정론적이기에 비트 단위로 정확하다. v1에서는 매끄럽지만 무거운 것보다 정확하고 값싼 것이 낫다.
- **왜 텐서별이 아니라 그룹별 int8인가?** residual stream에는 이상치(outlier) 채널이 있어서, 텐서당 스케일 하나로는 나머지 전부를 짓뭉갠다(실측: 0% 일치). 128차원 블록당 스케일 하나는 fidelity를 쓸 만하게 유지하면서도 여전히 와이어를 약 절반으로 줄인다.
- **왜 와이어를 M0에서 고정하는가?** 노드 내부가 플래그 데이(flag day) 없이 언제까지나 바뀔 수 있도록 하기 위함이다. `route` / `collect` / `scale`은 모두 *선택적* 필드로 추가되었을 뿐, 결코 호환성을 깨뜨리지 않았다.

---

## 마일스톤

| # | 마일스톤 | 상태 |
|---|---|---|
| **M0–M3** | 환경 · 레퍼런스 오라클 · 인프로세스 + TCP 2-샤드 패리티 | ✅ **비트 단위** |
| **M4** | 머신 간: Mac MPS + NVIDIA CUDA | ✅ **실측**, 130/130 토큰 |
| **M6** | 우아한 노드 종료 감지 | ✅ 깔끔한 `NodeUnavailable` |
| **M7** | 피어투피어 체인 데이터 플레인 | ✅ **비트 단위** · 2N→N+1 횡단, O(1) 헤드 |
| **M8** | 암호화 + 인증된 와이어 (PSK + X25519 + ChaCha20) | ✅ **비트 단위** · 조작-터널 봉쇄 |
| **M9** | 비트 단위 페일오버: 재분할 + 리플레이 | ✅ 실행 도중 kill 후 **48/48 비트 단위** |
| **M10** | 얇은 헤드: 가중치 없는 오케스트레이터 | ✅ **비트 단위** |
| **M11** | 실시간 트래픽 위의 서명 영수증 검증 | ✅ 조작 **실시간 적발**, 정직 실행 깔끔 |
| **M12** | 가십 멤버십 + 동적 조인 | ✅ 시드가 전부 학습, 헤드가 확장 + 분할 |
| **M13** | 기여 원장 (`drift ledger`) | ✅ 집계 대사 일치, 위조된 라인 거부 |
| **M14** | WAN 성능: 그룹별 int8 와이어 | ✅ **와이어 절반**, 실측 fidelity ~67% |
| **M15** | 문서 정비: 이 README | ✅ |

위의 모든 것은 `drift itest`(실제 로컬 노드를 띄우고 분할을 인프로세스 레퍼런스에 대해 게이트한다)로 검증된다. 추측 디코딩(speculative decoding), 리더 없는 합의, 그리고 토큰 이코노미는 아직 구현된 것이 아니라 비전이다. 아래를 참조하라.

---

## 빠른 시작

Python **3.12**와 [`uv`](https://github.com/astral-sh/uv)가 요구된다. 두 기본 모델 모두 **게이트가 없어**, Hugging Face 로그인이 필요하지 않다.

**1 · 설치**, 각 머신에서:

```bash
git clone https://github.com/TaewoooPark/DRIFT && cd DRIFT
bash scripts/install.sh          # macOS / Linux   ·   Windows: powershell -File scripts\install.ps1
drift doctor                     # checks Python, torch, device, config, ports
```

**2 · 한 대의 머신에서 시도:**

```bash
drift up 2                       # 2 local nodes, auto-split, open a chat
drift up 3 --chain               # peer-to-peer: nodes stream to each other
drift up 2 --thin                # weightless head (embed + lm_head on the nodes)
drift up 2 --int8                # half-size wire (lossy, opt-in)
```

**3 · 당신의 Mac + CUDA PC에 걸쳐 하나의 모델을 실행**, 진짜배기:

```bash
# Windows/Linux PC (NVIDIA)     - one terminal
drift node --port 52601        # device = cuda, announced on the LAN

# Mac (Apple)                  - terminal 1: a worker
drift node --port 52600        # device = mps

# Mac                          - terminal 2: the head (type the prompt)
drift run --chain --prompt "hello world"
```

**와이어 암호화**(머신 사이에 키 하나를 공유):

```bash
drift keygen                     # prints DRIFT_NETWORK_KEY=<hex>
export DRIFT_NETWORK_KEY=<hex>   # on every machine: the wire is now encrypted + authenticated
```

**어디서나 참여**, NAT 뒤 노드가 터널을 열고 네트워크로 가십한다:

```bash
drift node --tunnel --join bore.pub:PORT      # needs a network key (no open compute)
drift run --expand --nodes bore.pub:PORT      # discover the whole membership, split across it
```

**누가 무엇을 계산했는가:**

```bash
export DRIFT_JOURNAL=~/drift.jsonl && drift run --chain --prompt "…"
drift ledger ~/drift.jsonl --verify           # per-node contribution, signatures re-checked
```

**OpenAI 호환 로컬 백엔드처럼 서빙하기** — 모델은 여전히 DRIFT 노드들에
분산되어 실행되고, 클라이언트와 맞닿는 표면만 HTTP/SSE가 된다:

```bash
drift serve --nodes 127.0.0.1:52600,127.0.0.1:52601 --api-key local-dev
```

지원되는 text-generation 표면은 `/v1/models`, `/v1/chat/completions`,
`/v1/completions`, `/v1/responses`, `/v1/embeddings`(hidden state를 노출할 수
있는 모드), tokenizer helper, health/readiness, metrics를 포함한다. 여러 선택지
(`n`)와 OpenAI 형태의 logprobs를 받으며, logits를 노출할 수 있는 DRIFT 실행은
선택 토큰/top-k logprob를 실제 logits 기반으로 돌려준다. Tool-call과 JSON
response-format은 API shape 호환 계층으로 제공되고, Responses streaming은
semantic SSE event를 내보낸다. DRIFT는 tool을 직접 실행하지 않으며 엄격한
schema-constrained decoding을 보장하지 않는다. Multimodal/audio와 thin-mode
sampling/embedding은 명시적인 OpenAI 형태 unsupported error로 응답한다. 자세한
지원 범위는 [docs/openai-compatibility.md](docs/openai-compatibility.md), 체크리스트
감사와 Python/JS SDK smoke 및 남은 full-stack gate는
[docs/openai-compatibility-audit.md](docs/openai-compatibility-audit.md)에 정리되어 있다.

**커스터마이즈 & 파인튜닝**, 곧 모델, 분할 지점, 디바이스, 트러블슈팅은 모두 **운영 매뉴얼 → [docs/manual.ko.md](docs/manual.ko.md)** ([English](docs/manual.md) · [中文](docs/manual.zh.md) · [日本語](docs/manual.ja.md))에 있다.

**라이브로 보기** — [**DRIFT-Demo**](https://github.com/TaewoooPark/DRIFT-Demo): 실제 실행을 두 화면으로 보여주는 비주얼 데모다 — 와이어를 건너는 residual stream, 레이어별 ‖Δh‖, tail이 직접 계산하는 top-k, 서명 영수증, 기여 정산까지 모든 픽셀이 라이브 트래픽에서 그려지며, DRIFT 소스는 건드리지 않는다.

---

## 저장소 지도: 어디를 봐야 하는가

```text
drift/
  protocol.py       # 계약 그 자체. 4바이트 길이 프리픽스 + msgpack; fp16/int8 텐서 직렬화/역직렬화
  crypto.py         # 네트워크 키 + 노드 신원; X25519+ChaCha20 채널; keygen
  engine_torch.py   # PyTorch 샤드: 인트로스펙션 레이어 호출, 로컬 KV 재인덱싱, 자체 RoPE  ← 핵심
  loader.py         # 슬라이스 가중치: init_empty_weights + 노드가 실행하는 shard만
  shard_server.py   # 동시 TCP 서버: ping / configure / prefill / decode / relay / gossip
  orchestrator.py   # 헤드 + 주입 가능한 전송 계층 (인프로세스 / 스타 / 체인) + 디코드 루프 + 검증기
  run.py, node.py   # `drift run` 헤드 + `drift node` 워커 (자동 분할, 디스커버리, 터널, --join)
  receipts.py       # 홉마다 서명된 영수증 + 실시간 검증기 + 저널 (원장의 원천)
  membership.py     # 가십 피어 테이블: 서명된 항목, 안티엔트로피, --expand
  ledger.py         # `drift ledger`: 영수증 저널로부터 노드별 기여
  verify.py         # 트러스트리스 스팟체크 (재계산 감사, 실시간 영수증을 보완)
  parity_test.py    # 인프로세스 / TCP 비트 단위 게이트 + 다중 프롬프트 --selftest
  itest.py          # 실제 노드 위의 통합 게이트: chain / secure / thin / kill / tamper / expand / ledger / int8
  bench.py, bench_m4.py   # 단일 머신 + 크로스머신(M4) 벤치마크
config.yaml         # 모델, dtype, 포트, 샤드 테이블
```

**리뷰어를 위한 짧은 목록:** `engine_torch.py`(KV 재인덱싱 + 인트로스펙션), `protocol.py`(고정된 와이어), `orchestrator.py`(주입 가능한 전송 계층 + 체인 + 검증기), `receipts.py`(신뢰 계층).

---

## FAQ

**이거 그냥 파이프라인 병렬화 아닌가?** *아이디어*는 그렇다. 그러나 기여는 **경계**에 있다: vLLM/Megatron의 PP는 `torch.distributed`+NCCL에 용접되어 MPS↔CUDA를 연결하지 못한다. DRIFT의 경계는 피어투피어로 흐르는 중립적이고 암호화된 바이트이며, 비트 단위로 정확함이 입증되고 스스로를 검증한다.

**네트워크가 내 토큰을 보는가?** 냉정하게 짚자. `input_ids`는 정수 토큰 id이지만 이는 *가역* 인코딩이어서, (공개된) 토크나이저를 가진 누구든 이를 당신의 텍스트로 되돌리며, 다운스트림 샤드는 그것을 필요로 한다. 그러니 와이어를 암호화하지 않는 한 **노드 운영자가 당신의 프롬프트를 읽을 수 있다.** `drift keygen` + `DRIFT_NETWORK_KEY`는 키를 공유한 노드에게만 스트림을 기밀로 만든다. 그것이 없으면 DRIFT는 평문이다(당신이 소유한 LAN이라면 괜찮다). 그리고 노드가 자기 연산에 대해 거짓말하지 않는지도 확인할 수 있다: 매 홉이 헤드가 실시간으로 검증하는 영수증에 서명하고, `drift verify`는 당신 소유가 아닌 노드를 재계산으로 감사한다.

**생성 도중 노드가 죽으면 어떻게 되는가?** 세션은 살아남는다: DRIFT는 생존 노드들(그리고 여분이 있다면 그것)에 걸쳐 재분할하고, 지금까지의 시퀀스를 리플레이하며, 애초에 떨어진 적 없는 경우와 비트 단위로 동일하게 이어 나간다. 매끄러운(리플레이 없는) 페일오버는 아직 없다. 그것은 복제를 필요로 한다.

**세 번째 노드를 추가할 수 있는가?** 그렇다. `drift up 3`, 또는 모든 멤버를 가십으로 발견해 그 전부에 걸쳐 분할하는 `drift run --expand`를 쓰면 된다. 와이어 계약은 바뀌지 않는다.

**왜 레퍼런스가 CPU가 아니라 MPS에서 도는가?** 연산 dtype이 fp16인 데 반해 PyTorch의 CPU fp16 커널은 신뢰할 수 없다. MPS는 fp16을 올바르고 결정론적으로 실행하기에 패리티 기준선이 MPS에 놓인다. CPU/CUDA도 설정 가능하다.

---

## 무엇이 구현됐고 무엇이 아직 비전인가

어려운 핵심, 곧 정확하고 **비트 단위로 검증된** 이종 분할은 끝났다. 그 위의 탈중앙 계층은 다이어그램이 아니라 **구현되고 게이트를 통과했다**:

| 능력 | 구현됨 | 마일스톤 |
|---|---|---|
| 한 대엔 너무 큰 모델 실행 (샤드별 적재) | ✅ | v0.10–0.16 |
| 피어투피어 데이터 플레인 (헤드 허브 없음) | ✅ | M7 |
| 암호화 + 인증된 와이어 | ✅ | M8 |
| 떨어져 나간 노드에 대한 비트 단위 페일오버 | ✅ | M9 |
| 무게 없는 헤드 | ✅ | M10 |
| 자기 검증: 실시간 트래픽 위의 서명 영수증 | ✅ | M11 |
| 가십 멤버십 + 어디서나 조인 | ✅ | M12 |
| 기여 원장 | ✅ | M13 |
| 절반 크기 int8 와이어 | ✅ | M14 |

**여전히 비전인 것**(정직하게): 리더 없는 **합의**(여전히 오케스트레이터가 매 실행을 시작한다), **시빌(Sybil) 저항**(가십 항목은 스스로 주장한 것이며, 가입 통제가 없다), 가격 책정 / 지급 / 온체인 정산을 갖춘 **토큰 이코노미**(원장은 입력일 뿐 정산이 아니다), **매끄러운 페일오버**(복제, 하여 리플레이 없음), **추측 디코딩**(샤드별 KV 롤백이 필요하다), 그리고 **N-of-M 중복 실행**(일관되게 잘못 계산하는 노드를 잡기 위함이며, 실시간 영수증만으로는 불가능하다). 이것들이 로드맵이다. 곧 *"P2P이고, 암호화되고, 스스로를 검증하며, 내결함성을 갖춘 이종 추론 네트워크로서 모든 단계가 단일 머신과 증명 가능하게 동일하다"*(오늘 참이다)와 *"완성된 탈중앙 토큰 이코노미"*(아직 아니다) 사이의 차이다.

---

## 연락처

<p align="center">
  <a href="https://github.com/TaewoooPark"><img src="https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github&logoColor=white&cacheSeconds=3600" alt="GitHub"></a>
  <a href="https://x.com/theoverstrcture"><img src="https://img.shields.io/badge/-X-000000?style=for-the-badge&logo=x&logoColor=white&cacheSeconds=3600" alt="X (Twitter)"></a>
  <a href="https://www.linkedin.com/in/taewoo-park-427a05352"><img src="https://img.shields.io/badge/-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white&cacheSeconds=3600" alt="LinkedIn"></a>
  <a href="https://www.instagram.com/t.wo0_x/"><img src="https://img.shields.io/badge/-Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white&cacheSeconds=3600" alt="Instagram"></a>
  <a href="https://taewoopark.com"><img src="https://img.shields.io/badge/-taewoopark.com-000000?style=for-the-badge&logo=safari&logoColor=white&cacheSeconds=3600" alt="Personal site"></a>
  <a href="mailto:ptw151125@kaist.ac.kr"><img src="https://img.shields.io/badge/-Email-D14836?style=for-the-badge&logo=gmail&logoColor=white&cacheSeconds=3600" alt="Email"></a>
</p>

<p align="center"><sub>데이터센터 없이. torch.distributed 없이. 당신의 머신과 다른 누군가의 머신이 하나의 정신을 돌린다, 피어투피어로, 암호화된 채, 그리고 서명된 채, 비트 하나까지.</sub></p>
