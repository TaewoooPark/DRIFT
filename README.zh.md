<h1 align="center">DRIFT</h1>

<p align="center"><b>Decentralized Routed Inference For Tokens —— 一个模型，拆分到你自己的多台机器上运行，无需数据中心。</b></p>

<p align="center">
  <a href="./README.md">English</a> ·
  <a href="./README.ko.md">한국어</a> ·
  <b>中文</b> ·
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
  <img src="docs/img/hero.png" alt="一个模型，跨越地球两端拆分 —— 纽约与首尔共同运行一个模型，无需数据中心" width="900">
</p>

<p align="center"><sub>纽约的朋友在睡觉时留着一个节点，你在首尔。DRIFT 把<b>一个</b>模型拆到两台机器上——他的 GPU 计算前面的层，你的计算后面的，hidden state 通过一条<b>加密</b>线缆<b>节点到节点</b>地串流，而且每一跳都<b>签署一份回执</b>——于是你们一起运行一个任何一台都放不下的模型，并且可证明地给出与单机相同的答案。</sub></p>

**DRIFT** 让**一个**大语言模型跨**异构个人设备**运行——一台 Mac（Apple GPU，PyTorch **MPS**）和一台 Windows/Linux PC（NVIDIA GPU，PyTorch **CUDA**）——做法是把模型**逐层切分**（pipeline parallelism），并只在节点之间通过一套**框架中立的字节协议**（TCP + msgpack）流式传输 **hidden state**。没有数据中心，没有 `torch.distributed`，没有 NCCL，没有厂商锁定。数据平面*不绑定任何*框架，于是那些本来永远无法对话的运行时——一张 Apple Metal 计算图和一张 NVIDIA CUDA 计算图——如今得以共同运行一个模型，而其输出与在单机上运行整个模型**逐位（bit-for-bit）完全相同**。

在这个精确内核之上，DRIFT 已经长出了一层真正的**去中心化层**：hidden state 如今**点对点**串流（头节点不再是带宽枢纽），线缆**加密且经过成员认证**，掉线的节点会被**逐位**恢复，头节点可以是**无权重**的，每一跳都**签署一份回执**并由头节点在实时流量上验证，节点之间通过 **gossip 相互发现**，它们的贡献则记入一本**账本**。

**一句话讲清差异：** [Exo](https://github.com/exo-explore/exo) 把节点间通信绑定在 MLX（`mx.distributed`）上，因此只能*在 Apple 芯片之间*工作。DRIFT 把这条边界抬升为一套**中立、加密的线缆协议**——*不同的运行时、不同的 GPU 厂商、同一个模型*——用一道**逐位一致性门禁（bitwise parity gate）**证明这种切分是精确的，并用逐跳签名回执让它**自我验证**。一个不绑定任何框架、可证明精确、且无需信任节点即可核查的数据平面——这正是核心贡献。

**扩展。** 每个解码器层一个节点——默认 Qwen 最多可把一个模型拆到 **28** 台机器上（Gemma 为 **35** 台），并在所有机器之间流式运行。当前的最佳区间是 **2–4** 台。

> *"这段对话记录就是模型的输出。有意思的地方在于这段计算实际上**在哪里**运行——它逐位地对上了账、线缆是加密的，而且每一跳都为自己的工作签了名。"*

[**taewoopark.com** —— 作者主页](https://taewoopark.com)

---

## 目录

- [为何与众不同](#为何与众不同) —— 工程师们冲着来看的那张对比表
- [什么是 DRIFT](#什么是-drift) —— 名字、愿景与范围
- [五个平面](#五个平面) —— 控制 / 数据 / KV / 安全 / 信任
- [线缆契约](#线缆契约真正跨越边界的东西) —— schema + 每 token 字节数
- [三个正确性问题](#正确的切分必须解决的三个问题) —— KV 重索引、RoPE、mask
- [点对点、无权重头节点](#点对点以及无权重的头节点) —— 链式 + 瘦头节点
- [无需信任节点的信任](#无需信任节点的信任) —— 加密、签名回执、故障转移
- [正确性与一致性](#正确性一致性门禁) —— 逐位门禁 + 实测结果
- [基准测试](#基准测试) —— fidelity 100% · int8 上线缆减半 · O(1) 头节点带宽
- [通过自省实现模型无关](#通过自省实现模型无关) —— Qwen、Gemma，零硬编码
- [设计取舍（为何不）](#设计取舍为何不) —— 那些决定及其理由
- [里程碑](#里程碑) · [快速开始](#快速开始) · [仓库地图](#仓库地图该看哪里) · [常见问题](#常见问题) · [仍是愿景](#已交付-vs-仍是愿景)

---

## 为何与众不同

DRIFT 的全部要点都落在**节点之间的边界**上。下面把这条边界与已有工作做个对比：

| | **DRIFT** | Exo | Petals | llama.cpp RPC | vLLM / Megatron PP |
|---|---|---|---|---|---|
| **切分单元** | decoder 层 | 层 | transformer 块 | 层 / 张量 | 层（stage） |
| **节点↔节点传输** | **TCP + msgpack** | MLX `mx.distributed` | gRPC（torch 张量） | 自定义 RPC（ggml） | `torch.distributed` + NCCL |
| **框架中立的线缆** | **✅ 是** | ❌ 绑定 MLX | ❌ 绑定 torch | 绑定 ggml | ❌ 绑定 torch/NCCL |
| **异构 GPU 厂商** | **✅ MPS + CUDA 同时** | ❌ 仅 Apple | 部分 | ✅（ggml 后端） | ❌ NCCL 无法桥接 |
| **数据平面拓扑** | **✅ 点对点链式** | activation | activation | activation | activation |
| **线缆加密 + 节点认证** | **✅ X25519 + ChaCha20 + PSK** | ❌ | ❌ | ❌ | ❌ |
| **自我验证（逐跳签名）** | **✅ Ed25519 回执，实时** | ❌ | ❌ | ❌ | ❌ |
| **逐位精确的故障转移** | **✅ 重新切分 + 重放** | ❌ | ~（重路由） | ❌ | ❌ |
| **每 token 跨越边界的量** | **~1.5–3 KB（仅 hidden）** | activation | activation | activation | activation |
| **正确性契约** | **对比单机的逐位一致** | — | — | — | — |

从上到下读完这张表，论点自然浮现：**所有人都在传递 activation；唯有 DRIFT 让这种传递做到框架中立、加密、点对点，*并且*可证明逐位精确——然后还让你无需重跑模型就能核查一个节点有没有撒谎。** NCCL 无法把一块 Apple GPU 和一块 NVIDIA GPU 放进同一个进程组，MLX 走不出 Apple 生态。DRIFT 的答案是让线缆*只承载字节*——没有 torch 对象、没有 MLX 数组、没有 CUDA 句柄——于是两个世界在一份双方都能实现的契约上相遇，然后再把这份契约加固。

---

## 什么是 DRIFT

一个无服务器、点对点的推理网络：异构个人设备按层切分**同一个**模型并**协同**运行。不再经由超大规模云厂商的数据中心中转，而是*你的机器和别人的机器*汇聚起来，共同运行同一个 AI。

这个名字本身就是这套系统：

| 字母 | 含义 |
|---|---|
| **D** —— Decentralized（去中心化） | 没有数据中心；hidden state **点对点**地节点→节点串流，线缆经过加密 + 成员认证，掉线的节点会被恢复。仍由一个编排器启动整场运行，且头节点可以做成无权重的——完整的无领导者共识仍是愿景（见[仍是愿景](#已交付-vs-仍是愿景)）。 |
| **R** —— Routed（路由） | 由编排器把 hidden state *路由*穿过各节点，推动推理向前 |
| **I** —— Inference（推理） | 工作负载是 LLM 推理（可扩展到训练） |
| **For T** —— For Tokens（为了 token） | "token" 的双重含义：**推理** token（机器思维的原子）**以及**价值 token（靠贡献赚取、用于支付推理）。如今每一跳都会签署一份回执，`drift ledger` 会统计贡献——这正是分账层所消费的输入。DRIFT 的愿景是让思维的单位与价值的单位合而为一。 |

> **本仓库的范围。** 技术硬核——*一个跨 Mac 与 Windows 切分的模型能否给出正确答案？*——已交付并经**逐位**证明。在其之上，**"For Tokens"** 的底座不再只是一张图：**点对点加密数据平面**、**逐位故障转移**、**无权重头节点**、**实时流量上的签名回执验证**、**gossip 成员协议**，以及一本**贡献账本**，全部已实现并纳入门禁。完整的 token 经济、链上结算与无领导者共识仍是愿景。

---

## 五个平面

<p align="center"><img src="docs/img/arch.png" alt="DRIFT architecture — orchestrator head, per-layer shards, neutral wire" width="900"></p>

DRIFT 干净地分成几个平面：

- **控制平面**——编排器为每个节点分配一段层范围（`configure`）并驱动解码循环。节点有四种发现方式：零配置局域网发现（mDNS）、显式的 `--nodes host:port` 列表、由 NAT 后节点用 `drift node --tunnel` 开启的公共 `bore.pub` 隧道，或 **gossip**——一个节点 `--join` 某个种子，网络便自行学得成员全貌，随后 `drift run --expand` 据此完成切分。
- **数据平面**——跨越 stage 边界的只有 `hidden_states`（浮点）加上 `position_ids` + `input_ids`（整数）。与框架无关，而且——关键在于——它的大小取决于 `hidden_size`，而非参数量。**它如今点对点流动**（`--chain`）：head → n0 → n1 → … → tail → head，于是每 token 的张量跨越次数从 2N 降到 **N+1**，头节点的带宽也从 O(N) 变为节点数上的 **O(1)**。还可选 **int8**（`--int8`）把字节数减半。
- **KV 缓存平面**——每个分片按会话、在自己的设备上保存*属于自己*那段层范围的 KV。缓存永远不跨越线缆（否则将是每 token 数 MB，会葬送整个设计）。只有残差流（residual stream）在传输。
- **安全平面**——一个网络共享一把预共享密钥（`drift keygen`）。此后每条连接都跑一条 X25519 ECDH → HKDF（混入 PSK）→ ChaCha20-Poly1305 的通道，于是数据流是机密的，没有密钥的拨号方会被丢弃。`drift node --tunnel` 拒绝在无密钥下运行（不提供开放的公共算力），且长度前缀设有上限（杜绝 alloc-DoS）。
- **信任平面**——每一跳都对 `(in_hash, out_hash, layer range)` 签署一份 **Ed25519 回执**。头节点在真实流量的**每一个 token** 上校验签名 + 相邻性 + 端点锚定（而非另设一次盘问），于是线缆损坏、丢跳/伪造跳，以及节点在"算了什么"与"发了什么"之间撒谎，都会被实时抓出。掉线的节点则通过在幸存者上重新切分并重放而被**逐位**恢复。

**切分可以扩展到两台以上**——每个解码器层一个节点，最多 28 个（Gemma 为 35 个），而头节点与线缆保持不变：

<p align="center"><img src="docs/img/scale.png" alt="DRIFT scales one model across 2 to 28 nodes, one decoder layer per node" width="900"></p>

---

## 线缆契约（真正跨越边界的东西）

这份契约（`drift/protocol.py`）是**冻结不变的**：每条消息都是**一个 4 字节大端长度前缀 + 一个 msgpack dict**（设置网络密钥后，会作为单个 ChaCha20-Poly1305 帧加密）。任何未来的运行时——MLX、ggml、JAX、一个 Rust 节点——只需实现这套帧格式就能加入。线缆上没有任何 PyTorch。

```jsonc
// 请求（编排器 → 分片，或链式模式下分片 → 分片）
{
  "type":         "prefill" | "decode" | "reset" | "ping" | "configure",
  "session_id":   "s0",               // 一个生成序列
  "seq_id":       42,                 // 单调递增，用于排序 / 调试
  "shape":        [1, 1, 1536],       // hidden_states 形状（decode 时 S=1）
  "dtype":        "float16" | "int8",  // int8 → 线缆减半（有损）
  "scale":        "<per-group fp16>",  // int8 反量化 scale（fp16 时无此字段）
  "position_ids": [37],               // 绝对位置  → RoPE，在分片上计算
  "input_ids":    [785],              // token id → 逐层嵌入（PLE）/ 瘦头节点嵌入
  "tensor":       "<raw bytes>",       // 行主序 hidden_states
  "route":        [["10.0.0.2", 52601]], // 链式模式：下游节点
  "collect":      ["10.0.0.9", 6000]     // 链式模式：头节点的汇聚端
}

// 响应（分片 → 下一跳 / 头节点）
{ "ok": true, "shape": [1,1,1536], "dtype": "float16", "tensor": "<bytes>",
  "receipt": { "node": "<pubkey>", "in_hash", "out_hash", "start", "end", "sig" },
  "token":  785 }   // 瘦头节点的尾节点返回一个 token id 而非张量
```

`route` / `collect` 是**增量且可选的**——没有它们的节点行为与经典星形完全一致。`configure` 为一个**可替换（fungible）**节点分配层范围（以及瘦头节点的边缘职责），因此用户永远无需手写范围。

**每 token 的字节数。** 在 decode 阶段，activation 是 `[1, 1, hidden]`。对 Qwen 的 `hidden = 1536` 来说，fp16 是 **3 072 字节**，int8 是 **1 560 字节**（H 的 int8 + 逐组 fp16 scale ≈ 0.51×）。一条链每 token 做 `N+1` 次这样的跨越，星形则做 `2N` 次。在局域网上，这相比计算量微不足道。

**为什么线缆上用 fp16 是安全的（逐位）。** 序列化是一次 CPU fp16 往返。如果计算 dtype 本就是 fp16，这次往返就是**位无损的**——正是这个前提，让切分路径能够*精确*复现单机结果，而非近似。int8 *不是*无损的，且需显式启用；它跑在一道放宽的门禁下，绝不会走逐位门禁。

---

## 正确的切分必须解决的三个问题

把层切分到多个进程听起来微不足道，直到你试图让输出与未切分的模型*完全一致*。有三件事会咬人，DRIFT 逐一显式处理。

### 1 · KV 缓存索引——最微妙的一个

Hugging Face 的 `DynamicCache` 从**第 0 层**的槽位来报告"历史长度（past length）"。一个保留全局层 `[14, 28)` 的分片，若沿用这些全局索引，就会让缓存槽 0 **空着**——于是在 decode 时因果掩码会当作*没有历史*来构建，而一致性在第一个 token 之后就悄然崩坏。

<p align="center"><img src="docs/img/kv-reindex.png" alt="KV-cache local re-indexing — the fix that keeps decode parity" width="900"></p>

DRIFT 在加载时把每个分片保留的层重索引为**本地、从 0 开始**的缓存槽，并把每会话的 `DynamicCache` 大小设为该分片的本地层数。

### 2 · RoPE 自计算——让线缆保持极小

旋转位置嵌入只依赖 `position_ids`。因此每个分片都通过模型自带的 `rotary_emb`，从**绝对**位置各自计算 `cos/sin`。边界上携带的是寥寥几个整数，而不是一整个 `cos/sin` 张量，每个节点都保持自给自足。

### 3 · 每阶段的注意力掩码

prefill 时掩码是完整因果（causal-full）；decode 时它感知 KV 长度。DRIFT 在每个分片上用所安装的 Transformers 掩码工具重建掩码，**逐层**根据该层自身的 attention 类型来选择（Gemma 交替使用 local/global）——没有任何硬编码。

---

## 点对点，以及无权重的头节点

**链式串流（`--chain`）。** 不再把每一跳都星形路由回头节点，而是让 hidden state 沿路由节点→节点地流动，尾节点把最终状态交付到头节点的收集端。两个收益：每 token 的张量跨越次数从 **2N 降到 N+1**，而且——重点——头节点的数据平面带宽在节点数上变为 **O(1)**，而非 O(N)。头节点不再是每个 activation 都要穿过的枢纽。

**瘦头节点（`--thin`）。** 头节点可以持有**零模型权重**：`embed_tokens` 移交给第一个节点，`norm` + `lm_head` + `argmax` 移交给最后一个节点。与链式结合后，头节点向流水线送入**一个整数 token id**，再取回**一个整数 token id**——它不做任何张量运算，也不实体化任何参数。一致性依然成立，因为 `norm`+`lm_head`+`argmax` 是在同一设备上、用同一套（绑定的）权重、对逐位相同的 hidden state 运行的——argmax 与它由头节点还是尾节点计算无关。

解码循环只写**一次**，跑在一个可注入的传输层之上；被替换的只有传输层（进程内 / 星形 / 链式），所以里程碑之间唯一的变量就是网络，任何回归都*可证明地*是传输 bug，而绝不是逻辑 bug。

<p align="center"><img src="docs/img/decode-loop.png" alt="The decode loop over an injectable transport (in-process / TCP / chain)" width="900"></p>

---

## 无需信任节点的信任

**加密且经过认证的线缆（`drift keygen`）。** 一个网络共享一把 32 字节的预共享密钥。带密钥的连接会做一次 X25519 ECDH（临时密钥 → 前向保密）→ 混入 PSK 的 HKDF-SHA256 → 带逐方向计数器 nonce 的 ChaCha20-Poly1305。把 PSK 混入 KDF 就是成员身份校验：没有它的对端会推导出不同的密钥，其第一帧便无法解密。无密钥时保持明文，方便本地开发；一旦加密，则是全网范围。

**实时流量上的签名回执。** 每一跳都对 `(session, seq, mode, layer range, in_hash, out_hash)` 签署一份 Ed25519 回执。头节点在**每一个 token** 上校验签名、相邻性（第 *i* 跳的 `out_hash` == 第 *i+1* 跳的 `in_hash`）以及端点锚定（第一跳的输入与头节点所发一致，最后一跳的输出与它所收一致）。篡改的节点会在普通生成过程中被抓出——没有另设的盘问可供选择性地诚实——并在本地声誉表中被标记为 SUSPECT。*它能抓到什么：* 线缆损坏、丢跳/乱序/伪造跳，以及节点在"算了什么"与"发了什么"之间撒谎。*它抓不到什么*（一个始终错算并对结果签名的节点）——那是重算审计（`drift verify`）或 N-of-M 冗余执行（未来）的职责。

<p align="center"><img src="docs/img/parity-gate.png" alt="The parity gate — strict bitwise on one device, relaxed across GPU vendors" width="900"></p>

**逐位故障转移。** 生成中途死掉一个节点，不再会杀死会话。编排器在幸存者（外加任意备用节点）上重新切分模型，对至此的序列重新 prefill 以重建每个节点的 KV，然后续跑。因为贪心解码在固定前缀上是确定性的，续跑的延续与从未掉线**逐位相同**——这一点通过在 decode 中途 kill 一个节点、再把完成的序列与不中断的参考比对来验证。

**贡献账本。** 头节点把每一条已验证的回执记入日志；`drift ledger` 将其折叠成逐节点的统计（承载的 token、服务的 layer-token、会话数），`--verify` 会重新核验每一个签名，`--csv` 可导出。这正是结算层的输入底座。

---

## 正确性——一致性门禁

DRIFT 是**正确性优先**的：在任何性能或去中心化工作之前，每一个带网络的步骤都必须**逐位**复现单机参考结果。速度不是重点——*异构切分推理做到精确*才是，而上面每一项特性都以此为门禁把关。

**实测结果**——Qwen2.5-1.5B-Instruct，Apple MPS，fp16：

| 门禁 | 隔离的对象 | 结果 |
|---|---|---|
| **进程内** 2 分片 | 切分 · RoPE · KV · mask | ✅ **6 / 6 个 prompt 逐位**（`n = 1…180`） |
| **TCP 星形** 2 进程 | 序列化 / 帧格式 | ✅ **逐位 == 参考** |
| **链式** 2 与 3 节点 | 点对点中继 | ✅ **逐位 == 参考** |
| **链式 + 加密** | AEAD 通道透明性 | ✅ **逐位**（加密不扰动 token） |
| **瘦头节点** 2 与 3 节点 | 无权重头节点、边缘 embed/lm_head | ✅ **逐位** |
| **decode 中途 kill**（链式 / 星形，入口 / 中间 / 尾） | 故障转移重放 | ✅ **48 / 48 逐位**，恢复被触发 |
| **篡改一个节点** | 实时回执验证 | ✅ **在实时流量上被抓出**，诚实运行 0 个可疑 |
| **MPS ↔ CUDA（M4）** | 跨厂商 fp16 舍入 | ✅ 3 个 prompt 上 **130 / 130 token** 一致 |

**MPS ↔ CUDA（M4）。** 前半在 Mac（Apple MPS）、后半在 Colab NVIDIA T4（CUDA）上运行，切分路径**逐字复现了单机（130/130 token）**——尽管两家厂商的 fp16 kernel 把首步 logit 差距拉大到 ~2×10⁻²（单设备为 ~8×10⁻³），却不足以在这里翻转 argmax。在更大规模上该差距可能翻转靠后的 token，为此才有**放宽的门禁** `python -m drift.parity_test --prefix-match K`。

<p align="center"><img src="docs/img/m4-result.png" alt="M4 measured — Mac Apple MPS + Colab NVIDIA T4 CUDA, 130/130 token match vs one machine" width="900"></p>

---

## 基准测试

*用 `python -m drift.bench` 复现单机数字；用 `python -m drift.itest …` 复现集成门禁。*

**Fidelity——切分会改变输出吗？** *（切分路径 vs 单机 oracle，greedy）*

| 指标 | 结果 |
|---|---|
| token 精确匹配——6 个 prompt，`n = 1…180` | **411 / 411 = 100.00 %** |
| 首步 logit 最大绝对差 (fp32) | 7.81 × 10⁻³ *(fp16 ULP)* |
| KL 散度 (nats) | ≤ 2.82 × 10⁻¹⁰ |

**Footprint——没有任何单节点承载整个模型**（最重的节点 = 权重的 **42 %**，设备上实测，而不仅是计算份额）。每个节点只实体化自己那一片（`init_empty_weights` + 按需读取 safetensors），因此完整模型永远不会驻留在任何一台机器上。

**线缆又薄、点对点，还可选减半**

| 指标 | 值 |
|---|---|
| 每 token 每跳的线缆量 (fp16) | **3.10 KB**——只有 hidden state |
| 每 token 每跳的线缆量 (**int8**) | **1.52 KB**——fp16 的 51 %（实测 fidelity ~67 %，放宽） |
| 每 token 张量跨越次数——星形 → **链式** | 2N → **N+1** |
| 头节点数据平面带宽——星形 → **链式** | O(N) → **O(1)** |
| 协议开销（localhost，fp16 星形） | 每跳约 1.2 ms，被约 41 ms/token 的算力压得微不足道 |

> 绝对、可复现的数字——不是精挑细选的胜利。在纯 Apple 集群上，Exo 原生的 MLX 路径在原始吞吐上更快；DRIFT 的轴是*异构、精确、可验证*——在这条轴上，没有任何竞品能跑起来。

---

## 通过自省实现模型无关

引擎从不硬编码模型架构。在加载时，它**自省（introspect）**所加载的模型并随之适配——以加载的模型为准，而非某个固定的类。两个截然不同的模型家族能落进*同一个*引擎：

| 模型 | 层数 → 切分 | DRIFT 处理的架构怪癖（自省得来，从不硬编码） |
|---|---|---|
| **Qwen/Qwen2.5-1.5B-Instruct** *(主力)* | 28 → `0–14 / 14–28` | 普通 decoder、单一 RoPE θ、`DynamicCache`、绑定（tied）的 `lm_head`——正确性基线 |
| **google/gemma-4-E2B-it** *(次要)* | 35 → `0–18 / 18–35` | **Per-Layer Embeddings** · sqrt(hidden) 嵌入缩放 · **双 RoPE θ** · **混合式** sliding/global attention · `HybridCache` + KV 共享组；需要 `transformers ≥ 5.5` |

每个怪癖都干净地映射到某个平面上，并且都是在加载时从 `config`/签名发现的，所以运行 Qwen 的代码原封不动就能运行 Gemma：*依赖你能观察到的东西，什么都不硬编码。*

**超出这两个家族：** 上表列出的是经过奇偶校验套件逐位验证的*已守门*家族。同一套内省机制被设计为可承载 DRIFT 约束内的**任何 decoder-only Hugging Face causal LM**（架构须被已安装的 `transformers` 支持；fp16 权重能装进各节点的合计内存）。用 `drift run --model <hf-id>`（或 `config.yaml` 的 `model_id`）指向它——切分、线上字节数与层计划都会自行重新推导——然后用 `python -m drift.parity_test` 让新模型通过同样的标准。详见[操作手册 §6](docs/manual.zh.md)。

---

## 设计取舍（为何不）

- **为什么不在节点间用 `torch.distributed` / NCCL？** NCCL 无法把一台 Apple Metal 设备和一台 NVIDIA CUDA 设备放进同一个进程组——没有余地。而且它会把数据平面耦合到某个后端，这恰恰是 DRIFT 所拒绝的。
- **为什么用点对点链式，而非星形？** 星形把头节点变成一个 O(N) 的带宽枢纽——一个每个 activation 都要穿过的单点。链式把它降到 O(1)，并且是去特权化头节点的前提。
- **为什么在每一跳都签署回执，而不是抽查？** 一个固定的盘问会被只在盘问时诚实的节点绕过。把验证绑定到真实流量，就没有可供选择性诚实的对象了。
- **为什么在故障转移时重新 prefill，而不是复制 KV？** 复制是这个设计所拒绝的带宽；重新 prefill 是一次 O(序列长度) 的操作，而且——因为贪心是确定性的——是逐位精确的。对一个 v1 而言，正确且廉价胜过无缝而沉重。
- **为什么用逐组 int8，而非逐张量？** 残差流有离群通道；每张量一个 scale 会把其余一切压垮（实测：0% 匹配）。每 128 维块一个 scale 既让 fidelity 保持可用，又依然把线缆约减半。
- **为什么在 M0 就冻结线缆？** 这样节点内部就能永远演进，而不必来一次"大切换日（flag day）"。`route` / `collect` / `scale` 全都是作为*可选*字段加入的——从不构成破坏性变更。

---

## 里程碑

| # | 里程碑 | 状态 |
|---|---|---|
| **M0–M3** | 环境 · 参考 oracle · 进程内 + TCP 2 分片一致性 | ✅ **逐位** |
| **M4** | 跨机——Mac MPS + NVIDIA CUDA | ✅ **已实测**——130/130 token |
| **M6** | 优雅的 kill-node 检测 | ✅ 干净的 `NodeUnavailable` |
| **M7** | 点对点链式数据平面 | ✅ **逐位** · 2N→N+1 跨越，O(1) 头节点 |
| **M8** | 加密 + 认证的线缆（PSK + X25519 + ChaCha20） | ✅ **逐位** · 篡改隧道已封堵 |
| **M9** | 逐位故障转移——重新切分 + 重放 | ✅ 运行中途 kill 后 **48/48 逐位** |
| **M10** | 瘦头节点——零权重编排器 | ✅ **逐位** |
| **M11** | 实时流量上的签名回执验证 | ✅ 篡改**被实时抓出**，诚实运行干净 |
| **M12** | gossip 成员协议 + 动态加入 | ✅ 种子学得全体，头节点扩展 + 切分 |
| **M13** | 贡献账本（`drift ledger`） | ✅ 统计对账一致，伪造条目被拒 |
| **M14** | WAN 性能——逐组 int8 线缆 | ✅ **线缆减半**，实测 fidelity ~67% |
| **M15** | 文档大修——本 README | ✅ |

以上所有内容都由 `drift itest` 检验（它会启动真实的本地节点，并把切分对照进程内参考进行门禁把关）。推测解码、无领导者共识与 token 经济是愿景，尚未交付——见下文。

---

## 快速开始

需要 Python **3.12** 和 [`uv`](https://github.com/astral-sh/uv)。两个默认模型都是**无需授权（ungated）**的——不需要 Hugging Face 登录。

**1 · 安装**——在每台机器上：

```bash
git clone https://github.com/TaewoooPark/DRIFT && cd DRIFT
bash scripts/install.sh          # macOS / Linux   ·   Windows: powershell -File scripts\install.ps1
drift doctor                     # checks Python, torch, device, config, ports
```

**2 · 在一台机器上试跑：**

```bash
drift up 2                       # 2 local nodes, auto-split, open a chat
drift up 3 --chain               # peer-to-peer: nodes stream to each other
drift up 2 --thin                # weightless head (embed + lm_head on the nodes)
drift up 2 --int8                # half-size wire (lossy, opt-in)
```

**3 · 让一个模型跨你的 Mac + 一台 CUDA PC 运行**——正式的玩法：

```bash
# Windows/Linux PC (NVIDIA)     — one terminal
drift node --port 52601        # device = cuda, announced on the LAN

# Mac (Apple)                  — terminal 1: a worker
drift node --port 52600        # device = mps

# Mac                          — terminal 2: the head (type the prompt)
drift run --chain --prompt "hello world"
```

**加密线缆**（在多台机器间共享一把密钥）：

```bash
drift keygen                     # prints DRIFT_NETWORK_KEY=<hex>
export DRIFT_NETWORK_KEY=<hex>   # on every machine — the wire is now encrypted + authenticated
```

**随处加入**——NAT 后的节点开启一条隧道并 gossip 进入网络：

```bash
drift node --tunnel --join bore.pub:PORT      # needs a network key (no open compute)
drift run --expand --nodes bore.pub:PORT      # discover the whole membership, split across it
```

**谁计算了什么：**

```bash
export DRIFT_JOURNAL=~/drift.jsonl && drift run --chain --prompt "…"
drift ledger ~/drift.jsonl --verify           # per-node contribution, signatures re-checked
```

**定制与微调**——模型、切分点、设备、故障排查——全都在**操作手册 → [docs/manual.zh.md](docs/manual.zh.md)**（[English](docs/manual.md) · [한국어](docs/manual.ko.md) · [日本語](docs/manual.ja.md)）里。

**现场演示** — [**DRIFT-Demo**](https://github.com/TaewoooPark/DRIFT-Demo)：用两块屏幕可视化一次真实运行——跨越网络的 residual stream、逐层 ‖Δh‖、尾节点自己算出的 top-k、签名回执与贡献账本——每一个像素都来自真实流量，且完全不改动 DRIFT 源码。

---

## 仓库地图——该看哪里

```text
drift/
  protocol.py       # 契约 —— 4 字节长度前缀 + msgpack；fp16/int8 张量序列化/反序列化
  crypto.py         # 网络密钥 + 节点身份；X25519+ChaCha20 通道；keygen
  engine_torch.py   # PyTorch 分片：自省式 layer 调用、本地 KV 重索引、self-RoPE  ← 核心
  loader.py         # 切片权重 —— init_empty_weights + 只加载节点运行的分片
  shard_server.py   # 并发 TCP 服务器：ping / configure / prefill / decode / relay / gossip
  orchestrator.py   # 头节点 + 可注入传输层（进程内 / 星形 / 链式）+ 解码循环 + 验证器
  run.py, node.py   # `drift run` 头 + `drift node` 工作节点（自动切分、发现、隧道、--join）
  receipts.py       # 逐跳签名回执 + 实时验证器 + 日志（账本的来源）
  membership.py     # gossip 对等表 —— 签名条目、反熵、--expand
  ledger.py         # `drift ledger` —— 从回执日志得出的逐节点贡献
  verify.py         # 无信任抽查（重算审计 —— 与实时回执互补）
  parity_test.py    # 进程内 / TCP 逐位门禁 + 多 prompt --selftest
  itest.py          # 基于真实节点的集成门禁：chain / secure / thin / kill / tamper / expand / ledger / int8
  bench.py, bench_m4.py   # 单机 + 跨机（M4）基准
config.yaml         # 模型、dtype、端口、分片表
```

**审阅者的重点清单：** `engine_torch.py`（KV 重索引 + 自省）、`protocol.py`（冻结的线缆）、`orchestrator.py`（可注入传输层 + 链式 + 验证器）、`receipts.py`（信任层）。

---

## 常见问题

**这不就是 pipeline parallelism 吗？** *想法*是的，但贡献在于那条**边界**：vLLM/Megatron 里的 PP 焊死在 `torch.distributed`+NCCL 上，无法桥接 MPS↔CUDA。DRIFT 的边界是中立、加密、点对点流动的字节——已被证明逐位精确且能自我验证。

**网络能看到我的 token 吗？** 请清醒看待：`input_ids` 是整数 token id，但那是一种*可逆*编码——任何持有（公开）分词器的人都能把它还原成你的文本，而下游分片确实需要它。所以除非你加密线缆，否则**节点运营者可以读到你的提示词。** `drift keygen` + `DRIFT_NETWORK_KEY` 会让数据流对共享密钥的节点保持机密；没有它，DRIFT 是明文的（对你自己拥有的局域网没问题）。而且你可以核查一个节点没有在其算力上撒谎——每一跳都签署一份回执并由头节点实时验证，`drift verify` 还能对你并不拥有的节点做重算审计。

**如果一个节点在生成中途死掉会怎样？** 会话得以存活：DRIFT 在幸存者（外加任意备用节点）上重新切分，重放至此的序列，然后继续——与从未掉线逐位相同。尚无无缝（零重放）的故障转移；那需要复制副本。

**我能加第三个节点吗？** 能——`drift up 3`，或者用 `drift run --expand` 去 gossip 发现每一个成员并跨全体切分。线缆契约不变。

**为什么参考跑在 MPS 而不是 CPU 上？** 计算 dtype 是 fp16，而 PyTorch 里的 CPU fp16 kernel 并不可靠；MPS 能正确且确定性地运行 fp16，所以一致性基线跑在 MPS 上。CPU/CUDA 都是可配置的。

---

## 已交付 vs 仍是愿景

技术硬核——一个正确、**逐位验证**的异构切分——已完成。在其之上的去中心化层是**已实现并纳入门禁**的，而非一张图：

| 能力 | 已交付 | 里程碑 |
|---|---|---|
| 运行单机放不下的模型（按分片加载） | ✅ | v0.10–0.16 |
| 点对点数据平面（无头节点枢纽） | ✅ | M7 |
| 加密 + 认证的线缆 | ✅ | M8 |
| 掉线节点上的逐位故障转移 | ✅ | M9 |
| 无权重头节点 | ✅ | M10 |
| 自我验证——实时流量上的签名回执 | ✅ | M11 |
| gossip 成员协议 + 随处加入 | ✅ | M12 |
| 贡献账本 | ✅ | M13 |
| 减半的 int8 线缆 | ✅ | M14 |

**仍是愿景**（据实相告）：无领导者**共识**（仍由一个编排器启动每场运行）、**抗女巫攻击（Sybil resistance）**（gossip 条目是自我声明的；没有准入控制）、带定价 / 分账 / 链上结算的 **token 经济**（账本是输入，而非结算）、**无缝故障转移**（靠复制副本，从而免去重放）、**推测解码**（需要逐分片的 KV 回滚），以及 **N-of-M 冗余执行**（用于抓出一个始终错算的节点——这是单靠实时回执做不到的）。这些是路线图——是*"一个 P2P、加密、自我验证、可容错的异构推理网络，每一步都可证明与单机相同"*（今天已然为真）与*"一个完工的去中心化 token 经济"*（尚未）之间的差别。

---

## 联系方式

<p align="center">
  <a href="https://github.com/TaewoooPark"><img src="https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github&logoColor=white&cacheSeconds=3600" alt="GitHub"></a>
  <a href="https://x.com/theoverstrcture"><img src="https://img.shields.io/badge/-X-000000?style=for-the-badge&logo=x&logoColor=white&cacheSeconds=3600" alt="X (Twitter)"></a>
  <a href="https://www.linkedin.com/in/taewoo-park-427a05352"><img src="https://img.shields.io/badge/-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white&cacheSeconds=3600" alt="LinkedIn"></a>
  <a href="https://www.instagram.com/t.wo0_x/"><img src="https://img.shields.io/badge/-Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white&cacheSeconds=3600" alt="Instagram"></a>
  <a href="https://taewoopark.com"><img src="https://img.shields.io/badge/-taewoopark.com-000000?style=for-the-badge&logo=safari&logoColor=white&cacheSeconds=3600" alt="Personal site"></a>
  <a href="mailto:ptw151125@kaist.ac.kr"><img src="https://img.shields.io/badge/-Email-D14836?style=for-the-badge&logo=gmail&logoColor=white&cacheSeconds=3600" alt="Email"></a>
</p>

<p align="center"><sub>没有数据中心。没有 torch.distributed。你的机器和别人的机器，共同运行一个心智——点对点、加密、且逐位地为之签名。</sub></p>
