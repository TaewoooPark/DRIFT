# DRIFT —— 操作手册

**如何真正把 DRIFT 跑起来——从头到尾。** 语言：[English](manual.md) ·
[한국어](manual.ko.md) · **中文** · [日本語](manual.ja.md)

前半部分就是全部要点：安装、试跑、让一个模型跨你的机器运行。
后半部分——**定制与微调**——只在默认设置不够用时才需要。
关于基准测试方法论与数字，参见 [`benchmarks.md`](benchmarks.md)。

---

## 目录

**让它跑起来**
1. [安装](#1--安装)
2. [在一台机器上运行](#2--在一台机器上运行)
3. [跨你的机器运行——一个实操示例](#3--跨你的机器运行一个实操示例)

**定制与微调**
4. [`config.yaml` 参考](#4--configyaml-参考)
5. [选择切分点](#5--选择切分点)
6. [模型](#6--模型)
7. [设备与 dtype](#7--设备与-dtype)
8. [生成机制](#8--生成机制)
9. [手动驱动各分片](#9--手动驱动各分片)
10. [CLI 参考](#10--cli-参考)
11. [线缆与会话](#11--线缆与会话)
12. [内存](#12--内存)
13. [故障排查](#13--故障排查)

---

## 1 · 安装

需要 **Python 3.12** 和 [`uv`](https://github.com/astral-sh/uv)。两个内置模型都是
**无需授权（ungated）**的——不需要 Hugging Face 登录。请**在每台机器上**运行：

```bash
git clone https://github.com/TaewoooPark/DRIFT && cd DRIFT
bash scripts/install.sh          # macOS / Linux
# Windows (NVIDIA):  powershell -ExecutionPolicy Bypass -File scripts\install.ps1
drift doctor                     # checks Python, torch, device, config, ports
```

安装脚本会建一个 3.12 的 venv 并安装 DRIFT（`drift` CLI）。与平台匹配的 torch
wheel 会自动挑选 GPU 后端——Apple 上用 MPS，Linux 上用 CUDA；在 Windows 上脚本会拉取
CUDA 构建。`drift doctor` 应当显示你的设备（`mps` 或 `cuda`）。

---

## 2 · 在一台机器上运行

```bash
drift up 2                        # spawn 2 local nodes, auto-split the model, open a chat
drift up 2 --prompt "hello world" # …or a one-shot answer
```

`drift up N` 会在本机启动 N 个 worker 节点，读取模型的层数，把它均匀切分，给每个节点
分配自己的范围，然后生成。没有层范围，没有端口，没有 device 标志。这是看它跑起来的
最快方式；下一节则把节点放到*不同的*机器上。

---

## 3 · 跨你的机器运行——一个实操示例

**目标：** 在你的 **Mac** 上输入 `hello world`，让答案由 Mac（Apple/MPS）**和**一台
Windows PC（NVIDIA/CUDA）**共同**计算出来。

**角色。** **head** 负责输入 prompt 并持有 `embed` + `lm_head`；解码器层则住在**节点**上。
所以要把*两块* GPU 都用在层上，就让 Mac 同时跑一个**节点***和* head，PC 跑一个**节点**：

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

**你会看到什么**——head 发现了两个节点，切分模型，然后流式输出：

```
[run] discovering nodes on the LAN …
[run] found 192.168.0.22:52601(cuda), 127.0.0.1:52600(mps)

  model : Qwen/Qwen2.5-1.5B-Instruct
  head  : embed + norm + lm_head  · device=mps
  node  : 127.0.0.1:52600     layers [0:14)   · device=mps      ← the Mac computes these
  node  : 192.168.0.22:52601  layers [14:28)  · device=cuda     ← the PC computes these

Hello! How can I help you today?
```

**如果 head 找不到 PC**（在访客 / 企业 Wi-Fi 上 mDNS 常被阻断），就显式给节点点名——
这正是上面我们钉死端口的原因：

```bash
drift run --nodes 192.168.0.22:52601,127.0.0.1:52600 --prompt "hello world"
```

（Windows 机器用它的局域网 IP；Mac 自己的节点用 `127.0.0.1`。）先用
`drift doctor --nodes 192.168.0.22:52601` 检查可达性。

**同样的命令，任意组合。** 两台 Mac 或两台 Windows PC 的工作方式完全相同——`drift node`
自动检测各自的设备，`drift run` 负责发现并切分。只有两件事是 Mac + Windows 混合场景
特有的：

- **跨厂商浮点漂移。** MPS 与 CUDA 对 fp16 的舍入略有不同，所以一段长的贪心回答
  可能在*较靠后*的 token 上偏离单机结果。这是预期之内的，不是 bug（前若干 token 吻合，
  文本保持连贯）。同厂商的两个节点会**逐位**复现单机结果。
- **两个操作系统。** Mac 上用 `install.sh` 安装，PC 上用 `install.ps1`；之后的一切都完全相同。

---

**定制与微调**——下面的一切都是可选的，只在上面那套单命令流程不够用时才需要（换一个
模型、不均匀的切分、精确的端口、手动驱动各个部件）。

---

## 4 · `config.yaml` 参考

`config.yaml` 是模型、精度以及（对手动流程而言）分片表的唯一事实来源。`drift up` /
`drift run` 从中读取 `model_id`、`dtype` 和 `generation`；切分则由它们自己计算。

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

| 键 | 含义 |
|---|---|
| `model_id` | Hugging Face 模型 id。会下载一次到本地 HF 缓存。 |
| `dtype` | 计算**以及**线缆 dtype。`float16`（默认，CPU 往返无损）或 `float32`。`bfloat16` 在线缆上**无效**——见 §7。 |
| `device` | head 以及未指定自身设备的分片所用的默认设备。`mps` / `cuda` / `cpu`。 |
| `port` | 某个分片既无 `port` 又无 `DRIFT_PORT` 时的兜底端口。 |
| `shards[]` | 供手动流程（§9）以及发现服务一无所获时 `drift run` 兜底所用的有序分片表。 |
| `shards[].host` / `port` | 编排器拨号到该分片的地址。本地用 `127.0.0.1`；远程用局域网 IP。 |
| `shards[].start_layer` / `end_layer` | 半开的解码器层范围 `[start, end)`。 |
| `shards[].device` | 该分片的设备。 |
| `generation.max_new_tokens` | 默认 token 预算（用 `--max-new-tokens` 覆盖）。 |
| `generation.prompt` | 省略 `--prompt` 时的默认 prompt。 |

---

## 5 · 选择切分点

`drift run` 会按节点数量均匀切分；只有在手动流程（§9）或不均匀切分时你才需要考虑这个。
各范围必须**平铺（tile）**解码器层：连续、有序、无空隙、无重叠，且覆盖
`[0, num_hidden_layers)`。head 持有 `embed_tokens`、最终 norm 和 `lm_head`——它们绝不
属于任何分片范围。

```
model: 28 decoder layers (Qwen2.5-1.5B)
        └── [0, 14)  /  [14, 28)                 ✅ tiles 0..28 (the even 2-way split)
        └── [0, 10) / [10, 20) / [20, 28)        ✅ three shards, also valid
        └── [0, 14) / [16, 28)                   ❌ gap at 14–15
        └── [0, 16) / [14, 28)                   ❌ overlap at 14–15
```

在哪里切对正确性毫无影响（在单一设备上，任何平铺都是逐位精确的）；它只改变各节点
承担多少计算量与权重内存。如果两台机器速度不同，就把切分往更快的那台偏斜。层数：
Qwen2.5-1.5B = 28，Gemma-4-E2B = 35。

---

## 6 · 模型

引擎会**自省（introspect）**所加载的模型（解码器层类、`rotary_emb`、缓存类型、逐层
注意力），而不是硬编码某种架构——因此新的模型家族只需给出 id 即可接入。在
`config.yaml` 中设好 `model_id`（或 `drift run --model <id>`）；别无其他。

| 模型 | 层数 | 均匀切分 | 备注 |
|---|---:|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct`（默认） | 28 | `0–14 / 14–28` | 普通 decoder；一致性基线 |
| `google/gemma-4-E2B-it` | 35 | `0–18 / 18–35` | Per-Layer Embeddings（节点从 `input_ids` 重建）、双 RoPE θ、混合式注意力、`HybridCache`；需要 `transformers ≥ 5.5` |

更大的模型无非是有更多层可切——只要让各范围在其层数上保持连续即可。

---

## 7 · 设备与 dtype

**设备**——`mps`（Apple GPU）、`cuda`（NVIDIA GPU）、`cpu`（可移植，慢）。每个节点的
设备相互独立；这种独立性正是全部意义所在。`drift node` 会自动检测它；用 `--device`
覆盖。

**dtype**——`float16`（默认）或 `float32`。线缆把张量序列化为该 dtype 的原始字节，
而 fp16 的 CPU 往返是位无损的，所以序列化绝不会扰动结果。`bfloat16` 在线缆上**不受
支持**——请用 `float16`。

**同厂商 vs 混合**——同一设备家族上的两个分片会**逐位**复现单机结果。把 `mps` 与
`cuda` 混用会带来 fp16 舍入的位级差异，所以贪心解码可能在较靠后的 token 上发散
（这是预期之内的——§3）。

---

## 8 · 生成机制

- **贪心。** 参考 oracle 和编排器每一步都取 `argmax`；CLI 里还没有 temperature/top-p
  采样。一致性测试强制使用贪心以保证确定性。
- **EOS。** `drift run` / `drift up` 会在模型的序列结束（end-of-sequence）id 处停止
  （一个很窄的集合，而非每一个特殊 token）。一致性 / 参考路径运行固定的
  `max_new_tokens` 而不提前停止。
- **先 prefill 再 decode。** 整个 prompt 被处理一次（位置 `0…S-1`），然后一次一个
  token。每个节点跨步骤保留自己的 KV 缓存。

---

## 9 · 手动驱动各分片

`drift node` / `drift run` 这套流程是省心的路径。更底层的命令给你精确的控制
（固定的端口/范围、无发现服务），也正是一致性门禁和基准测试所使用的。

**1）让 `config.yaml` 指向每台机器**——设好每个分片的 `host`/`device` 并开放端口：

```yaml
shards:
  - { name: "mac",     host: "192.168.0.11", port: 52600, start_layer: 0,  end_layer: 14, device: "mps"  }
  - { name: "windows", host: "192.168.0.22", port: 52601, start_layer: 14, end_layer: 28, device: "cuda" }
```

**2）在每台机器上启动一个预分配的分片服务器**（绑定 `0.0.0.0` 以接受远程对等方）：

```bash
# on the Mac
DRIFT_PORT=52600 python -m drift.shard_server --name mac     --start 0  --end 14 --device mps  --host 0.0.0.0 --preload
# on the Windows PC
set DRIFT_PORT=52601
python -m drift.shard_server --name windows --start 14 --end 28 --device cuda --host 0.0.0.0 --preload
```

**3）从 head 驱动**——它从 `config.yaml` 读取 host/port：

```bash
python -m drift.orchestrator --ping                                  # both shards reply
python -m drift.orchestrator --prompt "Explain pipeline parallelism." # generate over the wire
```

在每台机器的防火墙中打开这些端口。编排器节点也会加载模型（用于
embed/norm/head），所以把它放在哪台机器上方便就放哪台。

---

## 10 · CLI 参考

每个命令都接受 `--config`（默认 `config.yaml`）。

### `drift` —— 高层命令

| 命令 | 作用 |
|---|---|
| `drift doctor` | 预检：Python/torch/device、依赖、`config.yaml` 平铺、端口可达性（`--nodes`）、防火墙提示 |
| `drift up N` | localhost：启动 N 个节点、自动切分、聊天（或 `--prompt` 做一次性生成） |
| `drift node` | 把**本**机作为 worker 运行：自动 device、`--port`、局域网宣告、等待 head |
| `drift run` | head：发现节点（或 `--nodes host:port,…`）、自动切分、配置、流式/聊天 |

`up`、`node`、`run` 都接受 `--max-new-tokens`；`run` 还接受 `--model`、`--nodes`、
`--no-discover`。在 `run`/`up` 上省略 `--prompt` 即可进入交互式聊天。

### 更底层的模块

| 模块 | 关键 flag |
|---|---|
| `python -m drift.shard_server` | `--name --start --end --device --host --port --preload`（+ `DRIFT_PORT`） |
| `python -m drift.orchestrator` | `--ping` · `--prompt` · `--max-new-tokens` · `--ports` |
| `python -m drift.reference` | `--device --out` —— 单机 oracle |
| `python -m drift.parity_test` | `--mode inprocess\|socket` · `--ports` · `--selftest` |
| `python -m drift.bench` | `--quick --no-socket --json`（见 [`benchmarks.md`](benchmarks.md)） |

---

## 11 · 线缆与会话

- **契约（`drift/protocol.py`，冻结不变）：** 每条消息都是一个 4 字节大端长度
  前缀 + 一个 msgpack dict。任何实现了这套帧格式的运行时都可以成为节点——线缆上
  没有 PyTorch。消息类型：`ping` / `configure` / `prefill` / `decode` / `reset`。
- **跨越边界的东西：** 只有 `hidden_states`（fp16）+ `position_ids` + `input_ids`。**KV 缓存
  永远不跨越**——每个节点保留自己的。每 token 的流量是 `hidden_size × 2` 字节外加
  几个整数（几 KB），与参数量无关。
- **可替换的节点。** 一个 `drift node` 启动时是未分配的；head 发来一个 `configure`
  （模型 + 层范围），所以你永远无需手写范围。预分配的服务器（§9）会跳过这一步。
- **会话。** 一次生成就是一个 `session_id`；每个节点持有一份按会话的 KV 缓存，一次
  生成结束时 head 会发送 `reset`。一个节点**一次只处理一个连接**——不要把两个 head
  指向同一个节点。

---

## 12 · 内存

按今天的实现，请为**每个节点上完整的模型驻留 RAM/VRAM** 做好准备：每个节点都会加载
整个 checkpoint，然后只使用它自己的那段层切片，而 head 也会加载模型（用于
embed/norm/head）。每个节点的*活跃*参数量更小（在默认的 2 路切分下，最重节点自己的层
约占模型的 ~42%——见 [`benchmarks.md`](benchmarks.md)），但把加载量削减到只加载那一段切片
是未来工作。在此之前：使用一个能装进每个节点的模型，或切分到**更多**节点上，以缩小
每个节点的活跃份额。

---

## 13 · 故障排查

| 症状 | 可能原因 → 修复 |
|---|---|
| `drift run` 找不到节点 | mDNS 被阻断（访客/企业 Wi-Fi）→ 给它们点名：`drift run --nodes host:port,…`。确认每个 `drift node` 都打印了自己的地址。 |
| `ConnectionRefusedError` | 节点未启动，或 host/port 有误。先启动节点；检查端口；`drift doctor --nodes host:port`。 |
| 本地能跑，跨机器不行 | 某个节点绑定到了 `127.0.0.1`。`drift node` 默认绑定 `0.0.0.0`；手动服务器请传 `--host 0.0.0.0`。打开防火墙端口。 |
| Windows：对等方够不着它 | 允许 `python.exe` 通过 Defender 防火墙（Private），启用网络发现（Network Discovery）。 |
| 输出仅在**靠后**的 token 上漂移（MPS↔CUDA） | 预期之内的厂商 fp16 舍入（§3、§7）。不是 bug。 |
| 一致性在 **token 1–2 处 FAIL** | 这是真正的 bug（mask/KV/RoPE），不是浮点噪声。 |
| 加载时内存溢出 | 每个进程都会加载完整 checkpoint（§12）。用更小的模型、更多节点，或给基准测试加 `--no-socket`。 |
| `unsupported wire dtype` | `dtype` 必须是 `float16` 或 `float32`（§7）。 |
| Mac 上出现罕见的 MPS op 报错 | 确保在启动进程的那个 shell 中设置了 `export PYTORCH_ENABLE_MPS_FALLBACK=1`。 |

---

用 `python -m drift.bench` 复现已发布的数字；方法论见
[`benchmarks.md`](benchmarks.md)。
