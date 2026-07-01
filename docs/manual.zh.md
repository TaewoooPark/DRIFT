# DRIFT —— 运维手册

**真正跑起 DRIFT 时，你能掌控的一切。** 语言：[English](manual.md) ·
[한국어](manual.ko.md) · **中文** · [日本語](manual.ja.md)

关于基准测试方法论与实测数字，参见
[`benchmarks.md`](benchmarks.md)。本文讲的是如何*运维*这套系统。

---

## 目录

1. [安装](#1--安装)
2. [60 秒快速运行](#2--60-秒快速运行)
3. [`config.yaml` 参考](#3--configyaml-参考)
4. [选择切分点](#4--选择切分点)
5. [跨两台机器运行（Mac + Windows）](#5--跨两台机器运行mac--windows)
6. [CLI 参考](#6--cli-参考)
7. [模型](#7--模型)
8. [设备与 dtype](#8--设备与-dtype)
9. [生成机制](#9--生成机制)
10. [线缆与会话](#10--线缆与会话)
11. [内存](#11--内存)
12. [故障排查](#12--故障排查)

---

## 1 · 安装

需要 **Python 3.12**（PyTorch 尚无 3.14 的 wheel）和
[`uv`](https://github.com/astral-sh/uv)。两个内置模型都是**无需授权（ungated）**的——不需要 Hugging
Face 登录。

```bash
git clone https://github.com/TaewoooPark/DRIFT && cd DRIFT
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install "torch" "transformers>=5.5" safetensors msgpack numpy huggingface_hub accelerate pyyaml
export PYTORCH_ENABLE_MPS_FALLBACK=1        # lets rare unimplemented MPS ops fall back to CPU
```

在 Windows/NVIDIA 上，请安装与你的工具包匹配的 CUDA 版 PyTorch，而不是默认的
wheel；其余一切完全相同。`PYTORCH_ENABLE_MPS_FALLBACK` 仅在 Mac 上有效，在其他平台上无害。

---

## 2 · 60 秒快速运行

只需安装一次——`bash scripts/install.sh`（macOS/Linux）或 `powershell -File scripts\install.ps1`
（Windows）——然后用 `drift doctor` 做一次健全性检查。要运行：

**在一台机器上：**

```bash
drift up 2      # spawn 2 local nodes, auto-split the model, and chat
                # add --prompt "…" for a one-shot answer
```

**跨机器：**

```bash
drift node      # on each worker — auto device, announced on the LAN
drift run       # on the head — auto-discovers the workers, splits, streams
```

**没有层范围，没有 IP，没有 device 标志。** `drift run` 读取模型的层数，
把它切分到自己找到的各个节点上，每个节点只计算自己那一片。下面的所有内容
都是关于如何改变一次运行的行为——而 §5–§6 还涵盖了手动驱动各分片的方法。

---

## 3 · `config.yaml` 参考

`config.yaml` 是唯一的事实来源。编排器、分片服务器、参考
oracle 和基准测试都读取它。

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

| 键 | 含义 |
|---|---|
| `model_id` | Hugging Face 模型 id。会下载一次到本地 HF 缓存。 |
| `dtype` | 计算**以及**线缆 dtype。`float16`（默认，CPU 往返无损）或 `float32`。`bfloat16` 在线缆上**无效**——见 §8。 |
| `device` | 当某个分片省略了自己的设备时使用的默认设备。`mps` / `cuda` / `cpu`。 |
| `port` | 某个分片省略了 `port` 且未设置 `DRIFT_PORT` 时的兜底端口。 |
| `shards[]` | 有序的分片列表。编排器**按此顺序**路由穿过它们。 |
| `shards[].name` | 逻辑名称，供 `--ports`/路由使用，并在 `--ping` 中显示。 |
| `shards[].host` | 编排器拨号到该分片的地址。本地用 `127.0.0.1`；远程用局域网 IP（§5）。 |
| `shards[].port` | 分片监听 / 编排器拨号的 TCP 端口。 |
| `shards[].start_layer` / `end_layer` | 该分片持有的半开解码器层范围 `[start, end)`。 |
| `shards[].device` | 该分片的设备（Mac 上是 `mps`，PC 上是 `cuda`……）。 |
| `generation.max_new_tokens` | `reference` 与编排器演示的默认 token 数。 |
| `generation.prompt` | 省略 `--prompt` 时，`reference` 与编排器的默认 prompt。 |

---

## 4 · 选择切分点

`shards[]` 的各个范围必须**平铺（tile）**模型的解码器层：连续、有序、无
空隙、无重叠，且覆盖 `[0, num_hidden_layers)`。编排器自身持有
`embed_tokens`、最终 norm 和 `lm_head`——它们**不属于**任何分片范围。

```
model: 28 decoder layers (Qwen2.5-1.5B)
        └── shard A: [0, 14)   ── shard B: [14, 28)      ✅ tiles 0..28
        └── [0, 10) / [10, 20) / [20, 28)                ✅ three shards, also valid
        └── [0, 14) / [16, 28)                           ❌ gap at 14–15
        └── [0, 16) / [14, 28)                           ❌ overlap at 14–15
```

- **分片数量**就是 `shards[]` 的长度——2 是演示用的，更多也没问题；
  编排器会按列表顺序路由穿过所有分片。
- **在哪里切**对正确性没有任何影响（在单一设备上，任何平铺都是逐位精确的）；
  它只改变*各节点承担多少计算量与权重内存*。默认是均匀的层切分；
  如果某台机器更快，可以让切分偏斜。
- **层数：** Qwen2.5-1.5B = 28，Gemma-4-E2B = 35。可以从任意一次运行的启动日志中读取
  （`reference` 会打印 `layers=…`），或从模型 config 中读取。

---

## 5 · 跨两台机器运行（Mac + Windows）

**简单的做法。** 在每个 worker 上运行 `drift node`（它会自动检测设备并宣告
自己）；在 head 上运行 `drift run`（它会在局域网上自动发现各 worker，切分
模型，并流式返回）。没有 IP，没有范围。如果局域网阻断了 mDNS，就显式列出各 worker：
`drift run --nodes 192.168.0.22:PORT,192.168.0.11:PORT`。本节其余部分是
**手动**方法——在你想钉死确切端口/范围，或想要完全手动控制时很有用。


§2 里的 localhost 运行，只需三处改动就能变成一个真正的集群。

**1）让配置指向各台机器。** 在编排器节点上，把每个分片的 `host` 设为
该机器的局域网 IP，并挑选开放的端口：

```yaml
shards:
  - { name: "mac",     host: "192.168.0.11", port: 52600, start_layer: 0,  end_layer: 14, device: "mps"  }
  - { name: "windows", host: "192.168.0.22", port: 52601, start_layer: 14, end_layer: 28, device: "cuda" }
```

**2）把每个分片服务器绑定到一个可达的地址。** 服务器默认绑定 `127.0.0.1`
（仅本地）。要接受远程连接，请用 `--host 0.0.0.0` 启动：

```bash
# on the Mac (192.168.0.11)
DRIFT_PORT=52600 python -m drift.shard_server --name mac     --start 0  --end 14 --device mps  --host 0.0.0.0 --preload
# on the Windows PC (192.168.0.22)
set DRIFT_PORT=52601
python -m drift.shard_server --name windows --start 14 --end 28 --device cuda --host 0.0.0.0 --preload
```

**3）从持有头部（head）的节点运行编排器。** 它直接从 `config.yaml` 读取
`host`/`port`，所以**省略 `--ports`**（它只覆盖端口，不覆盖 host）：

```bash
python -m drift.orchestrator --ping                                  # both shards should reply
python -m drift.orchestrator --prompt "Write a haiku about winter."  # front half on Apple, back half on NVIDIA
```

注意事项：
- 在每台机器的防火墙中打开所选端口。
- 编排器节点也会加载模型（用于 embed/norm/head），所以把它放在哪台
  机器上方便就放哪台——通常与分片 A 同机。
- 跨**不同 GPU 厂商**（MPS ↔ CUDA）时，fp16 在各家的舍入略有不同，因此
  贪心输出可能在*较靠后*的 token 上发散。前若干 token 吻合，文本保持连贯；
  这是预期之内的。在**同一**设备家族上，切分是逐位精确的（见
  [`benchmarks.md`](benchmarks.md)）。

---

## 6 · CLI 参考

每个入口点都接受 `--config`（默认 `config.yaml`）。

### `drift` —— 高层命令

| 命令 | 作用 |
|---|---|
| `drift doctor` | 预检：Python/torch/device、依赖、`config.yaml` 平铺、端口可达性、防火墙提示 |
| `drift up N` | localhost：启动 N 个节点、自动切分、聊天（或 `--prompt` 做一次性生成） |
| `drift node` | 把**本**机作为 worker 运行：自动 device、局域网宣告、等待 head |
| `drift run` | head：发现节点（或 `--nodes host:port,…`）、自动切分、配置、流式/聊天 |

`drift up`、`node` 和 `run` 都接受 `--max-new-tokens`；`run` 还接受 `--model` 和
`--nodes`。它们封装了下面这些更底层的模块——若要跑一致性门禁和基准测试，请直接
使用这些模块。


### `drift.shard_server` —— 运行一个分片

```bash
DRIFT_PORT=<port> python -m drift.shard_server [flags]
```

| Flag | 默认值 | 含义 |
|---|---|---|
| `--name` | `shard` | 逻辑分片名称。 |
| `--start` / `--end` | 取自 config 的 shard[0] | 解码器层范围 `[start, end)`。 |
| `--device` | config 的 `device` | `mps` / `cuda` / `cpu`。 |
| `--host` | `127.0.0.1` | 绑定地址。用 `0.0.0.0` 以接受远程节点。 |
| `--port` | `$DRIFT_PORT` 或 config 的 `port` | 监听端口。 |
| `--preload` | 关闭 | 在监听**之前**加载权重（推荐；避免第一次请求的冷启动）。 |

### `drift.orchestrator` —— 健康检查与生成

```bash
python -m drift.orchestrator [--ping] [--prompt "…"] [--max-new-tokens N] [--ports P1,P2]
```

| Flag | 含义 |
|---|---|
| `--ping` | 经 TCP ping 每个分片后退出（这就是 "M0" 可达性检查）。 |
| `--prompt` | 用于生成的 prompt。兜底为 `generation.prompt`。 |
| `--max-new-tokens` | token 预算。兜底为 `generation.max_new_tokens`。 |
| `--ports` | 逗号分隔的端口，覆盖每个分片 config 中的端口（host 不变——本地使用）。 |

这里的生成是贪心的，并且**在 EOS 处停止**。

### `drift.reference` —— 单机 oracle

```bash
python -m drift.reference [--device DEV] [--out reference_out.npz]
```

在一个设备上加载整个模型，并从 `generation.prompt` 贪心生成
`generation.max_new_tokens` 个 token，保存 token id + 首步 logits。这是
切分路径据以校验的基准真值（ground truth）。

### `drift.parity_test` —— 正确性门禁

```bash
python -m drift.parity_test --mode inprocess               # split in one process, no sockets
python -m drift.parity_test --mode socket --ports 52600,52601   # split over TCP (servers must be up)
python -m drift.parity_test --selftest                     # 6 prompts (EN/code/Korean, n=1…180)
```

| Flag | 含义 |
|---|---|
| `--mode` | `inprocess`（直接调用）或 `socket`（经线缆）。 |
| `--ports` | socket 模式所用的端口。 |
| `--ref` | 用于对比的参考文件（默认 `reference_out.npz`）。 |
| `--selftest` | 重新推导一份全新的参考，并在若干 prompt/长度上对比；无需 npz。 |

### `drift.bench` —— 基准测试

```bash
python -m drift.bench [--quick] [--no-socket] [--json out.json]
```

见 [`benchmarks.md`](benchmarks.md)。`--no-socket` 会跳过在低内存机器上
测量服务器启动开销的那一步。

---

## 7 · 模型

引擎会**自省（introspect）**所加载的模型（解码器层类、`rotary_emb`、缓存类型、
逐层注意力），而不是硬编码某种架构，因此新的模型家族只需给出 id 即可接入。

| 模型 | 层数 | 切分示例 | 备注 |
|---|---:|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct`（默认） | 28 | `0–14 / 14–28` | 普通 decoder；一致性基线 |
| `google/gemma-4-E2B-it` | 35 | `0–18 / 18–35` | Per-Layer Embeddings（分片从 `input_ids` 重建）、双 RoPE θ、混合式注意力、`HybridCache`；需要 `transformers ≥ 5.5` |

要切换模型，只需在 `config.yaml` 中设好 `model_id` 和一个有效的平铺——别无其他。更大的
模型无非是有更多层可切；只要让各范围在其层数上保持连续即可。

---

## 8 · 设备与 dtype

**设备**——`mps`（Apple GPU）、`cuda`（NVIDIA GPU）、`cpu`（可移植，慢）。一个分片的
`device` 与其他分片相互独立；这种独立性正是全部意义所在。

**dtype**——`float16`（默认）或 `float32`。线缆把张量序列化为该 dtype 的原始字节，
而 fp16 的 CPU 往返是位无损的，所以序列化绝不会扰动结果。`bfloat16` 在线缆上**不受支持**
——如果你需要 bf16 计算，那尚未打通；请用 `float16`。

**同设备 vs 混合厂商：** 同一设备家族上的两个分片能**逐位**复现单机结果。
把 `mps` 与 `cuda` 混用，会在两家厂商之间引入 fp16 舍入的位级差异，因此
贪心解码可能在较靠后的 token 上发散（这是预期之内的，不是 bug——§5）。

---

## 9 · 生成机制

- **仅贪心。** 参考 oracle 和编排器每一步都取 `argmax`；
  CLI 里没有暴露 temperature/top-p 采样。一致性测试强制使用贪心，好让
  输出确定且可比较。
- **EOS。** 编排器的 `--prompt` 路径会在模型的序列结束（end-of-sequence）id 处停止
  （一个很窄的集合——只包含真正的 EOS，而非每一个特殊 token）。一致性 / 参考路径
  运行固定的 `max_new_tokens` 而不提前停止，以便做精确对比。
- **先 prefill 再 decode。** 整个 prompt 被处理一次（prefill，位置 `0…S-1`），
  然后一次一个 token（decode，`S, S+1, …`）。每个分片跨步骤保留自己的 KV 缓存。

---

## 10 · 线缆与会话

- **契约（`drift/protocol.py`，冻结不变）：** 每条消息都是一个 4 字节大端长度
  前缀 + 一个 msgpack dict。任何实现了这套帧格式的运行时都可以成为节点——
  线缆上没有 PyTorch。
- **跨越边界的东西：** 只有 `hidden_states`（fp16）+ `position_ids` + `input_ids`。**KV 缓存
  永远不跨越**——每个分片保留自己的。每 token 的流量是 `hidden_size × 2` 字节外加
  几个整数（几 KB），与参数量无关。
- **会话。** 一次生成就是一个 `session_id`（默认 `s0`）。每个分片按会话持有一份
  KV 缓存；一次生成结束时，编排器会发送一个 `reset`。一个分片服务器
  **一次只处理一个连接**（顺序处理；并发是未来工作）——所以不要同时把两个
  编排器指向同一个分片。
- **TCP 调优。** 连接设置了 `TCP_NODELAY`；服务器设置了 `SO_REUSEADDR`。

---

## 11 · 内存

按今天的实现，请为**每个节点上完整的模型驻留 RAM/VRAM** 做好准备：每个分片服务器目前都会加载
整个 checkpoint，然后只使用它自己的那段层切片，而编排器也会加载模型（用于 embed/norm/head）。
每个节点的*活跃*参数量更小——在默认的 2 路切分下，最重节点自己的层约占模型的
~42%（见 [`benchmarks.md`](benchmarks.md)）——但把磁盘加载量削减到只加载那一段切片是未来
工作。在此之前：

- 使用一个能装进每个节点内存的模型，或切分到**更多**节点上，以缩小每个
  节点的*活跃*份额。
- 在内存吃紧的 Mac 上，`python -m drift.bench --no-socket` 会跳过额外启动
  完整模型的服务器进程。

---

## 12 · 故障排查

| 症状 | 可能原因 → 修复 |
|---|---|
| 编排器报 `ConnectionRefusedError` | 分片尚未启动，或 `host`/`port` 有误。先启动服务器；确认它打印了 `listening on …`；检查端口是否匹配。 |
| localhost 能跑，跨机器不行 | 服务器绑定到了 `127.0.0.1`。用 `--host 0.0.0.0` 重启它，并打开防火墙端口。 |
| `--ping` 对某个分片失败 | 该分片的进程挂了，或它的端口/host 有误。重新检查它的 `--start/--end/--device` 以及模型是否已加载。 |
| 一致性在 **token 1–2 处 FAIL** | 这是真正的 bug（mask/KV/RoPE），不是浮点噪声——切分逻辑发散了。 |
| 贪心输出仅在**靠后**的 token 上漂移，MPS↔CUDA | 预期之内的厂商 fp16 舍入（§5）。不是 bug。 |
| 加载时内存溢出 | 每个进程都会加载完整 checkpoint（§11）。用更小的模型、更多分片，或给基准测试加 `--no-socket`。 |
| `unsupported wire dtype` | `dtype` 必须是 `float16` 或 `float32`（§8）。 |
| Mac 上出现罕见的 MPS op 报错 | 确保在启动进程的那个 shell 中设置了 `export PYTORCH_ENABLE_MPS_FALLBACK=1`。 |
| 健康检查之后某个分片像是卡住了 | 有一个多余的第二连接连到了一次只处理一个连接的服务器。使用单个编排器；复用它的连接（§10）。 |

---

用 `python -m drift.bench` 复现已发布的数字；方法论见
[`benchmarks.md`](benchmarks.md)。
