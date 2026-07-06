# DRIFT — 運用マニュアル

**DRIFT を実際に動かす方法 — 最初から最後まで。** 言語:
[English](manual.md) · [한국어](manual.ko.md) · [中文](manual.zh.md) · **日本語**

前半がこの仕事のすべてです。インストールし、試し、あなたのマシンにまたがって 1 つのモデルを動かし、
OpenAI 互換のローカル endpoint として公開します。
後半 — **カスタマイズとチューニング** — は、デフォルトでは足りないときのためだけのものです。
ベンチマークの方法論と実測値については [`benchmarks.md`](benchmarks.md) を参照してください。

---

## 目次

**動かすまで**
1. [インストール](#1--インストール)
2. [1 台のマシンで実行する](#2--1-台のマシンで実行する)
3. [あなたのマシンにまたがって実行する — 実践例](#3--あなたのマシンにまたがって実行する--実践例)
4. [OpenAI 互換 API として配信する](#4--openai-互換-api-として配信する)

**カスタマイズとチューニング**
5. [`config.yaml` リファレンス](#5--configyaml-リファレンス)
6. [分割点の選び方](#6--分割点の選び方)
7. [モデル](#7--モデル)
8. [デバイスと dtype](#8--デバイスと-dtype)
9. [生成の仕組み](#9--生成の仕組み)
10. [シャードを手動で操作する](#10--シャードを手動で操作する)
11. [CLI リファレンス](#11--cli-リファレンス)
12. [ワイヤとセッション](#12--ワイヤとセッション)
13. [メモリ](#13--メモリ)
14. [トラブルシューティング](#14--トラブルシューティング)

---

## 1 · インストール

**Python 3.12** と [`uv`](https://github.com/astral-sh/uv) が必要です。バンドルされた 2 つのモデルはどちらも
**ゲートなし** です — Hugging Face のログインは不要です。これを **すべてのマシンで** 実行します:

```bash
git clone https://github.com/TaewoooPark/DRIFT && cd DRIFT
bash scripts/install.sh          # macOS / Linux
# Windows (NVIDIA):  powershell -ExecutionPolicy Bypass -File scripts\install.ps1
drift doctor                     # checks Python, torch, device, config, ports
```

インストーラーは 3.12 の venv を作り、DRIFT（`drift` CLI）をインストールします。プラットフォームに合った
torch wheel が GPU バックエンドを自動で選びます — Apple なら MPS、Linux なら CUDA。Windows では
スクリプトが CUDA ビルドを取得します。`drift doctor` は、あなたのデバイス（`mps` または `cuda`）を表示するはずです。

---

## 2 · 1 台のマシンで実行する

```bash
drift up 2                        # spawn 2 local nodes, auto-split the model, open a chat
drift up 2 --prompt "hello world" # …or a one-shot answer
```

`drift up N` は、このマシン上で N 個のワーカーノードを立ち上げ、モデルのレイヤー数を読み取り、それを
均等に分割し、各ノードに範囲を割り当てて生成します。レイヤー範囲も、ポートも、デバイスフラグも不要です。
これが動作を最速で確認する方法です。次の節では、ノードを *別々の* マシンに置きます。

---

## 3 · あなたのマシンにまたがって実行する — 実践例

**ゴール:** **Mac** で `hello world` と打ち込み、その答えを Mac（Apple/MPS）**と** Windows PC（NVIDIA/CUDA）
の **両方** を使って計算させる。

**役割。** **head** はプロンプトを打ち込み、`embed` + `lm_head` を保持します。デコーダーレイヤーは
**ノード** 上に置かれます。したがって、レイヤーに *両方* の GPU を使うには、Mac が **ノード** *かつ* head を動かし、
PC が **ノード** を動かします:

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

**表示される内容** — head は両方のノードを検出し、モデルを分割し、ストリーミングします:

```
[run] discovering nodes on the LAN …
[run] found 192.168.0.22:52601(cuda), 127.0.0.1:52600(mps)

  model : Qwen/Qwen2.5-1.5B-Instruct
  head  : embed + norm + lm_head  · device=mps
  node  : 127.0.0.1:52600     layers [0:14)   · device=mps      ← the Mac computes these
  node  : 192.168.0.22:52601  layers [14:28)  · device=cuda     ← the PC computes these

Hello! How can I help you today?
```

**head が PC を見つけられない場合**（ゲスト/企業の Wi-Fi では mDNS がブロックされていることがよくあります）、
ノードを明示的に名前で指定します — 上でポートを固定したのはそのためです:

```bash
drift run --nodes 192.168.0.22:52601,127.0.0.1:52600 --prompt "hello world"
```

（Windows マシンはその LAN IP で、Mac 自身のノードは `127.0.0.1` として。）まず `drift doctor --nodes 192.168.0.22:52601`
で到達性を確認してください。

**同じコマンド、どんな組み合わせでも。** 2 台の Mac でも 2 台の Windows PC でも同じように動きます — `drift node`
が各デバイスを自動検出し、`drift run` が見つけて分割します。Mac + Windows の混在ケースに固有なのは、
次の 2 点だけです:

- **ベンダー間の浮動小数点ドリフト。** MPS と CUDA は fp16 の丸めがわずかに異なるため、長い greedy な
  答えは *後半の* トークンで 1 台のマシンから分岐することがあります。これは想定内であり、バグではありません
  （序盤のトークンは一致し、テキストは一貫したままです）。同一ベンダーの 2 ノードは、1 台のマシンを
  **ビット単位で** 再現します。
- **2 つの OS。** Mac では `install.sh`、PC では `install.ps1` でインストールします。それ以降はすべて同一です。

---

## 4 · OpenAI 互換 API として配信する

OpenAI Python/JS SDK、`curl`、LangChain、LiteLLM、LlamaIndex、エディタプラグイン、または
自作 HTTP クライアントのように、OpenAI 風のローカルバックエンドを期待するクライアントには
`drift serve` を使います。モデルは引き続き DRIFT のノード間プロトコルで動き、クライアントに見える
入口だけが HTTP/SSE になります。

**1) 先に worker ノードを起動します。** `drift serve` は worker を自分では spawn しません。
LAN discovery を使ってもよいですし、ポートを固定して明示的に渡してもかまいません:

```bash
# terminal 1
drift node --port 52600

# terminal 2
drift node --port 52601
```

**2) OpenAI 互換 server を起動します。** デフォルトの bind はローカル専用の
`127.0.0.1:8000` です。他のマシンから HTTP で呼ぶ必要がある場合だけ `--host 0.0.0.0` を使います。

```bash
export DRIFT_API_KEY=local-dev
drift serve \
  --nodes 127.0.0.1:52600,127.0.0.1:52601 \
  --port 8000
```

クライアントの base URL は `http://127.0.0.1:8000/v1` です。HTTP server を自分のマシンの外へ
公開するなら、`--api-key` / `DRIFT_API_KEY` を有効にしておいてください。

**3) `curl` で呼びます。**

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

**4) OpenAI Python SDK からもそのまま使えます。**

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

**サポートされる surface。** 主な route は `GET /v1/models`、`POST /v1/chat/completions`、
`POST /v1/completions`、`POST /v1/responses`、選択した mode が hidden state を公開できる場合の
`POST /v1/embeddings`、`POST /v1/chat/completions/input_tokens`、`/tokenize`、
`/detokenize`、`/health`、`/ready`、`/metrics` です。Chat/Completions streaming は
`[DONE]` で終わる SSE を使い、Responses streaming は `response.created`、
`response.output_text.delta`、`response.completed` などの semantic events を出します。

**互換性の範囲。** Adapter は `n`、`temperature`、`top_p`、`top_k`、`min_p`、penalties、
`seed`、`stop`、logprobs、tool-call shape、JSON response format など、一般的な
OpenAI/vLLM 風の text-generation parameters を受け付けます。DRIFT は tools を代わりに実行せず、
厳密な schema-constrained decoding も保証しません。Multimodal/audio requests や thin-head
sampling/embeddings には、明示的な OpenAI 形状の error を返します。完全な matrix は
[`openai-compatibility.md`](openai-compatibility.md)、監査記録は
[`openai-compatibility-audit.md`](openai-compatibility-audit.md) を参照してください。

**よく使う `serve` flags。**

| Flag | 意味 |
|---|---|
| `--nodes host:port,...` | すでに動いている `drift node` worker を使う。 |
| `--no-discover` / `--discover-timeout` | `--nodes` を省略したときの LAN discovery を制御する。 |
| `--model` | `config.yaml` の `model_id` を上書きする。 |
| `--served-model-name` | `/v1/models` に公開する model name を変える。クライアントもこの名前を送る必要がある。 |
| `--max-new-tokens` | request が token limit を省略したときのデフォルト出力予算。 |
| `--chain`, `--thin`, `--int8`, `--expand` | `drift run` と同じ DRIFT routing modes を使う。 |
| `--api-key` / `DRIFT_API_KEY` | `Authorization: Bearer ...` または `x-api-key` を要求する。 |
| `--cors-origin` / `DRIFT_CORS_ORIGINS` | browser clients に許可する origin を指定する。 |
| `--max-concurrent-requests` | HTTP request の同時数上限。1 つの backend 上の generation は引き続き直列化される。 |

**起動中の server を smoke-test します。**

```bash
python scripts/openai_compat_smoke.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --api-key local-dev
```

---

**カスタマイズとチューニング** — 以下はすべて任意であり、上のワンコマンドフローでは足りないとき
（異なるモデル、不均等な分割、正確なポート、各パーツの手動操作）のためのものです。

---

## 5 · `config.yaml` リファレンス

`config.yaml` は、モデル、精度、そして（手動フロー向けの）シャードテーブルについての唯一の信頼できる
情報源です。`drift up` / `drift run` / `drift serve` はここから `model_id`、`dtype`、`generation` を読み取り、
分割は自分たちで計算します。

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

| キー | 意味 |
|---|---|
| `model_id` | Hugging Face のモデル id。ローカルの HF キャッシュへ一度だけダウンロードされます。 |
| `dtype` | 計算 **かつ** ワイヤの dtype。`float16`（デフォルト、CPU ラウンドトリップで無損失）または `float32`。`bfloat16` はワイヤ上では **無効** です — §8。 |
| `device` | head と、自分のデバイスを省略したシャードのためのデフォルトデバイス。`mps` / `cuda` / `cpu`。 |
| `port` | `port` も `DRIFT_PORT` も持たないシャード向けのフォールバックポート。 |
| `shards[]` | 手動フロー（§10）と、ディスカバリが何も見つけられなかったときの `drift run` フォールバックで使う、順序付きシャードテーブル。 |
| `shards[].host` / `port` | オーケストレーターがこのシャードに接続する先。ローカルなら `127.0.0.1`、リモートなら LAN IP。 |
| `shards[].start_layer` / `end_layer` | 半開区間のデコーダーレイヤー範囲 `[start, end)`。 |
| `shards[].device` | このシャードのデバイス。 |
| `generation.max_new_tokens` | デフォルトのトークン予算（`--max-new-tokens` で上書き）。 |
| `generation.prompt` | `--prompt` が省略されたときのデフォルトプロンプト。 |

---

## 6 · 分割点の選び方

`drift run` と `drift serve` はノード数で均等に分割するので、これを考えるのは手動フロー（§10）や不均等な分割のときだけです。
範囲はデコーダーレイヤーを **タイル状に** 敷き詰めなければなりません。すなわち連続し、順序どおりで、隙間なく、
重なりなく、`[0, num_hidden_layers)` を覆うことです。head は `embed_tokens`、最終ノルム、`lm_head` を
保持します — これらは決してシャード範囲の一部にはなりません。

```
model: 28 decoder layers (Qwen2.5-1.5B)
        └── [0, 14)  /  [14, 28)                 ✅ tiles 0..28 (the even 2-way split)
        └── [0, 10) / [10, 20) / [20, 28)        ✅ three shards, also valid
        └── [0, 14) / [16, 28)                   ❌ gap at 14–15
        └── [0, 16) / [14, 28)                   ❌ overlap at 14–15
```

どこで切っても正しさには何のコストもかかりません（どのタイリングも 1 台のデバイス上ではビット単位で厳密です）。
変わるのは、各ノードにどれだけの計算量と重みメモリが乗るかだけです。マシンの性能が異なるなら、速いほうへ
寄せてください。レイヤー数: Qwen2.5-1.5B = 28、Gemma-4-E2B = 35。

---

## 7 · モデル

エンジンは、アーキテクチャをハードコードする代わりに、ロードされたモデルを **イントロスペクト（内省）** します
（デコーダーレイヤーのクラス、`rotary_emb`、キャッシュの型、レイヤーごとのアテンション）。そのため、
新しいファミリーは id を指定するだけで組み込めます。`config.yaml` に `model_id` を設定するだけ（または
`drift run --model <id>`）。他には何もありません。

| モデル | レイヤー | 均等分割 | 備考 |
|---|---:|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct`（デフォルト） | 28 | `0–14 / 14–28` | プレーンなデコーダー、パリティのベースライン |
| `google/gemma-4-E2B-it` | 35 | `0–18 / 18–35` | Per-Layer Embeddings（ノードが `input_ids` から再構築）、二重 RoPE θ、ハイブリッドアテンション、`HybridCache`。`transformers ≥ 5.5` が必要 |

より大きなモデルは、単に分割すべきレイヤーが増えるだけです — 範囲はそのレイヤー数にわたって連続させてください。

---

## 8 · デバイスと dtype

**デバイス** — `mps`（Apple GPU）、`cuda`（NVIDIA GPU）、`cpu`（ポータブル、低速）。各ノードの
デバイスは独立しています。その独立性こそがすべての主眼です。`drift node` が自動検出します。
`--device` で上書きします。

**dtype** — `float16`（デフォルト）または `float32`。ワイヤはテンソルをこの dtype の生バイトとして
シリアライズし、fp16 の CPU ラウンドトリップはビット単位で無損失なので、シリアライズが結果を乱すことは
決してありません。`bfloat16` はワイヤ上で **サポートされていません** — `float16` を使ってください。

**同一ベンダー vs 混在** — 同じデバイスファミリー上の 2 つのシャードは、1 台のマシンを **ビット単位で**
再現します。`mps` と `cuda` を混在させると、fp16 の丸めにビットレベルの差が生じるため、greedy
デコーディングは後半のトークンで分岐する可能性があります（想定内 — §3）。

---

## 9 · 生成の仕組み

- **greedy。** リファレンスオラクルもオーケストレーターも、各ステップで `argmax` を選びます。CLI にはまだ
  temperature/top-p サンプリングはありません。OpenAI 互換 HTTP API は non-thin mode で一般的な
  sampling controls をサポートします。パリティテストは決定論のために greedy を強制します。
- **EOS。** `drift run` / `drift up` は、モデルの end-of-sequence id で停止します（あらゆる特殊トークンではなく、
  狭い集合）。パリティ/リファレンス経路は、停止なしで固定の `max_new_tokens` を実行します。
- **prefill してから decode。** プロンプト全体が一度処理され（位置 `0…S-1`）、その後一度に 1 トークンずつ
  処理されます。各ノードは、ステップをまたいで自分自身の KV キャッシュを保持します。

---

## 10 · シャードを手動で操作する

`drift node` / `drift run` のフローが簡単な道です。より低レベルなコマンドは、正確な制御（固定のポート/範囲、
ディスカバリなし）を与え、パリティゲートとベンチマークが使っているのもこれです。

**1) `config.yaml` を各マシンに向ける** — 各シャードの `host`/`device` を設定し、ポートを開けます:

```yaml
shards:
  - { name: "mac",     host: "192.168.0.11", port: 52600, start_layer: 0,  end_layer: 14, device: "mps"  }
  - { name: "windows", host: "192.168.0.22", port: 52601, start_layer: 14, end_layer: 28, device: "cuda" }
```

**2) 各マシンで事前割り当て済みのシャードサーバーを起動する**（リモートのピアを受け付けるには `0.0.0.0` にバインド）:

```bash
# on the Mac
DRIFT_PORT=52600 python -m drift.shard_server --name mac     --start 0  --end 14 --device mps  --host 0.0.0.0 --preload
# on the Windows PC
set DRIFT_PORT=52601
python -m drift.shard_server --name windows --start 14 --end 28 --device cuda --host 0.0.0.0 --preload
```

**3) head から操作する** — ホスト/ポートを `config.yaml` から読みます:

```bash
python -m drift.orchestrator --ping                                  # both shards reply
python -m drift.orchestrator --prompt "Explain pipeline parallelism." # generate over the wire
```

各ファイアウォールでポートを開けてください。オーケストレーターのノードもモデルをロードするので
（embed/norm/head のため）、都合のよいマシンで実行してください。

---

## 11 · CLI リファレンス

どのコマンドも `--config`（デフォルト `config.yaml`）を取ります。

### `drift` — 高レベルコマンド

| コマンド | 何をするか |
|---|---|
| `drift doctor` | プリフライト: Python/torch/デバイス、依存関係、`config.yaml` のタイリング、ポート到達性（`--nodes`）、ファイアウォールのヒント |
| `drift up N` | localhost: N 個のノードを立ち上げ、自動分割し、チャット（または `--prompt` でワンショット） |
| `drift node` | このマシンをワーカーとして実行: デバイス自動、`--port`、LAN でアナウンス、ヘッドを待機 |
| `drift run` | ヘッド: ノードを検出（または `--nodes host:port,…`）、自動分割、構成、ストリーミング/チャット |
| `drift serve` | DRIFT オーケストレーター上の OpenAI 互換 HTTP/SSE API（`/v1/chat/completions`、`/v1/completions`、`/v1/responses`、`/v1/embeddings`） |

`up`、`node`、`run` は `--max-new-tokens` を取ります。`run` はさらに `--model`、`--nodes`、
`--no-discover` を取ります。`serve` は `--api-key`、`--cors-origin`、`--served-model-name`、
`--host`、`--port`、`--max-concurrent-requests` と、`run` と同じノード/モデル routing flags を取ります。
`run`/`up` で `--prompt` を省略すると、インタラクティブなチャットになります。

### 低レベルモジュール

| モジュール | 主なフラグ |
|---|---|
| `python -m drift.shard_server` | `--name --start --end --device --host --port --preload`（+ `DRIFT_PORT`） |
| `python -m drift.orchestrator` | `--ping` · `--prompt` · `--max-new-tokens` · `--ports` |
| `python -m drift.reference` | `--device --out` — 単一マシンのオラクル |
| `python -m drift.parity_test` | `--mode inprocess\|socket` · `--ports` · `--selftest` |
| `python -m drift.bench` | `--quick --no-socket --json`（[`benchmarks.md`](benchmarks.md) を参照） |

---

## 12 · ワイヤとセッション

- **契約（`drift/protocol.py`、凍結済み）:** すべてのメッセージは、4 バイトのビッグエンディアンの長さ
  プレフィックス + msgpack の辞書です。このフレーミングを実装するランタイムなら何でもノードになれます — ワイヤ上に
  PyTorch は存在しません。メッセージタイプ: `ping` / `configure` / `prefill` / `decode` / `reset`。
- **何が越えるか:** `hidden_states`（fp16）+ `position_ids` + `input_ids` だけです。**KV キャッシュが
  越えることは決してありません** — 各ノードが自分自身のものを保持します。トークンあたりのトラフィックは
  `hidden_size × 2` バイトに数個の整数を加えたもの（数 KB）で、パラメータ数とは無関係です。
- **交換可能なノード。** `drift node` は未割り当てで起動します。head が `configure`（モデル + レイヤー範囲）を
  送るので、範囲を手で書く必要は決してありません。事前割り当て済みのサーバー（§10）はこれをスキップします。
- **セッション。** 1 つの生成は 1 つの `session_id` です。各ノードはセッションごとの KV キャッシュを保持し、
  head は生成が終わると `reset` を送ります。ノードは **一度に 1 つの接続** を扱います — 2 つの head を
  1 つのノードに向けないでください。

---

## 13 · メモリ

今日のところは **すべてのノードで、モデル全体が RAM/VRAM に乗る** ことを前提に計画してください。各ノードは
チェックポイント全体をロードし、その後自分のレイヤースライスだけを使います。head もモデルをロードします
（embed/norm/head のため）。ノードあたりの *アクティブな* パラメータはより小さくなります（デフォルトの
2 分割では、最も重いノード自身のレイヤーはモデルの約 42% です — [`benchmarks.md`](benchmarks.md) を参照）
が、ロードをスライスだけに削減するのは将来の課題です。それまでは: 各ノードに収まるモデルを使うか、
**より多くの** ノードに分割して各ノードのアクティブな取り分を縮めてください。

---

## 14 · トラブルシューティング

| 症状 | 考えられる原因 → 対処 |
|---|---|
| `drift run` がノードを見つけられない | mDNS がブロックされている（ゲスト/企業の Wi-Fi）→ 名前で指定する: `drift run --nodes host:port,…`。各 `drift node` が自分のアドレスを表示したことを確認する。 |
| `ConnectionRefusedError` | ノードが起動していないか、host/port が間違っている。まずノードを起動し、ポートを確認する。`drift doctor --nodes host:port`。 |
| ローカルでは動くが、マシン間では動かない | ノードが `127.0.0.1` にバインドされている。`drift node` はデフォルトで `0.0.0.0` にバインドします。手動サーバーの場合は `--host 0.0.0.0` を渡す。ファイアウォールのポートを開ける。 |
| Windows: ピアが到達できない | Defender Firewall（Private）で `python.exe` を許可し、Network Discovery を有効にする。 |
| 出力が **後半の** トークンでだけずれる（MPS↔CUDA） | 想定内のベンダー fp16 丸め（§3、§8）。バグではない。 |
| **トークン 1〜2 でパリティ FAIL** | 本物のバグ（マスク/KV/RoPE）であって浮動小数点ノイズではない。 |
| ロード時にメモリ不足 | 各プロセスがチェックポイント全体をロードする（§13）。より小さなモデル、より多くのノード、またはベンチでは `--no-socket` を使う。 |
| `unsupported wire dtype` | `dtype` は `float16` または `float32` でなければならない（§8）。 |
| `drift serve` が 0 個のノードしか見つけない | 先に `drift node` を起動し、`--nodes host:port,...` を渡すか、`--discover-timeout` を増やす。 |
| OpenAI client が `401` を受け取る | server が `--api-key` / `DRIFT_API_KEY` 付きで起動している。`Authorization: Bearer <key>` または `x-api-key: <key>` を送る。 |
| OpenAI client が `model ... is not served` を受け取る | `GET /v1/models` が返す id を使うか、`--served-model-name` を設定してその名前を正確に送る。 |
| `--thin` mode で `/v1/embeddings` や sampling が失敗する | thin mode では logits/hidden states が head にない。embeddings と sampling には non-thin mode を使う。 |
| Mac で稀な MPS op エラー | プロセスを起動したシェルで `export PYTORCH_ENABLE_MPS_FALLBACK=1` が設定されていることを確認する。 |

---

公開された数字は `python -m drift.bench` で再現できます。方法論は
[`benchmarks.md`](benchmarks.md) にあります。
