# DRIFT — 運用マニュアル

**DRIFT を実運用するときに制御できるすべて。** 言語:
[English](manual.md) · [한국어](manual.ko.md) · [中文](manual.zh.md) · **日本語**

ベンチマークの方法論と実測値については
[`benchmarks.md`](benchmarks.md) を参照してください。本書はシステムを *運用する* ことについて述べます。

---

## 目次

1. [インストール](#1--インストール)
2. [60 秒で実行](#2--60-秒で実行)
3. [`config.yaml` リファレンス](#3--configyaml-リファレンス)
4. [分割点の選び方](#4--分割点の選び方)
5. [2 台のマシンにまたがって実行する（Mac + Windows）](#5--2-台のマシンにまたがって実行するmac--windows)
6. [CLI リファレンス](#6--cli-リファレンス)
7. [モデル](#7--モデル)
8. [デバイスと dtype](#8--デバイスと-dtype)
9. [生成の仕組み](#9--生成の仕組み)
10. [ワイヤとセッション](#10--ワイヤとセッション)
11. [メモリ](#11--メモリ)
12. [トラブルシューティング](#12--トラブルシューティング)

---

## 1 · インストール

**Python 3.12**（PyTorch にはまだ 3.14 の wheel がありません）と
[`uv`](https://github.com/astral-sh/uv) が必要です。バンドルされた 2 つのモデルはどちらも **ゲートなし** です — Hugging
Face のログインは不要です。

```bash
git clone https://github.com/TaewoooPark/DRIFT && cd DRIFT
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install "torch" "transformers>=5.5" safetensors msgpack numpy huggingface_hub accelerate pyyaml
export PYTORCH_ENABLE_MPS_FALLBACK=1        # lets rare unimplemented MPS ops fall back to CPU
```

Windows/NVIDIA では、デフォルトの wheel の代わりに、お使いのツールキット向けの CUDA ビルドの PyTorch を
インストールしてください。それ以外はすべて同一です。`PYTORCH_ENABLE_MPS_FALLBACK` は Mac 専用で、
他の環境では無害です。

---

## 2 · 60 秒で実行

一度だけインストールします — `bash scripts/install.sh`（macOS/Linux）または `powershell -File scripts\install.ps1`
（Windows）— そのあと `drift doctor` で健全性を確認します。実行するには:

**1 台のマシンで:**

```bash
drift up 2      # spawn 2 local nodes, auto-split the model, and chat
                # add --prompt "…" for a one-shot answer
```

**複数のマシンにまたがって:**

```bash
drift node      # on each worker — auto device, announced on the LAN
drift run       # on the head — auto-discovers the workers, splits, streams
```

**レイヤー範囲も、IP も、デバイスフラグもありません。** `drift run` はモデルのレイヤー数を読み取り、
見つけたノード群にまたがって分割し、各ノードは自分のスライスだけを計算します。以下はすべて、
その実行の挙動を変える方法についてです — そして §5–§6 では、シャードを手動で操作する方法も扱います。

---

## 3 · `config.yaml` リファレンス

`config.yaml` が唯一の信頼できる情報源です。オーケストレーター、シャードサーバー、リファレンス
オラクル、ベンチマークは、すべてこれを読みます。

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

| キー | 意味 |
|---|---|
| `model_id` | Hugging Face のモデル id。ローカルの HF キャッシュへ一度だけダウンロードされます。 |
| `dtype` | 計算 **かつ** ワイヤの dtype。`float16`（デフォルト、CPU ラウンドトリップで無損失）または `float32`。`bfloat16` はワイヤ上では **無効** です — §8 を参照。 |
| `device` | シャードが自分のデバイスを省略したときのデフォルトデバイス。`mps` / `cuda` / `cpu`。 |
| `port` | `port` を省略し、`DRIFT_PORT` も設定していないシャード向けのフォールバックポート。 |
| `shards[]` | シャードの順序付きリスト。オーケストレーターは **この順序で** ルーティングします。 |
| `shards[].name` | 論理名。`--ports`/ルーティングで使われ、`--ping` で表示されます。 |
| `shards[].host` | オーケストレーターがこのシャードに接続する先。ローカルなら `127.0.0.1`、リモートなら LAN IP（§5）。 |
| `shards[].port` | シャードが待ち受け、オーケストレーターが接続する TCP ポート。 |
| `shards[].start_layer` / `end_layer` | このシャードが保持する半開区間のデコーダーレイヤー範囲 `[start, end)`。 |
| `shards[].device` | このシャードのデバイス（Mac なら `mps`、PC なら `cuda`…）。 |
| `generation.max_new_tokens` | `reference` とオーケストレーターのデモ向けのデフォルトトークン数。 |
| `generation.prompt` | `--prompt` が省略されたときの、`reference` とオーケストレーター向けのデフォルトプロンプト。 |

---

## 4 · 分割点の選び方

`shards[]` の範囲は、モデルのデコーダーレイヤーを **タイル状に** 敷き詰めなければなりません。すなわち連続し、順序どおりで、
隙間なく、重なりなく、`[0, num_hidden_layers)` を覆うことです。オーケストレーター自身が
`embed_tokens`、最終ノルム、`lm_head` を保持します — これらはどのシャード範囲にも **含まれません**。

```
model: 28 decoder layers (Qwen2.5-1.5B)
        └── shard A: [0, 14)   ── shard B: [14, 28)      ✅ tiles 0..28
        └── [0, 10) / [10, 20) / [20, 28)                ✅ three shards, also valid
        └── [0, 14) / [16, 28)                           ❌ gap at 14–15
        └── [0, 16) / [14, 28)                           ❌ overlap at 14–15
```

- **シャードの数** は単に `shards[]` の長さです — 2 がデモですが、それより多くても構いません。
  オーケストレーターはリストの順序どおりにすべてを経由してルーティングします。
- **どこで切るか** は正しさとは無関係です（どのタイリングも 1 台のデバイス上ではビット単位で厳密です）。
  変わるのは *各ノードにどれだけの計算量と重みメモリ* が乗るかだけです。レイヤーを均等に分割するのが
  デフォルトです。片方のマシンが速いなら、そちらに寄せてください。
- **レイヤー数:** Qwen2.5-1.5B = 28、Gemma-4-E2B = 35。どの実行の起動ログからでも読めます
  （`reference` は `layers=…` を表示します）し、モデルの config からも読めます。

---

## 5 · 2 台のマシンにまたがって実行する（Mac + Windows）

**簡単な方法。** 各ワーカーで `drift node` を実行し（デバイスを自動検出し、自分自身をアナウンスします）、
ヘッドで `drift run` を実行します（LAN 越しにワーカーを自動検出し、モデルを分割し、ストリーミングします）。
IP も範囲も不要です。LAN が mDNS をブロックする場合は、ワーカーを明示的に列挙します:
`drift run --nodes 192.168.0.22:PORT,192.168.0.11:PORT`。本節の残りは **手動** の方法です —
正確なポート/範囲を固定したい場合や、完全な手動制御が欲しい場合に便利です。


§2 の localhost 実行は、3 つの変更で本物のクラスタになります。

**1) config を各マシンに向ける。** オーケストレーターのノード上で、各シャードの `host` を
そのマシンの LAN IP に設定し、開いているポートを選びます。

```yaml
shards:
  - { name: "mac",     host: "192.168.0.11", port: 52600, start_layer: 0,  end_layer: 14, device: "mps"  }
  - { name: "windows", host: "192.168.0.22", port: 52601, start_layer: 14, end_layer: 28, device: "cuda" }
```

**2) 各シャードサーバーを到達可能なアドレスにバインドする。** サーバーはデフォルトで `127.0.0.1`
（ローカルのみ）です。リモート接続を受け付けるには、`--host 0.0.0.0` を付けて起動します。

```bash
# on the Mac (192.168.0.11)
DRIFT_PORT=52600 python -m drift.shard_server --name mac     --start 0  --end 14 --device mps  --host 0.0.0.0 --preload
# on the Windows PC (192.168.0.22)
set DRIFT_PORT=52601
python -m drift.shard_server --name windows --start 14 --end 28 --device cuda --host 0.0.0.0 --preload
```

**3) ヘッドを保持するノードからオーケストレーターを実行する。** これは `host`/`port` を
`config.yaml` から直接読むので、**`--ports` は省略します**（`--ports` はポートのみを上書きし、ホストは上書きしません）。

```bash
python -m drift.orchestrator --ping                                  # both shards should reply
python -m drift.orchestrator --prompt "Write a haiku about winter."  # front half on Apple, back half on NVIDIA
```

注意:
- 各マシンのファイアウォールで、選んだポートを開けてください。
- オーケストレーターのノードもモデル（embed/norm/head のため）をロードするので、都合のよいマシンで
  実行してください — 一般的にはシャード A と同じマシンです。
- **異なる GPU ベンダー間**（MPS ↔ CUDA）では、fp16 の丸めが各ベンダーでわずかに異なるため、
  greedy 出力は *後半の* トークンで分岐する可能性があります。序盤のトークンは一致し、テキストは
  一貫したままです。これは想定内です。**同一の** デバイスファミリー上では、分割はビット単位で厳密です（
  [`benchmarks.md`](benchmarks.md) を参照）。

---

## 6 · CLI リファレンス

どのエントリポイントも `--config`（デフォルト `config.yaml`）を取ります。

### `drift` — 高レベルコマンド

| コマンド | 何をするか |
|---|---|
| `drift doctor` | プリフライト: Python/torch/デバイス、依存関係、`config.yaml` のタイリング、ポート到達性、ファイアウォールのヒント |
| `drift up N` | localhost: N 個のノードを立ち上げ、自動分割し、チャット（または `--prompt` でワンショット） |
| `drift node` | このマシンをワーカーとして実行: デバイス自動、LAN でアナウンス、ヘッドを待機 |
| `drift run` | ヘッド: ノードを検出（または `--nodes host:port,…`）、自動分割、構成、ストリーミング/チャット |

`drift up`、`node`、`run` は `--max-new-tokens` を取ります。`run` はさらに `--model` と
`--nodes` を取ります。これらは以下の低レベルモジュールをラップしています — パリティゲートとベンチマークには
モジュールを直接使ってください。


### `drift.shard_server` — 1 つのシャードを実行する

```bash
DRIFT_PORT=<port> python -m drift.shard_server [flags]
```

| フラグ | デフォルト | 意味 |
|---|---|---|
| `--name` | `shard` | 論理シャード名。 |
| `--start` / `--end` | config の shard[0] から | デコーダーレイヤー範囲 `[start, end)`。 |
| `--device` | config の `device` | `mps` / `cuda` / `cpu`。 |
| `--host` | `127.0.0.1` | バインドアドレス。リモートノードを受け付けるには `0.0.0.0` を使用。 |
| `--port` | `$DRIFT_PORT` または config の `port` | 待ち受けポート。 |
| `--preload` | オフ | 待ち受けの **前に** 重みをロード（推奨。最初のリクエストのコールドスタートを回避します）。 |

### `drift.orchestrator` — ヘルスチェックと生成

```bash
python -m drift.orchestrator [--ping] [--prompt "…"] [--max-new-tokens N] [--ports P1,P2]
```

| フラグ | 意味 |
|---|---|
| `--ping` | すべてのシャードに TCP 越しに ping を打って終了します（これが「M0」の到達性チェックです）。 |
| `--prompt` | 生成の起点となるプロンプト。`generation.prompt` にフォールバックします。 |
| `--max-new-tokens` | トークン予算。`generation.max_new_tokens` にフォールバックします。 |
| `--ports` | 各シャードの config ポートを上書きするカンマ区切りのポート（ホストは変わりません — ローカル用途）。 |

ここでの生成は greedy であり、**EOS で停止** します。

### `drift.reference` — 単一マシンのオラクル

```bash
python -m drift.reference [--device DEV] [--out reference_out.npz]
```

モデル全体を 1 台のデバイスにロードし、`generation.prompt` から `generation.max_new_tokens` を
greedy に生成し、トークン id + 最初のステップのロジットを保存します。これが、分割経路を照合する
グラウンドトゥルースです。

### `drift.parity_test` — 正しさのゲート

```bash
python -m drift.parity_test --mode inprocess               # split in one process, no sockets
python -m drift.parity_test --mode socket --ports 52600,52601   # split over TCP (servers must be up)
python -m drift.parity_test --selftest                     # 6 prompts (EN/code/Korean, n=1…180)
```

| フラグ | 意味 |
|---|---|
| `--mode` | `inprocess`（直接呼び出し）または `socket`（ワイヤ越し）。 |
| `--ports` | socket モード用のポート。 |
| `--ref` | 比較対象のリファレンスファイル（デフォルト `reference_out.npz`）。 |
| `--selftest` | 新鮮なリファレンスを改めて導出し、複数のプロンプト/長さにわたって比較します。npz は不要です。 |

### `drift.bench` — ベンチマーク

```bash
python -m drift.bench [--quick] [--no-socket] [--json out.json]
```

[`benchmarks.md`](benchmarks.md) を参照してください。`--no-socket` は、低 RAM のマシンにおいて
サーバー起動のオーバーヘッド計測をスキップします。

---

## 7 · モデル

エンジンは、アーキテクチャをハードコードする代わりに、ロードされたモデルを **イントロスペクト（内省）** します
（デコーダーレイヤーのクラス、`rotary_emb`、キャッシュの型、レイヤーごとのアテンション）。そのため、
新しいファミリーは id を指定するだけで組み込めます。

| モデル | レイヤー | 分割例 | 備考 |
|---|---:|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct`（デフォルト） | 28 | `0–14 / 14–28` | プレーンなデコーダー、パリティのベースライン |
| `google/gemma-4-E2B-it` | 35 | `0–18 / 18–35` | Per-Layer Embeddings（シャードが `input_ids` から再構築）、二重 RoPE θ、ハイブリッドアテンション、`HybridCache`。`transformers ≥ 5.5` が必要 |

モデルを切り替えるには、`config.yaml` に `model_id` と有効なタイリングを設定するだけです — 他には何もありません。
より大きなモデルは、単に分割すべきレイヤーが増えるだけです。範囲はそのレイヤー数にわたって連続させてください。

---

## 8 · デバイスと dtype

**デバイス** — `mps`（Apple GPU）、`cuda`（NVIDIA GPU）、`cpu`（ポータブル、低速）。あるシャードの
`device` は他のシャードから独立しています。その独立性こそがすべての主眼です。

**dtype** — `float16`（デフォルト）または `float32`。ワイヤはテンソルをこの dtype の生バイトとして
シリアライズし、fp16 の CPU ラウンドトリップはビット単位で無損失なので、シリアライズが結果を乱すことは
決してありません。`bfloat16` はワイヤ上で **サポートされていません** — bf16 での計算が必要な場合、それはまだ
配線されていません。`float16` を使ってください。

**同一デバイス vs ベンダー混在:** 同じデバイスファミリー上の 2 つのシャードは、1 台のマシンを **ビット単位で**
再現します。`mps` と `cuda` を混在させると、ベンダー間で fp16 の丸めにビットレベルの差が生じるため、
greedy デコーディングは後半のトークンで分岐する可能性があります（想定内であり、バグではありません — §5）。

---

## 9 · 生成の仕組み

- **greedy のみ。** リファレンスオラクルもオーケストレーターも、各ステップで `argmax` を選びます。
  CLI に露出した temperature/top-p サンプリングはありません。パリティテストは greedy を強制するので、
  出力は決定論的で比較可能です。
- **EOS。** オーケストレーターの `--prompt` 経路は、モデルの end-of-sequence id で停止します
  （狭い集合 — あらゆる特殊トークンではなく、真の EOS のみ）。パリティ/リファレンス経路は、厳密な比較のため、
  早期停止なしで固定の `max_new_tokens` を実行します。
- **prefill してから decode。** プロンプト全体が一度処理され（prefill、位置 `0…S-1`）、その後
  一度に 1 トークンずつ処理されます（decode、`S, S+1, …`）。各シャードは、ステップをまたいで
  自分自身の KV キャッシュを保持します。

---

## 10 · ワイヤとセッション

- **契約（`drift/protocol.py`、凍結済み）:** すべてのメッセージは、4 バイトのビッグエンディアンの長さ
  プレフィックス + msgpack の辞書です。このフレーミングを実装するランタイムなら何でもノードになれます — ワイヤ上に
  PyTorch は存在しません。
- **何が越えるか:** `hidden_states`（fp16）+ `position_ids` + `input_ids` だけです。**KV キャッシュが
  越えることは決してありません** — 各シャードが自分自身のものを保持します。トークンあたりのトラフィックは
  `hidden_size × 2` バイトに数個の整数を加えたもの（数 KB）で、パラメータ数とは無関係です。
- **セッション。** 1 つの生成は 1 つの `session_id`（デフォルト `s0`）です。各シャードはセッションごとの
  KV キャッシュを保持し、オーケストレーターは生成が終わると `reset` を送ります。シャードサーバーは
  **一度に 1 つの接続** を扱います（逐次的。並行処理は将来の課題です） — したがって、2 つの
  オーケストレーターを同時に 1 つのシャードに向けないでください。
- **TCP チューニング。** 接続は `TCP_NODELAY` を設定し、サーバーは `SO_REUSEADDR` を設定します。

---

## 11 · メモリ

今日のところは **すべてのノードで、モデル全体が RAM/VRAM に乗る** ことを前提に計画してください。各シャード
サーバーは現在チェックポイント全体をロードし、その後自分のレイヤースライスだけを使います。オーケストレーターも
モデルをロードします（embed/norm/head のため）。ノードあたりの *アクティブな* パラメータはより小さくなります —
デフォルトの 2 分割では、最も重いノード自身のレイヤーはモデルの約 42% です（
[`benchmarks.md`](benchmarks.md) を参照） — が、ディスクからのロードをスライスだけに削減するのは将来の
課題です。それまでは:

- 各ノードのメモリに収まるモデルを使うか、**より多くの** ノードに分割して各ノードの *アクティブな* 取り分を
  縮めてください。
- メモリのきつい Mac では、`python -m drift.bench --no-socket` が追加のフルモデルサーバープロセスの起動を
  スキップします。

---

## 12 · トラブルシューティング

| 症状 | 考えられる原因 → 対処 |
|---|---|
| オーケストレーターから `ConnectionRefusedError` | シャードがまだ起動していないか、`host`/`port` が間違っている。まずサーバーを起動し、`listening on …` が表示されたことを確認し、ポートが一致しているか確認する。 |
| localhost では動くが、マシン間では動かない | サーバーが `127.0.0.1` にバインドされている。`--host 0.0.0.0` で再起動し、ファイアウォールのポートを開ける。 |
| 1 つのシャードで `--ping` が失敗する | そのシャードのプロセスが死んだか、ポート/ホストが間違っている。その `--start/--end/--device` と、モデルがロードされたことを再確認する。 |
| **トークン 1〜2 でパリティ FAIL** | 本物のバグ（マスク/KV/RoPE）であって浮動小数点ノイズではない — 分割ロジックが分岐している。 |
| greedy 出力が **後半の** トークンでだけずれる、MPS↔CUDA | 想定内のベンダー fp16 丸め（§5）。バグではない。 |
| ロード時にメモリ不足 | 各プロセスがチェックポイント全体をロードする（§11）。より小さなモデル、より多くのシャード、またはベンチでは `--no-socket` を使う。 |
| `unsupported wire dtype` | `dtype` は `float16` または `float32` でなければならない（§8）。 |
| Mac で稀な MPS op エラー | プロセスを起動したシェルで `export PYTORCH_ENABLE_MPS_FALLBACK=1` が設定されていることを確認する。 |
| ヘルスチェックの後、シャードがハングしたように見える | 一度に 1 接続のサーバーへの、迷い込んだ 2 つ目の接続。単一のオーケストレーターを使い、その接続を再利用する（§10）。 |

---

公開された数字は `python -m drift.bench` で再現できます。方法論は
[`benchmarks.md`](benchmarks.md) にあります。
