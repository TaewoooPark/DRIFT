<h1 align="center">DRIFT</h1>

<p align="center"><b>Decentralized Routed Inference For Tokens — 1 つのモデルを、あなたの複数のマシンにまたがって分割。データセンター不要。</b></p>

<p align="center">
  <a href="./README.md">English</a> ·
  <a href="./README.ko.md">한국어</a> ·
  <a href="./README.zh.md">中文</a> ·
  <b>日本語</b>
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
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Transformers%205.12-000000?style=flat-square&labelColor=000000" alt="Transformers">
  &nbsp;
  <img src="https://img.shields.io/badge/Pipeline--parallel-000000?style=flat-square&labelColor=000000" alt="Pipeline parallel">
  <img src="https://img.shields.io/badge/Framework--neutral%20wire-000000?style=flat-square&labelColor=000000" alt="Framework neutral">
  <img src="https://img.shields.io/badge/No%20torch.distributed-000000?style=flat-square&labelColor=000000" alt="No torch.distributed">
  <img src="https://img.shields.io/badge/Bitwise%20parity-000000?style=flat-square&labelColor=000000" alt="Bitwise parity">
  <img src="https://img.shields.io/badge/MPS%20%E2%86%94%20CUDA-000000?style=flat-square&labelColor=000000" alt="MPS to CUDA">
  <img src="https://img.shields.io/badge/Up%20to%2035%20nodes-000000?style=flat-square&labelColor=000000" alt="Up to 35 nodes">
</p>

<p align="center">
  <img src="docs/img/hero.png" alt="1 つのモデルを地球の裏側にまたがって分割 — ニューヨークとソウルが 1 つのモデルを一緒に、データセンター不要" width="900">
</p>

<p align="center"><sub>ニューヨークの友人が眠っている間にノードを立てておき、あなたはソウルにいる。DRIFT は<b>1 つ</b>のモデルを 2 台に分割する。友人の GPU が前半の層を、あなたの GPU が後半を計算し、hidden state だけが大洋を渡る。こうして、どちらの 1 台にも収まらないモデルを一緒に動かし、その答えは単一マシンとビット単位で同じだ。</sub></p>

**DRIFT** は、**1 つ** の大規模言語モデルを **異種混在のパーソナルマシン** — Mac（Apple GPU、PyTorch **MPS**）と Windows PC（NVIDIA GPU、PyTorch **CUDA**）— にまたがって実行します。モデルを **レイヤー単位** で分割し（パイプライン並列）、ノード間では **hidden state** だけを **フレームワーク中立なバイトプロトコル**（TCP + msgpack）でストリーミングします。データセンターも、`torch.distributed` も、NCCL も、ベンダーロックもありません。データプレーンは *どの* フレームワークにも束縛されないため、本来なら決して会話できないはずのランタイム — Apple Metal のグラフと NVIDIA CUDA のグラフ — が、いまや 1 つのモデルを一緒に動かし、その出力は 1 台のマシンでモデル全体を実行した場合と **ビット単位で完全に一致** します。

**差別化点を一言で:** [Exo](https://github.com/exo-explore/exo) はノード間通信を MLX（`mx.distributed`）に束縛しているため、*Apple シリコン間でしか動きません*（ロードマップ上、Windows は「Longer term」）。DRIFT はその境界を **中立なワイヤプロトコル** へと引き上げ — *異なるランタイム、異なる GPU ベンダー、1 つのモデル* — 分割が厳密であることを **ビット単位のパリティゲート** で証明します。どのフレームワークにも束縛されないデータプレーンこそが中核的な貢献です。

**スケール。** デコーダーレイヤー 1 つにつきノード 1 つ — デフォルトの Qwen で最大 **28 台**（Gemma は **35 台**）にまたがって 1 つのモデルをストリーミングします。現在のモデル特性では **2〜4 台** がスイートスポットです。

> *「トランスクリプトはモデルの出力そのものです。興味深いのは、その計算が実際に **どこで** 走ったのか — そしてそれがビット単位で辻褄が合ったということです。」*

[**taewoopark.com** — 著者サイト](https://taewoopark.com)

---

## 目次

- [何が違うのか](#何が違うのか) — エンジニアが見に来た比較表
- [DRIFT とは](#drift-とは) — 名前、ビジョン、スコープ
- [アーキテクチャ](#アーキテクチャ) — 制御 / データ / KV プレーン
- [ワイヤ契約](#ワイヤ契約境界を実際に越えるもの) — スキーマ + トークンあたりのバイト数
- [3 つの正しさの問題](#正しい分割が解くべき-3-つの問題) — KV 再インデックス、RoPE、マスク
- [デコードループ](#デコードループと差し替え可能なトランスポート) — シーケンス + 差し替え可能なトランスポート
- [正しさとパリティ](#正しさ--パリティゲート) — ビット単位のゲート + 実測結果
- [ベンチマーク](#ベンチマーク) — fidelity 100% · ノードあたりモデルの ≤ 42% · プロトコルオーバーヘッド ≈ 0
- [イントロスペクションによるモデル非依存](#イントロスペクションによるモデル非依存) — Qwen、Gemma 4、そしてハードコーディングなし
- [設計上の根拠（why-not）](#設計上の根拠why-not) — その判断とその理由
- [マイルストーン](#マイルストーン) · [クイックスタート](#クイックスタート) · [リポジトリマップ](#リポジトリマップ--どこを見るか) · [FAQ](#faq) · [ロードマップ](#ロードマップ)

---

## 何が違うのか

DRIFT の核心はすべて **ノード間の境界** にあります。その境界を従来技術と比較すると、次のようになります。

| | **DRIFT** | Exo | Petals | llama.cpp RPC | vLLM / Megatron PP |
|---|---|---|---|---|---|
| **分割単位** | デコーダーレイヤー | レイヤー | Transformer ブロック | レイヤー / テンソル | レイヤー（ステージ） |
| **ノード↔ノード間トランスポート** | **TCP + msgpack** | MLX `mx.distributed` | gRPC（torch テンソル） | カスタム RPC（ggml） | `torch.distributed` + NCCL |
| **境界ペイロード** | **生の fp16 バイト + 整数** | MLX 配列 | torch オブジェクト | ggml テンソル | torch テンソル / NCCL バッファ |
| **フレームワーク中立なワイヤ** | **✅ はい** | ❌ MLX 依存 | ❌ torch 依存 | ggml 依存 | ❌ torch/NCCL 依存 |
| **異種 GPU ベンダー** | **✅ MPS + CUDA を同時に** | ❌ Apple のみ | 部分的 | ✅（ggml バックエンド） | ❌ NCCL は橋渡し不可 |
| **Mac + Windows を併用** | **✅** | ❌（「Longer term」） | ~ | ✅ | ❌ |
| **インターフェース越しにエンジン差し替え可能** | **✅ `ShardEngine` ABC** | ❌ | ❌ | n/a | ❌ |
| **KV キャッシュの配置** | シャードごと、ローカル | シャードごと | ブロックごと | ノードごと | ステージごと |
| **トークンあたりに越えるもの** | **約 3 KB（hidden のみ）** | activation | activation | activation | activation |
| **正しさの契約** | **1 台のマシンに対するビット単位パリティ** | — | — | — | — |

この表を上から下まで読めば、主張はおのずと浮かび上がります。**どの実装も activation を受け渡している。だが、その受け渡しをフレームワーク中立にし、*しかも* 結果がビット単位で厳密であることを証明したのは DRIFT だけだ。** NCCL は Apple GPU と NVIDIA GPU を同じプロセスグループに入れられません。MLX は Apple のエコシステムから出られません。DRIFT の答えは、ワイヤに *バイト以外の何も* 運ばせないこと — torch オブジェクトも、MLX 配列も、CUDA ハンドルもなし — です。そうすることで、2 つの世界は互いに実装可能な 1 つの契約の上で出会えるのです。

---

## DRIFT とは

サーバーレスな P2P 推論ネットワーク。異種混在のパーソナルデバイスが **1 つ** のモデルをレイヤー単位で分割し、**一緒に** 実行します。ハイパースケーラーのデータセンターを経由する代わりに、*あなたのマシンと誰か他の人のマシン* が寄り集まって単一の AI を動かします。

名前がそのままシステムを表しています。

| 文字 | 意味 |
|---|---|
| **D** — Decentralized（非中央集権） | 単一のコントローラーも、単一障害点もない。異種混在のデバイスが対等な P2P ノードとして参加する |
| **R** — Routed（経路制御） | オーケストレーターが hidden state をノード群へと *ルーティング* し、推論を前へ進める |
| **I** — Inference（推論） | ワークロードは LLM 推論（学習へも拡張可能） |
| **For T** — For Tokens（トークンのために） | 「トークン」の二重の意味 ― **推論** トークン（機械的思考の最小単位）**と**、**価値** トークン（貢献によって得られ、推論に費やされる）。思考の単位と価値の単位を 1 つにすること、それが DRIFT のビジョンである |

> **本リポジトリのスコープ。** 技術的中核 — *Mac と Windows マシンにまたがって分割されたモデルは正しい答えを出すのか？* — は完成し、ビット単位で検証済みです。その上で、**「For Tokens」** の土台には今や動く v1 の芽が育っています。トラストレスな抜き取り検証、どこからでも参加、耐障害性、そして 1 台に収まらないモデルの実行です（下記「分割の先へ」を参照）。完全なトークンエコノミーと合意形成は依然としてビジョンです。

---

## アーキテクチャ

<p align="center"><img src="docs/img/arch.png" alt="DRIFT architecture — orchestrator head, per-layer shards, neutral wire" width="900"></p>

DRIFT は 3 つのプレーンにきれいに分離されます。

- **制御プレーン** — オーケストレーターは、設定で定めた固定の順序でシャードを呼び出します。ディスカバリサービスもリーダー選出もなく、アドレスのリストは `config.yaml` にあります。（ディスカバリは「For Tokens」側の関心事であり、スコープ外です。）
- **データプレーン** — ステージの境界を越えるのは `hidden_states`（浮動小数点）と `position_ids` + `input_ids`（整数）だけです。フレームワーク非依存であり、そして — ここが肝心ですが — **そのサイズはパラメータ数ではなく `hidden_size` に依存します。** `hidden_size` が一致すれば、1.5 B のモデルも 70 B のモデルも、同じ約 3 KB/token を送出します。
- **KV キャッシュプレーン** — 各シャードは *自分自身* のレイヤー範囲の KV を、セッションごとに、自分のデバイス上で保持します。**キャッシュがワイヤを越えることは決してありません**（そうなればトークンあたり数メガバイトになり、設計全体が台無しになります）。移動するのは残差ストリームだけです。

**分割は 2 台を超えて拡張します** — デコーダーレイヤー 1 つにつきノード 1 つ、最大 28 台（Gemma は 35 台）まで、ヘッドとワイヤは不変のままです。

<p align="center"><img src="docs/img/scale.png" alt="DRIFT scales one model across 2 to 28 nodes, one decoder layer per node" width="900"></p>

---

## ワイヤ契約（境界を実際に越えるもの）

この契約（`drift/protocol.py`）は **凍結** されています。すべてのメッセージは **4 バイトのビッグエンディアン長さプレフィックス + msgpack の辞書** です。将来のどんなランタイム — MLX、ggml、JAX、Rust ノード — も、このフレーミングを実装しさえすればパイプラインに参加できます。ワイヤ上に PyTorch は存在しません。

```jsonc
// リクエスト（オーケストレーター → シャード）
{
  "type":         "prefill" | "decode" | "reset" | "ping",
  "session_id":   "s0",               // 1 つの生成シーケンス
  "seq_id":       42,                 // 単調増加、順序付け / デバッグ用
  "shape":        [1, 1, 1536],       // hidden_states の形状（decode: S=1）
  "dtype":        "float16",
  "position_ids": [37],               // 絶対位置  → RoPE、シャード上で計算
  "input_ids":    [785],              // トークン id → レイヤーごとの埋め込み（PLE, Gemma 4）
  "tensor":       "<raw fp16 bytes>"  // 行優先の hidden_states
}

// レスポンス（シャード → オーケストレーター）
{ "ok": true, "shape": [1,1,1536], "dtype": "float16", "tensor": "<bytes>", "error": null }

// ping レスポンス  →  { "ok": true, "name", "start_layer", "end_layer", "device" }
```

**トークンあたりのバイト数。** デコード中、activation は fp16 の `[1, 1, hidden]` = `hidden × 2` バイトです。Qwen の `hidden = 1536` なら **3 072 バイト ≈ 3 KB** となり、これに `position_id` が 1 つ、`input_id` が 1 つ、そして数バイトの msgpack フレーミングが加わります。2 シャードのパイプラインでは、トークンあたり約 4 回の越境が発生し（オーケストレーター→A、A→オーケストレーター、オーケストレーター→B、B→オーケストレーター）、**トークンあたり約 12 KB のワイヤトラフィック** になります — LAN では、計算量に比べれば取るに足らない量です。

**なぜこの 3 つのフィールドだけなのか:**

- `hidden_states` — 残差ストリーム。下流のレイヤーが本当に必要とする唯一のもの。
- `position_ids` — 各シャードが絶対位置から **自分自身の** RoPE を計算するため（後述）。事前計算した `cos/sin` ではなく位置そのものを送ることで、ペイロードは極小に、ノードは自己完結的に保たれます。
- `input_ids` — **M0** の時点で予約済み。これにより、契約を再凍結することなく **Per-Layer-Embedding** モデル（Gemma 4）が動作します。下流のシャードが、レイヤーごとの埋め込み信号をトークン id からローカルに再構築するのです。プレーンなモデル（Qwen）は単にこれを無視します。

**なぜワイヤ上の fp16 が安全なのか。** シリアライズは CPU 上での fp16 ラウンドトリップです。送出時は `tensor.detach().to("cpu", float16).contiguous().numpy().tobytes()`、受信時は `np.frombuffer(buf, np.float16).reshape(shape).copy()`。計算 dtype がすでに fp16 であれば、このラウンドトリップは **ビット単位で無損失** です — これこそが、分割経路が 1 台のマシンを近似的にではなく *厳密に* 再現できる前提なのです。

---

## 正しい分割が解くべき 3 つの問題

レイヤーをプロセス間で分割するのは些細なことに聞こえます — 出力を未分割のモデルと *完全に同一* にしようとするまでは。厄介な点が 3 つあり、DRIFT はそのそれぞれを明示的に処理します。ここにこそ本当のエンジニアリングがあり — レビュアーが精査すべき箇所です。

### 1 · KV キャッシュのインデックス付け — 見落としやすい問題

Hugging Face の `DynamicCache` はレイヤーの `layer_idx` でインデックスされ、「過去の長さ（past length）」を **レイヤー 0 の** スロットから報告します。グローバルなレイヤー `[14, 28)` を保持するシャードが、そのグローバルインデックスを素朴にそのまま再利用すると、キャッシュのスロット 0 が **空** のままになります — その結果、デコード中に因果マスクが *過去が存在しない* かのように構築され、ごく最初のトークンの後でパリティが静かに壊れます。

<p align="center"><img src="docs/img/kv-reindex.png" alt="KV-cache local re-indexing — the fix that keeps decode parity" width="900"></p>

DRIFT は、ロード時に各シャードが保持するレイヤーを **ローカルな 0 始まり** のキャッシュスロットへ再インデックスし、セッションごとの `DynamicCache` のサイズをそのシャードのローカルなレイヤー数に合わせます。プロセス内では、2 つのシャードが 1 つのロード済みモデルを共有できます — それぞれが **互いに素な** レイヤーオブジェクトを所有しているため、一方を再インデックスしても他方には決して影響しないからです。

### 2 · RoPE の自己計算 — ワイヤを小さく保つ

回転位置埋め込み（RoPE）は `position_ids` にのみ依存し、どのレイヤーがそれを消費するかには依存しません。そのため各シャードは、モデル自身の `rotary_emb` モジュールを通じて **絶対** 位置から自分の `cos/sin` を計算します — レイヤー `[14, 28)` を保持するシャードでも、正しく求められます。したがって境界を越えるのは、完全な `[S, head_dim]` の `cos/sin` テンソルではなく、ほんの一握りの整数だけであり、各ノードは自己完結的なままです。

### 3 · ステージごとのアテンションマスク

prefill ではマスクは因果的にフル（causal-full）であり、decode では KV 長を考慮したものになります。DRIFT は、インストール済みの Transformers のマスク生成ユーティリティ（`create_causal_mask`、および Gemma のようにレイヤーごとにローカル/グローバルアテンションを交互に切り替えるモデル向けの `create_sliding_window_causal_mask`）を使って、各シャード上でマスクを再構築します。マスクはレイヤー自身のアテンションタイプに基づいて **レイヤーごとに** 選択されます — 何一つハードコードされていません。

---

## デコードループと差し替え可能なトランスポート

<p align="center"><img src="docs/img/decode-loop.png" alt="The decode loop over an injectable transport (in-process / TCP)" width="900"></p>

このループは、単一のシグネチャを持つ **差し替え可能なトランスポート** — `transport(shard, session, hidden, position_ids, input_ids, mode)` — を経由します。デコードループは **一度だけ** 書かれ、差し替えられるのはトランスポートだけです。

| トランスポート | マイルストーン | 境界 | 目的 |
|---|---|---|---|
| **プロセス内呼び出し** | M2 | 直接 `engine.forward(...)`、ソケットなし | 分割ロジックを単独で検証する |
| **ソケットクライアント** | M3+ | TCP 上の §6 プロトコル | シリアライズ / フレーミングを検証する |

ループが同一であるため、**M2 と M3 の間の唯一の変数はネットワークだけ** です — したがって M3 で回帰が起きれば、それは *証明可能な形で* シリアライズのバグであり、ロジックのバグではあり得ません。これはこのコードベースにおいて最も重要な構造上の決定です。

---

## 正しさ — パリティゲート

DRIFT は **正しさ優先（correctness-first）** です。ネットワークを介するすべてのステップは、いかなる性能改善作業よりも前に、1 台のマシンによるリファレンスを **ビット単位で** 再現しなければなりません。速度はこのデモの主眼ではありません — *異種混在の分割推論が厳密であること* こそが主眼です。

<p align="center"><img src="docs/img/parity-gate.png" alt="The parity gate — strict bitwise on one device, relaxed across GPU vendors" width="900"></p>

**実測結果** — Qwen2.5-1.5B-Instruct、Apple MPS、fp16:

| ゲート | 何を切り分けるか | 結果 |
|---|---|---|
| **M0** ping | 中立プロトコルの到達性 | ✅ 両シャードが応答 |
| **M2** プロセス内 2 シャード | シャーディング · RoPE · KV · マスク | ✅ **50 / 50 トークン id がビット単位で == リファレンス** |
| **M3** TCP 2 プロセス | シリアライズ / フレーミング | ✅ **50 / 50 ビット単位で == リファレンス** |
| **`--selftest`**（6 プロンプト） | 1 つのプロンプトへの過適合 | ✅ **6 / 6 ビット単位** — 英語 · コード · 韓国語; `n = 1, 40, 50, 60, 80, 180` |

`--selftest` は最も強力な証拠です。新鮮なリファレンスを改めて導出し、プロンプトの *種類*（散文、ソースコード、韓国語）と *長さ*（単一トークンの生成から 180 トークンのデコードまで）にわたって比較します。すべてのトークン id が一致し — 6 つすべてで最初の分岐インデックスは `None` です。

**MPS ↔ CUDA（M4）— 実測済み。** 前半を Mac（Apple MPS）、後半を Colab の NVIDIA T4（CUDA）で実行したところ、分割経路は単一マシンの参照を**完全に再現しました。3 プロンプトで 130/130 トークン一致、分岐なし**です — 2 つのベンダーの fp16 カーネル差（および torch 2.11 と 2.12 の差）で初手の logit 差は ~2×10⁻²（同一デバイスは ~8×10⁻³）に広がりましたが、argmax を反転させるには至りませんでした。より大きな規模ではこの差が後半のトークンを反転させ得るため、そのための**緩和されたゲート** `python -m drift.parity_test --prefix-match K` があります。公開トンネル経由のマシン間スループットは約 2.7 tok/s（ネットワーク律速）でした。トークン 1〜2 での分岐は浮動小数点ノイズではなく **バグ** です → 二分探索へ。

<p align="center"><img src="docs/img/m4-result.png" alt="M4 measured — Mac Apple MPS + Colab NVIDIA T4 CUDA, 130/130 token match vs one machine" width="900"></p>

---

## ベンチマーク

*方法論・制御変数・公正な競合比較プロトコルは **[docs/benchmarks.md](docs/benchmarks.md)**。すべての数字は `python -m drift.bench` で再現できる。*

`tokens/sec` を先頭に据えるのは誤った軸だ。Apple のみのクラスタでは Exo のネイティブ MLX 経路が生スループットで勝り、DRIFT が独占する軸——Mac(MPS)↔Windows(CUDA)——では競合はそもそも動かない（[表を参照](#何が違うのか)）。だから数字は*正しい*分割が実際に先行する場所で、すべて**1 台**の Mac 上で出す——Qwen2.5-1.5B-Instruct · fp16 · Apple MPS。

**Fidelity — 分割は出力を変えるか？** *（分割経路 vs 単一マシンのオラクル、greedy）*

| 指標 | 結果 |
|---|---|
| トークン完全一致 — プロンプト 6 個、`n = 1…180` | **411 / 411 = 100.00 %** |
| ビット単位で同一のケース | **6 / 6** |
| 初手 logit 最大絶対差 (fp32) | 7.81 × 10⁻³ *(fp16 ULP)* |
| KL ダイバージェンス (nats) | ≤ 2.82 × 10⁻¹⁰ |

トークン id は単一マシンと**ビット単位で同一**。logit は fp16 ULP まで一致し、argmax はそのノイズに対して不変だ。この軸を*測定*する——ましてや*保証*する——ツールはこの分野に他にない。*これが DRIFT の独占軸だ。*

**Footprint — どの単一ノードもモデル全体を保持しない**

| ノード | 保持 | fp16 | 全体比 |
|---|---|---:|---:|
| オーケストレータ | embed + norm + lm_head | 0.47 GB | 15.1 % |
| シャード · mac | デコーダ層 [0, 14) | 1.31 GB | 42.4 % |
| シャード · windows | デコーダ層 [14, 28) | 1.31 GB | 42.4 % |
| **全モデル** | — | **3.09 GB** | 100 % |

最も重い単一ノードでもモデルの **42.4 %** しか担わない——どちらか 1 台には 2 倍大きすぎるモデルが 2 台にまたがって動く。パイプライン分割が存在する理由そのものだ。**これらは各ノードの計算分担ではなく、デバイス上で実測されたメモリ割り当てです。**各ノードは自分のスライスだけを実体化するため（`init_empty_weights` + safetensors の選択的読み込み）、モデル全体がどの 1 台にも常駐せず、スライスロードが単一マシンのロードとビット単位で同一であることは parity ゲートが証明します。

**中立ワイヤは薄く、ほぼ無料だ**

| 指標 | 値 |
|---|---|
| トークンあたり・ホップあたりのワイヤ | **3.10 KB** — fp16 hidden state のみ |
| 重み : トークンあたりワイヤ | **≈ 970,000 ×** |
| TPOT — プロセス内 (M2) → TCP (M3) | 40.7 → 43.1 ms/token |
| プロトコルオーバーヘッド | **+2.45 ms/token**（ホップあたり約 1.2 ms、TPOT の約 6 %） |

**同一のデコードループ**が両トランスポート上で走るので、M3 − M2 の差はフレームワーク中立プロトコルの*純粋な*コスト——MPS と CUDA を協調させるまさにそれ——だ。localhost ではホップあたり約 1.2 ms の小さな往復コストで、~41 ms/token の計算に圧倒される。この規模では測定ノイズが大きく、実行ごとに 0 を挟んで数 ms 揺れる（以前の実行ではわずかに負と測定されたこともある）。（実際の LAN は RTT が上乗せされるが、それは DRIFT とは無関係だ。）

> チェリーピックした勝利ではなく、絶対的で再現可能な数字だ。Exo / llama.cpp RPC との `tok/s` 直接対決には同一マシンへのインストールが要る。その公正なプロトコルは **[docs/benchmarks.md](docs/benchmarks.md)**。今日下せる比較の主張は、上の能力マトリクス**に加えて**単一マシンと*証明可能に*同一な分散出力だ。

---

## イントロスペクションによるモデル非依存

エンジンはモデルアーキテクチャを一切ハードコードしません。ロード時に、ロードされたモデルを **イントロスペクト（内省）** して適応します。

```python
# drift/engine_torch.py — 真実の源はロードされたモデルであって、固定のクラスではない
layer_cls   = type(self.layers[0])                       # Qwen2DecoderLayer / Gemma4DecoderLayer / …
self._layer_params = set(inspect.signature(layer_cls.forward).parameters)
self.rotary       = self.inner.rotary_emb                # 自己計算する RoPE、どのモデルでも
self.has_sliding  = getattr(self.inner, "has_sliding_layers", False)
self.layer_types  = [cfg.layer_types[i] for i in range(start, end)]   # レイヤーごとのアテンションタイプ
# … 呼び出し時には、このバージョンのレイヤーが実際に受け取る kwargs だけを渡す:
call_kwargs = {k: v for k, v in call_kwargs.items() if k in self._layer_params}
```

だからこそ、まったく異なる 2 つのモデルファミリーが *同じ* エンジンにそのまま乗るのです。

| モデル | レイヤー → 分割 | ゲート | DRIFT が処理するアーキテクチャ上の癖（イントロスペクトで判別、決してハードコードしない） |
|---|---|---|---|
| **Qwen/Qwen2.5-1.5B-Instruct** *(主)* | 28 → `0–14 / 14–28` | なし | プレーンなデコーダー、単一の RoPE θ、`DynamicCache`、tied な `lm_head` — 正しさのベースライン |
| **google/gemma-4-E2B-it** *(副)* | 35 → `0–18 / 18–35` | なし（Apache-2.0） | **Per-Layer Embeddings**（シャードが `input_ids` から自己計算）· sqrt(hidden) の埋め込みスケーリング（オーケストレーター）· ローカル/グローバルの **二重 RoPE θ** · レイヤーごとのスライディング/グローバルアテンションの **ハイブリッド** · `HybridCache` + KV 共有グループ · 最終ロジットの softcap なし; `transformers ≥ 5.5` が必要 |

Gemma 4 の癖はそれぞれ、いずれかのプレーンにきれいに対応づけられます — **オーケストレーター**（埋め込みスケーリング）、**シャード**（二重 θ の RoPE、ハイブリッドマスク、ハイブリッドキャッシュ）、あるいは **ワイヤ**（PLE のための `input_ids`）— そしてそのどれもが、ロード時に `config`/シグネチャから発見されます。だからこそ、Qwen を動かすコードが、変更なしで Gemma 4 を動かすのです。これは、中立なワイヤが体現するのと同じ原則がもたらす、モデル非依存の恩恵です。*観測できるものに依存せよ、何もハードコードするな。*

---

## 設計上の根拠（why-not）

興味深い決定とは、DRIFT が採らなかった選択肢のことです。そのそれぞれが、意図的に課された厳格な制約です。

- **なぜノード間で `torch.distributed` / NCCL / gloo を使わないのか？** NCCL は Apple Metal デバイスと NVIDIA CUDA デバイスを 1 つのプロセスグループに置くことができません — 議論の余地なく。しかもこれらはいずれも *データプレーン* を特定のバックエンドに結合してしまい、それこそが DRIFT の拒むものです。ワイヤは中立なバイトなので、ランタイム同士はフレーミング以外に何も合意する必要がありません。
- **なぜノード間で KV キャッシュを送らないのか？** KV はトークンあたり数メガバイトに達し、シーケンス長とともに増大します。これを送れば約 3 KB の残差をはるかに上回り、コスト効率を破壊してしまいます。各シャードは自分の KV をローカルに保持し、移動するのは残差ストリームだけです。
- **なぜワイヤ上は fp16 なのか（fp32 ではなく）？** fp16 計算であれば、CPU 上の fp16 ラウンドトリップはビット単位で無損失なので、シリアライズがパリティを乱すことはあり得ません — しかも fp32 に対してワイヤのバイト数を半減できます。（fp16 計算は速度の出る GPU 上で動きます。CPU の fp16 カーネルは信頼できず、それがパリティのベースラインを CPU ではなく MPS で走らせる理由です。）
- **なぜまず逐次・単一セッションなのか？** 並行処理、バッチ処理、投機的デコーディングは最適化です。このデモの価値は *異種混在の下での正しさ* にあるので、それらはパリティが証明されるまで先送りされます — そして実際に証明済みです。
- **なぜ各ノードにモデル全体を置かないのか？** 各ノードは自分のスライスだけを実体化します：`init_empty_weights` で meta デバイス上に骨組みを作り、そのノードが実際に実行するテンソル（自分のデコーダ層、またはヘッドの `embed`/`norm`/`lm_head`）だけを safetensors から読み、デバイスに載せます。最も重いノードでも重みの **42 %** しか実メモリに持たず、スライスロードが単一マシンのロードとビット単位で同一であることは parity ゲートが証明します。
- **なぜ M0 の時点でワイヤ契約を凍結するのか？** ノードの内部を、いつまでも一斉切り替え（flag day）なしに変更できるようにするためです。`input_ids` フィールドは、まさに PLE モデル（Gemma 4）が破壊的変更を一切強いないよう、凍結の *前に* 追加されました。

---

## マイルストーン

| # | マイルストーン | 必要なもの | ステータス |
|---|---|---|---|
| **M0** | 環境 + 中立プロトコルのフレーミング（ping） | Mac | ✅ 完了 |
| **M1** | 1 台のマシンによるリファレンスオラクル | Mac | ✅ 完了 |
| **M2** | プロセス内 2 シャードのパリティ（ネットワークなし） | Mac | ✅ **ビット単位** |
| **M3** | localhost 2 プロセスのパリティ（TCP） | Mac | ✅ **ビット単位** |
| **M4** | マシン間 — Mac MPS + NVIDIA CUDA | Mac + CUDA（Colab 可） | ✅ **実測** — 100% トークン一致、~2.7 tok/s |
| **M5** | ブース展示 + インタラクティブなストリーミング | + Windows | ⬜ |
| **M6** | ノード切断からの穏当な回復 | 完了 | ✅ 綺麗な `NodeUnavailable` + 再接続 |

Mac のみで完結するトラック（M0–M3）は、エンジニアリングの約 80 %、そして **正しさに関するリスクの 100 %** を占めます — 完了かつレビュー済みです。M4–M6 が加えるのは、2 台目のマシンと展示だけです。

---

## クイックスタート

Python **3.12** と [`uv`](https://github.com/astral-sh/uv) が必要です。デフォルトの 2 つのモデルはどちらも **ゲートなし** です — Hugging Face のログインは不要です。以下はすべて本物の `drift` CLI です。

**1 · インストール** — 各マシンで:

```bash
git clone https://github.com/TaewoooPark/DRIFT && cd DRIFT
bash scripts/install.sh          # macOS / Linux   ·   Windows: powershell -File scripts\install.ps1
drift doctor                     # checks Python, torch, device, config, ports
```

**2 · 1 台のマシンで試す:**

```bash
drift up 2                       # 2 local nodes, auto-split the model, open a chat
                                 # (add --prompt "…" for a one-shot answer)
```

**3 · あなたの Mac + CUDA PC にまたがって 1 つのモデルを実行する** — これが本命です。

**head** はプロンプトを打ち込み、`embed`/`lm_head` を保持します。デコーダーレイヤーは **ノード** 上に置かれます。*両方* の GPU を使うには、Mac がノードを **かつ** head を動かし、PC がノードを動かします:

```bash
# Windows PC (NVIDIA)          — one terminal
drift node --port 52601        # device = cuda, announced on the LAN

# Mac (Apple)                  — terminal 1: a worker
drift node --port 52600        # device = mps

# Mac                          — terminal 2: the head (type the prompt)
drift run --prompt "hello world"
```

```text
  node : 127.0.0.1:52600     layers [0:14)   · device=mps      ← Mac がこれらを計算
  node : 192.168.0.22:52601  layers [14:28)  · device=cuda     ← PC がこれらを計算

  Hello! How can I help you today?
```

2 台の Mac でも 2 台の Windows PC でも、**同じ 3 つのコマンド** で動きます — デバイスは自動検出され、`drift run` が見つけて分割します。Wi-Fi が mDNS をブロックする場合は、ノードを名前で指定します: `drift run --nodes 192.168.0.22:52601,127.0.0.1:52600 --prompt "hello world"`。GPU ベンダーをまたぐ場合（MPS↔CUDA）、fp16 の丸めがわずかに異なるため、長い答えは後半のトークンでずれることがあります — 想定内であり、バグではありません。

**カスタマイズとチューニング** — モデル、分割点、デバイス、シャードを手動で操作する方法、そしてトラブルシューティング — はすべて **運用マニュアル → [docs/manual.ja.md](docs/manual.ja.md)** にあります（[English](docs/manual.md) · [한국어](docs/manual.ko.md) · [中文](docs/manual.zh.md)）。

---

## リポジトリマップ — どこを見るか

```text
drift/
  protocol.py       # 契約そのもの — 4B の長さプレフィックス + msgpack; fp16 テンソルの ser/deser
  engine_base.py    # ShardEngine ABC — ランタイム差し替えの継ぎ目
  engine_torch.py   # PyTorch シャード: イントロスペクトされたレイヤー呼び出し、ローカル KV 再インデックス、self-RoPE  ← 要
  shard_server.py   # TCP サーバー: ping / reset / prefill / decode
  orchestrator.py   # embed + norm + lm_head + sampler; 差し替え可能なトランスポート; デコードループ
  reference.py      # M1 1 台のマシンによるオラクル
  parity_test.py    # M2/M3 ゲート + マルチプロンプト --selftest
  common.py         # config + 同一のトークン化（オラクルと分割経路で共有）
config.yaml         # モデル、dtype、ポート、シャードテーブル
docs/               # 公開ドキュメント — benchmarks.md（方法論 + 結果）· manual.ja.md（実行方法）
```

**レビュアー向けの要点リスト:** `engine_torch.py`（KV 再インデックス + イントロスペクション）、`protocol.py`（凍結されたワイヤ）、`orchestrator.py`（差し替え可能なトランスポート + デコードループ）。

---

## FAQ

**これは単なるパイプライン並列では？** *アイデア* はそうですが、貢献は **境界** にあります。vLLM/Megatron の PP は `torch.distributed`+NCCL に溶接されており、MPS↔CUDA を橋渡しできません。DRIFT の境界は中立なバイトなので、異種のベンダーが参加でき — しかもビット単位で厳密であることが証明されています。

**ネットワークは私のトークンを見るのか？** ワイヤを越えるのは整数の `input_ids` と浮動小数点の `hidden_states` だけです — テキストもなければ、KV もありません。LAN 上であれば、これはあなたのマシン群の中にとどまります。（暗号化や信頼は「For Tokens」側の関心事であり、ここではスコープ外です。）

**3 つ目のノードを追加できるのか？** はい — 分割は `config.yaml` 内のレイヤー範囲のリストです。シャードのエントリを追加すれば、オーケストレーターが順番にそこを経由してルーティングします。ワイヤ契約は変わりません。

**なぜリファレンスは CPU ではなく MPS 上なのか？** 計算 dtype が fp16 であり、PyTorch の CPU fp16 カーネルが信頼できないからです。MPS は fp16 を正しく決定論的に実行するので、M1–M3 はすべて MPS 上で行われ、ビット単位で一致します。CPU/CUDA は設定で切り替え可能です。

**バッチ処理やスループットは？** 設計上、先送りしています（正しさ優先）。分割が厳密であることを証明するには逐次の単一セッションで十分であり、バッチ処理は将来の課題です。

**なぜ特に Qwen と Gemma 4 なのか？** どちらもゲートなし（ライセンスの壁がない）で、アーキテクチャ空間の両極 — プレーンなデコーダーと、Per-Layer Embeddings + ハイブリッドアテンションを備えたもの — をカバーします。これが「イントロスペクトせよ、ハードコードするな」というエンジンのストレステストになります。

---

## 分割の先へ — 分散化の芽

技術的中核（正しく、ビット単位で検証された異種分割）は完了しました。その上で、*分散*ネットワークに必要な部品には今や動く v1 の芽が育っています:

| 能力 | 何をするか | 試す |
|---|---|---|
| **1 台に収まらないモデルの実行** | 各ノードは**自分の層を持つシャードだけ**をダウンロード・ロード — ディスクも VRAM も自分のスライスだけ | 十分なノードに分割 |
| **どこからでも参加** | `drift node --tunnel` が NAT 越し / Colab / VM のノードを公開 `bore.pub:PORT` として公開（アカウント・トークン不要） | `drift node --tunnel` |
| **ノード脱落への耐性** | 実行中の脱落を検出して 1 度再接続、だめならノード名付きの綺麗な `NodeUnavailable` に（生のトレースバックではなく） | デコード中にノードを kill |
| **持っていないノードを信頼** | ヘッドが固定チャレンジでノードを問い、M4 で実測した誠実な包絡から外れた出力を検出（誠実 → ビット一致、改ざん → 検出） | `python -m drift.verify --nodes …` |

完成した経済ではなく芽です。検証は抜き取り（合意/暗号ではない）、トンネルはリレー（DHT ではない）、kill-node は検出するが複製はしません。しかし*1 台にできないことをする・任意のノードに届く・脱落に耐える・見知らぬ相手を信頼する*は、もはや図だけではありません。

---

## ロードマップ

- **M4 — マシン間（完了）。** Mac（Apple MPS）+ Colab の NVIDIA T4（CUDA）で実測。分割経路は単一マシンを**完全に再現**し（3 プロンプト 130/130 トークン）、公開トンネル経由で約 2.7 tok/s、バージョン / バイトオーダー確認が torch のバージョン差を検出しました。`scripts/colab_node.py --bore` + `python -m drift.bench_m4` で再現できます。
- **M5 — ブース展示。** 各ノードが自分のライブなレイヤー範囲 + デバイスを表示。オーケストレーターは *「前半は Apple GPU が考え、後半は NVIDIA が考える」* という形でトークンをストリーミングします。
- **M6 — 穏当なノード切断（完了）。** デコード中に脱落したノードを検出して 1 度再接続し、だめならノード名付きの綺麗な `NodeUnavailable` として表面化します。シームレスなフェイルオーバーはまだありません（レプリケーションが必要）。
- **v2 — エンジン差し替え。** 同じ `ShardEngine` インターフェースの背後に置く `engine_mlx.py` — ワイヤは凍結されたまま、変わるのはノードの内部だけです。ここでフレームワーク中立という主張が報われます。MLX シャードと CUDA シャードで、1 つのモデル。

---

## 連絡先

<p align="center">
  <a href="https://github.com/TaewoooPark"><img src="https://img.shields.io/badge/-GitHub-181717?style=for-the-badge&logo=github&logoColor=white&cacheSeconds=3600" alt="GitHub"></a>
  <a href="https://x.com/theoverstrcture"><img src="https://img.shields.io/badge/-X-000000?style=for-the-badge&logo=x&logoColor=white&cacheSeconds=3600" alt="X (Twitter)"></a>
  <a href="https://www.linkedin.com/in/taewoo-park-427a05352"><img src="https://img.shields.io/badge/-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white&cacheSeconds=3600" alt="LinkedIn"></a>
  <a href="https://www.instagram.com/t.wo0_x/"><img src="https://img.shields.io/badge/-Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white&cacheSeconds=3600" alt="Instagram"></a>
  <a href="https://taewoopark.com"><img src="https://img.shields.io/badge/-taewoopark.com-000000?style=for-the-badge&logo=safari&logoColor=white&cacheSeconds=3600" alt="Personal site"></a>
  <a href="mailto:ptw151125@kaist.ac.kr"><img src="https://img.shields.io/badge/-Email-D14836?style=for-the-badge&logo=gmail&logoColor=white&cacheSeconds=3600" alt="Email"></a>
</p>

<p align="center"><sub>データセンターなし。torch.distributed なし。あなたのマシンと誰か他の人のマシンが、1 つの精神を動かす — そしてそれはビット単位で辻褄が合う。</sub></p>

