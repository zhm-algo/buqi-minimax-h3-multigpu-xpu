# buqi-minimax-h3-multigpu

[English](README.md) | [中文](README_zh.md) | **日本語**

ComfyUI MiniMax H3 用の Ulysses シーケンス並列ローダーです。複数の NVIDIA GPU または Intel Arc Pro B70（XPU）に対応します。

## 互換性

- ComfyUI commit **`8d534945`** 以降で導入された MiniMax H3 API が必要です。release tag ではなく起動時の capability check で判定します。`shifts` 形式の final layer API（例: ComfyUI 0.39）とステップ別 PDD `grids` 形式の API（ComfyUI 0.45）の両方に対応します。
- 現在の ComfyUI 0.37 development commit **`7fbcfa8be9a8f47cf905ec47978b5bd754959ea7`** で検証済みです。
- ComfyUI 0.30 は旧 release の対象で、main branch の対象外です。
- NVIDIA: Linux または WSL2 + NCCL。Intel: Linux + XCCL 対応 PyTorch XPU（[Intel Arc Pro B70](#intel-arc-pro-b70xpu) 参照）。Windows native の multi-GPU は非対応です。
- `world_size`: 1 / 2 / 4 / 7 / 8

CPU テストは package discovery、現在の H3 API、fail-fast、patch 同期、2-rank Gloo parity を確認します。実 GPU の可否・性能は対象マシンで GPU acceptance を実行して確認してください。

## インストールと起動

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/buqi-code/buqi-minimax-h3-multigpu.git
cd /path/to/ComfyUI
MINIMAX_SP_DEVICES=0,1 python main.py --cuda-device 0,1 --highvram
```

ワークフローでは `UNETLoader` だけを `MiniMaxH3SPUNETLoader` に置き換えます。SP は 1 prompt の遅延を下げる方式で、各 GPU に完全な DiT weight を保持します。別の ComfyUI MultiGPU wrapper と重ねないでください。DP は GPU ごとに別 prompt を処理する throughput 向け方式です。

`devices="auto"` は `MINIMAX_SP_DEVICES`、次に `CUDA_VISIBLE_DEVICES` の物理 ID（`--cuda-device 2,3` を含む）を使います。論理番号 `0,1` へ置き換えません。明示指定は重複不可で `world_size` と同数、すべて可視リスト内、先頭は ComfyUI の primary GPU と一致する必要があります。

## Intel Arc Pro B70（XPU）

Intel GPU では PyTorch XPU + XCCL（oneCCL）backend を使います。NVIDIA/NCCL の経路は変わりません。backend は自動選択（CUDA、次に Intel XPU）で、`MINIMAX_SP_BACKEND=xpu` で強制できます。

> **状態:** T2V は PCIe 接続の Arc Pro B70 2 枚でエンドツーエンドに検証済みです（ComfyUI 0.39.0、PyTorch 2.14.0+xpu、INT8 H3 checkpoint、ComfyUI-OmniXPU + `--enable-dynamic-vram`）。2 枚の出力は 1 枚の出力と同じシーンで、画素差は INT8 kernel の丸めによるものです。1 ステップのサンプリング時間（1 枚 → 2 枚）: 832×480×25 フレーム 1.17 → 0.85 s/it（1.4×）、832×480×97 フレーム 4.43 → 2.64 s/it（1.7×）、1280×720×97 フレーム 16.6 → 9.0 s/it（1.8×）。その他のワークフローは CPU/mock テストのみで確認しています。利用前に対象マシンで下記の XPU acceptance を実行してください。

- Linux 専用（XCCL は Linux 版 PyTorch XPU のみ）。PyTorch 2.13+ 推奨。`python -c "import torch, torch.distributed as d; print(torch.xpu.device_count(), d.is_xccl_available())"` が 2 以上と `True` を返すこと。Arc Pro B70（Battlemage）対応の新しい kernel と Intel GPU driver/compute-runtime が必要です。

```bash
cd /path/to/ComfyUI
ZE_AFFINITY_MASK=0,1 MINIMAX_SP_DEVICES=0,1 python main.py
```

- device ID は `ZE_AFFINITY_MASK` の Level Zero 番号です。mask 未設定の状態で `sycl-ls` で確認してください（iGPU があると番号がずれます）。`CUDA_VISIBLE_DEVICES` と異なり mask の記述順でデバイス順は変わらず、ComfyUI の primary は常に最小の可視番号なので `MINIMAX_SP_DEVICES` はそれから始めてください。`--oneapi-device-selector level_zero:0,1` も使えますが、単純な `level_zero:<ids>`（または `level_zero:*`）のみ対応です。
- worker は rank 0 の DynamicVRAM 状態に従います。XPU DynamicVRAM（`--enable-dynamic-vram`）は [ComfyUI-OmniXPU](https://github.com/intel/llm-scaler/tree/main/omni/ComfyUI-OmniXPU) の `comfy_aimdo.xpu` runtime provider 経由で対応します。各 worker は rank 0 の環境（`AIMDO_XPU_ALLOCATOR_MODE`・`LD_PRELOAD`・`OMNIXPU_*`）を継承し、ComfyUI と同じ順序で処理します：PyTorch の import 前に OmniXPU の `prestartup_script.py` を実行し、自分のカードで AIMDO を初期化してから OmniXPU adapter を読み込みます。worker の有効 runtime provider が rank 0 と異なる場合は起動エラーになります。adapter の差異（rank 0 でのみ実行される SeedVR2/VAE 前処理 adapter など）は worker ログに警告として出力されます。provider なしで XPU DynamicVRAM が有効な場合は worker 起動前に拒否します。DynamicVRAM 無効時（`--disable-dynamic-vram` または OmniXPU なし）は通常の ComfyUI メモリ管理を使います。OmniXPU がインストールされていれば provider と adapter は引き続き再現され、全 rank が同じ XPU kernel を使います。NVIDIA の DynamicVRAM 動作は変わりません。最初は `--highvram` を付けず、FP8-scaled checkpoint で確認してください。
- `MINIMAX_SP_XPU_VISIBILITY=shared`（既定）: worker は ComfyUI の可視性を継承し番号で選択するため、全 SP カードを ComfyUI から見える状態にしてください。`isolated`: 各 worker は `ZE_AFFINITY_MASK=<自分のカード>` で起動します。非推奨：Arc Pro B70 + PyTorch 2.14 XCCL では rank ごとに mask が異なると oneCCL がデバイスを対応付けられず（worker ログに `comm_dev_uuids is not sub-vector of node_dev_uuids`）、最初の all-to-all がハングします。
- XPU では XCCL の tensor group を default process group とし、Gloo の control plane を subgroup にします。CUDA と同じ構成（Gloo default + NCCL subgroup）では B70 上で XCCL の最初の collective がハングします。
- XCCL には NCCL のような watchdog がないため、XPU では startup probe、各 all-to-all、最後の gather を `MINIMAX_SP_COLLECTIVE_TIMEOUT` で明示的に待ちます（既定は collective ごとに 120 秒、先行 GPU 処理を含む）。各交換で待つことで長いシーケンスの一時バッファ寿命も制限します。独立した CPU 監視スレッドは worker 終了とホスト側 enqueue 停滞を記録しますが、ドライバ呼び出しを中断したり ComfyUI を自動終了したりはしません。XCCL 失敗後は ComfyUI を、メモリ転送がまだ停止している場合はコンテナを再起動してください。
- サンプリング後の `ON_CLEANUP` callback は worker に cleanup コマンドを送り、ACK を待ちます。worker は prefetch 状態と、DynamicVRAM 使用時は cast buffer / VBAR watermark limit をリセットします。全シーケンスの Q/K/V は出力交換前に解放します。メモリ圧力を軽減しますが、任意の長さ・解像度が VRAM に収まる保証はありません。

**OmniXPU/container**（例: llm-scaler の `start_comfyui.sh`）: 既定の `--enable-dynamic-vram` のままにし、entrypoint に `native_hook` allocator とその `LD_PRELOAD` を設定させてください。worker はそれを継承します。DynamicVRAM を無効にする場合は entrypoint に `--disable-dynamic-vram` を渡し、shell／コンテナ環境に `AIMDO_XPU_ALLOCATOR_MODE=native_hook` が残っていないことを確認してください。あわせて `LD_PRELOAD` から `aimdo_xpu.so` を外してください。そうしないと ComfyUI は `[OmniXPU] explicit native_hook requires a compatible AIMDO provider and DynamicVRAM`（entrypoint では `native_hook requires DynamicVRAM`）で終了します。

起動に失敗した場合は、例外に示された worker ログを確認してください。既定はシステム一時ディレクトリ（Linux では通常 `/tmp/minimax_sp_worker1.log`）、変更は `MINIMAX_SP_LOGDIR` です。次回起動で上書きされるため先に保存してください。プロセスグループ参加前に worker が終了した場合（例: OmniXPU prestartup の失敗）は終了コードとともに即座に報告されます。worker ログの `Could not autodetect AIMDO implementation, assuming Nvidia` は、その worker に OmniXPU provider が適用されていないことを示します。rank 0 の ComfyUI ログで OmniXPU が provider を読み込んでいるか確認してください。

XPU acceptance:

```bash
COMFYUI_ROOT=/path/to/ComfyUI torchrun --standalone --nproc-per-node=2 \
  tests/latent_parity.py --backend xpu --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors
MINIMAX_SP_BACKEND=xpu COMFYUI_ROOT=/path/to/ComfyUI python tests/runtime_recovery.py \
  --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors --devices auto
```

XPU で `latent_parity` が失敗した場合は、変更する前に偏差の原因を切り分けてください：

```bash
COMFYUI_ROOT=/path/to/ComfyUI torchrun --standalone --nproc-per-node=2 \
  tests/sp_diagnose.py --backend xpu --unet <同じ checkpoint>
```

SP の collective をビット単位で検証し、ランダムな小型 H3 を float32 でカード上で実行します。`--unet` を付けると実モデルの kernel の形状不変性も確認し、SP の偏差をモデル自身の ±1 eps ノイズ基準および float32 での同じ偏差と比較します。collective の不一致は XCCL/oneCCL、小型 float32 の失敗は SP ロジックの問題です。どちらも正常なら、偏差は行数や head 数で丸めが変わる XPU kernel によるもので、SP はビット一致になりません。ノイズ基準と end-to-end の生成結果で判断してください。

## INT8 と VAE

事前量子化済み INT8/ConvRot checkpoint は **`weight_dtype=default`** を使用します。例では次を使用します。

- `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- `minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- `minimax_h3_video_vae_int8_convrot.safetensors`
- `minimax_h3_audio_vae_fp32.safetensors`
- `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`

公式 INT8/FP16 video VAE はどちらも標準 **`VAEDecode`** を使い、ComfyUI の quantized metadata、streaming、memory policy を保持します。旧 `MiniMaxH3SPVAEDecode` node ID は workflow 互換性のためだけに残り、warning を一度出して公式 `vae.decode` を呼ぶ deprecated alias です。VAE の並列実行は行いません。Audio は標準 `VAEDecodeAudio` を使います。

## サポート範囲

対応: T2V、I2V first、I2V first+last、R2V、`MiniMaxH3AddGuide`、mask、PDD/static options、実行前に適用する static Turbo LoRA。

非対応（distributed 実行前に停止）: Fun ControlNet、Sparse Attention、dynamic hooks/patches、ComfyUI MultiGPU/threaded MultiGPU との併用。

## サンプル

- UI workflow: [`T2V`](examples/workflow_api_t2v_2gpu.json), [`I2V first`](examples/workflow_api_i2v_2gpu.json), [`I2V first+last`](examples/workflow_api_i2v_first_last_2gpu.json), [`R2V`（参照画像 2 枚）](examples/workflow_api_r2v_2gpu.json)。

いずれも ComfyUI の UI workflow で、`MiniMaxH3SPUNETLoader`（`world_size=2`、`devices=auto`）を使い、Intel Arc Pro B70 2 枚で検証済みです。API 形式が必要な場合は workflow を開き、Developer Mode の **Save (API Format)** で export してください。`LoadImage` の画像名は検証環境のもので、リポジトリには含まれません。`ComfyUI/input` に自分の画像を置き、実行前に選び直してください。

## 検証とトラブルシューティング

```bash
COMFYUI_ROOT=/path/to/ComfyUI PYTHONPATH=/path/to/ComfyUI \
  python -m unittest discover -s tests -p 'test_*.py' -v
COMFYUI_ROOT=/path/to/ComfyUI torchrun --standalone --nproc-per-node=2 tests/test_current_api.py
python -m compileall -q __init__.py minimax_sp tests
```

GPU tensor parity と実際の loader/SPGroup lifecycle command は [English README](README.md#verification) を参照してください。失敗は nonzero で終了します。

主な環境変数:

- `MINIMAX_SP_DEVICES=0,1`: device mapping。先頭は ComfyUI の main GPU。
- `MINIMAX_SP_BACKEND=auto`: `auto` / `cuda` / `xpu`（`cpu` はテスト専用）。
- `MINIMAX_SP_XPU_VISIBILITY=shared`: Intel XPU worker の可視性（`shared` / `isolated`）。
- `MINIMAX_SP_STARTUP_TIMEOUT=300`: startup timeout（秒）。
- `MINIMAX_SP_COLLECTIVE_TIMEOUT=120`: collective timeout（秒）。Intel XPU では各 probe/all-to-all/最終 gather の待機上限とホスト enqueue 停滞警告。
- `MINIMAX_SP_COMM_BUFFER_MB=512`: Intel XPU の固定 collective staging buffer（MiB、プロセスごと）。B70 では一度 XCCL に渡したデバイスメモリが解放されないため、すべての交換をこの buffer 経由で分割転送します。`0` で無効化（数 prompt 後に `Fault failed: 2` が出る可能性あり）。
- `MINIMAX_SP_LOGDIR=/path`: `minimax_sp_worker<N>.log` の保存先。

ノードが出ない場合は clone 先と root `__init__.py`、ComfyUI 起動ログを確認してください。P2P/startup/collective failure は `nvidia-smi`、device mapping、NCCL と各 worker log を確認します。Intel XPU の XCCL probe が失敗する場合は `sycl-ls` と `ZE_AFFINITY_MASK`/`MINIMAX_SP_DEVICES` を照合し、`MINIMAX_SP_XPU_VISIBILITY=shared`、ACS/IOMMU による PCIe P2P のブロック、`CCL_LOG_LEVEL=info` のログを確認してください。旧 VAE node の deprecated warning が出たら標準 `VAEDecode` に置き換えてください。checkpoint/dtype/world size を変えると（例: T2V の後に R2V）、実行中の worker を終了して新しい SP group を起動し、worker の重みを再ロードします。

MIT License。詳細は [LICENSE](LICENSE) を参照してください。
