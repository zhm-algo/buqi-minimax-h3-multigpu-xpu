# buqi-minimax-h3-multigpu

**English** | [中文](README_zh.md) | [日本語](README_ja.md)

A ComfyUI custom node that runs the MiniMax H3 DiT with Ulysses sequence parallelism across multiple NVIDIA GPUs or Intel Arc Pro B70 (XPU) GPUs.

## Compatibility and status

- Requires the MiniMax H3 API introduced at or after ComfyUI commit **`8d534945`**; startup uses a runtime capability check instead of a release-version claim. Both the `shifts` final-layer API (e.g. ComfyUI 0.39) and the per-step PDD `grids` API (ComfyUI 0.45) are supported.
- Tested against the current ComfyUI 0.37 development commit **`7fbcfa8be9a8f47cf905ec47978b5bd754959ea7`**.
- The old ComfyUI 0.30 integration belongs to older releases; the main branch no longer claims compatibility with it.
- Platform: Linux or WSL2 with PyTorch/NCCL for NVIDIA; Linux with a PyTorch XPU/XCCL build for Intel (see [Intel Arc Pro B70](#intel-arc-pro-b70-xpu)). Native Windows multi-GPU is not supported.
- `world_size` must divide MiniMax H3's 56 attention heads: **1, 2, 4, 7, or 8**.

The CPU suite checks package discovery, current API contracts, fail-fast behavior, patch synchronization, and two-rank Gloo parity. Current-commit GPU execution still requires the manual self-hosted acceptance workflow or the commands below; do not interpret CPU tests as a GPU performance claim.

## Install

Clone the repository directly as one ComfyUI custom-node directory:

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/buqi-code/buqi-minimax-h3-multigpu.git
```

Restart ComfyUI. No extra pip package is required by this node; PyTorch and ComfyUI are intentionally not declared as pip dependencies.

## Models and INT8

Current example filenames are:

- DiT: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- Reference DiT: `minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- Text encoder: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- Video VAE: `minimax_h3_video_vae_int8_convrot.safetensors`
- Audio VAE: `minimax_h3_audio_vae_fp32.safetensors`
- Turbo LoRA: `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`

For a pre-quantized INT8/ConvRot checkpoint, select **`weight_dtype=default`**. Its quantization metadata is loaded by ComfyUI; do not force another dtype in the SP loader.

Use the standard **`VAEDecode`** node for both official INT8 and FP16 video VAEs. This preserves ComfyUI's quantized metadata, streaming behavior, and memory policy. The old `MiniMaxH3SPVAEDecode` node ID remains only so existing workflows load: it logs one deprecation warning and calls the same official `vae.decode` path; it no longer performs parallel VAE decode.

| VAE path | Supported behavior |
|---|---|
| Standard `VAEDecode` + official INT8 or FP16 video VAE | Recommended; official decode and streaming path |
| Legacy `MiniMaxH3SPVAEDecode` node | Compatibility alias for standard `vae.decode`; deprecated and not parallel |
| Audio VAE | Standard `VAEDecodeAudio`; not sharded by this project |

## Supported workflow surface

| Feature | Status |
|---|---|
| T2V | Supported by the current H3 conditioning path |
| I2V first frame | Supported |
| I2V first + last frame | Supported |
| R2V and chained `MiniMaxH3AddGuide` | Supported |
| Denoise masks | Supported |
| PDD/static model options | Supported |
| Static Turbo LoRA loaded before execution | Supported; patches are synchronized before denoising |
| Fun ControlNet | Not supported; rejected before distributed execution |
| Sparse Attention | Not supported; rejected before distributed execution |
| Dynamic hooks/patches | Not supported; rejected before distributed execution |
| Stacking with ComfyUI MultiGPU/threaded MultiGPU | Not supported; rejected before distributed execution |

“Supported” describes the capability-checked H3 interface. Run the GPU acceptance tests for the exact hardware, model files, and workflow combination you deploy.

## SP versus DP

- **Sequence parallelism (this project):** one prompt is split across GPUs by sequence/head communication. Every rank holds the full DiT weights. It targets lower latency for one generation; it does not divide model-weight VRAM.
- **Data parallelism:** each GPU runs a separate prompt. It increases throughput, not the latency of one prompt, and needs an external queue/orchestrator.

Do not stack this SP loader with another MultiGPU wrapper.

## Start and device mapping

Two GPUs:

```bash
cd /path/to/ComfyUI
MINIMAX_SP_DEVICES=0,1 python main.py --cuda-device 0,1 --highvram
```

Then replace only `UNETLoader` with `MiniMaxH3SPUNETLoader`. Keep standard `VAEDecode`, especially with the INT8 video VAE.

`devices="auto"` reads `MINIMAX_SP_DEVICES` first, then the physical identifiers in `CUDA_VISIBLE_DEVICES` (including values set by `--cuda-device 2,3`). It does not replace those with logical `0,1`. Explicit mappings must be unique, contain exactly `world_size` visible physical identifiers, and start with ComfyUI's primary visible GPU. `world_size=1` behaves as a normal single-GPU load.

## Intel Arc Pro B70 (XPU)

Intel GPUs use PyTorch XPU with the XCCL (oneCCL) backend; the NVIDIA/NCCL path is unchanged. The backend is chosen automatically (CUDA first, then Intel XPU) or forced with `MINIMAX_SP_BACKEND=xpu`.

> **Status:** T2V validated end to end on two Arc Pro B70 cards over PCIe (ComfyUI 0.39.0, PyTorch 2.14.0+xpu, INT8 H3 checkpoint, ComfyUI-OmniXPU with `--enable-dynamic-vram`). The two-card video shows the same scene as the one-card video; pixel differences come from INT8 kernel rounding (see Verification). Sampling time per step, one card → two cards: 832×480×25 frames 1.17 → 0.85 s/it (1.4×), 832×480×97 frames 4.43 → 2.64 s/it (1.7×), 1280×720×97 frames 16.6 → 9.0 s/it (1.8×). Other workflows are covered only by the CPU/mocked tests; run the XPU acceptance commands below on your machine before relying on it.

Requirements:

- Linux. XCCL is only shipped in Linux PyTorch XPU builds; native Windows is rejected.
- A PyTorch XPU build with XCCL, PyTorch 2.13+ recommended: `python -c "import torch, torch.distributed as d; print(torch.xpu.device_count(), d.is_xccl_available())"` must report 2+ devices and `True`.
- A recent kernel and Intel GPU driver/compute-runtime with Arc Pro B70 (Battlemage) support; follow Intel's GPU driver installation guide for your distribution.
- The same ComfyUI H3 API requirement as above.

Two B70 cards:

```bash
cd /path/to/ComfyUI
ZE_AFFINITY_MASK=0,1 MINIMAX_SP_DEVICES=0,1 python main.py
```

- Device ids are Level Zero indices, as used by `ZE_AFFINITY_MASK`. List them with `sycl-ls` while no mask is set; an integrated GPU can shift the numbering (then use e.g. `ZE_AFFINITY_MASK=1,2 MINIMAX_SP_DEVICES=1,2`). Unlike `CUDA_VISIBLE_DEVICES`, the mask order does not reorder devices: ComfyUI's primary card is always the lowest visible index, so `MINIMAX_SP_DEVICES` must start with it. `--oneapi-device-selector level_zero:0,1` also works; only the simple `level_zero:<ids>` (or `level_zero:*`) form is accepted.
- Workers follow rank 0's DynamicVRAM state. XPU DynamicVRAM (`--enable-dynamic-vram`) is supported through the [ComfyUI-OmniXPU](https://github.com/intel/llm-scaler/tree/main/omni/ComfyUI-OmniXPU) `comfy_aimdo.xpu` runtime provider: each worker inherits rank 0's environment (`AIMDO_XPU_ALLOCATOR_MODE`, `LD_PRELOAD`, `OMNIXPU_*`), runs OmniXPU's `prestartup_script.py` before importing PyTorch, initializes AIMDO on its card, then loads the OmniXPU adapters, mirroring ComfyUI's startup order. Startup fails if a worker's active runtime providers differ from rank 0; adapter differences (e.g. SeedVR2/VAE preprocessing adapters for nodes that only run on rank 0) are logged as warnings in the worker log. Active XPU DynamicVRAM without that provider is rejected before workers start. With DynamicVRAM off (`--disable-dynamic-vram`, or no OmniXPU), workers use ordinary ComfyUI memory management; when OmniXPU is installed they still replay its providers and adapters, so all ranks run the same XPU kernels. NVIDIA DynamicVRAM behavior is unchanged.
- Avoid `--highvram` at first: rank 0 also hosts the text encoder and VAE, so let ComfyUI offload them.
- Validate with the FP8-scaled checkpoint first; quantized formats depend on the XPU kernels in your ComfyUI/PyTorch stack.

**OmniXPU/container launchers** (e.g. llm-scaler `start_comfyui.sh`): keep the default `--enable-dynamic-vram` and let the launcher set up the `native_hook` allocator and its `LD_PRELOAD`; workers inherit both. To run without DynamicVRAM, pass `--disable-dynamic-vram` to the entrypoint and make sure `AIMDO_XPU_ALLOCATOR_MODE=native_hook` is not set in the shell or container environment. Also drop `aimdo_xpu.so` from `LD_PRELOAD`. Otherwise ComfyUI exits with `[OmniXPU] explicit native_hook requires a compatible AIMDO provider and DynamicVRAM` (the entrypoint prints `native_hook requires DynamicVRAM`).

If startup fails, inspect the worker log named in the error, not just the Gloo traceback. Logs default to the system temporary directory (usually `/tmp/minimax_sp_worker1.log` on Linux), or `MINIMAX_SP_LOGDIR`; save them before retrying because the next startup overwrites them. A worker that exits before joining the process group, for example in OmniXPU prestartup, is reported immediately with its exit code. `Could not autodetect AIMDO implementation, assuming Nvidia` in a worker log means the OmniXPU provider was not applied there; check that rank 0's ComfyUI log shows OmniXPU loading its providers.

Worker visibility (`MINIMAX_SP_XPU_VISIBILITY`):

| Mode | Behavior |
|---|---|
| `shared` (default) | Workers inherit ComfyUI's visibility and select their card by index; every SP card must be visible to ComfyUI (e.g. `ZE_AFFINITY_MASK=0,1`). |
| `isolated` | Each worker starts with `ZE_AFFINITY_MASK=<its card>`, mirroring the CUDA design, so ComfyUI only needs to see rank 0's card. Not recommended: with PyTorch 2.14 XCCL on Arc Pro B70, oneCCL cannot map ranks whose masks differ (worker log: `comm_dev_uuids is not sub-vector of node_dev_uuids`) and the first all-to-all hangs. |

Process groups: on XPU the XCCL tensor group is the default process group and the Gloo control plane is a subgroup. The CUDA layout (Gloo default group, NCCL subgroup) hangs XCCL's first collective on B70.

Timeouts: XCCL has no NCCL-style watchdog, so on XPU the startup probe, each all-to-all, and the final gathers wait with an explicit `MINIMAX_SP_COLLECTIVE_TIMEOUT` bound (default 120 s per collective, including earlier queued GPU work). Waiting at each exchange also bounds temporary-buffer lifetime on long sequences. An independent CPU monitor logs worker exits and host-side enqueue stalls even if XCCL never returns a work handle. It only logs: it does not cancel a blocked driver call or automatically terminate ComfyUI. A failed XCCL collective can leave GPU work queued and block ComfyUI's cleanup (`torch.xpu.synchronize`); restart ComfyUI, or the container if pinned-memory transfers remain stuck.

Worker memory lifecycle: the loader's `ON_CLEANUP` callback sends an acknowledged cleanup command after sampling. Workers release prefetch state and, with DynamicVRAM enabled, reset cast buffers and VBAR watermark limits just as ComfyUI does after a node. This prevents sampler state from persisting across different resolutions/durations. Full-sequence Q/K/V tensors are released before the output exchange. These changes reduce memory pressure but cannot guarantee that every long-video shape fits in VRAM.

XPU acceptance:

```bash
COMFYUI_ROOT=/path/to/ComfyUI torchrun --standalone --nproc-per-node=2 \
  tests/latent_parity.py --backend xpu --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors
MINIMAX_SP_BACKEND=xpu COMFYUI_ROOT=/path/to/ComfyUI python tests/runtime_recovery.py \
  --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors --devices auto
```

If `latent_parity` fails on XPU, attribute the drift before changing anything:

```bash
COMFYUI_ROOT=/path/to/ComfyUI torchrun --standalone --nproc-per-node=2 \
  tests/sp_diagnose.py --backend xpu --unet <same checkpoint>
```

It checks the SP collectives bit for bit and a tiny random H3 in float32 on the cards. With `--unet` it also checks the real model's kernel shape invariance, and compares the SP drift with the model's own ±1 eps noise floor and with the same drift in float32. Mismatched collectives point at XCCL/oneCCL, and a failing tiny float32 run points at SP logic. Otherwise the drift comes from XPU kernels whose rounding depends on the row or head count, so SP cannot be bit-exact there; judge it against the noise floor and an end-to-end sample.

The manual GPU acceptance workflow runs the same steps with `backend=xpu` on a self-hosted runner labeled `xpu`.

## Examples

| File | Format | Coverage |
|---|---|---|
| [`examples/workflow_api_t2v_2gpu.json`](examples/workflow_api_t2v_2gpu.json) | UI | T2V |
| [`examples/workflow_api_i2v_2gpu.json`](examples/workflow_api_i2v_2gpu.json) | UI | First-frame I2V |
| [`examples/workflow_api_i2v_first_last_2gpu.json`](examples/workflow_api_i2v_first_last_2gpu.json) | UI | First+last-frame I2V |
| [`examples/workflow_api_r2v_2gpu.json`](examples/workflow_api_r2v_2gpu.json) | UI | R2V with two reference images |
| [`examples/workflow_api_t2v_4gpu.json`](examples/workflow_api_t2v_4gpu.json) | UI | Four-GPU T2V |
| [`examples/workflow_api_i2v_4gpu.json`](examples/workflow_api_i2v_4gpu.json) | UI | Four-GPU first-frame I2V |
| [`examples/workflow_api_i2v_first_last_4gpu.json`](examples/workflow_api_i2v_first_last_4gpu.json) | UI | Four-GPU first+last-frame I2V |
| [`examples/workflow_api_r2v_4gpu.json`](examples/workflow_api_r2v_4gpu.json) | UI | Four-GPU R2V with two reference images |
| [`examples/workflow_api_t2v_turbo_lora_2gpu.json`](examples/workflow_api_t2v_turbo_lora_2gpu.json) | UI | T2V with the static 8-step Turbo LoRA |
| [`examples/workflow_api_i2v_turbo_lora_2gpu.json`](examples/workflow_api_i2v_turbo_lora_2gpu.json) | UI | First-frame I2V with the static 8-step Turbo LoRA |
| [`examples/workflow_api_i2v_first_last_turbo_lora_2gpu.json`](examples/workflow_api_i2v_first_last_turbo_lora_2gpu.json) | UI | First+last-frame I2V with the static 8-step Turbo LoRA |

All examples are ComfyUI UI graphs (load them by dragging into ComfyUI or via **Open**) that use `MiniMaxH3SPUNETLoader` with `devices=auto`. The `_2gpu.json` examples set `world_size=2`; the `_4gpu.json` examples set `world_size=4`. Expose the matching number of GPUs; for four B70s, use `ZE_AFFINITY_MASK=0,1,2,3 MINIMAX_SP_DEVICES=0,1,2,3` when starting ComfyUI. For an API graph, open the workflow, enable ComfyUI Developer Mode, then use **Save (API Format)**.

The `LoadImage` nodes name images from the validation setup that are not shipped with this repository. Put your own files in `ComfyUI/input` and select them before running.

## Verification

CPU checks:

```bash
cd /path/to/ComfyUI/custom_nodes/buqi-minimax-h3-multigpu
COMFYUI_ROOT=/path/to/ComfyUI PYTHONPATH=/path/to/ComfyUI \
  python -m unittest discover -s tests -p 'test_*.py' -v
COMFYUI_ROOT=/path/to/ComfyUI torchrun --standalone --nproc-per-node=2 tests/test_current_api.py
python -m compileall -q __init__.py minimax_sp tests
python -c 'import json,pathlib; [json.loads(p.read_text()) for p in pathlib.Path("examples").glob("*.json")]'
```

Two-GPU checks:

```bash
COMFYUI_ROOT=/path/to/ComfyUI torchrun --standalone --nproc-per-node=2 \
  tests/latent_parity.py --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors
COMFYUI_ROOT=/path/to/ComfyUI python tests/runtime_recovery.py \
  --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors --devices auto
```

The parity and lifecycle scripts exit nonzero on failure. `tests/selftest.py` is an end-to-end server smoke test and also propagates submission/execution failures, but encoded-video hashes are informational rather than a numerical parity check.

## Timeouts, logs, and troubleshooting

| Variable | Default | Purpose |
|---|---:|---|
| `MINIMAX_SP_DEVICES` | `CUDA_VISIBLE_DEVICES` / `ZE_AFFINITY_MASK` | Optional physical device mapping used by `devices=auto` |
| `MINIMAX_SP_BACKEND` | `auto` | `auto` (CUDA, then Intel XPU), `cuda`, or `xpu`; `cpu` is for tests only |
| `MINIMAX_SP_XPU_VISIBILITY` | `shared` | Intel XPU worker visibility: `shared` or `isolated` |
| `MINIMAX_SP_STARTUP_TIMEOUT` | 300 s | Worker/process-group startup timeout |
| `MINIMAX_SP_COLLECTIVE_TIMEOUT` | 120 s | NCCL collective timeout; on Intel XPU, each probe/all-to-all/final gather wait and host enqueue stall warning |
| `MINIMAX_SP_COMM_BUFFER_MB` | 512 on XPU, else 0 | Fixed per-process collective staging buffer on Intel XPU (MiB). On B70, device memory once handed to XCCL is not returned after PyTorch frees it, so every exchange is chunked through this buffer; `0` disables it (prompts may then fail with `Fault failed: 2` after a few runs) |
| `MINIMAX_SP_LOGDIR` | system temp directory | Worker log directory (`minimax_sp_worker<N>.log`) |
| `MINIMAX_SP_VERIFY=1` | off | Compare the first SP result with native H3 before accepting it |
| `MINIMAX_SP_PROFILE=1` | off | Forward timing logs |
| `MINIMAX_SP_PROFILE_OPS=1` | off | Per-region timing logs |

Common failures:

- **Node is missing after clone:** confirm the repository directory itself is under `custom_nodes`, contains the root `__init__.py`, and inspect the ComfyUI startup traceback.
- **Native Windows error:** use WSL2 or Linux; NCCL is required. Intel XPU multi-GPU needs Linux.
- **GPU visibility/P2P/startup probe failure:** verify `--cuda-device`, `MINIMAX_SP_DEVICES`, `nvidia-smi`, and NCCL/P2P availability. Increase `MINIMAX_SP_STARTUP_TIMEOUT` only if model loading is genuinely slow.
- **XCCL startup probe failure (Intel XPU):** verify `ZE_AFFINITY_MASK`/`ONEAPI_DEVICE_SELECTOR`/`MINIMAX_SP_DEVICES` against `sycl-ls`, keep the default `MINIMAX_SP_XPU_VISIBILITY=shared` (a `comm_dev_uuids is not sub-vector` warning in the worker log means the ranks see different cards), make sure PCIe P2P between the cards is not blocked (ACS/IOMMU settings), and rerun with `CCL_LOG_LEVEL=info`. oneCCL settings reported to help on multi-Arc systems, one at a time: `CCL_ZE_IPC_EXCHANGE=sockets`, `CCL_TOPO_P2P_ACCESS=0` (disables P2P; slower), `CCL_ENABLE_SYCL_KERNELS=0`, `SYCL_UR_USE_LEVEL_ZERO_V2=0`. Restart ComfyUI after a timed-out probe.
- **Collective timeout or worker exit:** inspect every worker log in `MINIMAX_SP_LOGDIR`; the rank-specific traceback is authoritative. Level Zero `OUT_OF_DEVICE_MEMORY` / `OUT_OF_RESOURCES` on one rank can leave its peer blocked inside XCCL. The monitor reports this without auto-exiting ComfyUI. Restart ComfyUI (the container if GPU transfers remain stuck), and reduce resolution/duration if needed rather than repeatedly extending the timeout.
- **Two-card XPU sampling slower than expected:** the all-to-all exchanges depend on the platform's PCIe peer-to-peer support, not on CPU speed. On a tested Core Ultra 9 275HX desktop, Linux does not enable peer-to-peer between the CPU root ports, so XCCL copies through host memory. There the all-to-all reached about 4.9 GB/s per direction, versus about 19.6 GB/s on a Xeon server; CPU load stayed low, and pinning oneCCL workers to P- or E-cores made no difference. Expect a smaller two-card speedup on such systems; a Xeon/Xeon W platform or a PCIe switch shared by both cards avoids the host-memory copy.
- **Deprecated VAE node warning:** replace `MiniMaxH3SPVAEDecode` with standard `VAEDecode`; both now call the official decode path.
- **Changed model/dtype/world size:** loading a different checkpoint (e.g. T2V then R2V) shuts down the running workers and starts a new SP group, which reloads the worker weights; alternating checkpoints therefore pays that startup each time.

## License

MIT; see [LICENSE](LICENSE).
