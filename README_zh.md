# buqi-minimax-h3-multigpu

[English](README.md) | **中文** | [日本語](README_ja.md)

用于 ComfyUI MiniMax H3 的 Ulysses 序列并行多卡加载器，支持多张 NVIDIA GPU 或 Intel Arc Pro B70（XPU）。

## 兼容性与验证范围

- 要求 ComfyUI commit **`8d534945`** 或之后引入的 MiniMax H3 API；启动时由能力检查判断，不声明精确 release tag 范围。同时支持 `shifts` 形式的 final layer API（如 ComfyUI 0.39）和按步 PDD `grids` 形式的 API（ComfyUI 0.45）。
- 已测试当前 ComfyUI 0.37 开发提交：**`7fbcfa8be9a8f47cf905ec47978b5bd754959ea7`**。
- ComfyUI 0.30 接口只属于旧版本；main 不再声明兼容。
- NVIDIA：Linux 或 WSL2 + PyTorch/NCCL；Intel：Linux + 带 XCCL 的 PyTorch XPU 版本（见 [Intel Arc Pro B70](#intel-arc-pro-b70xpu)）。不支持原生 Windows 多卡。
- `world_size` 必须整除 56 个注意力头：**1、2、4、7、8**。

CPU 测试覆盖安装发现、当前 H3 API 契约、fail-fast、patch 同步和双进程 Gloo parity。当前提交的真实 GPU 能力仍需在目标机器运行手动 GPU 验收；CPU 通过不等于 GPU 性能已验证。

## 安装

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/buqi-code/buqi-minimax-h3-multigpu.git
```

重启 ComfyUI。仓库根目录的 `__init__.py` 会直接导出扩展入口，不需要改 `sys.path`。本节点没有额外 pip 依赖；`torch` 和 ComfyUI 不作为 pip 依赖重复声明。

## 当前模型名与 INT8

- DiT：`minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- R2V DiT：`minimax_h3_ref2va_pruned_int8_convrot.safetensors`
- 文本编码器：`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
- 视频 VAE：`minimax_h3_video_vae_int8_convrot.safetensors`
- 音频 VAE：`minimax_h3_audio_vae_fp32.safetensors`
- Turbo LoRA：`minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors`

预量化 INT8/ConvRot checkpoint 必须使用 **`weight_dtype=default`**，由 ComfyUI 读取并保留量化 metadata，不要再次强制转换 dtype。

官方 INT8 和 FP16 视频 VAE 都使用标准 **`VAEDecode`**，以保留 ComfyUI 的量化 metadata、streaming 和显存策略。旧 `MiniMaxH3SPVAEDecode` node ID 只为兼容旧工作流保留：它仅警告一次并调用官方 `vae.decode`，不再执行并行 VAE decode。

| VAE 路径 | 实际支持 |
|---|---|
| 标准 `VAEDecode` + 官方 INT8 或 FP16 视频 VAE | 推荐；走官方 decode/streaming 路径 |
| 旧 `MiniMaxH3SPVAEDecode` 节点 | 标准 `vae.decode` 的 deprecated 兼容别名，不并行 |
| 音频 VAE | 使用标准 `VAEDecodeAudio`，本项目不分片 |

## 功能矩阵

| 功能 | 状态 |
|---|---|
| T2V | 支持当前 H3 conditioning 路径 |
| I2V first | 支持 |
| I2V first + last | 支持 |
| R2V / 链式 `MiniMaxH3AddGuide` | 支持 |
| denoise mask | 支持 |
| PDD / 静态 model options | 支持 |
| 运行前加载的静态 Turbo LoRA | 支持，去噪前同步 patch |
| Fun ControlNet | 不支持，进入分布式执行前失败 |
| Sparse Attention | 不支持，提前失败 |
| 动态 hooks/patches | 不支持，提前失败 |
| 与 ComfyUI MultiGPU/threaded MultiGPU 叠加 | 不支持，提前失败 |

这里的“支持”指通过运行时能力检查的 H3 接口；部署前仍应对你的硬件、模型文件和工作流运行 GPU 验收。

## SP 与 DP

- **SP（本项目）**：同一个 prompt 的序列/注意力计算分到多张卡，目标是降低单任务延迟。每张卡仍保存完整 DiT 权重，不会按卡数降低权重显存。
- **DP**：每张卡独立跑不同 prompt，提升吞吐量，不降低单个任务延迟，需要外部队列或调度器。

不要把本 SP loader 再套到其他 MultiGPU wrapper 上。

## 启动与设备映射

```bash
cd /path/to/ComfyUI
MINIMAX_SP_DEVICES=0,1 python main.py --cuda-device 0,1 --highvram
```

工作流中只把 `UNETLoader` 换成 `MiniMaxH3SPUNETLoader`；尤其使用 INT8 视频 VAE 时，继续使用标准 `VAEDecode`。

`devices="auto"` 优先读取 `MINIMAX_SP_DEVICES`，否则读取 `CUDA_VISIBLE_DEVICES` 中的物理 ID（包括 `--cuda-device 2,3` 写入的值），不会误用逻辑重编号后的 `0,1`。显式映射必须唯一、数量恰好等于 `world_size`、全部来自当前可见物理列表，且第一项与 ComfyUI 主卡一致。`world_size=1` 为普通单卡加载。

## Intel Arc Pro B70（XPU）

Intel 显卡使用 PyTorch XPU + XCCL（oneCCL）后端；NVIDIA/NCCL 路径保持不变。后端自动选择（先 CUDA，再 Intel XPU），也可用 `MINIMAX_SP_BACKEND=xpu` 强制指定。

> **状态：** T2V 已在两张通过 PCIe 互连的 Arc Pro B70 上端到端验证（ComfyUI 0.39.0、PyTorch 2.14.0+xpu、INT8 H3 checkpoint、ComfyUI-OmniXPU + `--enable-dynamic-vram`）。双卡视频与单卡视频画面一致，像素差异来自 INT8 kernel 舍入（见“验证命令”）。每步采样时间（单卡 → 双卡）：832×480×25 帧 1.17 → 0.85 s/it（1.4×），832×480×97 帧 4.43 → 2.64 s/it（1.7×），1280×720×97 帧 16.6 → 9.0 s/it（1.8×）。其他工作流目前只有 CPU/mock 测试覆盖；正式使用前请先在目标机器上运行下方 XPU 验收命令。

要求：

- Linux。XCCL 只包含在 Linux 版 PyTorch XPU 中；原生 Windows 会被拒绝。
- 带 XCCL 的 PyTorch XPU，推荐 PyTorch 2.13+：`python -c "import torch, torch.distributed as d; print(torch.xpu.device_count(), d.is_xccl_available())"` 应输出 ≥2 张卡和 `True`。
- 支持 Arc Pro B70（Battlemage）的较新内核与 Intel GPU 驱动/compute-runtime，按 Intel 官方对应发行版的驱动安装指南安装。
- ComfyUI H3 API 要求同上。

双卡 B70：

```bash
cd /path/to/ComfyUI
ZE_AFFINITY_MASK=0,1 MINIMAX_SP_DEVICES=0,1 python main.py
```

- 设备 ID 是 `ZE_AFFINITY_MASK` 使用的 Level Zero 序号。请在未设置 mask 时用 `sycl-ls` 查看；有核显时编号可能后移（此时改用如 `ZE_AFFINITY_MASK=1,2 MINIMAX_SP_DEVICES=1,2`）。与 `CUDA_VISIBLE_DEVICES` 不同，mask 中的书写顺序不会改变设备顺序：ComfyUI 主卡始终是可见序号最小的那张，`MINIMAX_SP_DEVICES` 必须以它开头。也可用 `--oneapi-device-selector level_zero:0,1`，但只接受简单的 `level_zero:<ids>`（或 `level_zero:*`）形式。
- worker 跟随 rank 0 的 DynamicVRAM 状态。XPU DynamicVRAM（`--enable-dynamic-vram`）需通过 [ComfyUI-OmniXPU](https://github.com/intel/llm-scaler/tree/main/omni/ComfyUI-OmniXPU) 的 `comfy_aimdo.xpu` runtime provider 使用。每个 worker 继承 rank 0 的环境（`AIMDO_XPU_ALLOCATOR_MODE`、`LD_PRELOAD`、`OMNIXPU_*`），按 ComfyUI 的启动顺序：在导入 PyTorch 前执行 OmniXPU 的 `prestartup_script.py`，在自己的卡上初始化 AIMDO，再加载 OmniXPU adapter。若 worker 的活动 runtime provider 与 rank 0 不一致，启动会报错；adapter 的差异（例如只在 rank 0 运行的 SeedVR2/VAE 预处理 adapter）只在 worker 日志中给出警告。rank 0 开启了 XPU DynamicVRAM 但没有该 provider 时，会在启动 worker 前拒绝。关闭 DynamicVRAM（`--disable-dynamic-vram`，或未安装 OmniXPU）时，worker 使用普通 ComfyUI 显存管理；只要装了 OmniXPU，worker 仍会重放其 provider 和 adapter，保证各 rank 使用相同 XPU kernel。NVIDIA 的 DynamicVRAM 行为不变。
- 初次运行不要加 `--highvram`：rank 0 还承载文本编码器和 VAE，应让 ComfyUI 自行 offload。
- 先用 FP8-scaled checkpoint 验证；量化格式能否运行取决于你的 ComfyUI/PyTorch 中的 XPU kernel。

**OmniXPU/容器入口脚本**（如 llm-scaler 的 `start_comfyui.sh`）：保持默认的 `--enable-dynamic-vram`，由入口脚本设置 `native_hook` 分配器及其 `LD_PRELOAD`，worker 会自动继承。若要关闭 DynamicVRAM，需向 entrypoint 传入 `--disable-dynamic-vram`，并确保 shell/容器环境中没有 `AIMDO_XPU_ALLOCATOR_MODE=native_hook`，同时从 `LD_PRELOAD` 去掉 `aimdo_xpu.so`。否则 ComfyUI 会报 `[OmniXPU] explicit native_hook requires a compatible AIMDO provider and DynamicVRAM`（入口脚本则输出 `native_hook requires DynamicVRAM`）并退出。

启动失败时，请检查异常中给出的 worker 日志，而不只是 Gloo 堆栈。日志默认在系统临时目录（Linux 通常为 `/tmp/minimax_sp_worker1.log`），也可用 `MINIMAX_SP_LOGDIR` 指定；重试前先保存，下一次启动会覆盖。worker 在加入进程组前退出（例如 OmniXPU prestartup 失败）时会立即报告退出码。若 worker 日志出现 `Could not autodetect AIMDO implementation, assuming Nvidia`，说明该 worker 未应用 OmniXPU provider，请确认 rank 0 的 ComfyUI 日志中 OmniXPU 已加载其 provider。

worker 可见性（`MINIMAX_SP_XPU_VISIBILITY`）：

| 模式 | 行为 |
|---|---|
| `shared`（默认） | worker 继承 ComfyUI 的可见性并按序号选卡；所有 SP 卡都必须对 ComfyUI 可见（如 `ZE_AFFINITY_MASK=0,1`）。 |
| `isolated` | 每个 worker 以 `ZE_AFFINITY_MASK=<自己的卡>` 启动，与 CUDA 设计一致，ComfyUI 只需看到 rank 0 的卡。不推荐：在 Arc Pro B70 + PyTorch 2.14 XCCL 上，各 rank 的 mask 不同时 oneCCL 无法映射设备（worker 日志出现 `comm_dev_uuids is not sub-vector of node_dev_uuids`），第一次 all-to-all 会卡死。 |

进程组：XPU 上 XCCL tensor 组是默认进程组，Gloo 控制面是子组。CUDA 的布局（Gloo 默认组 + NCCL 子组）在 B70 上会让 XCCL 的第一个 collective 卡死。

超时：XCCL 没有 NCCL 那样的 watchdog，因此 XPU 上 startup probe、每次 all-to-all 和最终 gather 都使用 `MINIMAX_SP_COLLECTIVE_TIMEOUT` 显式等待（默认每次 collective 120 秒，包含前面排队的 GPU 工作）。逐次等待也限制了长序列通信临时缓冲区的存活时间。独立 CPU 监视线程会记录 worker 退出及主机侧入队卡死，即使 XCCL 尚未返回 work handle 也能报错。它只记录日志，不会取消驱动调用或自动退出 ComfyUI。失败的 XCCL 可能让 GPU 工作一直排队，连 ComfyUI 清理时的 `torch.xpu.synchronize` 也会阻塞；请重启 ComfyUI，若锁页内存传输仍卡住则重启容器。

worker 内存生命周期：loader 的 `ON_CLEANUP` 回调会在采样结束后发送并确认 cleanup 命令。worker 清理 prefetch 状态；启用 DynamicVRAM 时，还会像 ComfyUI 节点结束时一样重置 cast buffer 和 VBAR watermark limit，避免不同分辨率/时长的任务之间残留采样状态。全序列 Q/K/V 在输出通信前释放。这些改动降低显存压力，但不保证任意长视频尺寸都能装入显存。

XPU 验收：

```bash
COMFYUI_ROOT=/path/to/ComfyUI torchrun --standalone --nproc-per-node=2 \
  tests/latent_parity.py --backend xpu --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors
MINIMAX_SP_BACKEND=xpu COMFYUI_ROOT=/path/to/ComfyUI python tests/runtime_recovery.py \
  --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors --devices auto
```

如果 XPU 上 `latent_parity` 失败，先定位偏差来源再改代码：

```bash
COMFYUI_ROOT=/path/to/ComfyUI torchrun --standalone --nproc-per-node=2 \
  tests/sp_diagnose.py --backend xpu --unet <同一个 checkpoint>
```

它会逐位校验 SP 的 collective，并在卡上跑一个随机初始化的小型 H3（float32）。加 `--unet` 时还会检查真实模型 kernel 的形状不变性，并把 SP 偏差与模型自身的 ±1 eps 噪声基线、以及 float32 下的同一偏差做对比。collective 不一致说明是 XCCL/oneCCL 问题；小模型 float32 失败说明是 SP 逻辑问题。若两者都正常，偏差来自舍入结果随行数或 head 数变化的 XPU kernel，SP 在该平台上无法逐位一致，应以噪声基线和端到端生成结果来判断。

手动 GPU 验收 workflow 选择 `backend=xpu` 时，会在带 `xpu` 标签的 self-hosted runner 上执行相同步骤。

## 示例

| 文件 | 格式/用途 |
|---|---|
| [`examples/workflow_api_t2v_2gpu.json`](examples/workflow_api_t2v_2gpu.json) | UI：T2V |
| [`examples/workflow_api_i2v_2gpu.json`](examples/workflow_api_i2v_2gpu.json) | UI：I2V first |
| [`examples/workflow_api_i2v_first_last_2gpu.json`](examples/workflow_api_i2v_first_last_2gpu.json) | UI：I2V first+last |
| [`examples/workflow_api_r2v_2gpu.json`](examples/workflow_api_r2v_2gpu.json) | UI：R2V（两张参考图） |
| [`examples/workflow_api_t2v_turbo_lora_2gpu.json`](examples/workflow_api_t2v_turbo_lora_2gpu.json) | UI：T2V + 静态 8-step Turbo LoRA |
| [`examples/workflow_api_i2v_turbo_lora_2gpu.json`](examples/workflow_api_i2v_turbo_lora_2gpu.json) | UI：I2V first + 静态 8-step Turbo LoRA |
| [`examples/workflow_api_i2v_first_last_turbo_lora_2gpu.json`](examples/workflow_api_i2v_first_last_turbo_lora_2gpu.json) | UI：I2V first+last + 静态 8-step Turbo LoRA |

所有示例都是 ComfyUI UI 工作流（拖入 ComfyUI 或通过 **Open** 打开），使用 `MiniMaxH3SPUNETLoader`，`world_size=2`、`devices=auto`，已在两张 Intel Arc Pro B70 上验证。需要 API 格式时，打开工作流后启用 ComfyUI Developer Mode，再使用 **Save (API Format)** 导出。

`LoadImage` 节点中的图片名来自验证环境，仓库未附带这些图片。请把自己的文件放进 `ComfyUI/input` 并在运行前重新选择。

## 验证命令

CPU：

```bash
cd /path/to/ComfyUI/custom_nodes/buqi-minimax-h3-multigpu
COMFYUI_ROOT=/path/to/ComfyUI PYTHONPATH=/path/to/ComfyUI \
  python -m unittest discover -s tests -p 'test_*.py' -v
COMFYUI_ROOT=/path/to/ComfyUI torchrun --standalone --nproc-per-node=2 tests/test_current_api.py
python -m compileall -q __init__.py minimax_sp tests
python -c 'import json,pathlib; [json.loads(p.read_text()) for p in pathlib.Path("examples").glob("*.json")]'
```

双卡验收：

```bash
COMFYUI_ROOT=/path/to/ComfyUI torchrun --standalone --nproc-per-node=2 \
  tests/latent_parity.py --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors
COMFYUI_ROOT=/path/to/ComfyUI python tests/runtime_recovery.py \
  --unet minimax_h3_fl2va_pruned_fp8_scaled.safetensors --devices auto
```

张量 parity 和真实 loader/SPGroup 生命周期脚本失败都会返回非零。`tests/selftest.py` 是连接已运行 ComfyUI 服务的端到端冒烟测试，也会传递提交/执行错误；编码视频 hash 只作信息展示，不代表数值一致性。

## 超时、worker 日志与排障

| 环境变量 | 默认 | 作用 |
|---|---:|---|
| `MINIMAX_SP_DEVICES` | `CUDA_VISIBLE_DEVICES` / `ZE_AFFINITY_MASK` | `devices=auto` 的可选物理设备映射 |
| `MINIMAX_SP_BACKEND` | `auto` | `auto`（先 CUDA 再 Intel XPU）、`cuda` 或 `xpu`；`cpu` 仅用于测试 |
| `MINIMAX_SP_XPU_VISIBILITY` | `shared` | Intel XPU worker 可见性：`shared` 或 `isolated` |
| `MINIMAX_SP_STARTUP_TIMEOUT` | 300 秒 | worker/进程组启动超时 |
| `MINIMAX_SP_COLLECTIVE_TIMEOUT` | 120 秒 | NCCL collective 超时；Intel XPU 上每次 probe/all-to-all/最终 gather 的等待上限及主机入队停滞告警 |
| `MINIMAX_SP_COMM_BUFFER_MB` | XPU 512，其他 0 | Intel XPU 固定通信暂存缓冲区（MiB，每进程一份）。B70 上曾交给 XCCL 的显存在 PyTorch 释放后不会归还，因此所有交换都经此缓冲区分块传输；设为 `0` 可关闭（多次 prompt 后可能出现 `Fault failed: 2`） |
| `MINIMAX_SP_LOGDIR` | 系统临时目录 | `minimax_sp_worker<N>.log` 所在目录 |
| `MINIMAX_SP_VERIFY=1` | 关闭 | 首次 SP 输出与原生 H3 对比后才接受 |
| `MINIMAX_SP_PROFILE=1` | 关闭 | forward 耗时 |
| `MINIMAX_SP_PROFILE_OPS=1` | 关闭 | 分区耗时 |

- **clone 后没有节点**：确认仓库目录本身位于 `custom_nodes`，根目录有 `__init__.py`，查看 ComfyUI 启动 traceback。
- **原生 Windows 报错**：改用 WSL2 或 Linux，SP 需要 NCCL；Intel XPU 多卡需要 Linux。
- **GPU visibility/P2P/startup probe 失败**：检查 `--cuda-device`、`MINIMAX_SP_DEVICES`、`nvidia-smi` 与 NCCL/P2P。只有确实加载较慢时才提高 startup timeout。
- **XCCL startup probe 失败（Intel XPU）**：对照 `sycl-ls` 检查 `ZE_AFFINITY_MASK`/`ONEAPI_DEVICE_SELECTOR`/`MINIMAX_SP_DEVICES`，保持默认的 `MINIMAX_SP_XPU_VISIBILITY=shared`（worker 日志中的 `comm_dev_uuids is not sub-vector` 警告表示各 rank 看到的卡不同），确认卡间 PCIe P2P 未被 ACS/IOMMU 设置阻断，并用 `CCL_LOG_LEVEL=info` 重跑。多 Arc 系统上有报告称以下 oneCCL 设置有帮助，请逐个尝试：`CCL_ZE_IPC_EXCHANGE=sockets`、`CCL_TOPO_P2P_ACCESS=0`（关闭 P2P，较慢）、`CCL_ENABLE_SYCL_KERNELS=0`、`SYCL_UR_USE_LEVEL_ZERO_V2=0`。probe 超时后请重启 ComfyUI。
- **collective timeout/worker 退出**：查看 `MINIMAX_SP_LOGDIR` 中每个 rank 的日志。单个 rank 的 Level Zero `OUT_OF_DEVICE_MEMORY` / `OUT_OF_RESOURCES` 可能让另一端一直卡在 XCCL 中；监视线程会报错，但不自动退出 ComfyUI。重启 ComfyUI（GPU 传输仍卡住时重启容器），必要时降低分辨率/时长，不要只反复加大 timeout。
- **VAE 节点 deprecated warning**：把 `MiniMaxH3SPVAEDecode` 替换为标准 `VAEDecode`；两者现在都走官方解码。
- **更换 checkpoint/dtype/world size**：加载不同 checkpoint（如先 T2V 后 R2V）时会关闭正在运行的 worker 并新建 SP 组，worker 需要重新加载权重；交替使用不同 checkpoint 每次都要付出这段启动时间。

## 许可证

MIT，见 [LICENSE](LICENSE)。
