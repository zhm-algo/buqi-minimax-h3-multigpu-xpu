# Changelog

## Unreleased

- Add Intel Arc Pro B70 / Intel XPU support through PyTorch XPU and the XCCL backend (Linux only). The backend is selected automatically (CUDA first, then XPU) or with `MINIMAX_SP_BACKEND`; the NVIDIA/NCCL path is unchanged.
- Map XPU devices through `ZE_AFFINITY_MASK` and simple `ONEAPI_DEVICE_SELECTOR=level_zero:<ids>` values, following driver enumeration order (the mask order does not reorder devices, unlike `CUDA_VISIBLE_DEVICES`). Workers keep ComfyUI's visibility by default (`MINIMAX_SP_XPU_VISIBILITY=shared`); `isolated` gives each worker its own mask, but oneCCL's all-to-all hangs on B70 when the ranks' masks differ.
- On XPU, make the XCCL tensor group the default process group and the Gloo control plane a subgroup: an XCCL subgroup under a Gloo default group hangs in its first collective on B70. The CUDA layout is unchanged.
- Because XCCL has no NCCL-style watchdog, XPU waits on the startup probe, every all-to-all, and the final gathers with an explicit `MINIMAX_SP_COLLECTIVE_TIMEOUT` bound, limiting queued temporary-buffer lifetime. Release full-sequence Q/K/V before output exchange/projection to reduce peak memory.
- Mirror ComfyUI's post-sampling cleanup in workers through an acknowledged `ON_CLEANUP` command, including prefetch state, DynamicVRAM cast buffers, and VBAR watermark limits. Add repeated-lifecycle and cleanup-failure regression tests.
- Log worker exits and host-side XCCL enqueue stalls from an independent CPU monitor. It does not terminate ComfyUI automatically or claim to cancel a blocked driver call; recovery may require restarting the container.
- Add XPU backend unit tests, a CPU/Gloo end-to-end SPGroup lifecycle test, a bounded-wait case in the two-rank parity check, `latent_parity.py --backend`, and an opt-in self-hosted XPU acceptance job.
- Add `tests/sp_diagnose.py`, which attributes a parity failure to communication, SP logic, or device kernel rounding, and report `rel_l2` and cosine similarity in `latent_parity.py`.
- Support XPU DynamicVRAM through ComfyUI-OmniXPU: workers follow rank 0's DynamicVRAM state, run OmniXPU's prestartup before importing PyTorch, and load its adapters after AIMDO device init, failing startup if their active runtime providers differ from rank 0 and logging adapter differences (e.g. preprocessing adapters for nodes that only run on rank 0) as worker warnings. Active XPU DynamicVRAM without the OmniXPU `comfy_aimdo.xpu` provider, or an inherited `native_hook` allocator without active DynamicVRAM, is rejected before workers start. Report workers that exit before the rendezvous immediately, include worker log paths in startup errors, and retain worker exception tracebacks before disconnecting.
- Support the ComfyUI 0.45 H3 API, where `FinalLayer.forward` takes per-step PDD `grids` instead of `shifts` and the output projections may stack `pdd_steps` heads. The capability check accepts either form, the SP forward builds the schedule the running ComfyUI expects, and gathers use the per-head output width. Add a stacked-head (`pdd_steps=3`) two-rank parity case; ComfyUI 0.39 behavior is unchanged.
- After worker cleanup, release rank 0's cached device memory so VAE decode on rank 0 does not compete with blocks left by sampling.
- Fix workers exiting when ComfyUI is idle for longer than `MINIMAX_SP_STARTUP_TIMEOUT` (300 s by default) between prompts: the idle worker sat in a Gloo control broadcast, whose receive deadline then expired. Rank 0 now announces each control command through the TCPStore, and workers wait there without a deadline before entering the broadcast; active control and tensor collectives keep their timeouts.
- Validate XPU T2V end to end on two PCIe-connected Arc Pro B70 cards with ComfyUI 0.39.0, PyTorch 2.14.0+xpu and OmniXPU DynamicVRAM: 1.4× to 1.8× faster sampling than one card, with the gain growing with resolution and length. Other workflows are covered only by CPU/mocked tests.

## 0.1.0

- Add a ComfyUI-discoverable package entrypoint and Comfy Registry metadata.
- Require the MiniMax H3 API introduced at or after ComfyUI commit `8d534945`, enforce it with runtime capability checks, and test against current 0.37 development commit `7fbcfa8b` without claiming an exact release-tag range.
- Cover T2V, first/first+last-frame I2V, reference/AddGuide, masks, PDD, and static Turbo LoRA workflows.
- Retire parallel VAE handover/decode; official INT8 and FP16 VAEs use standard `VAEDecode`, while `MiniMaxH3SPVAEDecode` remains only as a deprecated compatibility alias.
- Add CPU unit/two-rank checks, workflow validation, and opt-in self-hosted GPU parity plus real loader/SPGroup lifecycle acceptance.

Local acceptance on two NVIDIA L20 GPUs against ComfyUI `7fbcfa8b`:

- Real INT8 DiT SP2 parity passed (`video rel=1.199e-6`, audio bit-exact).
- Real loader, Gloo/NCCL group creation, all-reduce/all-to-all startup probe, and clean shutdown passed.
- One-step T2V, first-frame I2V, first+last-frame I2V, AddGuide, and R2V completed through the ComfyUI server with SP2.
- Four-GPU execution and a real Turbo LoRA checkpoint were not available in this environment and remain self-hosted acceptance items.

Breaking scope: the main branch no longer supports the old ComfyUI 0.30 MiniMax API. Use an older repository release with an old ComfyUI checkout if that API is required.
