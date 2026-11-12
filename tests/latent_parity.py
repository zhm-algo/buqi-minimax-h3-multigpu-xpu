"""Numerical parity of the sequence-parallel DiT against the single-GPU forward.

Compares the velocity tensors the DiT actually produces, so nothing is hidden by
the VAE or by lossy video encoding. Runs the official MiniMaxH3Model._forward as
the reference on rank 0 and the all-to-all sequence-parallel forward on all ranks,
then reports max absolute deviation.

    torchrun --nproc_per_node=2 tests/latent_parity.py --unet <file.safetensors>
    torchrun --nproc_per_node=2 tests/latent_parity.py --backend xpu --unet <file.safetensors>
"""
import argparse
import os
from pathlib import Path
import sys
from datetime import timedelta

import torch
import torch.distributed as dist


def configure_imports(comfyui_root):
    if not comfyui_root:
        raise RuntimeError("Set COMFYUI_ROOT or pass --comfyui-root")
    root = Path(comfyui_root).expanduser().resolve()
    if not (root / "comfy/ldm/minimax/model.py").is_file():
        raise RuntimeError(f"Not a ComfyUI checkout: {root}")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_inputs(dit, args, device, dtype):
    torch.manual_seed(1234)
    lat_t, lat_h, lat_w = args.latent_t, args.height // 16, args.width // 16
    video = torch.randn(1, dit.latents_dim, lat_t, lat_h, lat_w,
                        dtype=torch.float32, device=device)
    audio = torch.randn(1, 32, 2, args.audio_t, dtype=torch.float32, device=device)
    context = torch.randn(1, args.text_len, 5120, dtype=dtype, device=device)
    timestep = torch.full((1,), 500.0, dtype=torch.float32, device=device)
    return [video, audio], timestep, context


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfyui-root", default=os.environ.get("COMFYUI_ROOT"))
    ap.add_argument("--unet", required=True)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--latent-t", type=int, default=32)
    ap.add_argument("--audio-t", type=int, default=216)
    ap.add_argument("--text-len", type=int, default=64)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--backend", choices=("auto", "cuda", "xpu"), default="auto",
                    help="auto uses CUDA when available, otherwise Intel XPU (XCCL)")
    args = ap.parse_args()
    configure_imports(args.comfyui_root)

    local_rank = int(os.environ["LOCAL_RANK"])
    backend = args.backend
    if backend == "auto":
        backend = "cuda" if torch.cuda.is_available() else "xpu"
    if backend == "cuda":
        dist.init_process_group("nccl", timeout=timedelta(minutes=40),
                                device_id=torch.device("cuda", local_rank))
        torch.cuda.set_device(local_rank)
    else:
        # Bind the device before ComfyUI import: model_management picks the current XPU.
        torch.xpu.set_device(local_rank)
        dist.init_process_group("xccl", timeout=timedelta(minutes=40))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(backend, local_rank)

    import comfy.model_management
    import comfy.sd
    import folder_paths
    from minimax_sp import sp_forward as spf

    path = folder_paths.get_full_path_or_raise("diffusion_models", args.unet)
    patcher = comfy.sd.load_diffusion_model(path)
    comfy.model_management.load_models_gpu([patcher], force_full_load=True)
    dit = patcher.model.diffusion_model
    dtype = next(p for p in dit.parameters()).dtype

    x, timestep, context = build_inputs(dit, args, device, dtype)

    with torch.no_grad():
        ref = None
        if rank == 0:
            out = dit._forward(x, timestep, context, transformer_options={})
            ref = [t.float().clone() for t in out]
        got = spf.sp_forward(dit, x, timestep, context, {}, None, rank, world, None)
        if rank == 0:
            got = [tensor.float() for tensor in got]

    failed = False
    if rank == 0:
        print(f"world={world} shapes video={tuple(ref[0].shape)} audio={tuple(ref[1].shape)}")
        for label, expected, actual in zip(("video", "audio"), ref, got):
            delta = (actual - expected).abs()
            max_abs = delta.max().item()
            relative = max_abs / max(expected.abs().max().item(), 1e-12)
            rel_l2 = (delta.norm() / expected.norm().clamp_min(1e-12)).item()
            cos = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
            close = torch.allclose(actual, expected, atol=args.atol, rtol=args.rtol)
            failed |= not close
            print(f"  vs 1gpu  all_to_all {label:<5} max_abs={max_abs:.3e} "
                  f"rel={relative:.3e} rel_l2={rel_l2:.3e} cos={cos:.6f} close={close} "
                  f"exact={torch.equal(actual, expected)} nonzero={int((delta > 0).sum())}/{delta.numel()}")
        print("FAIL: latent parity exceeded tolerance" if failed else "PASS: latent parity within tolerance")
        if failed:
            print("  To attribute the drift (communication, SP logic, or kernel rounding), run tests/sp_diagnose.py.")
    dist.destroy_process_group()
    return 1 if rank == 0 and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
