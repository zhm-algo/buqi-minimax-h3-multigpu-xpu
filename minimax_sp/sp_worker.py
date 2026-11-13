"""Worker process for the MiniMax-H3 sequence-parallel group (rank >= 1).

CUDA workers use their own device mask; XPU workers normally share visibility
and bind the assigned logical device. Holds only the DiT: no text encoder, no
VAE. Parks on a gloo broadcast waiting for work.
"""

import argparse
import importlib.util
import json
import logging
import os
import sys
from datetime import timedelta

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_local_module(name):
    # Importing minimax_sp runs its __init__, which imports ComfyUI and torch; the device must be
    # bound (and OmniXPU providers routed) before that, so load standalone helpers by path.
    module_name = f"minimax_sp_worker_{name}"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(PACKAGE_DIR, f"{name}.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


def load_device_module():
    return load_local_module("device")


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--unet", required=True)
    ap.add_argument("--weight-dtype", default="default")
    ap.add_argument("--comfy-root", required=True)
    ap.add_argument("--startup-timeout", type=int, required=True)
    ap.add_argument("--collective-timeout", type=int, required=True)
    ap.add_argument("--runtime-options", default="{}")
    ap.add_argument("--omnixpu", default="null")
    ap.add_argument("--backend", default="cuda")
    ap.add_argument("--local-device", type=int, default=0)
    return ap.parse_args(argv)


def prepare_runtime(a):
    """Apply rank 0's ComfyUI flags and runtime providers; runs before this process imports torch."""
    sys.path.insert(0, a.comfy_root)
    sys.path.insert(0, os.path.dirname(PACKAGE_DIR))
    from comfy.cli_args import args, enables_dynamic_vram
    for key, value in json.loads(a.runtime_options).items():
        setattr(args, key, value)
    omnixpu_state = json.loads(a.omnixpu)
    if omnixpu_state is not None:
        load_local_module("omnixpu").run_prestartup(omnixpu_state)
    elif enables_dynamic_vram():
        import comfy_aimdo.control
        headroom = None if args.reserve_vram is None else int(args.reserve_vram * 2**30)
        comfy_aimdo.control.init(simple_vram_headroom=headroom, nvml_pressure=not args.disable_nvml_pressure)
    return omnixpu_state


if __name__ == "__main__":
    # OmniXPU's Linux allocator takeover refuses to run once PyTorch is imported.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    WORKER_ARGS = parse_args()
    try:
        WORKER_OMNIXPU = prepare_runtime(WORKER_ARGS)
    except Exception:
        logging.exception("MiniMax SP worker rank %s failed before importing PyTorch", WORKER_ARGS.rank)
        raise SystemExit(1)

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402


def worker_forward(meta, previous_operation_id, rank, world, device, control_pg, tensor_pg,
                   dit, patches, spg, spf, collective_timeout=None):
    expected = spg.forward_status(meta)
    tensors = {}
    try:
        spg.validate_operation_id(meta["operation_id"], previous_operation_id)
        patches.require_ready()
        video = spg.alloc_from_meta(meta["video"], device)
        audio = spg.alloc_from_meta(meta["audio"], device)
        timestep = spg.alloc_from_meta(meta["timestep"], device)
        context = spg.alloc_from_meta(meta["context"], device)
        denoise_mask = spg.alloc_from_meta(meta["denoise_mask"], device)
        audio_denoise_mask = spg.alloc_from_meta(meta["audio_denoise_mask"], device)
        tensors["tags"] = spg.alloc_from_meta(meta["tags"], device)
        tensors["cond_video"] = [spg.alloc_from_meta(m, device) for m in meta["cond_video"]]
        tensors["cond_audio"] = [spg.alloc_from_meta(m, device) for m in meta["cond_audio"]]
        tensors["sample_sigmas"] = spg.alloc_from_meta(meta["sample_sigmas"], device)
        local_status = expected
    except Exception as exc:
        local_status = spg.forward_status(meta, ready=False, error=exc)

    replies = [None] * world
    dist.all_gather_object(replies, local_status, group=control_pg)
    spg.validate_forward_status(expected, replies)

    for tensor in [video, audio, timestep, context, tensors["tags"],
                   *tensors["cond_video"], *tensors["cond_audio"], tensors["sample_sigmas"],
                   denoise_mask, audio_denoise_mask]:
        if tensor is not None:
            # Same device module as sp_forward, so both share one staging buffer.
            spg.spdev.broadcast_tensor(tensor, 0, tensor_pg)

    payload = spg.payload_from_meta(meta, tensors)
    options = dict(meta["options"])
    if tensors["sample_sigmas"] is not None:
        options["sample_sigmas"] = tensors["sample_sigmas"]
    with torch.no_grad():
        spf.sp_forward(dit, [video, audio], timestep, context, options,
                       payload, rank, world, tensor_pg, denoise_mask, audio_denoise_mask,
                       collective_timeout=collective_timeout)
    return meta["operation_id"]


def main(a, omnixpu_state=None):
    spdev = load_device_module()
    backend = spdev.get_backend(a.backend)
    from comfy.cli_args import args, enables_dynamic_vram

    store = None
    control_pg = None
    tensor_pg = None
    owns_default_pg = False
    try:
        spdev.apply_process_env(backend)
        store = dist.TCPStore("127.0.0.1", a.port, a.world, False,
                              timeout=timedelta(seconds=a.startup_timeout), wait_for_workers=False)
        store.set(spdev.joined_key(a.rank), "1")
        spdev.set_device(backend, a.local_device)
        collective_timeout = timedelta(seconds=a.collective_timeout)
        control_pg, tensor_pg = spdev.init_groups(
            backend, store, a.rank, a.world, timedelta(seconds=a.startup_timeout), collective_timeout)
        owns_default_pg = True
        bounded_timeout = spdev.bounded_timeout(backend, collective_timeout)

        import comfy.model_management
        import comfy.sd
        if enables_dynamic_vram():
            import comfy_aimdo.control
            import comfy.memory_management
            import comfy.model_patcher
            if not comfy_aimdo.control.init_devices([(a.local_device, int(args.vram_headroom * 2**30))]):
                raise RuntimeError("worker could not initialize native DynamicVRAM")
            # Same worker-local startup selection as main.py, not a class rewrite.
            comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
            comfy.memory_management.aimdo_enabled = True
        # ComfyUI loads custom nodes after DynamicVRAM device init; OmniXPU adapters follow suit.
        if omnixpu_state is not None:
            load_local_module("omnixpu").load_custom_node(omnixpu_state)
        import folder_paths
        from minimax_sp import sp_forward as spf
        from minimax_sp import sp_group as spg
        from minimax_sp.sp_patches import PatchManifest, WorkerPatches

        model_options = {}
        if a.weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif a.weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif a.weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        path = a.unet if os.path.isabs(a.unet) else folder_paths.get_full_path_or_raise("diffusion_models", a.unet)
        stream = os.environ.get("MINIMAX_SP_STREAM") == "1"
        logging.info(f"rank {a.rank}: loading {a.unet} (stream_weights={stream})")
        patcher = comfy.sd.load_diffusion_model(path, model_options=model_options)
        comfy.model_management.load_models_gpu([patcher], force_full_load=not stream)
        dit = patcher.model.diffusion_model
        device = patcher.load_device
        logging.info(f"rank {a.rank}: loaded on {device}, "
                     f"{spdev.memory_allocated(device) / 2**30:.1f} GiB allocated")

        loaded = [None] * a.world
        dist.all_gather_object(loaded, {"loaded": True}, group=control_pg)
        if any(reply != {"loaded": True} for reply in loaded):
            raise RuntimeError(f"MiniMax SP worker load failed: {loaded}")
        spg.startup_probe(a.rank, a.world, tensor_pg, device, bounded_timeout)
        logging.info(f"rank {a.rank}: ready")

        steps = 0
        previous_operation_id = 0
        patches = WorkerPatches(patcher, a.rank)
        command_seq = 0
        while True:
            # Idle time between prompts is unbounded; only enter Gloo once a command is sent.
            command_seq += 1
            spdev.wait_for_command(store, command_seq)
            box = [None]
            dist.broadcast_object_list(box, src=0, group=control_pg)
            meta = box[0]
            if meta is None or meta.get("op") == "shutdown":
                logging.info(f"rank {a.rank}: shutdown after {steps} forwards")
                break

            if meta.get("op") in ("prepare", "patches", "cleanup"):
                try:
                    if meta["op"] == "prepare":
                        reply = patches.prepare(meta["memory_required"], meta["version"])
                    elif meta["op"] == "cleanup":
                        reply = patches.cleanup()
                        logging.info("rank %s: sampler cleanup complete", a.rank)
                    else:
                        manifest = PatchManifest.from_message(meta)
                        reply = patches.apply(manifest)
                        logging.info("rank %s: applied %d patched keys, sha256=%s",
                                     a.rank, len(manifest.keys), manifest.sha256)
                except Exception as exc:
                    logging.exception("worker %s failed", meta["op"])
                    reply = {"rank": a.rank, "error": str(exc)}
                dist.all_gather_object([None] * a.world, reply, group=control_pg)
                continue
            previous_operation_id = worker_forward(
                meta, previous_operation_id, a.rank, a.world, device, control_pg, tensor_pg,
                dit, patches, spg, spf, bounded_timeout)
            steps += 1
    except Exception as exc:
        # Record the cause before closing Gloo wakes rank 0's worker cleanup.
        logging.exception("MiniMax SP worker rank %s failed", a.rank)
        raise SystemExit(1) from exc
    finally:
        try:
            if dist.is_initialized():
                spdev.destroy_groups(backend, control_pg, tensor_pg, owns_default_pg)
        finally:
            control_pg = tensor_pg = None
            store = None


if __name__ == "__main__":
    try:
        main(WORKER_ARGS, WORKER_OMNIXPU)
    except Exception:
        logging.exception("MiniMax SP worker failed")
        raise SystemExit(1)
