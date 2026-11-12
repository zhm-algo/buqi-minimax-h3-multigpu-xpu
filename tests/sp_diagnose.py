"""Attribute a latent_parity failure to communication or to device kernel numerics.

Checks the SP building blocks layer by layer, from exact data movement up to the full DiT:

  [1] collectives  the SP all_to_all / all_gather calls with the forward's layouts, bit-for-bit
  [2] tiny model   a random two-layer MiniMax H3 on the target device, fp32 and bf16
  [3] real model   (--unet) reference repeatability, cross-card reference, the model's sensitivity
                   to +-1 eps noise, row/head shape invariance of block-0 kernels, one SP attention
                   layer, the full forward, and the full forward again in float32 at a small shape

    torchrun --standalone --nproc-per-node=2 tests/sp_diagnose.py --backend xpu \
        --unet minimax_h3_fl2va_pruned_int8_convrot.safetensors

Exits nonzero only for a communication or SP-logic failure ([1] or tiny fp32 in [2]).
"""
import argparse
from datetime import timedelta
import math
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist


def init_distributed(backend, local_rank):
    if backend == "auto":
        backend = "cuda" if torch.cuda.is_available() else "xpu"
    if backend == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", timeout=timedelta(minutes=40), device_id=device)
    elif backend == "xpu":
        device = torch.device("xpu", local_rank)
        # Bind the device before ComfyUI import: model_management picks the current XPU.
        torch.xpu.set_device(device)
        dist.init_process_group("xccl", timeout=timedelta(minutes=40))
    else:
        device = torch.device("cpu")
        dist.init_process_group("gloo", timeout=timedelta(minutes=40))
    return backend, device


def configure_imports(comfyui_root, cpu):
    if not comfyui_root:
        raise RuntimeError("Set COMFYUI_ROOT or pass --comfyui-root")
    root = Path(comfyui_root).expanduser().resolve()
    if not (root / "comfy/ldm/minimax/model.py").is_file():
        raise RuntimeError(f"Not a ComfyUI checkout: {root}")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if cpu:
        from comfy.cli_args import args
        args.cpu = True


class Diff:
    """Mergeable actual-vs-expected error summary."""

    def __init__(self, values=None):
        # max_abs, sum d^2, sum e^2, sum a*e, sum a^2, max |e|, differing, numel
        self.v = list(values) if values is not None else [0.0] * 8

    @classmethod
    def of(cls, actual, expected):
        a = actual.detach().float().flatten()
        e = expected.detach().float().flatten()
        d = a - e
        if not d.numel():
            return cls()
        return cls([d.abs().max().item(), (d * d).sum().item(), (e * e).sum().item(), (a * e).sum().item(),
                    (a * a).sum().item(), e.abs().max().item(), float((d != 0).sum().item()), float(d.numel())])

    def merge(self, other):
        x, y = self.v, other.v
        return Diff([max(x[0], y[0]), x[1] + y[1], x[2] + y[2], x[3] + y[3], x[4] + y[4],
                     max(x[5], y[5]), x[6] + y[6], x[7] + y[7]])

    @property
    def exact(self):
        return self.v[6] == 0

    @property
    def rel_l2(self):
        return math.sqrt(self.v[1] / max(self.v[2], 1e-30))

    def __str__(self):
        max_abs, _, ee, ae, aa, emax, nd, n = self.v
        cos = ae / max(math.sqrt(aa * ee), 1e-30)
        return (f"max_abs={max_abs:.3e} rel_max={max_abs / max(emax, 1e-30):.3e} rel_l2={self.rel_l2:.3e} "
                f"cos={cos:.6f} differing={nd / max(n, 1):.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfyui-root", default=os.environ.get("COMFYUI_ROOT"))
    ap.add_argument("--backend", choices=("auto", "cuda", "xpu", "cpu"), default="auto",
                    help="auto uses CUDA when available, otherwise Intel XPU; cpu (Gloo) checks this script")
    ap.add_argument("--unet", help="real DiT checkpoint for section [3]; omit to run [1] and [2] only")
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--latent-t", type=int, default=32)
    ap.add_argument("--audio-t", type=int, default=216)
    ap.add_argument("--text-len", type=int, default=64)
    ap.add_argument("--collective-timeout", type=float, default=600.0,
                    help="seconds for the XPU bounded waits, as MINIMAX_SP_COLLECTIVE_TIMEOUT in production")
    args = ap.parse_args()

    backend, device = init_distributed(args.backend, int(os.environ.get("LOCAL_RANK", 0)))
    configure_imports(args.comfyui_root, backend == "cpu")
    rank, world = dist.get_rank(), dist.get_world_size()

    import comfy.ldm.common_dit
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model, PackedLayout, rope_rotation_table
    from minimax_sp import device as spdev
    from minimax_sp import sp_forward as spf
    from minimax_sp import sp_group as spg

    # XPU production waits on the final gather with a bound; exercise the same path here.
    timeout = timedelta(seconds=args.collective_timeout) if backend == "xpu" else None
    gather = spdev.all_gather_single_op()

    def log(msg=""):
        if rank == 0:
            print(msg, flush=True)

    def gather_rows(values):
        local = torch.tensor(values, dtype=torch.float32, device=device)
        out = torch.empty(world * local.numel(), dtype=torch.float32, device=device)
        gather(out, local)
        return out.view(world, -1).cpu().tolist()

    def all_ranks(flag):
        return all(row[0] > 0 for row in gather_rows([1.0 if flag else 0.0]))

    def merged(diff):
        rows = gather_rows(diff.v)
        total = Diff(rows[0])
        for row in rows[1:]:
            total = total.merge(Diff(row))
        return total

    def ranges(seq_len):
        return [(c.start, c.stop) for c in (spf.SPContext(r, world, None, seq_len) for r in range(world))]

    def cpu_randn(seed, *shape):
        return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))

    log(f"backend={backend} world={world} torch={torch.__version__} device={device}")
    if backend == "xpu":
        log(f"xpu device: {torch.xpu.get_device_name(device)}")

    # [1] -----------------------------------------------------------------------------
    def check_collectives(shapes):
        ok = True
        for seq_len, heads, dim in shapes:
            hp = heads // world
            ctx = spf.SPContext(rank, world, None, seq_len, timeout)
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                full = cpu_randn(seq_len * 1009 + heads * 31 + dim, seq_len, heads, dim).to(dtype).to(device)
                local = full[ctx.start:ctx.stop]
                mine = full[:, rank * hp:(rank + 1) * hp]
                # SP hands the all-to-all a strided q/k/v view of the fused qkv buffer.
                fused = torch.cat([local.reshape(ctx.local, heads * dim)] * 3, dim=1)
                strided = fused[:, heads * dim:2 * heads * dim].view(ctx.local, heads, dim)
                results = {
                    "a2a seq->heads": torch.equal(spf.a2a_gather_seq_scatter_heads(local.contiguous(), ctx), mine),
                    "a2a seq->heads (strided input)": torch.equal(spf.a2a_gather_seq_scatter_heads(strided, ctx), mine),
                    "a2a heads->seq": torch.equal(
                        spf.a2a_scatter_seq_gather_heads(mine.reshape(seq_len, hp * dim), ctx),
                        local.reshape(ctx.local, heads * dim)),
                }
                if dtype == torch.float32:
                    rows = full[:, 0, :7].contiguous()
                    got = spf._gather_rows(rows[ctx.start:ctx.stop], ctx.splits, ctx, 7, device)
                    results["all_gather rows" + (" (bounded wait)" if timeout else "")] = torch.equal(got, rows)
                for name, good in results.items():
                    good = all_ranks(good)
                    ok &= good
                    log(f"  S={seq_len:<6} H={heads:<3} D={dim:<4} {str(dtype)[6:]:<9} {name:<32} "
                        f"{'exact' if good else 'MISMATCH'}")
        return ok

    # [2] -----------------------------------------------------------------------------
    def tiny_parity(dtype):
        torch.manual_seed(17)
        model = MiniMaxH3Model(hidden_size=64, num_layers=2, token_refiner_num_layers=1,
                               num_attention_heads=2 * world, attention_head_dim=128, ffn_hidden_size=128,
                               text_dim=64, timestep_input_dim=16, time_embed_hidden_size=64,
                               time_embed_dim=32, dtype=dtype, device="cpu", operations=comfy.ops.manual_cast)
        model.requires_grad_(False)
        for p in model.parameters():
            p.uniform_(-0.05, 0.05)
        model.rope.inv_freq.fill_(0.01)
        model.to(device)
        x = [cpu_randn(1, 1, 24, 2, 4, 4).to(device), cpu_randn(2, 1, 32, 2, 3).to(device)]
        t = torch.tensor([500.0], device=device)
        context = cpu_randn(3, 1, 3, 64).to(dtype).to(device)
        image = cpu_randn(4, 1, 24, 1, 4, 4).to(device)
        audio = cpu_randn(5, 1, 32, 2, 2).to(device)
        cases = {
            "t2v": {},
            "reference": {"refs": [{"kind": "image", "latent_h": 4, "latent_w": 4}, {"kind": "audio", "ref_audio_t": 2}],
                          "cond_video_latents": [image], "cond_audio_latents": [audio]},
        }
        worst = Diff()
        with torch.no_grad():
            for name, payload in cases.items():
                ref = model._forward(x, t, context, transformer_options={}, minimax_payload=payload) if rank == 0 else None
                meta = spg.build_meta(x, t, context, {}, payload)
                wire = spg.payload_from_meta(meta, {"tags": None, "cond_video": payload.get("cond_video_latents", []),
                                                    "cond_audio": payload.get("cond_audio_latents", [])})
                got = spf.sp_forward(model, x, t, context, {}, wire, rank, world, None, collective_timeout=timeout)
                if rank == 0:
                    for label, actual, expected in zip(("video", "audio"), got, ref):
                        diff = Diff.of(actual, expected)
                        worst = worst if worst.rel_l2 >= diff.rel_l2 else diff
                        log(f"  {str(dtype)[6:]:<9} {name:<10} {label:<5} {diff}")
        return worst

    # [3] -----------------------------------------------------------------------------
    def kitchen_backends():
        try:
            import comfy_kitchen as ck
        except ImportError:
            return "not installed"
        return ", ".join(f"{name}={'on' if info.get('available') and not info.get('disabled') else 'off'}"
                         for name, info in ck.list_backends().items())

    def real_model():
        import comfy.model_management
        import comfy.sd
        import folder_paths

        path = folder_paths.get_full_path_or_raise("diffusion_models", args.unet)
        patcher = comfy.sd.load_diffusion_model(path)
        comfy.model_management.load_models_gpu([patcher], force_full_load=True)
        dit = patcher.model.diffusion_model
        dtype = next(p for p in dit.parameters()).dtype
        attn = dit.blocks[0].attn
        layout_cls = getattr(attn.qkv_proj.weight, "layout_cls", None)
        log(f"  model={args.unet} param dtype={dtype} blocks={len(dit.blocks)} "
            f"qkv weight={layout_cls.__name__ if layout_cls else attn.qkv_proj.weight.dtype}")
        log(f"  comfy_kitchen backends: {kitchen_backends()}")
        lat_h, lat_w = args.height // 16, args.width // 16

        def inputs(latent_t, cdtype):
            # Same draws as tests/latent_parity.py.
            torch.manual_seed(1234)
            video = torch.randn(1, dit.latents_dim, latent_t, lat_h, lat_w, dtype=torch.float32, device=device)
            audio = torch.randn(1, 32, 2, args.audio_t, dtype=torch.float32, device=device)
            context = torch.randn(1, args.text_len, dit.condition_proj.in_features, dtype=cdtype, device=device)
            timestep = torch.full((1,), 500.0, dtype=torch.float32, device=device)
            return [video, audio], timestep, context

        def reference(x, timestep, context, options=None):
            return [t.float().clone() for t in dit._forward(x, timestep, context, transformer_options=options or {})]

        def sequence_parallel(x, timestep, context):
            out = spf.sp_forward(dit, x, timestep, context, {}, None, rank, world, None, collective_timeout=timeout)
            return [t.float() for t in out] if rank == 0 else None

        def eps_noise_before_block0():
            """Reference patch: +-1 eps relative noise on the hidden state entering block 0."""
            def block0(block_args, extra):
                img = block_args["img"]
                step = torch.randint(-1, 2, img.shape, dtype=torch.int8,
                                     generator=torch.Generator().manual_seed(11)).to(device)
                noisy = img.float() * (1 + step.float() * torch.finfo(img.dtype).eps)
                return extra["original_block"](dict(block_args, img=noisy.to(img.dtype)))
            return {"patches_replace": {"dit": {("double_block", 0): block0}}}

        def row_split(fn, *tensors):
            full = fn(*tensors)
            diff = Diff()
            for a, b in ranges(full.shape[0]):
                diff = diff.merge(Diff.of(fn(*(t[a:b] for t in tensors)), full[a:b]))
            return diff

        x, timestep, context = inputs(args.latent_t, dtype)
        cdtype = context.dtype
        padded = comfy.ldm.common_dit.pad_to_patch_size(x[0], dit.patch_size)
        layout = PackedLayout(args.text_len, *padded.shape[2:5], args.audio_t)
        seq_len = layout.seq_len
        heads, dim, hp = attn.heads, attn.head_dim, attn.heads // world
        log(f"  seq_len={seq_len} heads={heads} head_dim={dim} compute dtype={cdtype} splits={ranges(seq_len)}")
        summary = {}

        log("\n[3a] collectives at the real attention shape")
        summary["collectives"] = check_collectives([(seq_len, heads, dim)])

        with torch.no_grad():
            log("\n[3b] reference repeatability, cross-card reference, and the model's rounding sensitivity")
            ref = reference(x, timestep, context)
            again = reference(x, timestep, context)
            noisy = reference(x, timestep, context, eps_noise_before_block0())
            for i, label in enumerate(("video", "audio")):
                repeat = merged(Diff.of(again[i], ref[i]))
                log(f"  run-to-run (worst card)        {label:<5} {repeat}")
                summary[f"repeat_{label}"] = repeat
                rows = torch.empty(world * ref[i].numel(), dtype=torch.float32, device=device)
                gather(rows, ref[i].contiguous().view(-1))
                rows = rows.view(world, *ref[i].shape)
                for r in range(1, world):
                    log(f"  card {r} vs card 0              {label:<5} {Diff.of(rows[r], rows[0])}")
                floor = merged(Diff.of(noisy[i], ref[i]))
                log(f"  +-1 eps noise before block 0   {label:<5} {floor}")
                summary[f"floor_{label}"] = floor
            del again, noisy, rows

            log("\n[3c] kernel shape invariance on one card (no communication; nonzero -> SP cannot be bit-exact)")
            if rank == 0:
                h = cpu_randn(7, seq_len, dit.hidden_size).to(cdtype).to(device)
                log(f"  qkv_proj  rows split as SP vs full M      {row_split(attn.qkv_proj, h)}")
                inner = cpu_randn(8, seq_len, heads * dim).to(cdtype).to(device)
                log(f"  out_proj  rows split as SP vs full M      {row_split(attn.out_proj, inner)}")
                w = (cpu_randn(9, dit.hidden_size, dit.hidden_size) / dit.hidden_size ** 0.5).to(cdtype).to(device)
                plain = row_split(lambda t: torch.nn.functional.linear(t, w), h)
                log(f"  {'plain ' + str(cdtype)[6:] + ' linear rows split':<42}{plain}")
                if dit.hidden_size % 256 == 0:
                    rot = (cpu_randn(10, 256, 256) / 16).to(cdtype).to(device)
                    log(f"  ConvRot-style [M,G,256]@[256,256] split   "
                        f"{row_split(lambda t: torch.matmul(t.view(t.shape[0], -1, 256), rot), h)}")
                q, k, v = (cpu_randn(11 + i, seq_len, heads, dim).to(cdtype).to(device) for i in range(3))

                def attention(q_, k_, v_, n):
                    return spf.run_optimized_attention(attn, q_.transpose(0, 1).unsqueeze(0), k_.transpose(0, 1).unsqueeze(0),
                                                       v_.transpose(0, 1).unsqueeze(0), n, {})
                full = attention(q, k, v, heads)
                log(f"  attention repeat                          {Diff.of(attention(q, k, v, heads), full)}")
                contiguous = spf.run_optimized_attention(
                    attn, q.transpose(0, 1).contiguous().unsqueeze(0), k.transpose(0, 1).contiguous().unsqueeze(0),
                    v.transpose(0, 1).contiguous().unsqueeze(0), heads, {})
                log(f"  attention strided vs contiguous input     {Diff.of(contiguous, full)}")
                head = Diff()
                for r in range(world):
                    s = slice(r * hp, (r + 1) * hp)
                    sub = attention(q[:, s].contiguous(), k[:, s].contiguous(), v[:, s].contiguous(), hp)
                    head = head.merge(Diff.of(sub, full[..., r * hp * dim:(r + 1) * hp * dim]))
                log(f"  {f'attention {hp}/{heads} heads vs all heads':<42}{head}")
                del h, inner, w, q, k, v, full, contiguous, sub

            log("\n[3d] one SP attention layer (real all-to-all) vs the same layer on one card")
            h = cpu_randn(7, seq_len, dit.hidden_size).to(cdtype).to(device)
            rope = rope_rotation_table(dit.rope_freqs(layout.position_ids, device), cdtype)
            expected = attn(h, rope, transformer_options={})
            ctx = spf.SPContext(rank, world, None, seq_len, timeout)
            local = spf.sp_attention(attn, h[ctx.start:ctx.stop], rope[:, ctx.start:ctx.stop].contiguous(), ctx, {})
            log(f"  SP attention layer                        {merged(Diff.of(local, expected[ctx.start:ctx.stop]))}")
            del h, rope, expected, local

            log("\n[3e] full SP forward vs reference (as latent_parity)")
            got = sequence_parallel(x, timestep, context)
            if rank == 0:
                for i, label in enumerate(("video", "audio")):
                    summary[f"sp_{label}"] = Diff.of(got[i], ref[i])
                    log(f"  SP vs reference                {label:<5} {summary[f'sp_{label}']}")
            del got, ref

            small_t = min(args.latent_t, 8)
            log(f"\n[3f] SP drift by compute dtype at latent_t={small_t} (much smaller in float32 -> rounding, not a bug)")
            for run_dtype in dict.fromkeys((cdtype, torch.float32)):
                name = str(run_dtype)[6:]
                try:
                    xs, ts, cs = inputs(small_t, run_dtype)
                    expected = reference(xs, ts, cs) if rank == 0 else None
                    error = None
                except Exception as e:  # e.g. a kernel without float32 support on this device
                    error = f"{type(e).__name__}: {e}"
                if not all_ranks(error is None):
                    log(f"  {name:<9} skipped, reference failed: {error}")
                    continue
                try:
                    got = sequence_parallel(xs, ts, cs)
                except Exception as e:
                    log(f"  {name:<9} skipped, SP failed: {type(e).__name__}: {e}")
                    continue
                if rank == 0:
                    for i, label in enumerate(("video", "audio")):
                        summary[f"small_{name}_{label}"] = Diff.of(got[i], expected[i])
                        log(f"  {name:<9} SP vs reference      {label:<5} {summary[f'small_{name}_{label}']}")
        return summary

    failed = False
    log("\n[1] collectives (bit-exact expected)")
    coll_ok = check_collectives([(7, 4 * world, 8), (4097, 8 * world, 128)])
    failed |= not coll_ok

    log("\n[2] tiny random MiniMax H3 on this device: SP vs single-card _forward")
    tiny32 = tiny_parity(torch.float32)
    tiny16 = tiny_parity(torch.bfloat16)
    tiny_ok = all_ranks(rank != 0 or tiny32.rel_l2 < 1e-4)
    failed |= not tiny_ok

    summary = {}
    if args.unet:
        log("\n[3] real model")
        summary = real_model()
        failed |= not summary["collectives"]

    log("\nsummary")
    log(f"  collectives exact:            {'yes' if coll_ok and summary.get('collectives', True) else 'NO -> communication bug'}")
    log(f"  tiny fp32 SP parity:          {'ok' if tiny_ok else 'FAILED -> SP logic/device bug'} (rel_l2={tiny32.rel_l2:.1e})")
    log(f"  tiny bf16 SP drift:           rel_l2={tiny16.rel_l2:.1e}")
    if rank == 0 and summary:
        def pair(key):
            return f"{summary[key + '_video'].rel_l2:.1e} / {summary[key + '_audio'].rel_l2:.1e}"
        log(f"  reference run-to-run:         rel_l2={pair('repeat')}  (video / audio)")
        log(f"  SP vs reference:              rel_l2={pair('sp')}")
        log(f"  +-1 eps noise floor:          rel_l2={pair('floor')}")
        for name in dict.fromkeys(k.split("_")[1] for k in summary if k.startswith("small_")):
            log(f"  small-shape SP, {name:<9}     rel_l2={pair('small_' + name)}")
    log("  Exact collectives and a passing tiny fp32 rule out communication and SP logic. Nonzero [3c] rows mean\n"
        "  this device's kernels depend on the row or head count, so SP cannot be bit-exact; SP drift near the\n"
        "  +-1 eps floor that shrinks in float32 ([3f]) is rounding noise amplified by depth and quantization.")
    dist.destroy_process_group()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
