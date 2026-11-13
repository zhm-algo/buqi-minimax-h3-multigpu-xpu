"""Accelerator backends for the MiniMax-H3 sequence-parallel group.

CUDA keeps the original design: NCCL tensor group, one CUDA_VISIBLE_DEVICES id per
worker. Intel XPU (Arc Pro B60/B70, Data Center GPU Max) uses XCCL/oneCCL and Level
Zero device masks. CPU/Gloo exists only so the full worker lifecycle can be tested
on machines without accelerators.

This module only depends on torch so the worker can load it before ComfyUI.
"""

import logging
import os
import threading
import time
from typing import NamedTuple

import torch
import torch.distributed as dist

XPU_VISIBILITY_MODES = ("isolated", "shared")


class Backend(NamedTuple):
    # Value semantics on purpose: the worker loads this module separately from the package.
    name: str  # torch device type
    dist_backend: str
    label: str
    visible_env: str | None
    # XCCL has no watchdog, so tensor collectives need explicit bounded waits.
    bounded_waits: bool
    # On Arc Pro B70 (PyTorch 2.14 XCCL) a communicator created with new_group under a
    # Gloo default group hangs in its first collective, so XPU makes the tensor group the
    # default and puts the Gloo control plane in the subgroup instead (see init_groups).
    tensor_group_is_default: bool = False


CUDA = Backend("cuda", "nccl", "NCCL", "CUDA_VISIBLE_DEVICES", False)
XPU = Backend("xpu", "xccl", "XCCL", "ZE_AFFINITY_MASK", True, True)
CPU = Backend("cpu", "gloo", "Gloo", None, False)
BACKENDS = {backend.name: backend for backend in (CUDA, XPU, CPU)}


def get_backend(name):
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown MiniMax SP backend {name!r}; expected one of {', '.join(BACKENDS)}") from None


def xpu_is_available():
    xpu = getattr(torch, "xpu", None)
    try:
        return xpu is not None and xpu.is_available()
    except Exception:
        return False


def select_backend(name=None):
    """Pick the tensor backend: MINIMAX_SP_BACKEND, else the accelerator PyTorch exposes."""
    name = (name or os.environ.get("MINIMAX_SP_BACKEND") or "auto").strip().lower()
    if name != "auto":
        return get_backend(name)
    if torch.cuda.is_available():
        return CUDA
    if xpu_is_available():
        return XPU
    raise RuntimeError("MiniMax SP needs CUDA or Intel XPU devices, but PyTorch sees neither; "
                       "install a CUDA or XPU build of PyTorch")


def check_available(backend):
    if backend == XPU:
        if not getattr(dist, "is_xccl_available", lambda: False)():
            raise RuntimeError("this PyTorch build has no XCCL backend; Intel XPU multi-GPU SP needs a Linux "
                               "PyTorch XPU build with XCCL (oneCCL), e.g. torch 2.13+xpu")
    elif backend == CUDA and not dist.is_nccl_available():
        raise RuntimeError("this PyTorch build has no NCCL backend; CUDA multi-GPU SP needs NCCL")


def platform_error(backend, platform):
    """Return why native Windows cannot host the tensor group, or None."""
    if platform != "win32":
        return None
    if backend == CUDA:
        return ("multi-GPU SP needs NCCL, which is not available on native Windows; "
                "run ComfyUI inside WSL2 or on Linux")
    if backend == XPU:
        return ("multi-GPU SP on Intel XPU needs XCCL (oneCCL), which PyTorch only ships for Linux; "
                "run ComfyUI on Linux")
    return None


def xpu_visibility_mode(environ=None):
    environ = os.environ if environ is None else environ
    # oneCCL cannot map ranks whose Level Zero masks differ (all_to_all hangs on B70), so
    # workers keep ComfyUI's visibility by default.
    mode = environ.get("MINIMAX_SP_XPU_VISIBILITY", "shared").strip().lower()
    if mode not in XPU_VISIBILITY_MODES:
        raise ValueError(f"MINIMAX_SP_XPU_VISIBILITY must be one of {', '.join(XPU_VISIBILITY_MODES)}, got {mode!r}")
    return mode


def parse_device_list(value, source, kind="CUDA"):
    picked = [item.strip() for item in value.split(",")]
    if not picked or any(not item for item in picked):
        raise ValueError(f"{source} must be a comma-separated list of {kind} device identifiers")
    if len(set(picked)) != len(picked):
        raise ValueError(f"{source} contains duplicate {kind} devices: {value}")
    return picked


def parse_xpu_ids(value, source):
    picked = parse_device_list(value, source, "XPU")
    bad = [item for item in picked if not item.isdigit()]
    if bad:
        raise ValueError(f"{source} must list whole Level Zero device indices such as 0,1 "
                         f"(sub-devices and other syntax are not supported): {bad}")
    return picked


def _oneapi_selector_ids(value):
    """Device indices from a simple ONEAPI_DEVICE_SELECTOR, or None for all devices."""
    selector = value.strip()
    backend, sep, ids = selector.partition(":")
    if sep and backend.strip().lower() == "level_zero":
        ids = ids.strip()
        if ids.lower() in ("*", "gpu"):
            return None
        if ";" not in ids and "!" not in ids:
            return parse_xpu_ids(ids, "ONEAPI_DEVICE_SELECTOR")
    raise ValueError(f"unsupported ONEAPI_DEVICE_SELECTOR {value!r} for MiniMax SP; use the simple form "
                     "level_zero:0,1 (or set ZE_AFFINITY_MASK=0,1 instead)")


def visible_devices(backend, environ=None):
    """(source, physical ids visible to this process) or (None, None) if unrestricted."""
    environ = os.environ if environ is None else environ
    if backend == CUDA:
        value = environ.get("CUDA_VISIBLE_DEVICES")
        if value is None:
            return None, None
        visible = parse_device_list(value, "CUDA_VISIBLE_DEVICES")
        if visible == ["-1"]:
            raise ValueError("CUDA_VISIBLE_DEVICES exposes no CUDA devices")
        return "CUDA_VISIBLE_DEVICES", visible
    if backend == XPU:
        mask_value = environ.get("ZE_AFFINITY_MASK")
        selector_value = environ.get("ONEAPI_DEVICE_SELECTOR")
        mask = parse_xpu_ids(mask_value, "ZE_AFFINITY_MASK") if mask_value is not None else None
        selected = _oneapi_selector_ids(selector_value) if selector_value is not None else None
        # Unlike CUDA_VISIBLE_DEVICES, neither variable reorders devices: Level Zero and SYCL
        # keep driver enumeration order, so xpu:k is the k-th lowest visible index.
        mask = sorted(mask, key=int) if mask is not None else None
        selected = sorted(selected, key=int) if selected is not None else None
        if selected is None:
            return ("ZE_AFFINITY_MASK", mask) if mask is not None else (None, None)
        if mask is None:
            return "ONEAPI_DEVICE_SELECTOR", selected
        # ONEAPI_DEVICE_SELECTOR indexes the devices left visible by ZE_AFFINITY_MASK.
        try:
            return "ZE_AFFINITY_MASK+ONEAPI_DEVICE_SELECTOR", [mask[int(i)] for i in selected]
        except IndexError:
            raise ValueError(f"ONEAPI_DEVICE_SELECTOR {selector_value!r} selects devices outside "
                             f"ZE_AFFINITY_MASK {mask_value!r}") from None
    return None, None


def current_index(backend):
    if backend == CUDA and torch.cuda.is_available():
        return torch.cuda.current_device()
    if backend == XPU and xpu_is_available():
        return torch.xpu.current_device()
    return None


def device_count(backend):
    if backend == CUDA:
        return torch.cuda.device_count()
    if backend == XPU:
        return torch.xpu.device_count() if xpu_is_available() else 0
    return 0


def required_visible_count(backend, world, environ=None):
    """Devices rank 0 itself must see; isolated XPU workers get their own Level Zero mask."""
    if backend == CPU:
        return 0
    if backend == XPU and xpu_visibility_mode(environ) == "isolated":
        return 1
    return world


def rank0_device(backend):
    index = current_index(backend)
    if backend == CPU:
        return torch.device("cpu")
    return torch.device(backend.name, 0 if index is None else index)


def worker_env(backend, base_env, device_id):
    """(environment, local device index) for a worker bound to physical device_id."""
    env = dict(base_env)
    if backend == CUDA:
        env["CUDA_VISIBLE_DEVICES"] = device_id
        env["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        env["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        return env, 0
    if backend == XPU:
        if xpu_visibility_mode(base_env) == "shared":
            _, visible = visible_devices(XPU, base_env)
            return env, visible.index(device_id) if visible is not None else int(device_id)
        # ZE_AFFINITY_MASK uses driver-level indices, so the worker sees exactly this card
        # as xpu:0. Drop the inherited selector: it would index the masked list.
        env["ZE_AFFINITY_MASK"] = device_id
        env.pop("ONEAPI_DEVICE_SELECTOR", None)
        return env, 0
    return env, 0


def joined_key(rank):
    """TCPStore key a worker sets once it reaches the rendezvous."""
    return f"minimax_sp/joined/{rank}"


def command_key(seq):
    """TCPStore key rank 0 sets before broadcasting control command ``seq``."""
    return f"minimax_sp/command/{seq}"


def wait_for_command(store, seq, max_poll=0.05):
    """Wait, without a deadline, until rank 0 announces control command ``seq``.

    Gloo applies its process-group timeout to a pending receive, so a worker idling in
    the control broadcast between prompts would time out and exit. Polling the store has
    no such deadline and still fails as soon as rank 0's store goes away.
    """
    key = command_key(seq)
    delay = 0.001
    while not store.check([key]):
        time.sleep(delay)
        delay = min(delay * 2, max_poll)


def apply_process_env(backend, environ=None):
    environ = os.environ if environ is None else environ
    if backend == CUDA:
        environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
        environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")


def set_device(backend, index):
    if backend == CUDA:
        torch.cuda.set_device(index)
    elif backend == XPU:
        torch.xpu.set_device(index)


def memory_allocated(device):
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device)
    if device.type == "xpu":
        return torch.xpu.memory_allocated(device)
    return 0


def memory_reserved(device):
    device = torch.device(device)
    if device.type == "cuda":
        return torch.cuda.memory_reserved(device)
    if device.type == "xpu":
        return torch.xpu.memory_reserved(device)
    return 0


def device_memory_used(device):
    """Physical memory in use on the device by all processes, or None if unknown."""
    device = torch.device(device)
    try:
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info(device)
        elif device.type == "xpu":
            free, total = torch.xpu.mem_get_info(device)
        else:
            return None
    except Exception:
        return None
    return total - free


def synchronize(device=None):
    """Wait for the accelerator (the given device, or the current CUDA/XPU device)."""
    if device is not None:
        kind = torch.device(device).type
    elif torch.cuda.is_available():
        kind = "cuda"
    else:
        kind = "xpu" if xpu_is_available() else "cpu"
    if kind == "cuda":
        torch.cuda.synchronize(device)
    elif kind == "xpu":
        torch.xpu.synchronize(device)


def init_groups(backend, store, rank, world, control_timeout, tensor_timeout):
    """Create this process's (control_pg, tensor_pg): Gloo for pickled control messages and
    the backend's collective library for tensors. One of them is the default group."""
    if backend.tensor_group_is_default:
        dist.init_process_group(backend.dist_backend, store=store, rank=rank, world_size=world,
                                timeout=tensor_timeout)
    else:
        dist.init_process_group("gloo", store=store, rank=rank, world_size=world, timeout=control_timeout)
    try:
        if backend.tensor_group_is_default:
            return dist.new_group(backend="gloo", timeout=control_timeout), dist.group.WORLD
        return dist.group.WORLD, dist.new_group(backend=backend.dist_backend, timeout=tensor_timeout)
    except BaseException:
        dist.destroy_process_group()
        raise


def destroy_groups(backend, control_pg, tensor_pg, owns_default):
    """Destroy the subgroup, then the default group if this process created it."""
    if backend is not None and backend.tensor_group_is_default:
        subgroup, default = control_pg, tensor_pg
    else:
        subgroup, default = tensor_pg, control_pg
    try:
        if subgroup is not None:
            dist.destroy_process_group(subgroup)
    finally:
        if owns_default and default is not None:
            dist.destroy_process_group(default)


def bounded_timeout(backend, timeout):
    return timeout if backend is not None and backend.bounded_waits else None


_COLLECTIVE_MONITOR = threading.local()


class CollectiveMonitor:
    """Log failed workers and host-side XCCL stalls without touching a wedged device."""

    _poll_interval = 1.0

    def __init__(self, processes, log_paths, timeout):
        self.processes = processes
        self.log_paths = log_paths
        self.timeout = timeout.total_seconds()
        self.active = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="minimax_sp_monitor", daemon=True)

    def __enter__(self):
        self._previous = getattr(_COLLECTIVE_MONITOR, "current", None)
        _COLLECTIVE_MONITOR.current = self
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        _COLLECTIVE_MONITOR.current = self._previous
        return False

    def stop(self):
        self._stop.set()
        self._thread.join()

    def _check(self):
        for rank, process in enumerate(self.processes, start=1):
            code = process.poll()
            if code is not None:
                logging.critical(
                    "[minimax_sp] worker rank %s exited with code %s; see %s. "
                    "XCCL may still be blocked in the main process. Restart ComfyUI or the container; "
                    "ComfyUI will NOT be terminated automatically.",
                    rank, code, self.log_paths[rank])
                return True
        active = self.active
        if active is not None and time.monotonic() - active[1] >= self.timeout:
            logging.critical(
                "[minimax_sp] XCCL %s exceeded %.1fs, including host-side enqueue time. "
                "Check worker logs: %s. A blocked XCCL call cannot be cancelled safely; "
                "restart ComfyUI or the container. ComfyUI will NOT be terminated automatically.",
                active[0], self.timeout, self.log_paths)
            return True
        return False

    def _run(self):
        while not self._stop.wait(self._poll_interval):
            if self._check():
                return


def stop_collective_monitor():
    monitor = getattr(_COLLECTIVE_MONITOR, "current", None)
    if monitor is not None:
        monitor.stop()


def run_collective(op, *args, timeout=None, **kwargs):
    """Issue a tensor collective; XPU waits bound both GPU work and buffer lifetime."""
    monitor = getattr(_COLLECTIVE_MONITOR, "current", None)
    if monitor is not None:
        monitor.active = (getattr(op, "__name__", type(op).__name__), time.monotonic())
    try:
        if timeout is None:
            return op(*args, **kwargs)
        work = op(*args, async_op=True, **kwargs)
        work.wait(timeout)
        return None
    finally:
        if monitor is not None:
            monitor.active = None


def all_gather_single_op():
    # all_gather_single is new in PyTorch 2.13; older XPU stacks only ship the legacy name.
    return getattr(dist, "all_gather_single", None) or dist.all_gather_into_tensor


# On Arc Pro B70 (PyTorch 2.14 XCCL/oneCCL) device memory that has ever been a
# collective buffer is not returned to the device when PyTorch frees it, not even by
# destroy_process_group; only process exit releases it. Fresh activation buffers
# therefore leak a little more every prompt until DynamicVRAM can no longer fault in
# weights. XPU collectives are staged through one fixed per-process buffer instead.
DEFAULT_XPU_COMM_BUFFER_MB = 512
_COMM_BUFFERS = {}


def comm_buffer_bytes(device):
    """Size of the fixed collective staging buffer for this device; 0 disables staging."""
    device = torch.device(device)
    value = os.environ.get("MINIMAX_SP_COMM_BUFFER_MB")
    if value is None or not value.strip():
        megabytes = DEFAULT_XPU_COMM_BUFFER_MB if device.type == "xpu" else 0
    else:
        try:
            megabytes = float(value)
        except ValueError:
            raise ValueError(f"MINIMAX_SP_COMM_BUFFER_MB must be a number, got {value!r}") from None
        if megabytes < 0:
            raise ValueError("MINIMAX_SP_COMM_BUFFER_MB must not be negative")
    return int(megabytes * 2**20) // 512 * 512


def comm_slots(device, dtype):
    """Two equal (send, receive) staging views of `dtype`, or None when staging is off.

    Every rank derives the size from the same environment, so chunking stays identical.
    """
    device = torch.device(device)
    total = comm_buffer_bytes(device)
    if not total:
        return None
    if device.type == "xpu" and device.index is None:
        device = torch.device("xpu", torch.xpu.current_device())
    key = (device.type, device.index)
    buf = _COMM_BUFFERS.get(key)
    if buf is None or buf.numel() != total:
        buf = _COMM_BUFFERS[key] = torch.empty(total, dtype=torch.uint8, device=device)
    half = total // 2 // 256 * 256
    itemsize = torch.empty((), dtype=dtype).element_size()
    if half < itemsize:
        return None
    return buf[:half].view(dtype), buf[half:2 * half].view(dtype)


def _column_chunks(columns, rows, slot_numel):
    width = max(1, min(columns, slot_numel // max(rows, 1)))
    if width * max(rows, 1) > slot_numel:
        raise RuntimeError(f"MINIMAX_SP_COMM_BUFFER_MB is too small for {rows} rows; increase it")
    for start in range(0, columns, width):
        yield start, min(start + width, columns)


def all_to_all_rows(out, send, out_splits, in_splits, group, timeout=None, max_rows=None):
    """all_to_all_single along rows of [..., C] tensors; leading dims flatten to rows.

    Split sizes count rows. `max_rows` must be identical on every rank (it fixes the
    column chunking when staging); `out` and `send` may be non-contiguous views.
    """
    columns = send.shape[-1]
    rows_in, rows_out = send.numel() // max(columns, 1), out.numel() // max(columns, 1)
    slots = comm_slots(send.device, send.dtype)
    if slots is None:
        source = send.reshape(rows_in, columns)
        direct = out.is_contiguous()
        target = out.view(rows_out, columns) if direct else torch.empty(
            (rows_out, columns), dtype=out.dtype, device=out.device)
        run_collective(dist.all_to_all_single, target, source.contiguous(),
                       output_split_sizes=list(out_splits), input_split_sizes=list(in_splits),
                       group=group, timeout=timeout)
        if not direct:
            out.copy_(target.view(out.shape))
        return out
    send_slot, recv_slot = slots
    rows = max(max_rows or 0, rows_in, rows_out)
    for start, stop in _column_chunks(columns, rows, send_slot.numel()):
        width = stop - start
        staged_send = send_slot[:rows_in * width].view(*send.shape[:-1], width)
        staged_send.copy_(send[..., start:stop])
        staged_recv = recv_slot[:rows_out * width].view(rows_out, width)
        run_collective(dist.all_to_all_single, staged_recv, staged_send.view(rows_in, width),
                       output_split_sizes=list(out_splits), input_split_sizes=list(in_splits),
                       group=group, timeout=timeout)
        out[..., start:stop].copy_(staged_recv.view(*out.shape[:-1], width))
    return out


def all_gather_rows(out, send, group, timeout=None):
    """all_gather_into_tensor of [rows, C] blocks into [world * rows, C]."""
    columns = send.shape[-1]
    slots = comm_slots(send.device, send.dtype)
    if slots is None:
        run_collective(all_gather_single_op(), out, send.contiguous(), group=group, timeout=timeout)
        return out
    send_slot, recv_slot = slots
    rows_in, rows_out = send.numel() // max(columns, 1), out.numel() // max(columns, 1)
    for start, stop in _column_chunks(columns, rows_out, send_slot.numel()):
        width = stop - start
        staged_send = send_slot[:rows_in * width].view(rows_in, width)
        staged_send.copy_(send.reshape(rows_in, columns)[:, start:stop])
        staged_recv = recv_slot[:rows_out * width].view(rows_out, width)
        run_collective(all_gather_single_op(), staged_recv, staged_send, group=group, timeout=timeout)
        out.view(rows_out, columns)[:, start:stop].copy_(staged_recv)
    return out


def broadcast_tensor(tensor, src, group, timeout=None):
    """In-place broadcast of a contiguous tensor, staged in fixed-size pieces on XPU."""
    slots = comm_slots(tensor.device, tensor.dtype)
    if slots is None:
        run_collective(dist.broadcast, tensor, src=src, group=group, timeout=timeout)
        return tensor
    slot = slots[0]
    flat = tensor.view(-1)
    is_source = dist.get_rank(group) == src
    for start in range(0, flat.numel(), slot.numel()):
        stop = min(start + slot.numel(), flat.numel())
        staged = slot[:stop - start]
        if is_source:
            staged.copy_(flat[start:stop])
        run_collective(dist.broadcast, staged, src=src, group=group, timeout=timeout)
        if not is_source:
            flat[start:stop].copy_(staged)
    return tensor
