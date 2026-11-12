"""Replay ComfyUI-OmniXPU's process setup in MiniMax SP Intel XPU workers.

Rank 0 is a normal ComfyUI process: OmniXPU's prestartup script routes comfy_aimdo
and comfy_kitchen to their XPU runtime providers before PyTorch loads, and its
custom node later installs XPU adapters (DynamicVRAM staging/writable weights,
INT8/H3 kernels, attention, ...). Workers start from a bare interpreter, so they
repeat both stages in ComfyUI's order. Providers must match rank 0; adapter
differences are only logged.

Torch-free on purpose: the worker loads this file by path before importing
PyTorch, which OmniXPU's Linux allocator takeover requires.
"""

import importlib.util
import logging
import os
import sys

# Module names chosen by OmniXPU's prestartup_script.py and __init__.py.
RUNTIME_MODULE = "_comfyui_omnixpu_runtime_bootstrap"
PACKAGE = "ComfyUI-OmniXPU"
AIMDO_PROVIDER = "comfy_aimdo.xpu"


def _root(module):
    path = getattr(module, "__file__", None)
    return os.path.dirname(os.path.abspath(path)) if path else None


def active_providers(runtime_state):
    providers = (runtime_state or {}).get("providers") or {}
    return sorted(name for name, state in providers.items() if (state or {}).get("status") == "active")


def applied_adapters(patch_status):
    return sorted(item["name"] for item in patch_status or () if item.get("status") == "applied")


def rank0_state(modules=None):
    """OmniXPU setup already applied to this ComfyUI process, or None without OmniXPU."""
    modules = sys.modules if modules is None else modules
    runtime = modules.get(RUNTIME_MODULE)
    node = modules.get(PACKAGE)
    root = _root(runtime) or _root(node)
    if root is None:
        return None
    patches = modules.get(PACKAGE + ".patches")
    return {
        "root": root,
        "providers": active_providers(runtime.get_state()) if runtime is not None else None,
        "custom_node": node is not None,
        "adapters": applied_adapters(patches.get_status()) if patches is not None else None,
    }


def adapter_differences(expected, patch_status):
    """Adapters applied on rank 0 but not here (with this worker's status/reason), and vice versa."""
    status = {item.get("name"): item for item in patch_status or ()}
    applied = applied_adapters(patch_status)
    missing = []
    for name in expected:
        if name not in applied:
            item = status.get(name) or {}
            missing.append(f"{name} ({item.get('status', 'absent')}: {item.get('reason') or '-'})")
    return missing, [name for name in applied if name not in expected]


def provider_active(state, provider=AIMDO_PROVIDER):
    return bool(state) and provider in (state.get("providers") or ())


def _require_same(what, expected, got):
    if got != expected:
        raise RuntimeError(f"OmniXPU {what} on this worker differ from rank 0: rank0={expected} worker={got}")


def run_prestartup(state):
    """Run OmniXPU's prestartup like ComfyUI main.py; must precede the worker's torch import."""
    if not state or state.get("providers") is None:
        return
    # main.py imports the official module first; OmniXPU overlays it with the XPU provider.
    # Its official init() is skipped: on XPU it would only assume NVIDIA, and the provider
    # accepts a pristine module.
    import comfy_aimdo.control  # noqa: F401
    script = os.path.join(state["root"], "prestartup_script.py")
    spec = importlib.util.spec_from_file_location(os.path.splitext(script)[0], script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = sys.modules.get(RUNTIME_MODULE)
    _require_same("runtime providers", state["providers"],
                  active_providers(runtime.get_state()) if runtime is not None else None)


def load_custom_node(state):
    """Load the OmniXPU custom node like ComfyUI nodes.py, after DynamicVRAM device init."""
    if not state or not state.get("custom_node"):
        return
    root = state["root"]
    name = root.replace(".", "_x_")
    spec = importlib.util.spec_from_file_location(name, os.path.join(root, "__init__.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if state.get("adapters") is not None:
        patches = sys.modules.get(PACKAGE + ".patches")
        missing, extra = adapter_differences(state["adapters"], patches.get_status() if patches is not None else ())
        # Not fatal: some adapters target nodes that only run on rank 0 (e.g. SeedVR2/VAE
        # preprocessing needs built-in nodes the DiT-only worker never loads).
        if missing or extra:
            logging.warning("[minimax_sp] OmniXPU adapters differ from rank 0; not applied here: %s; "
                            "applied only here: %s", ", ".join(missing) or "-", ", ".join(extra) or "-")
