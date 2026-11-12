"""CPU-only checks that SP workers replay ComfyUI-OmniXPU setup like rank 0."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from bootstrap import ROOT, setup
COMFY_ROOT = setup()
from minimax_sp import omnixpu


PRESTARTUP = """
import importlib.util, os, sys
root = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(root, "torch_seen"), "w") as f:
    f.write(str("torch" in sys.modules))
spec = importlib.util.spec_from_file_location(
    "_comfyui_omnixpu_runtime_bootstrap", os.path.join(root, "runtime_bootstrap.py"))
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
module.bootstrap()
"""

RUNTIME = """
import copy
_STATE = {{"providers": {{}}}}
def bootstrap():
    if {fail!r}:
        raise RuntimeError("fake prestartup stop")
    for name in {providers!r}:
        _STATE["providers"][name] = {{"status": "active", "reason": ""}}
    _STATE["providers"]["comfy_kitchen.cuda"] = {{"status": "skipped", "reason": "no CUDA"}}
def get_state():
    return copy.deepcopy(_STATE)
"""

CUSTOM_NODE = """
import sys, types
sys.modules.setdefault("ComfyUI-OmniXPU", sys.modules[__name__])
patches = types.ModuleType("ComfyUI-OmniXPU.patches")
patches.get_status = lambda: [{{"name": n, "status": "applied"}} for n in {adapters!r}] + [
    {{"name": "legacy_median_fix", "status": "skipped", "reason": "disabled by env"}}]
sys.modules[patches.__name__] = patches
"""


def fake_omnixpu(base, providers=(), adapters=(), fail=False):
    root = Path(base) / "custom_nodes" / "ComfyUI-OmniXPU.v1"
    root.mkdir(parents=True)
    (root / "prestartup_script.py").write_text(PRESTARTUP)
    (root / "runtime_bootstrap.py").write_text(RUNTIME.format(providers=list(providers), fail=fail))
    (root / "__init__.py").write_text(CUSTOM_NODE.format(adapters=list(adapters)))
    return str(root)


def fake_aimdo_modules():
    package = ModuleType("comfy_aimdo")
    control = ModuleType("comfy_aimdo.control")
    package.control = control
    return {"comfy_aimdo": package, "comfy_aimdo.control": control}


def isolated_modules():
    # Each test replays OmniXPU into a fresh worker-like process state.
    cleared = patch.dict(sys.modules, fake_aimdo_modules())
    cleared.start()
    for name in [n for n in sys.modules if n == omnixpu.RUNTIME_MODULE or n.startswith(omnixpu.PACKAGE)]:
        del sys.modules[name]
    return cleared


class Rank0StateTests(unittest.TestCase):
    def test_without_omnixpu_there_is_nothing_to_replay(self):
        self.assertIsNone(omnixpu.rank0_state({}))
        self.assertFalse(omnixpu.provider_active(None))

    def test_captures_active_providers_and_applied_adapters(self):
        root = os.path.abspath("omx")
        runtime = SimpleNamespace(__file__=os.path.join(root, "runtime_bootstrap.py"), get_state=lambda: {
            "providers": {"comfy_kitchen.xpu": {"status": "active"}, "comfy_aimdo.xpu": {"status": "active"},
                          "comfy_kitchen.cuda": {"status": "skipped"}}})
        node = SimpleNamespace(__file__=os.path.join(root, "__init__.py"))
        patches = SimpleNamespace(get_status=lambda: [
            {"name": "writable_weight_adapter", "status": "applied"},
            {"name": "fp8_gemm_adapter", "status": "failed"},
            {"name": "attention_adapter", "status": "applied"}])
        state = omnixpu.rank0_state({omnixpu.RUNTIME_MODULE: runtime, omnixpu.PACKAGE: node,
                                     omnixpu.PACKAGE + ".patches": patches})
        self.assertEqual(state, {"root": root, "providers": ["comfy_aimdo.xpu", "comfy_kitchen.xpu"],
                                 "custom_node": True,
                                 "adapters": ["attention_adapter", "writable_weight_adapter"]})
        self.assertTrue(omnixpu.provider_active(state))

    def test_custom_node_without_xpu_patches_still_replays_the_node(self):
        root = os.path.abspath("omx")
        node = SimpleNamespace(__file__=os.path.join(root, "__init__.py"))
        self.assertEqual(omnixpu.rank0_state({omnixpu.PACKAGE: node}),
                         {"root": root, "providers": None, "custom_node": True, "adapters": None})


class WorkerReplayTests(unittest.TestCase):
    def setUp(self):
        self.modules = isolated_modules()
        self.addCleanup(self.modules.stop)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def test_replay_reproduces_rank0_state(self):
        providers, adapters = ["comfy_aimdo.xpu", "comfy_kitchen.xpu"], ["attention_adapter", "xpu_ram_staging"]
        root = fake_omnixpu(self.tmp, providers, adapters)
        state = {"root": root, "providers": providers, "custom_node": True, "adapters": adapters}
        omnixpu.run_prestartup(state)
        omnixpu.load_custom_node(state)
        # Same module name ComfyUI's nodes.py uses, so OmniXPU's alias resolves to it.
        self.assertIs(sys.modules[omnixpu.PACKAGE], sys.modules[root.replace(".", "_x_")])
        self.assertEqual(omnixpu.rank0_state(), state)

    def test_provider_mismatch_fails_worker_startup(self):
        root = fake_omnixpu(self.tmp, ["comfy_kitchen.xpu"])
        state = {"root": root, "providers": ["comfy_aimdo.xpu", "comfy_kitchen.xpu"],
                 "custom_node": False, "adapters": None}
        with self.assertRaisesRegex(RuntimeError, "runtime providers on this worker differ from rank 0"):
            omnixpu.run_prestartup(state)

    def test_adapter_differences_are_logged_without_failing_startup(self):
        root = fake_omnixpu(self.tmp, adapters=["attention_adapter", "int8_ffn_adapter"])
        state = {"root": root, "providers": None, "custom_node": True,
                 "adapters": ["attention_adapter", "large_video_preprocess_adapter", "legacy_median_fix"]}
        with self.assertLogs(level="WARNING") as logs:
            omnixpu.load_custom_node(state)
        message = "\n".join(logs.output)
        self.assertIn("large_video_preprocess_adapter (absent: -), legacy_median_fix (skipped: disabled by env)",
                      message)
        self.assertIn("applied only here: int8_ffn_adapter", message)

    def test_matching_adapters_log_nothing(self):
        root = fake_omnixpu(self.tmp, adapters=["attention_adapter"])
        state = {"root": root, "providers": None, "custom_node": True, "adapters": ["attention_adapter"]}
        with self.assertNoLogs(level="WARNING"):
            omnixpu.load_custom_node(state)

    def test_stages_absent_on_rank0_are_not_replayed(self):
        root = fake_omnixpu(self.tmp, fail=True)
        state = {"root": root, "providers": None, "custom_node": False, "adapters": None}
        omnixpu.run_prestartup(None)
        omnixpu.run_prestartup(state)
        omnixpu.load_custom_node(state)
        self.assertNotIn(omnixpu.RUNTIME_MODULE, sys.modules)
        self.assertNotIn(omnixpu.PACKAGE, sys.modules)


class WorkerStartupOrderTests(unittest.TestCase):
    def test_worker_runs_omnixpu_prestartup_before_importing_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fake_omnixpu(tmp, ["comfy_aimdo.xpu"], fail=True)
            aimdo = Path(tmp) / "fake_site" / "comfy_aimdo"
            aimdo.mkdir(parents=True)
            (aimdo / "__init__.py").write_text("")
            (aimdo / "control.py").write_text(textwrap.dedent("""
                def init(*args, **kwargs):
                    raise AssertionError("official AIMDO init must be skipped on the OmniXPU path")
                """))
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(
                p for p in [str(aimdo.parent), os.environ.get("PYTHONPATH")] if p))
            state = {"root": root, "providers": ["comfy_aimdo.xpu"], "custom_node": True, "adapters": []}
            result = subprocess.run(
                [sys.executable, str(ROOT / "minimax_sp" / "sp_worker.py"), "--rank", "1", "--world", "2",
                 "--port", "1", "--unet", "model", "--comfy-root", str(COMFY_ROOT),
                 "--startup-timeout", "5", "--collective-timeout", "5",
                 "--runtime-options", json.dumps({"enable_dynamic_vram": True, "disable_dynamic_vram": False}),
                 "--omnixpu", json.dumps(state), "--backend", "xpu"],
                env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("fake prestartup stop", result.stderr)
            self.assertIn("failed before importing PyTorch", result.stderr)
            self.assertEqual((Path(root) / "torch_seen").read_text(), "False")


if __name__ == "__main__":
    unittest.main()
