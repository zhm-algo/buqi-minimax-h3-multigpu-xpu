"""CPU-only checks for the Intel XPU (XCCL) backend plumbing; no XPU hardware needed."""
from datetime import timedelta
import json
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import torch

from bootstrap import setup
setup()
import minimax_sp as node
from minimax_sp import device as spdev
from minimax_sp import sp_forward as spf
from minimax_sp import sp_group as spg


class XPUVisibilityTests(unittest.TestCase):
    def test_affinity_mask_lists_physical_level_zero_devices(self):
        self.assertEqual(spdev.visible_devices(spdev.XPU, {"ZE_AFFINITY_MASK": "2,3"}),
                         ("ZE_AFFINITY_MASK", ["2", "3"]))
        self.assertEqual(spdev.visible_devices(spdev.XPU, {}), (None, None))

    def test_xpu_visibility_follows_driver_order_not_mask_order(self):
        self.assertEqual(spdev.visible_devices(spdev.XPU, {"ZE_AFFINITY_MASK": "1,0"}),
                         ("ZE_AFFINITY_MASK", ["0", "1"]))
        with patch.dict(os.environ, {"ZE_AFFINITY_MASK": "1,0"}, clear=True):
            self.assertEqual(node.resolve_devices("auto", 2, spdev.XPU), ["0", "1"])
            with self.assertRaisesRegex(ValueError, "primary visible device"):
                node.resolve_devices("1,0", 2, spdev.XPU)
        env = {"ZE_AFFINITY_MASK": "3,1,2", "MINIMAX_SP_XPU_VISIBILITY": "shared"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(node.resolve_devices("auto", 2, spdev.XPU), ["1", "2"])
        self.assertEqual(spdev.worker_env(spdev.XPU, env, "3")[1], 2)

    def test_selector_composes_with_affinity_mask(self):
        env = {"ONEAPI_DEVICE_SELECTOR": "level_zero:1,0"}
        self.assertEqual(spdev.visible_devices(spdev.XPU, env), ("ONEAPI_DEVICE_SELECTOR", ["0", "1"]))
        env["ZE_AFFINITY_MASK"] = "6,4"
        self.assertEqual(spdev.visible_devices(spdev.XPU, env)[1], ["4", "6"])
        env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:1"
        self.assertEqual(spdev.visible_devices(spdev.XPU, env)[1], ["6"])
        for selector in ("level_zero:*", "LEVEL_ZERO:gpu"):
            self.assertEqual(spdev.visible_devices(spdev.XPU, {"ONEAPI_DEVICE_SELECTOR": selector}), (None, None))
        env["ONEAPI_DEVICE_SELECTOR"] = "level_zero:2"
        with self.assertRaisesRegex(ValueError, "outside ZE_AFFINITY_MASK"):
            spdev.visible_devices(spdev.XPU, env)

    def test_unsupported_xpu_visibility_syntax_is_rejected(self):
        for env in ({"ZE_AFFINITY_MASK": "0.1,1"}, {"ZE_AFFINITY_MASK": "0,0"},
                    {"ONEAPI_DEVICE_SELECTOR": "opencl:0"}, {"ONEAPI_DEVICE_SELECTOR": "!level_zero:1"},
                    {"ONEAPI_DEVICE_SELECTOR": "level_zero:0;level_zero:1"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                spdev.visible_devices(spdev.XPU, env)

    def test_cuda_visibility_is_unchanged(self):
        self.assertEqual(spdev.visible_devices(spdev.CUDA, {"CUDA_VISIBLE_DEVICES": "2,3"}),
                         ("CUDA_VISIBLE_DEVICES", ["2", "3"]))
        with self.assertRaisesRegex(ValueError, "exposes no CUDA devices"):
            spdev.visible_devices(spdev.CUDA, {"CUDA_VISIBLE_DEVICES": "-1"})
        self.assertEqual(spdev.visible_devices(spdev.CUDA, {"ZE_AFFINITY_MASK": "0"}), (None, None))

    def test_isolated_xpu_only_requires_rank0_visible(self):
        isolated = {"MINIMAX_SP_XPU_VISIBILITY": "isolated"}
        with patch.dict(os.environ, {"ZE_AFFINITY_MASK": "0", **isolated}, clear=True):
            self.assertEqual(node.resolve_devices("0,1", 2, spdev.XPU), ["0", "1"])
            with self.assertRaisesRegex(ValueError, "need 2"):
                node.resolve_devices("auto", 2, spdev.XPU)
            with self.assertRaisesRegex(ValueError, "missing \\['1'\\]"):
                node.resolve_devices("1,0", 2, spdev.XPU)
        with patch.dict(os.environ, {"ZE_AFFINITY_MASK": "0,1", **isolated}, clear=True):
            with self.assertRaisesRegex(ValueError, "primary visible device"):
                node.resolve_devices("1,0", 2, spdev.XPU)
        with patch.dict(os.environ, {"ZE_AFFINITY_MASK": "0", "MINIMAX_SP_DEVICES": "0,1", **isolated}, clear=True):
            self.assertEqual(node.resolve_devices("auto", 2, spdev.XPU), ["0", "1"])

    def test_shared_xpu_requires_every_worker_visible(self):
        for env in ({"ZE_AFFINITY_MASK": "0"}, {"ZE_AFFINITY_MASK": "0", "MINIMAX_SP_XPU_VISIBILITY": "shared"}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(ValueError, "missing \\['1'\\]"):
                    node.resolve_devices("0,1", 2, spdev.XPU)
        env = {"ZE_AFFINITY_MASK": "2,3", "MINIMAX_SP_XPU_VISIBILITY": "shared"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(node.resolve_devices("auto", 2, spdev.XPU), ["2", "3"])

    def test_xpu_device_ids_must_be_level_zero_indices(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "Level Zero"):
                node.resolve_devices("0,GPU-1", 2, spdev.XPU)
            self.assertEqual(node.resolve_devices("auto", 2, spdev.XPU), ["0", "1"])

    def test_cpu_backend_uses_logical_ranks(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "5"}, clear=True):
            self.assertEqual(node.resolve_devices("auto", 2, spdev.CPU), ["0", "1"])

    def test_visibility_mode_is_validated(self):
        self.assertEqual(spdev.xpu_visibility_mode({}), "shared")
        self.assertEqual(spdev.xpu_visibility_mode({"MINIMAX_SP_XPU_VISIBILITY": "Isolated"}), "isolated")
        with self.assertRaisesRegex(ValueError, "isolated, shared"):
            spdev.xpu_visibility_mode({"MINIMAX_SP_XPU_VISIBILITY": "all"})

    def test_required_visible_count(self):
        self.assertEqual(spdev.required_visible_count(spdev.CUDA, 4, {}), 4)
        self.assertEqual(spdev.required_visible_count(spdev.XPU, 4, {}), 4)
        self.assertEqual(spdev.required_visible_count(spdev.XPU, 4, {"MINIMAX_SP_XPU_VISIBILITY": "isolated"}), 1)
        self.assertEqual(spdev.required_visible_count(spdev.CPU, 4, {}), 0)


class WorkerEnvironmentTests(unittest.TestCase):
    def test_cuda_worker_env_matches_original_launch(self):
        env, local = spdev.worker_env(spdev.CUDA, {"PATH": "x", "CUDA_VISIBLE_DEVICES": "2,3"}, "3")
        self.assertEqual(local, 0)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "3")
        self.assertEqual(env["TORCH_NCCL_ASYNC_ERROR_HANDLING"], "1")
        self.assertEqual(env["TORCH_NCCL_BLOCKING_WAIT"], "1")
        self.assertEqual(env["PATH"], "x")

    def test_isolated_xpu_worker_gets_single_card_mask(self):
        base = {"ZE_AFFINITY_MASK": "0,1", "ONEAPI_DEVICE_SELECTOR": "level_zero:0,1",
                "MINIMAX_SP_XPU_VISIBILITY": "isolated"}
        env, local = spdev.worker_env(spdev.XPU, base, "1")
        self.assertEqual((env["ZE_AFFINITY_MASK"], local), ("1", 0))
        self.assertNotIn("ONEAPI_DEVICE_SELECTOR", env)
        self.assertNotIn("TORCH_NCCL_BLOCKING_WAIT", env)
        self.assertEqual(base["ZE_AFFINITY_MASK"], "0,1")

    def test_shared_xpu_worker_keeps_visibility_and_selects_local_index(self):
        base = {"ZE_AFFINITY_MASK": "2,3", "MINIMAX_SP_XPU_VISIBILITY": "shared"}
        env, local = spdev.worker_env(spdev.XPU, base, "3")
        self.assertEqual((env["ZE_AFFINITY_MASK"], local), ("2,3", 1))
        env, local = spdev.worker_env(spdev.XPU, {"MINIMAX_SP_XPU_VISIBILITY": "shared"}, "1")
        self.assertEqual(local, 1)
        env, local = spdev.worker_env(spdev.XPU, {"ZE_AFFINITY_MASK": "0,1"}, "1")
        self.assertEqual((env["ZE_AFFINITY_MASK"], local), ("0,1", 1))

    def test_cpu_worker_env_is_untouched(self):
        self.assertEqual(spdev.worker_env(spdev.CPU, {"A": "1"}, "1"), ({"A": "1"}, 0))

    def test_process_env_only_sets_nccl_defaults_for_cuda(self):
        environ = {}
        spdev.apply_process_env(spdev.XPU, environ)
        self.assertEqual(environ, {})
        spdev.apply_process_env(spdev.CUDA, environ)
        self.assertEqual(environ["TORCH_NCCL_BLOCKING_WAIT"], "1")


class WorkerRuntimeTests(unittest.TestCase):
    def test_xpu_explicitly_disables_worker_dynamic_vram(self):
        from comfy.cli_args import enables_dynamic_vram

        with patch.dict(os.environ, {}, clear=True), \
                patch.object(spg.comfy.memory_management, "aimdo_enabled", False), \
                patch.object(spg.comfy_args, "enable_dynamic_vram", False), \
                patch.object(spg.comfy_args, "disable_dynamic_vram", False), \
                patch.object(spg.comfy_args, "cpu", False):
            runtime = spg.worker_runtime_options(spdev.XPU)
            self.assertIs(runtime["disable_dynamic_vram"], True)
            self.assertIs(runtime["enable_dynamic_vram"], False)
            self.assertIs(spg.comfy_args.disable_dynamic_vram, False)
            with patch.multiple(spg.comfy_args, **runtime):
                self.assertFalse(enables_dynamic_vram())

    def test_xpu_follows_rank0_dynamic_vram_through_omnixpu(self):
        from comfy.cli_args import enables_dynamic_vram

        state = {"root": "/omnixpu", "providers": ["comfy_aimdo.xpu", "comfy_kitchen.xpu"],
                 "custom_node": True, "adapters": ["writable_weight_adapter"]}
        with patch.dict(os.environ, {"AIMDO_XPU_ALLOCATOR_MODE": "native_hook"}, clear=True), \
                patch.object(spg.comfy.memory_management, "aimdo_enabled", True), \
                patch.object(spg.comfy_args, "enable_dynamic_vram", True), \
                patch.object(spg.comfy_args, "disable_dynamic_vram", False), \
                patch.object(spg.comfy_args, "cpu", False):
            runtime = spg.worker_runtime_options(spdev.XPU, state)
            self.assertIs(runtime["enable_dynamic_vram"], True)
            self.assertIs(runtime["disable_dynamic_vram"], False)
            with patch.multiple(spg.comfy_args, **runtime):
                self.assertTrue(enables_dynamic_vram())

    def test_xpu_enable_flag_without_active_dynamic_vram_follows_rank0(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(spg.comfy.memory_management, "aimdo_enabled", False), \
                patch.object(spg.comfy_args, "enable_dynamic_vram", True):
            runtime = spg.worker_runtime_options(spdev.XPU, None)
        self.assertIs(runtime["enable_dynamic_vram"], False)
        self.assertIs(runtime["disable_dynamic_vram"], True)

    def test_xpu_rejects_unreproducible_dynamic_vram_before_startup(self):
        no_aimdo = {"root": "/omnixpu", "providers": ["comfy_kitchen.xpu"], "custom_node": True, "adapters": []}
        for active, allocator, state, message in (
                (True, "", None, "comfy_aimdo.xpu.*--disable-dynamic-vram"),
                (True, "native_hook", no_aimdo, "comfy_aimdo.xpu.*--disable-dynamic-vram"),
                (False, "native_hook", no_aimdo, "native_hook.*not active.*LD_PRELOAD")):
            with self.subTest(active=active, allocator=allocator, state=state), \
                    patch.dict(os.environ, {"AIMDO_XPU_ALLOCATOR_MODE": allocator}, clear=True), \
                    patch.object(spg.comfy.memory_management, "aimdo_enabled", active), \
                    patch.object(spg.omnixpu, "rank0_state", return_value=state), \
                    patch.object(spg.dist, "is_initialized", return_value=False), \
                    patch.object(spg.spdev, "check_available"), \
                    patch.object(spg.dist, "TCPStore") as store, \
                    patch.object(spg.subprocess, "Popen") as spawn, \
                    patch.object(spg.SPGroup, "destroy") as destroy:
                with self.assertRaisesRegex(RuntimeError, message):
                    spg.SPGroup(2, "model", "default", ["0", "1"], backend=spdev.XPU)
                store.assert_not_called()
                spawn.assert_not_called()
                destroy.assert_called_once()

    def test_rank0_omnixpu_state_is_passed_to_xpu_workers_only(self):
        state = {"root": "/omnixpu", "providers": ["comfy_kitchen.xpu"], "custom_node": True, "adapters": []}
        for backend, expected in ((spdev.XPU, state), (spdev.CUDA, None)):
            with self.subTest(backend=backend.name), \
                    patch.dict(os.environ, {}, clear=True), \
                    patch.object(spg.comfy.memory_management, "aimdo_enabled", False), \
                    patch.object(spg.omnixpu, "rank0_state", return_value=state), \
                    patch.object(spg.dist, "is_initialized", return_value=False), \
                    patch.object(spg.spdev, "check_available"), \
                    patch.object(spg.folder_paths, "get_full_path_or_raise", return_value="/models/model"), \
                    patch.object(spg.dist, "TCPStore"), \
                    patch.object(spg.subprocess, "Popen") as spawn, \
                    patch.object(spg.dist, "init_process_group", side_effect=RuntimeError("stop")), \
                    patch.object(spg.SPGroup, "destroy"), \
                    tempfile.TemporaryDirectory() as logs, \
                    patch.dict(os.environ, {"MINIMAX_SP_LOGDIR": logs}):
                with self.assertRaisesRegex(RuntimeError, "stop"):
                    spg.SPGroup(2, "model", "default", ["0", "1"], backend=backend)
                argv = spawn.call_args.args[0]
                self.assertEqual(json.loads(argv[argv.index("--omnixpu") + 1]), expected)

    def test_cuda_keeps_existing_dynamic_vram_options(self):
        with patch.object(spg.comfy.memory_management, "aimdo_enabled", True), \
                patch.object(spg.comfy_args, "disable_dynamic_vram", False):
            runtime = spg.worker_runtime_options(spdev.CUDA)
        self.assertIs(runtime["disable_dynamic_vram"], False)
        self.assertNotIn("enable_dynamic_vram", runtime)
        self.assertNotIn("cpu", runtime)

    def test_cpu_keeps_cpu_mode_and_disables_dynamic_vram(self):
        with patch.object(spg.comfy.memory_management, "aimdo_enabled", False):
            runtime = spg.worker_runtime_options(spdev.CPU)
        self.assertIs(runtime["cpu"], True)
        self.assertIs(runtime["disable_dynamic_vram"], True)
        self.assertIs(runtime["enable_dynamic_vram"], False)


class BackendSelectionTests(unittest.TestCase):
    def test_explicit_backend_and_validation(self):
        self.assertIs(spdev.select_backend("cpu"), spdev.CPU)
        with patch.dict(os.environ, {"MINIMAX_SP_BACKEND": "XPU"}):
            self.assertIs(spdev.select_backend(), spdev.XPU)
        with self.assertRaisesRegex(ValueError, "unknown MiniMax SP backend"):
            spdev.select_backend("rocm")

    def test_auto_prefers_cuda_then_xpu(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.object(torch.cuda, "is_available", return_value=True):
                self.assertIs(spdev.select_backend(), spdev.CUDA)
            with patch.object(torch.cuda, "is_available", return_value=False), \
                    patch.object(spdev, "xpu_is_available", return_value=True):
                self.assertIs(spdev.select_backend(), spdev.XPU)
            with patch.object(torch.cuda, "is_available", return_value=False), \
                    patch.object(spdev, "xpu_is_available", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "CUDA or Intel XPU"):
                    spdev.select_backend()

    def test_backend_identity_survives_separate_module_instances(self):
        from minimax_sp import sp_worker
        worker_device = sp_worker.load_device_module()
        self.assertEqual(worker_device.get_backend("xpu"), spdev.XPU)
        self.assertNotEqual(worker_device.get_backend("xpu"), spdev.CUDA)

    def test_xccl_availability_is_required(self):
        with patch.object(spdev.dist, "is_xccl_available", return_value=False, create=True):
            with self.assertRaisesRegex(RuntimeError, "no XCCL backend"):
                spdev.check_available(spdev.XPU)
        with patch.object(spdev.dist, "is_xccl_available", return_value=True, create=True):
            spdev.check_available(spdev.XPU)
        spdev.check_available(spdev.CPU)

    def test_native_windows_is_rejected_for_device_backends(self):
        self.assertRegex(spdev.platform_error(spdev.CUDA, "win32"), "NCCL.*WSL2")
        self.assertRegex(spdev.platform_error(spdev.XPU, "win32"), "XCCL.*Linux")
        self.assertIsNone(spdev.platform_error(spdev.XPU, "linux"))
        self.assertIsNone(spdev.platform_error(spdev.CPU, "win32"))

    def test_only_xpu_uses_bounded_waits(self):
        timeout = timedelta(seconds=5)
        self.assertEqual(spdev.bounded_timeout(spdev.XPU, timeout), timeout)
        self.assertIsNone(spdev.bounded_timeout(spdev.CUDA, timeout))
        self.assertIsNone(spdev.bounded_timeout(None, timeout))


class ProcessGroupLayoutTests(unittest.TestCase):
    def _run(self, backend):
        with tempfile.TemporaryDirectory() as tmp:
            store = torch.distributed.FileStore(os.path.join(tmp, "store"), 1)
            timeout = timedelta(seconds=30)
            control_pg, tensor_pg = spdev.init_groups(backend, store, 0, 1, timeout, timeout)
            try:
                world = torch.distributed.group.WORLD
                replies = [None]
                torch.distributed.all_gather_object(replies, {"ok": True}, group=control_pg)
                value = torch.ones(1)
                torch.distributed.all_reduce(value, group=tensor_pg)
                layout = (control_pg is world, tensor_pg is world)
            finally:
                spdev.destroy_groups(backend, control_pg, tensor_pg, True)
                del store
        self.assertFalse(torch.distributed.is_initialized())
        self.assertEqual(replies, [{"ok": True}])
        return layout

    def test_cuda_layout_keeps_gloo_default_group(self):
        self.assertFalse(spdev.CUDA.tensor_group_is_default)
        self.assertEqual(self._run(spdev.CPU), (True, False))

    def test_xpu_layout_makes_tensor_group_default(self):
        self.assertTrue(spdev.XPU.tensor_group_is_default)
        self.assertEqual(self._run(spdev.CPU._replace(tensor_group_is_default=True)), (False, True))

    def test_failed_subgroup_destroys_default_group(self):
        backend = spdev.CPU._replace(tensor_group_is_default=True)
        with tempfile.TemporaryDirectory() as tmp:
            store = torch.distributed.FileStore(os.path.join(tmp, "store"), 1)
            with patch.object(spdev.dist, "new_group", side_effect=RuntimeError("subgroup failed")):
                with self.assertRaisesRegex(RuntimeError, "subgroup failed"):
                    spdev.init_groups(backend, store, 0, 1, timedelta(seconds=30), timedelta(seconds=30))
            del store
        self.assertFalse(torch.distributed.is_initialized())

    def test_destroy_groups_removes_subgroup_first_and_respects_ownership(self):
        control, tensor = object(), object()
        with patch.object(spdev.dist, "destroy_process_group") as destroy:
            spdev.destroy_groups(spdev.XPU, control, tensor, True)
            spdev.destroy_groups(spdev.CUDA, control, tensor, True)
            spdev.destroy_groups(spdev.XPU, control, tensor, False)
            spdev.destroy_groups(None, control, None, True)
        self.assertEqual(destroy.call_args_list,
                         [call(control), call(tensor), call(tensor), call(control), call(control), call(control)])


class BoundedCollectiveTests(unittest.TestCase):
    def test_unbounded_collective_keeps_original_call(self):
        op = Mock(return_value="done")
        self.assertEqual(spdev.run_collective(op, 1, group="g"), "done")
        op.assert_called_once_with(1, group="g")

    def test_bounded_collective_waits_with_timeout(self):
        work = Mock()
        op = Mock(return_value=work)
        timeout = timedelta(seconds=7)
        spdev.run_collective(op, 1, group="g", timeout=timeout)
        op.assert_called_once_with(1, async_op=True, group="g")
        work.wait.assert_called_once_with(timeout)

    def test_bounded_wait_error_surfaces(self):
        work = Mock()
        work.wait.side_effect = RuntimeError("Work timed out after 7000 milliseconds.")
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            spdev.run_collective(Mock(return_value=work), timeout=timedelta(seconds=7))

    def test_bounded_wait_on_real_gloo_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = torch.distributed.FileStore(os.path.join(tmp, "store"), 1)
            torch.distributed.init_process_group("gloo", store=store, rank=0, world_size=1)
            try:
                ctx = spf.SPContext(0, 1, None, 3, timeout=timedelta(seconds=30))
                rows = torch.arange(6, dtype=torch.float32).reshape(3, 2)
                torch.testing.assert_close(spf._gather_rows(rows, [3], ctx, 2, "cpu"), rows)
            finally:
                torch.distributed.destroy_process_group()
                del store

    def test_startup_probe_bounds_both_checks(self):
        tensor_pg, timeout = object(), timedelta(seconds=9)
        works = []

        def make_work():
            work = Mock()
            works.append(work)
            return work

        def all_reduce(tensor, **kwargs):
            self.assertIs(kwargs["group"], tensor_pg)
            self.assertTrue(kwargs["async_op"])
            tensor.fill_(3)
            return make_work()

        def all_to_all(received, source, **kwargs):
            self.assertTrue(kwargs["async_op"])
            received.copy_(torch.tensor([0, 2]))
            return make_work()

        with patch.object(spg.dist, "all_reduce", side_effect=all_reduce), \
                patch.object(spg.dist, "all_to_all_single", side_effect=all_to_all):
            spg.startup_probe(0, 2, tensor_pg, "cpu", timeout)
        self.assertEqual(len(works), 2)
        for work in works:
            work.wait.assert_called_once_with(timeout)

    def test_xpu_probe_failure_points_at_xccl(self):
        with patch.object(spg.dist, "all_reduce", side_effect=RuntimeError("ze error")):
            with self.assertRaisesRegex(RuntimeError, "XCCL startup probe.*ZE_AFFINITY_MASK.*CCL_LOG_LEVEL"):
                spg.startup_probe(0, 2, object(), torch.device("xpu"))

    def test_rank0_passes_bound_to_sp_forward(self):
        group = spg.SPGroup.__new__(spg.SPGroup)
        self.assertIsNone(group.bounded_timeout)
        group.bounded_timeout = timedelta(seconds=11)
        group.world, group.control_pg, group.tensor_pg = 2, object(), object()
        group.patches_ready, group.operation_id, group.verify_pending = True, 0, False
        group.check_alive, group.destroy = Mock(), Mock()
        source = torch.ones(1)

        def gather(replies, local, **kwargs):
            replies[:] = [local for _ in replies]

        with patch.object(spg, "ordered_tensors", return_value=[source]), \
                patch.object(spg, "build_meta", return_value=_forward_meta()), \
                patch.object(spg.dist, "broadcast_object_list"), \
                patch.object(spg.dist, "all_gather_object", side_effect=gather), \
                patch.object(spg.dist, "broadcast"), \
                patch.object(spg.spf, "sp_forward", return_value="out") as forward:
            group.forward(None, [source], None, None, {}, None)
        self.assertEqual(forward.call_args.kwargs["collective_timeout"], timedelta(seconds=11))


def _forward_meta():
    return {"op": "forward", "video": {"shape": [1], "dtype": "float32"}, "audio": None,
            "timestep": None, "context": None, "tags": None, "cond_video": [], "cond_audio": [],
            "sample_sigmas": None, "options": {}}


class LoaderBackendTests(unittest.TestCase):
    def _execute(self, backend, device_count, env):
        dit = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(heads=56))])
        model = SimpleNamespace(get_model_object=lambda name: dit)
        with patch.dict(os.environ, env, clear=True), \
                patch.object(node, "check_h3_api"), \
                patch.object(node, "MiniMaxH3Model", SimpleNamespace), \
                patch.object(node.folder_paths, "get_full_path_or_raise", return_value="model.safetensors"), \
                patch.object(node.comfy.sd, "load_diffusion_model", return_value=model), \
                patch.object(node.spdev, "select_backend", return_value=backend), \
                patch.object(node.spdev, "platform_error", return_value=None), \
                patch.object(node.spdev, "check_available"), \
                patch.object(node.spdev, "device_count", return_value=device_count), \
                patch.object(node.sp_group, "get_group", side_effect=RuntimeError("group requested")) as get_group:
            with self.assertRaises(Exception) as caught:
                node.MiniMaxH3SPUNETLoader.execute("model", "default", 2, "auto")
        return caught.exception, get_group

    def test_xpu_loader_reports_missing_cards(self):
        error, get_group = self._execute(spdev.XPU, 1, {"ZE_AFFINITY_MASK": "0,1",
                                                        "MINIMAX_SP_XPU_VISIBILITY": "shared"})
        self.assertRegex(str(error), "only 1 Intel XPUs.*ZE_AFFINITY_MASK=0,1")
        get_group.assert_not_called()

    def test_xpu_loader_passes_backend_to_group(self):
        for env, count in (({"ZE_AFFINITY_MASK": "0,1"}, 2),
                           ({"ZE_AFFINITY_MASK": "0", "MINIMAX_SP_DEVICES": "0,1",
                             "MINIMAX_SP_XPU_VISIBILITY": "isolated"}, 1)):
            with self.subTest(env=env):
                error, get_group = self._execute(spdev.XPU, count, env)
                self.assertEqual(str(error), "group requested")
                get_group.assert_called_once_with(2, "model", "default", ["0", "1"], spdev.XPU)

    def test_xpu_loader_requires_every_card_by_default(self):
        error, get_group = self._execute(spdev.XPU, 1, {"ZE_AFFINITY_MASK": "0", "MINIMAX_SP_DEVICES": "0,1"})
        self.assertRegex(str(error), "missing \\['1'\\]")
        get_group.assert_not_called()

    def test_cuda_loader_message_is_unchanged(self):
        error, _ = self._execute(spdev.CUDA, 1, {"CUDA_VISIBLE_DEVICES": "0,1"})
        self.assertRegex(str(error), "only 1 GPUs are visible; start ComfyUI with e.g. --cuda-device 0,1")


if __name__ == "__main__":
    unittest.main()
