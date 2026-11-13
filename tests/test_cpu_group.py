"""End-to-end SP group lifecycle on CPU/Gloo with a real worker process.

Runs the loader node, worker spawn, patch handshake, a full forward and shutdown
through the same backend plumbing used for CUDA/NCCL and Intel XPU/XCCL, selected
here with MINIMAX_SP_BACKEND=cpu so no accelerator is needed.
"""
import os
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from bootstrap import setup
setup()
import comfy.ops
import comfy.utils
import folder_paths
from comfy.ldm.minimax.model import MiniMaxH3Model
import minimax_sp as node
from minimax_sp import device as spdev
from minimax_sp import sp_group


def save_tiny_h3(path, seed=17):
    torch.manual_seed(seed)
    model = MiniMaxH3Model(hidden_size=64, num_layers=2, token_refiner_num_layers=1,
                           num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=128,
                           text_dim=64, timestep_input_dim=16, time_embed_hidden_size=64,
                           time_embed_dim=32, dtype=torch.float32, device="cpu", operations=comfy.ops.manual_cast)
    with torch.no_grad():
        for p in model.parameters():
            p.uniform_(-0.05, 0.05)
        model.rope.inv_freq.fill_(0.01)
    comfy.utils.save_torch_file({k: v.contiguous() for k, v in model.state_dict().items()}, path)


class CPUGroupLifecycleTests(unittest.TestCase):
    def setUp(self):
        try:
            torch.distributed.TCPStore("127.0.0.1", 0, 1, True, wait_for_workers=False)
        except Exception as exc:  # native Windows wheels ship TCPStore without libuv
            self.skipTest(f"TCPStore unavailable: {exc}")

    def test_loader_spawns_worker_runs_forward_and_shuts_down(self):
        self._run_lifecycle({})

    def test_staged_collectives_match_native_forward_across_ranks(self):
        # A tiny staging buffer forces every rank through many identical collective chunks.
        self._run_lifecycle({"MINIMAX_SP_COMM_BUFFER_MB": "0.002"})

    def _run_lifecycle(self, extra_env):
        with tempfile.TemporaryDirectory() as tmp:
            save_tiny_h3(os.path.join(tmp, "tiny_h3.safetensors"))
            env = {"MINIMAX_SP_BACKEND": "cpu", "MINIMAX_SP_LOGDIR": tmp,
                   "MINIMAX_SP_STARTUP_TIMEOUT": "240", "MINIMAX_SP_COLLECTIVE_TIMEOUT": "120",
                   **extra_env}
            paths = folder_paths.folder_names_and_paths["diffusion_models"][0]
            paths.append(tmp)
            group = None
            try:
                with patch.dict(os.environ, env):
                    model = node.MiniMaxH3SPUNETLoader.execute("tiny_h3.safetensors", "default", 2, "auto").result[0]
                    group = sp_group.active_group()
                    self.assertIsNotNone(group)
                    self.assertEqual((group.world, group.backend), (2, spdev.CPU))
                    self.assertIsNone(group.bounded_timeout)
                    group.check_alive()

                    group.sync_patches(model)
                    dit = model.get_model_object("diffusion_model")
                    torch.manual_seed(3)
                    x = [torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 3)]
                    timestep = torch.tensor([500.0])
                    context = torch.randn(1, 3, 64)
                    with torch.no_grad():
                        expected = type(dit)._forward(dit, x, timestep, context, transformer_options={})
                        for _ in range(2):
                            got = group.forward(dit, x, timestep, context, {}, None)
                            for actual, reference in zip(got, expected):
                                torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)
                        model.cleanup()
                        self.assertFalse(group.patches_ready)
                        group.check_alive()
                        model.pre_run()
                        got = group.forward(dit, x, timestep, context, {}, None)
                        for actual, reference in zip(got, expected):
                            torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)
                        model.cleanup()
                    procs = list(group.procs)
                    group.shutdown()
                    self.assertIsNone(sp_group.active_group())
                    self.assertEqual([p.returncode for p in procs], [0])
            except Exception:
                log = os.path.join(tmp, "minimax_sp_worker1.log")
                if os.path.exists(log):
                    with open(log, encoding="utf-8", errors="replace") as f:
                        print("worker log:\n" + f.read()[-4000:])
                raise
            finally:
                if sp_group.active_group() is not None:
                    sp_group.active_group().destroy()
                paths.remove(tmp)

    def test_worker_survives_idle_longer_than_control_timeout(self):
        # Gloo's control timeout follows MINIMAX_SP_STARTUP_TIMEOUT; idling past it used to kill workers.
        with tempfile.TemporaryDirectory() as tmp:
            save_tiny_h3(os.path.join(tmp, "tiny_h3.safetensors"))
            env = {"MINIMAX_SP_BACKEND": "cpu", "MINIMAX_SP_LOGDIR": tmp,
                   "MINIMAX_SP_STARTUP_TIMEOUT": "30", "MINIMAX_SP_COLLECTIVE_TIMEOUT": "30"}
            paths = folder_paths.folder_names_and_paths["diffusion_models"][0]
            paths.append(tmp)
            try:
                with patch.dict(os.environ, env):
                    model = node.MiniMaxH3SPUNETLoader.execute("tiny_h3.safetensors", "default", 2, "auto").result[0]
                    group = sp_group.active_group()
                    group.sync_patches(model)
                    model.cleanup()
                    time.sleep(35)
                    group.check_alive()
                    model.pre_run()
                    dit = model.get_model_object("diffusion_model")
                    x = [torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 3)]
                    timestep, context = torch.tensor([500.0]), torch.randn(1, 3, 64)
                    with torch.no_grad():
                        expected = type(dit)._forward(dit, x, timestep, context, transformer_options={})
                        got = group.forward(dit, x, timestep, context, {}, None)
                    for actual, reference in zip(got, expected):
                        torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)
                    procs = list(group.procs)
                    group.shutdown()
                    self.assertEqual([p.returncode for p in procs], [0])
            except Exception:
                log = os.path.join(tmp, "minimax_sp_worker1.log")
                if os.path.exists(log):
                    with open(log, encoding="utf-8", errors="replace") as f:
                        print("worker log:\n" + f.read()[-4000:])
                raise
            finally:
                if sp_group.active_group() is not None:
                    sp_group.active_group().destroy()
                paths.remove(tmp)

    def test_loading_another_checkpoint_replaces_the_running_group(self):
        # e.g. a T2V workflow followed by R2V in the same ComfyUI process.
        with tempfile.TemporaryDirectory() as tmp:
            save_tiny_h3(os.path.join(tmp, "tiny_a.safetensors"), seed=17)
            save_tiny_h3(os.path.join(tmp, "tiny_b.safetensors"), seed=29)
            env = {"MINIMAX_SP_BACKEND": "cpu", "MINIMAX_SP_LOGDIR": tmp,
                   "MINIMAX_SP_STARTUP_TIMEOUT": "240", "MINIMAX_SP_COLLECTIVE_TIMEOUT": "120"}
            paths = folder_paths.folder_names_and_paths["diffusion_models"][0]
            paths.append(tmp)
            torch.manual_seed(3)
            x = [torch.randn(1, 24, 2, 4, 4), torch.randn(1, 32, 2, 3)]
            timestep, context = torch.tensor([500.0]), torch.randn(1, 3, 64)

            def run(model):
                group = sp_group.active_group()
                model.pre_run()
                dit = model.get_model_object("diffusion_model")
                with torch.no_grad():
                    expected = type(dit)._forward(dit, x, timestep, context, transformer_options={})
                    got = group.forward(dit, x, timestep, context, {}, None)
                for actual, reference in zip(got, expected):
                    torch.testing.assert_close(actual, reference, atol=1e-5, rtol=1e-4)
                model.cleanup()
                return group, got

            try:
                with patch.dict(os.environ, env):
                    first = node.MiniMaxH3SPUNETLoader.execute("tiny_a.safetensors", "default", 2, "auto").result[0]
                    group_a, out_a = run(first)
                    procs_a = list(group_a.procs)

                    second = node.MiniMaxH3SPUNETLoader.execute("tiny_b.safetensors", "default", 2, "auto").result[0]
                    group_b = sp_group.active_group()
                    self.assertIsNot(group_b, group_a)
                    self.assertEqual(group_b.unet_name, "tiny_b.safetensors")
                    self.assertEqual([p.returncode for p in procs_a], [0])
                    self.assertEqual(len(sp_group._GROUPS), 1)
                    _, out_b = run(second)
                    self.assertFalse(torch.allclose(out_a[0], out_b[0]))
                    sp_group.active_group().shutdown()
                    self.assertIsNone(sp_group.active_group())
            except Exception:
                log = os.path.join(tmp, "minimax_sp_worker1.log")
                if os.path.exists(log):
                    with open(log, encoding="utf-8", errors="replace") as f:
                        print("worker log:\n" + f.read()[-4000:])
                raise
            finally:
                if sp_group.active_group() is not None:
                    sp_group.active_group().destroy()
                paths.remove(tmp)

    def test_worker_load_failure_retains_traceback_and_reports_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "broken.safetensors"), "wb") as stream:
                stream.write(b"invalid checkpoint")
            env = {"MINIMAX_SP_BACKEND": "cpu", "MINIMAX_SP_LOGDIR": tmp,
                   "MINIMAX_SP_STARTUP_TIMEOUT": "60", "MINIMAX_SP_COLLECTIVE_TIMEOUT": "60"}
            paths = folder_paths.folder_names_and_paths["diffusion_models"][0]
            paths.append(tmp)
            try:
                with patch.dict(os.environ, env):
                    with self.assertRaisesRegex(RuntimeError, "Worker logs: rank 1:") as caught:
                        sp_group.SPGroup(2, "broken.safetensors", "default", ["0", "1"], backend=spdev.CPU)
                log = os.path.join(tmp, "minimax_sp_worker1.log")
                self.assertIn(log, str(caught.exception))
                self.assertIsNotNone(caught.exception.__cause__)
                with open(log, encoding="utf-8", errors="replace") as stream:
                    contents = stream.read()
                self.assertIn("MiniMax SP worker rank 1 failed", contents)
                self.assertIn("Traceback (most recent call last)", contents)
                self.assertIn("load_diffusion_model", contents)
                self.assertIn("broken.safetensors", contents)
                self.assertFalse(torch.distributed.is_initialized())
            finally:
                paths.remove(tmp)


if __name__ == "__main__":
    unittest.main()
