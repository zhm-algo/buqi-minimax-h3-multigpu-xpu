"""Static checks for the shipped UI workflow examples."""

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
WORKFLOWS = {
    "t2v": "workflow_api_t2v_2gpu.json",
    "i2v_first": "workflow_api_i2v_2gpu.json",
    "i2v_first_last": "workflow_api_i2v_first_last_2gpu.json",
    "r2v": "workflow_api_r2v_2gpu.json",
}
FOUR_GPU_WORKFLOWS = {
    "t2v": "workflow_api_t2v_4gpu.json",
    "i2v_first": "workflow_api_i2v_4gpu.json",
    "i2v_first_last": "workflow_api_i2v_first_last_4gpu.json",
    "r2v": "workflow_api_r2v_4gpu.json",
}
TURBO_WORKFLOWS = {
    "t2v": "workflow_api_t2v_turbo_lora_2gpu.json",
    "i2v_first": "workflow_api_i2v_turbo_lora_2gpu.json",
    "i2v_first_last": "workflow_api_i2v_first_last_turbo_lora_2gpu.json",
}


class WorkflowTests(unittest.TestCase):
    def load(self, name):
        with (EXAMPLES / name).open(encoding="utf-8") as handle:
            return json.load(handle)

    def node(self, workflow, node_id):
        return next(node for node in workflow["nodes"] if node["id"] == node_id)

    def linked_inputs(self, node):
        return {item["name"] for item in node.get("inputs", []) if item.get("link") is not None}

    def source(self, workflow, node, name):
        link_id = next(item["link"] for item in node["inputs"] if item["name"] == name)
        return next(link[1] for link in workflow["links"] if link[0] == link_id)

    def test_examples_use_sp_loader_and_standard_vae_decode(self):
        for world_size, workflows in ((2, WORKFLOWS), (4, FOUR_GPU_WORKFLOWS), (2, TURBO_WORKFLOWS)):
            for filename in workflows.values():
                with self.subTest(workflow=filename):
                    workflow = self.load(filename)
                    types = [node["type"] for node in workflow["nodes"]]
                    self.assertIn("MiniMaxH3SPUNETLoader", types)
                    self.assertNotIn("UNETLoader", types)
                    loader = self.node(workflow, "unet")
                    self.assertIn("int8_convrot", loader["widgets_values"][0])
                    self.assertEqual(loader["widgets_values"][1:], ["default", world_size, "auto"])
                    self.assertEqual(self.node(workflow, "vae_video")["widgets_values"][0],
                                     "minimax_h3_video_vae_int8_convrot.safetensors")
                    self.assertEqual(self.node(workflow, "decode_video")["type"], "VAEDecode")
                    self.assert_links_are_consistent(workflow)

    def test_required_variants_are_distinct(self):
        frames = {"t2v": set(), "i2v_first": {"first_frame"},
                  "i2v_first_last": {"first_frame", "last_frame"}}
        for workflows in (WORKFLOWS, FOUR_GPU_WORKFLOWS):
            for label, expected in frames.items():
                with self.subTest(workflow=workflows[label]):
                    conditioning = self.linked_inputs(
                        self.node(self.load(workflows[label]), "conditioning"))
                    self.assertEqual(conditioning & {"first_frame", "last_frame"}, expected)
            with self.subTest(workflow=workflows["r2v"]):
                r2v = self.node(self.load(workflows["r2v"]), "conditioning")
                self.assertEqual(r2v["type"], "MiniMaxH3ReferenceToVideo")
                self.assertIn("ref_images.ref_image_0", self.linked_inputs(r2v))
                self.assertIn("ref_images.ref_image_1", self.linked_inputs(r2v))

    def test_four_gpu_duration_and_resolution_controls_are_connected(self):
        for filename in FOUR_GPU_WORKFLOWS.values():
            with self.subTest(workflow=filename):
                workflow = self.load(filename)
                conditioning = self.node(workflow, "conditioning")
                links = {link[0]: link for link in workflow["links"]}
                inputs = {item["name"]: item["link"] for item in conditioning["inputs"]}
                self.assertEqual(links[inputs["width"]][1:3], [3, 0])
                self.assertEqual(links[inputs["height"]][1:3], [3, 1])
                self.assertEqual(links[inputs["length"]][1:3], [2, 1])
                self.assertEqual(self.source(workflow, self.node(workflow, 2), "values.a"), 1)

    def test_turbo_variants_sample_through_the_lora(self):
        frames = {"t2v": set(), "i2v_first": {"first_frame"},
                  "i2v_first_last": {"first_frame", "last_frame"}}
        for label, filename in TURBO_WORKFLOWS.items():
            with self.subTest(workflow=label):
                workflow = self.load(filename)
                lora = self.node(workflow, "turbo")
                self.assertEqual(lora["type"], "LoraLoaderModelOnly")
                self.assertIn("turbo_8step", lora["widgets_values"][0])
                self.assertEqual(self.source(workflow, lora, "model"), "unet")
                scheduler = self.node(workflow, "scheduler")
                self.assertEqual(self.source(workflow, self.node(workflow, "guider"), "model"), "turbo")
                self.assertEqual(self.source(workflow, scheduler, "model"), "turbo")
                self.assertEqual(scheduler["widgets_values"][1], 8)
                conditioning = self.linked_inputs(self.node(workflow, "conditioning"))
                self.assertEqual(conditioning & {"first_frame", "last_frame"}, frames[label])

    def assert_links_are_consistent(self, workflow):
        ids = {node["id"] for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}
        for link in workflow["links"]:
            self.assertIn(link[1], ids)
            self.assertIn(link[3], ids)
        for node in workflow["nodes"]:
            for item in node.get("inputs", []):
                if item.get("link") is not None:
                    self.assertIn(item["link"], links)
                    self.assertEqual(links[item["link"]][3], node["id"])


if __name__ == "__main__":
    unittest.main()
