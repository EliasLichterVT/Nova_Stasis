import unittest
from uuid import uuid4

from nova.anatomy import effective_negative
from nova.workflow import Settings, build_workflow


class AnatomyTests(unittest.TestCase):
    def test_prompt_reader_focuses_visible_regions(self):
        common = effective_negative("Two adults in a cafe.", "", "high")
        self.assertIn("(extra limbs:1.30)", common)
        self.assertIn("(malformed hands:1.25)", common)
        self.assertNotIn("extra fingers", common)
        self.assertNotIn("extra toes", common)

        focused = effective_negative("Full-body portrait of an adult holding a lantern.", "", "high")
        self.assertIn("(malformed hands:1.30)", focused)
        self.assertIn("(extra fingers:1.30)", focused)
        self.assertIn("(extra toes:1.25)", focused)
        self.assertIn("(misaligned eyes:1.20)", focused)

    def test_custom_weight_wins_and_off_preserves_custom_text(self):
        custom = "(Extra Fingers:0.85), grainy photo, FUSED LIMBS"
        merged = effective_negative("An adult holding a cup.", custom, "high")
        self.assertTrue(merged.endswith(custom))
        self.assertNotIn("(extra fingers:1.30)", merged)
        self.assertNotIn("(fused limbs:1.25)", merged)
        self.assertEqual(effective_negative("An adult holding a cup.", custom, "off"), custom)

    def test_workflow_routes_combined_prompt_into_sampler(self):
        images = {name: f"{name}.png" for name in ("ref_a", "ref_b", "region_a", "region_b")}
        settings = Settings(prompt="Two adults walking on a trail.", negative="noise",
                            checkpoint="checkpoint", adapter="adapter", vision="vision",
                            ref_a=uuid4(), ref_b=uuid4())
        for mode in ("high", "off"):
            with self.subTest(mode=mode):
                settings.anatomy = mode
                graph = build_workflow(settings, images, 0, "test")
                sampler = next(node for node in graph.values() if node["class_type"] == "KSampler")
                negative_key = sampler["inputs"]["negative"][0]
                self.assertEqual(graph[negative_key]["inputs"]["text"],
                                 effective_negative(settings.prompt, settings.negative, mode))
