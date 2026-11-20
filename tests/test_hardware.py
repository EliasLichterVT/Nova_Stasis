import unittest

from nova.hardware import GIB, hardware_report


class HardwareTests(unittest.TestCase):
    def test_cpu_mode_does_not_use_a_secondary_gpu(self):
        report = hardware_report({
            "system": {"ram_total": 8 * GIB},
            "devices": [{"type": "cpu", "vram_total": 8 * GIB},
                        {"type": "cuda", "vram_total": 24 * GIB}],
        })
        self.assertEqual(report["label"], "CPU mode")
        self.assertIsNone(report["device"]["memory"])
        self.assertEqual(report["defaults"], {"width": 384, "height": 384, "steps": 12, "batch": 1})

    def test_starter_defaults_preserve_hardware_limits(self):
        report = hardware_report({
            "system": {"ram_total": 32 * GIB},
            "devices": [{"name": "Test GPU", "type": "cuda", "vram_total": 16 * GIB, "vram_free": GIB}],
        }, starter=True)
        self.assertEqual(report["defaults"]["width"], 512)
        self.assertEqual(report["limits"]["width"], 1024)
        self.assertEqual(report["limits"]["batch"], 2)
        self.assertTrue(any("Free GPU memory is low" in note for note in report["notes"]))

    def test_apple_shared_memory_and_missing_stats(self):
        report = hardware_report({
            "system": {"ram_total": 16 * GIB},
            "devices": [{"type": "mps", "vram_total": 16 * GIB}],
        })
        self.assertEqual(report["label"], "Balanced")
        self.assertIsNone(report["device"]["memory"])
        self.assertTrue(any("shared" in note for note in report["notes"]))
        self.assertEqual(hardware_report({})["label"], "Conservative")
