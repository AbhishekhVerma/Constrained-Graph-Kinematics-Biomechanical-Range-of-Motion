"""
End-to-End System Integration Smoke Test.
Verifies all modules, models, network parsers, and live demo modes pass seamlessly.
"""

import subprocess
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestSystemIntegration(unittest.TestCase):
    def test_01_kinematics_unit_suite(self):
        res = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "tests" / "test_kinematics.py")],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Kinematics unit test failed:\n{res.stderr}")
        self.assertIn("ALL KINEMATICS & MODEL ARCHITECTURE TESTS PASSED", res.stdout)

    def test_02_mode_b_udp_diagnostics(self):
        res = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "data" / "iphone_parser.py"), "--mode", "test"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Mode B UDP diagnostics failed:\n{res.stderr}")
        self.assertIn("PASS: 10 live UDP frames", res.stdout)

    def test_03_replay_demo_headless(self):
        res = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "run_presentation_demo.py"),
                "--mode",
                "replay",
                "--headless",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"Replay demo failed:\n{res.stderr}")
        self.assertIn("100.0% VALID", res.stdout)

    def test_04_udp_demo_headless(self):
        res = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "run_presentation_demo.py"),
                "--mode",
                "udp",
                "--headless",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0, f"UDP demo failed:\n{res.stderr}")
        self.assertIn("100.0% VALID", res.stdout)


if __name__ == "__main__":
    unittest.main()
