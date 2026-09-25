"""check_pins.py range pins: a raised floor must reach an EXISTING install.

launch.py's requirements_met() compares only == pins, so once check_pins has
repaired those it reports "met" and pip never runs -- a raised floor (the way a
security fix ships, e.g. Pillow>=12.3.0) used to reach fresh installs only.
"""
import importlib.util
import os
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

REQS = """--extra-index-url https://download.pytorch.org/whl/cu126
numpy>=2.5.3   # a floor
numba>=0.67
setuptools>=70,<81
Pillow>=12.3.0
torch==2.13.0+cu126
pydantic==2.13.5
httpcore
windowsonly>=9; sys_platform == "never-a-platform"
"""


def load():
    spec = importlib.util.spec_from_file_location("check_pins_under_test", os.path.join(ROOT, "check_pins.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RangePins(unittest.TestCase):
    def setUp(self):
        self.cp = load()
        fd, self.path = tempfile.mkstemp(suffix=".txt")
        os.write(fd, REQS.encode())
        os.close(fd)
        self.cp.REQS = self.path

    def tearDown(self):
        os.unlink(self.path)

    def targets(self, have):
        return [t for _p, _g, _r, t in self.cp.range_drift(have)]

    def test_below_a_floor_installs_exactly_the_floor(self):
        # not "numpy>=2.5.3": pip would take the newest numpy, which can
        # overshoot a cap another package sets (numba 0.67 allows <2.6)
        self.assertEqual(self.targets({"numpy": "2.4.4"}), ["numpy==2.5.3"])
        self.assertEqual(self.targets({"numba": "0.66.0"}), ["numba==0.67"])

    def test_over_a_cap_passes_the_whole_range(self):
        self.assertEqual(self.targets({"setuptools": "81.0.0"}), ["setuptools<81,>=70"])

    def test_satisfied_ranges_are_left_alone(self):
        self.assertEqual(self.targets({"setuptools": "80.9.0", "pillow": "12.3.0", "numpy": "2.5.9"}), [])

    def test_exact_pins_unpinned_lines_and_other_platforms_are_skipped(self):
        self.assertEqual(self.targets({"pydantic": "2.8.2", "httpcore": "0.1", "windowsonly": "1.0"}), [])
        self.assertEqual(self.cp.pinned().get("pydantic"), "2.13.5")   # still handled by pinned()

    def test_check_mode_reports_range_drift_and_fails(self):
        self.cp.installed = lambda: {"pillow": "12.2.0", "numpy": "2.5.3"}
        argv, sys.argv = sys.argv, ["check_pins.py", "--check"]
        try:
            self.assertEqual(self.cp.main(), 1)
        finally:
            sys.argv = argv


if __name__ == "__main__":
    unittest.main()
