"""Mathematical unit tests; no retinal observations are synthesized."""
from fractions import Fraction
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / "scripts" / "exact_p_bounds.py"
SPEC = importlib.util.spec_from_file_location("exact_p_bounds", PATH)
bounds = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bounds)


class ExactBoundTests(unittest.TestCase):
    def test_drive(self):
        self.assertEqual(bounds.minimum_two_sided_exact(20, 20), Fraction(1, 68923264410))

    def test_chase(self):
        self.assertEqual(bounds.minimum_two_sided_exact(8, 8), Fraction(1, 6435))

    def test_stare(self):
        self.assertEqual(bounds.minimum_two_sided_exact(10, 10), Fraction(1, 92378))

    def test_symmetry_and_smallest_sample(self):
        self.assertEqual(bounds.minimum_two_sided_exact(2, 3), bounds.minimum_two_sided_exact(3, 2))
        self.assertEqual(bounds.minimum_two_sided_exact(1, 1), Fraction(1))

    def test_invalid_sample_sizes(self):
        for n in (0, -1, True, 2.5, "8"):
            with self.subTest(n=n), self.assertRaises(ValueError):
                bounds.minimum_two_sided_exact(n, 8)

    def test_all_comparisons_are_labelled_bounds(self):
        report = bounds.table7_bounds()
        self.assertEqual(len(report["comparisons"]), 21)
        self.assertTrue(all(row["is_empirical_p_value"] is False for row in report["comparisons"]))
        self.assertEqual({row["method"] for row in report["comparisons"]}, set(bounds.METHODS))
        self.assertFalse(any(row["method"] == row["reference"] for row in report["comparisons"]))

    def test_same_counts_same_bound(self):
        report = bounds.table7_bounds()
        for dataset in bounds.TEST_IMAGE_COUNTS:
            values = {row["p_min_fraction"] for row in report["comparisons"] if row["dataset"] == dataset}
            self.assertEqual(len(values), 1)


if __name__ == "__main__":
    unittest.main()
