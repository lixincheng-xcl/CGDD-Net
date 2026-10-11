#!/usr/bin/env python3
"""Report mathematical lower bounds, never empirical Table 7 p-values.

Assumptions: independent exchangeable groups, no tied observations, an exact
Mann-Whitney/Wilcoxon rank-sum null distribution, and twice the smaller tail.
The minimum 2 / comb(n1+n2, n1) occurs at complete rank separation. Dataset
counts below assume one AUC per test image and the requested STARE 10/10 split.
No AUC vectors or manuscript result files are read, generated, or overwritten.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
import json
from math import comb

METHODS = (
    "U-Net", "LadderNet", "CE-Net", "MPS-Net", "MFI-Net",
    "CS2-Net", "AACA-MLA-D-U-Net",
)
TEST_IMAGE_COUNTS = {"DRIVE": 20, "CHASE_DB1": 8, "STARE": 10}


def minimum_two_sided_exact(n1: int, n2: int) -> Fraction:
    """Return the untied exact two-sided rank-sum minimum as a rational number."""
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in (n1, n2)):
        raise ValueError("Both sample sizes must be positive integers")
    return Fraction(2, comb(n1 + n2, n1))


def bound_record(n1: int, n2: int) -> dict:
    value = minimum_two_sided_exact(n1, n2)
    with localcontext() as context:
        context.prec = 40
        decimal = str(Decimal(value.numerator) / Decimal(value.denominator))
    return {"n1": n1, "n2": n2, "p_min_fraction": str(value),
            "p_min_decimal": decimal, "is_empirical_p_value": False}


def table7_bounds() -> dict:
    rows = []
    for method in METHODS:
        for dataset, n in TEST_IMAGE_COUNTS.items():
            rows.append({"method": method, "reference": "CGDD-Net",
                         "dataset": dataset, **bound_record(n, n)})
    return {
        "purpose": "Sample-size feasibility bounds only; NOT experimental p-values",
        "assumptions": ["one AUC per test image", "independent exchangeable groups",
                        "no ties", "exact rank-sum distribution",
                        "two-sided p = twice the smaller tail"],
        "stare_protocol": "author-requested 10 train / 10 test; original run IDs not supplied",
        "reference_self_comparison": "not applicable; display --",
        "comparisons": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", nargs=2, type=int, metavar=("N1", "N2"),
                        help="Use explicitly supplied sample sizes instead of dataset assumptions")
    args = parser.parse_args()
    try:
        result = bound_record(*args.samples) if args.samples else table7_bounds()
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
