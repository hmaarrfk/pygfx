"""TEMPORARY: render the five flaky validation examples and hash the result.

Mirrors what examples/tests/test_examples.py does (same env vars, same mocked
time), but prints a sha256 per example instead of asserting, so that a run
reports a *variant fingerprint* whether or not it would have failed.
"""

import hashlib
import importlib.util
import os
import pathlib
import sys
from unittest.mock import patch

os.environ["WGPU_FORCE_OFFSCREEN"] = "true"
os.environ["PYGFX_DEFAULT_PPAA"] = "none"

import imageio.v3 as iio  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).parent
NAMES = [
    "validate_image_colormap",
    "validate_line_loop",
    "validate_outpass",
    "validate_points_markers",
    "validate_ppaa",
]


def render(name):
    filename = ROOT / "examples" / "validation" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, filename)
    module = importlib.util.module_from_spec(spec)
    with patch("time.time") as time_mock:
        time_mock.return_value = 1.23456
        spec.loader.exec_module(module)
        return np.asarray(module.renderer.target.draw())


def main():
    from examples.tests.testutils import adapter

    print("adapter:", adapter.info["device"], "|", adapter.info["description"])
    for name in NAMES:
        img = render(name)
        ref = iio.imread(ROOT / "examples" / "screenshots" / f"{name}.png")
        digest = hashlib.sha256(np.ascontiguousarray(img)).hexdigest()[:12]
        diff = np.abs(img.astype("i4") - ref.astype("i4"))
        n_above = int((diff > 1).sum())
        print(
            f"  {name:28s} sha={digest} maxdiff={diff.max():3d} "
            f"n_above_atol={n_above:5d} {'FAIL' if n_above else 'ok'}"
        )
    sys.stdout.flush()


main()
