import json
from pathlib import Path

import numpy as np


def test_winding_free_dssf_reconstruction_and_cubic_symmetry():
    root = Path(__file__).resolve().parents[1]
    stem = root / "campaign" / "outputs" / "dssf_cubic16_p0p046000"
    report = json.loads(stem.with_suffix(".json").read_text())
    data = np.load(stem.with_suffix(".npz"))

    assert report["frame_diagnostics"]["cached_operator_relative_error"] < 1.0e-10
    assert report["frame_diagnostics"]["isometry_error"] < 1.0e-10
    np.testing.assert_allclose(
        data["periodic_weights"][1:].sum(axis=1),
        data["periodic_weights"][1].sum(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        data["winding_free_weights"][1:].sum(axis=1),
        data["winding_free_weights"][1].sum(),
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    periodic_first = report["periodic_channels"][1]["first_active_frequency"]
    winding_free_first = report["winding_free_channels"][1][
        "first_active_frequency"
    ]
    assert winding_free_first < 0.1 * periodic_first
