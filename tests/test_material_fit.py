from pathlib import Path

import numpy as np

from qsi_campaign.material_fit import (
    MODEL_METHOD,
    abc_to_transverse,
    curve_score,
    load_fcc32_model_curve,
)


def test_abc_mapping_for_published_a_parameters():
    mapped = abc_to_transverse(0.050, 0.021, 0.004)
    assert mapped == {"Ja_meV": 0.050, "Jpm_meV": -0.00625, "Jpmpm_meV": 0.00425}


def test_curve_score_is_zero_for_matching_curve():
    temperature = np.geomspace(0.025, 2.0, 20)
    heat = np.sin(np.log(temperature)) ** 2
    score = curve_score(temperature, heat, temperature, heat, (0.025, 2.0))
    assert score["n_points"] == 20
    assert score["rmse_J_molCe_K"] < 1.0e-14


def test_model_contract_rejects_unconverged_curve(tmp_path: Path):
    path = tmp_path / "curve.npz"
    np.savez(
        path,
        method=MODEL_METHOD,
        n_sites=32,
        character_M=4,
        character_converged=False,
        complement_converged=True,
        Ja_meV=0.05,
        Jb_meV=0.02,
        Jc_meV=0.0,
        temperature_K=np.array([0.02, 0.03, 0.04]),
        heat_capacity_J_molCe_K=np.array([0.1, 0.2, 0.1]),
    )
    try:
        load_fcc32_model_curve(path)
    except ValueError as error:
        assert "character-grid convergence" in str(error)
    else:
        raise AssertionError("unconverged curve was accepted")
