from pathlib import Path

import numpy as np

from qsi_campaign.benchmarks import load_digitized_thermodynamics
from qsi_campaign.thermodynamics import thermal_observables


def test_two_level_entropy_limits():
    temperatures = np.array([1.0e-3, 1.0e3])
    result = thermal_observables(np.array([0.0, 1.0]), temperatures)
    assert result["entropy_per_site"][0] < 1.0e-10
    np.testing.assert_allclose(result["entropy_per_site"][1], np.log(2.0), rtol=1.0e-6)
    assert np.all(result["heat_capacity_per_site"] >= 0.0)


def test_huang_qmc_vector_extraction_has_reported_limits():
    root = Path(__file__).resolve().parents[1]
    temperature, heat, entropy = load_digitized_thermodynamics(
        root / "campaign" / "data" / "huang_2018_qmc_jpm_0p046.csv"
    )
    low = (temperature >= 5.0e-4) & (temperature <= 2.0e-2)
    peak_index = np.flatnonzero(low)[np.argmax(heat[low])]
    np.testing.assert_allclose(temperature[peak_index], 0.001100447383, rtol=1.0e-8)
    np.testing.assert_allclose(heat[peak_index], 0.1547068232, rtol=1.0e-8)
    assert entropy[-1] > 0.68
