import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notes"))

import recompute_finite_size_artifact as geometry  # noqa: E402


def test_cluster_loop_census():
    cubic = geometry.build_cluster("cubic", (1, 1, 1))
    fcc = geometry.build_cluster("fcc", (2, 2, 2))
    assert cubic.n_sites == 16 and cubic.n_ice == 90
    assert fcc.n_sites == 32 and fcc.n_ice == 2970
    assert geometry.cycle_summary(cubic.loops4)["contractible"] == 0
    assert geometry.cycle_summary(fcc.loops4)["contractible"] == 0
    assert geometry.cycle_summary(cubic.hexes)["contractible"] == 16
    assert geometry.cycle_summary(fcc.hexes)["contractible"] == 32


def test_zero_transport_removes_four_loops_and_keeps_hexagons():
    cubic = geometry.build_cluster("cubic", (1, 1, 1))
    rows = geometry.sw_order23(cubic, verbose=False)
    channels = geometry.channel_survival(cubic, rows)
    four = channels["H2"]["4_loop_wrapping"]
    hexagon = channels["H3"]["hexagon_contractible"]
    assert four["delta0_max"] == 0.0
    assert hexagon["delta0_min"] == 1.0
    assert four["finite_grid_retained"] == {"M2": 0.0, "M3": 0.0, "M4": 0.0}
    assert hexagon["finite_grid_retained"] == {"M2": 1.0, "M3": 1.0, "M4": 1.0}
    assert four["finite_grid_terms"] == {"M2": 0, "M3": 0, "M4": 0}
    assert hexagon["finite_grid_terms"] == {"M2": 16, "M3": 16, "M4": 16}
