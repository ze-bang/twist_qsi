"""Loop-amplitude tomography of the winding-free map.

The masked ice-block map is, entry by entry, an effective gauge Hamiltonian:
each off-diagonal element is the amplitude for one microscopic ice-to-ice
process, and the process is fully described by the geometry of the string of
spins it flips.  This module classifies every entry by that geometry --
perimeter `L`, number of connected components, graph diameter, minimum-image
spatial diameter -- and pools the connected single-string classes into the
class amplitudes `K_L`.

The physics question those amplitudes answer (WP1): the bare theory has one
ring exchange, the contractible hexagon at `L = 6`.  Virtual spinon excursions
dress it and also *generate* longer processes that were not there at all.  If
`K_L` falls off exponentially, `K_L ~ A exp(-(L - 6) / xi_gauge)`, then the
emergent gauge Hamiltonian has a finite range `xi_gauge` -- a number that can
be tracked against coupling, against the residue `1 - Z`, and against the
inverse continuum-edge gap.  A `xi_gauge` that grows with coupling is
delocalization: the photon survives but its Hamiltonian spreads.

Two things this module is careful about.

*Product processes.*  A flipped set with two disconnected components is two
independent shorter processes happening together, and its amplitude factorises.
Pooling those with genuine connected strings of the same perimeter flattens the
decay and inflates `xi_gauge`.  They are classified separately and excluded
from the `K_L` fit.

*Cost.*  On cubic-16 the ice manifold has 90 states and any implementation
works.  On FCC-32 it has 2970, so the pair table has 8.8M entries; the
per-entry work is vectorised (popcount for the perimeter, `np.unique` +
`bincount` for the aggregation) and the genuinely expensive geometric
classification runs once per *distinct short flip mask*, of which there are
few.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "graph_distance_matrix",
    "minimum_image_distance_matrix",
    "classify_flip",
    "loop_tomography",
    "connected_class_amplitudes",
    "fit_gauge_range",
]


def graph_distance_matrix(cluster) -> np.ndarray:
    """All-pairs nearest-neighbour hop distance on the periodic cluster."""
    n = int(cluster.n_sites)
    distance = np.full((n, n), 1 << 20, dtype=np.int64)
    for source in range(n):
        distance[source, source] = 0
        frontier = [source]
        depth = 0
        seen = {source}
        while frontier:
            depth += 1
            following = []
            for u in frontier:
                for v, _ in cluster.adj[u]:
                    if v not in seen:
                        seen.add(v)
                        distance[source, v] = depth
                        following.append(v)
            frontier = following
    return distance


def minimum_image_distance_matrix(cluster) -> np.ndarray:
    """All-pairs spatial distance under the minimum-image convention."""
    positions = np.asarray(cluster.positions, dtype=float)
    lvecs = np.asarray(cluster.Lvecs, dtype=float)
    best = np.full((len(positions),) * 2, np.inf)
    for shift in np.ndindex(3, 3, 3):
        offset = (np.asarray(shift, dtype=float) - 1.0) @ lvecs
        candidate = np.linalg.norm(
            positions[:, None, :] - positions[None, :, :] - offset, axis=2)
        best = np.minimum(best, candidate)
    return best


def classify_flip(cluster, mask: int, graph_distance, real_distance):
    """Geometry of one flipped string.

    Returns `(perimeter, components, simple, graph_diameter, spatial_diameter)`.

    ``components`` above one marks a *disconnected* product process: two or
    more disjoint strings flipped together.

    ``simple`` is the stricter and more important test.  Connectivity alone
    does NOT mean "one loop": two hexagons sharing a single site form a
    graph-connected eleven-site set, and two sharing an edge a ten-site set.
    Those are composite processes whose amplitude is a *product* of hexagon
    amplitudes -- large -- and pooling them with genuine long strings, whose
    amplitudes are small, inflates `K_10` and `K_12` and flattens the very
    decay that `xi_gauge` measures.  A genuine simple loop has every site at
    degree exactly two inside the induced subgraph; anything with a
    degree-three vertex is two loops joined at that vertex.
    """
    sites = [i for i in range(int(cluster.n_sites)) if (mask >> i) & 1]
    if not sites:
        return (0, 0, True, 0, 0.0)
    members = set(sites)
    remaining = set(sites)
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        while stack:
            u = stack.pop()
            for v, _ in cluster.adj[u]:
                if v in remaining:
                    remaining.discard(v)
                    stack.append(v)
    simple = all(
        sum(1 for v, _ in cluster.adj[u] if v in members) == 2
        for u in sites
    )
    index = np.asarray(sites)
    graph_diameter = int(graph_distance[np.ix_(index, index)].max())
    real_diameter = float(real_distance[np.ix_(index, index)].max())
    return (len(sites), components, bool(simple), graph_diameter,
            round(real_diameter, 6))


def loop_tomography(matrix, ice_states, cluster, labels, graph_distance,
                    real_distance, max_perimeter: int = 12) -> dict:
    """Class-resolved amplitudes of an ice-basis operator.

    Everything at or below ``max_perimeter`` is classified geometrically;
    longer processes are pooled into a single ``long`` bucket, since the
    `K_L` fit never uses them and their number grows combinatorially.
    """
    states = np.asarray([int(s) for s in ice_states], dtype=np.uint64)
    values = np.asarray(matrix).real
    difference = states[:, None] ^ states[None, :]
    perimeter = np.bitwise_count(difference)
    same_sector = np.asarray(labels)[:, None] == np.asarray(labels)[None, :]

    short = (perimeter > 0) & (perimeter <= max_perimeter)
    flat_masks = difference[short]
    flat_values = values[short]
    flat_kept = same_sector[short]

    summary: dict[str, dict] = {}
    if flat_masks.size:
        unique_masks, inverse = np.unique(flat_masks, return_inverse=True)
        keys = [
            classify_flip(cluster, int(mask), graph_distance, real_distance)
            for mask in unique_masks
        ]
        # distinct geometry classes, and the class each unique mask belongs to
        ordered = sorted(set(keys))
        class_of_key = {key: index for index, key in enumerate(ordered)}
        class_of_mask = np.array([class_of_key[key] for key in keys])
        class_id = class_of_mask[inverse]
        n_classes = len(ordered)

        count = np.bincount(class_id, minlength=n_classes)
        total = np.bincount(class_id, weights=flat_values, minlength=n_classes)
        absolute = np.bincount(class_id, weights=np.abs(flat_values),
                               minlength=n_classes)
        square = np.bincount(class_id, weights=flat_values ** 2,
                             minlength=n_classes)
        kept = np.bincount(class_id, weights=flat_kept.astype(float),
                           minlength=n_classes)
        masks_per_class = np.bincount(class_of_mask, minlength=n_classes)
        largest = np.zeros(n_classes)
        np.maximum.at(largest, class_id, np.abs(flat_values))

        safe = np.maximum(count, 1)
        mean = total / safe
        abs_mean = absolute / safe
        variance = np.maximum(square / safe - mean ** 2, 0.0)
        abs_variance = np.maximum(square / safe - abs_mean ** 2, 0.0)

        # Statistics over the entries the mask KEEPS.  This, not the average
        # over the whole class, is the amplitude of the winding-free theory:
        # the deleted entries are exactly zero by construction (they transport
        # between sectors), so averaging them in measures the survival fraction
        # rather than an amplitude.  The fractions differ sharply by perimeter
        # -- on FCC-32, 15360/62208 = 24.7% survive at L=6 against 10560/167424
        # = 6.3% at L=8 -- so pooling them corrupts every K_L ratio by the
        # ratio of survival fractions, and the within-class spread becomes a
        # measure of the zero/nonzero split rather than of the amplitudes.
        kept_weights = flat_kept.astype(float)
        kept_absolute = np.bincount(class_id,
                                    weights=np.abs(flat_values) * kept_weights,
                                    minlength=n_classes)
        kept_square = np.bincount(class_id,
                                  weights=(flat_values ** 2) * kept_weights,
                                  minlength=n_classes)
        kept_safe = np.maximum(kept, 1.0)
        kept_abs_mean = kept_absolute / kept_safe
        kept_abs_variance = np.maximum(
            kept_square / kept_safe - kept_abs_mean ** 2, 0.0)

        for index, key in enumerate(ordered):
            length, components, simple, graph_diameter, real_diameter = key
            name = (f"L{length}_c{components}_{'simple' if simple else 'joined'}"
                    f"_gd{graph_diameter}_rd{real_diameter:g}")
            summary[name] = {
                "perimeter": int(length),
                "components": int(components),
                "simple": bool(simple),
                "graph_diameter": int(graph_diameter),
                "real_diameter": float(real_diameter),
                "count": int(count[index]),
                "n_distinct_masks": int(masks_per_class[index]),
                "mean": float(mean[index]),
                "abs_mean": float(abs_mean[index]),
                "spread": float(np.sqrt(variance[index])),
                "abs_spread": float(np.sqrt(abs_variance[index])),
                "max_abs": float(largest[index]),
                "kept_by_mask": int(kept[index]),
                "kept_fraction": float(kept[index] / max(count[index], 1)),
                "kept_abs_mean": float(kept_abs_mean[index]),
                "kept_abs_spread": float(np.sqrt(kept_abs_variance[index])),
            }

    long_selection = perimeter > max_perimeter
    long_values = values[long_selection]
    summary["long"] = {
        "perimeter": -1,
        "components": -1,
        "graph_diameter": -1,
        "real_diameter": -1.0,
        "count": int(long_values.size),
        "n_distinct_masks": -1,
        "mean": float(long_values.mean()) if long_values.size else 0.0,
        "abs_mean": float(np.abs(long_values).mean()) if long_values.size else 0.0,
        "spread": float(long_values.std()) if long_values.size else 0.0,
        "abs_spread": float(np.abs(long_values).std()) if long_values.size else 0.0,
        "max_abs": float(np.abs(long_values).max()) if long_values.size else 0.0,
        "kept_by_mask": int((long_selection & same_sector).sum()),
    }
    return summary


def connected_class_amplitudes(summary) -> dict[int, dict]:
    """`K_L` per perimeter, pooling only the genuine simple loops.

    Both kinds of product process are excluded -- disconnected sets, and sets
    that are connected but join two loops at a shared vertex.  Each factorises
    into shorter processes and each would flatten the decay that defines
    `xi_gauge`, which is the one thing this measurement must not do.
    """
    pooled: dict[int, dict] = {}
    for name, entry in summary.items():
        if name == "long" or entry["components"] != 1:
            continue
        if not entry.get("simple", True):
            continue                # two loops joined at a vertex: a product
        length = int(entry["perimeter"])
        slot = pooled.setdefault(
            length, {"kept": 0.0, "sum": 0.0, "max_abs": 0.0, "all": 0,
                     "max_spread": 0.0, "n_classes": 0, "masks": 0})
        # weight by surviving entries, and average only those
        slot["kept"] += entry["kept_by_mask"]
        slot["sum"] += entry["kept_abs_mean"] * entry["kept_by_mask"]
        slot["all"] += entry["count"]
        slot["max_abs"] = max(slot["max_abs"], entry["max_abs"])
        slot["max_spread"] = max(slot["max_spread"], entry["kept_abs_spread"])
        slot["n_classes"] += 1
        slot["masks"] += entry["n_distinct_masks"]
    return {
        length: {
            "K": slot["sum"] / slot["kept"] if slot["kept"] else 0.0,
            "count": int(slot["kept"]),
            "count_all_entries": slot["all"],
            "kept_fraction": (slot["kept"] / slot["all"]
                              if slot["all"] else 0.0),
            "n_geometry_classes": slot["n_classes"],
            "n_distinct_masks": slot["masks"],
            "max_abs": slot["max_abs"],
            "within_class_spread": slot["max_spread"],
            "spread_over_K": (slot["max_spread"] * slot["kept"] / slot["sum"]
                              if slot["sum"] else float("inf")),
        }
        for length, slot in sorted(pooled.items())
    }


def fit_gauge_range(amplitudes, reference: int = 6,
                    generated_from: int = 8) -> dict:
    """Fit `K_L = A exp(-(L - reference) / xi_gauge)` over the connected classes.

    Two things this reports rather than hides.

    *The hexagon is not one of the generated processes.*  `L = 6` is the bare
    ring exchange -- it exists in the unperturbed theory at order `Jpm^3` --
    while every longer loop is generated purely by virtual spinon excursions.
    Pooling them into one exponential fits a straight line through a bare term
    and a dressed tail, and the `L = 6 -> 8` step dominates the slope.  The fit
    is therefore run over the *generated* sector (`L >= generated_from`) and
    `K_6` is reported separately, with the all-perimeter fit kept alongside for
    comparison.

    *Whether an exponential describes the data at all.*  The successive
    two-point estimates are returned in full.  If they disagree by more than a
    factor of ~1.5 there is no single `xi_gauge`, and the honest statement is a
    decay envelope with a quoted spread -- not one number with an error bar.
    """
    usable = {length: entry["K"] for length, entry in amplitudes.items()
              if entry["K"] > 0.0}
    if len(usable) < 2:
        return {"xi_gauge": None, "note": "fewer than two positive K_L"}

    def straight_line(lengths, values):
        logs = np.log(values)
        slope, intercept = np.polyfit(lengths, logs, 1)
        residual = float(np.sqrt(np.mean((logs - (slope * lengths + intercept))
                                         ** 2)))
        return slope, intercept, residual

    all_lengths = np.array(sorted(usable), dtype=float)
    all_values = np.array([usable[int(x)] for x in all_lengths])
    slope, intercept, residual = straight_line(all_lengths, all_values)

    # successive two-point estimates, the diagnostic that exposes curvature
    ladder = []
    order = sorted(usable)
    for first, second in zip(order, order[1:]):
        ratio = usable[second] / usable[first]
        if 0.0 < ratio != 1.0:
            ladder.append({"from": int(first), "to": int(second),
                           "xi": float(-(second - first) / np.log(ratio)),
                           "ratio": float(ratio)})

    generated = [length for length in order if length >= generated_from]
    generated_fit = None
    if len(generated) >= 2:
        lengths = np.array(generated, dtype=float)
        values = np.array([usable[int(x)] for x in lengths])
        g_slope, g_intercept, g_residual = straight_line(lengths, values)
        generated_fit = {
            "xi_gauge": float(-1.0 / g_slope) if g_slope < 0.0 else None,
            "log_residual_rms": g_residual,
            "perimeters": [int(x) for x in lengths],
            "growing": bool(g_slope >= 0.0),
        }

    positive = [entry["xi"] for entry in ladder
                if entry["xi"] is not None and entry["xi"] > 0]
    spread = (max(positive) / min(positive)) if len(positive) >= 2 else None

    return {
        "xi_gauge": (generated_fit or {}).get("xi_gauge"),
        "xi_gauge_all_perimeters": float(-1.0 / slope) if slope < 0.0 else None,
        "generated_sector_fit": generated_fit,
        "k_reference": float(usable.get(reference, np.nan)),
        "slope": float(slope),
        "log_residual_rms": residual,
        "n_perimeters": int(all_lengths.size),
        "perimeters": [int(x) for x in all_lengths],
        "two_point_ladder": ladder,
        "xi_spread_factor": spread,
        "is_exponential": bool(spread is not None and spread < 1.5),
        "growing": bool(slope >= 0.0),
    }
