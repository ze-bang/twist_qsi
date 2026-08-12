"""FCC-32 theta=0 band extraction, one momentum sector per job.

With M=2 exact for the winding artifact, the whole character grid follows from
h(0) by diagonal gauge phases, so this single band IS the campaign input for
both the heat capacity and the DSSF.

Writes the ice-projected band (n_ice x width, a few MB) rather than the block
itself (~295 GB), because the polar pullback only ever needs <ice|psi>.
"""
from __future__ import annotations
import argparse, json, os, resource, sys, time
import numpy as np

TW = "/lustre09/project/6003507/zhouzb79/twist_qsi"
sys.path[:0] = [TW + "/notes", TW + "/src", TW + "/campaign/pilot"]
import qed._core as core
from recompute_finite_size_artifact import build_cluster
from pilot_fcc32_stage_a import fcc32_translations, write_directory
from chebfsi import spectral_bounds
from chebfsi_lowmem import MatvecCounter, chebfsi_lowmem

MOMENTUM = {0: "Gamma", 3: "X", 5: "X", 6: "X", 1: "L", 2: "L", 4: "L", 7: "L"}


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", type=int, required=True)
    ap.add_argument("--margin", type=int, default=24)
    ap.add_argument("--degree", type=int, default=25)
    ap.add_argument("--outer", type=int, default=3)
    ap.add_argument("--fp64", action="store_true", help="fp64 storage (2x memory)")
    ap.add_argument("--device", action="store_true", default=True)
    ap.add_argument("--outdir", default=TW + "/campaign/outputs/fcc32_band")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    dtype = np.complex128 if args.fp64 else np.complex64

    cl = build_cluster("fcc", (2, 2, 2))
    n_up, n_ice = cl.n_sites // 2, len(cl.ice_states)
    shift = 0.5 * len(cl.tets)
    root = TW + "/campaign/pilot/fcc32_dir"
    os.makedirs(root, exist_ok=True)
    write_directory(root, cl, np.zeros(3), fcc32_translations(cl))

    t0 = time.perf_counter()
    sectors = core.sector_operators(root, cl.n_sites, 0.5, n_up)
    sector = sectors[args.sector]
    dim = int(sector.dimension)
    print(f"sector {args.sector} ({MOMENTUM[args.sector]}): dim={dim:,} "
          f"build={time.perf_counter()-t0:.1f}s rss={rss_gb():.1f} GB", flush=True)

    internal = np.array([(1 << cl.n_sites) - 1 - int(s) for s in cl.ice_states],
                        dtype=np.uint64)
    idx, amp = sector.project_states(internal)
    hit = np.nonzero(idx >= 0)[0]              # ice states living in this sector
    rows = idx[hit]                            # their sector basis indices
    unique_rows = np.unique(rows)
    share = len(unique_rows)
    width = min(share + args.margin, dim - 2)
    # NEVER form the dense (dim x n_ice) gather: that is 3.25 TiB here. Each ice
    # state projects onto exactly ONE sector basis vector, so the ice subspace
    # is spanned by the unit vectors at ``unique_rows`` and the pullback needs
    # only the block's rows there.
    print(f"  ice share={share} block width={width} "
          f"({width * dim * (16 if args.fp64 else 8) / 1e9:.0f} GB)", flush=True)

    if args.device:
        t0 = time.perf_counter()
        sector.use_device()
        print(f"  bind_cuda {time.perf_counter()-t0:.1f}s on_device={sector.on_device}",
              flush=True)

    # Seed: the ice unit vectors, padded to ``width`` with random columns so the
    # filter has room to work (the surplus is what the +24 margin buys).
    block = np.zeros((dim, width), dtype=dtype)
    block[unique_rows, np.arange(share)] = 1.0
    if width > share:
        rng = np.random.default_rng(args.sector)
        extra = rng.standard_normal((dim, width - share)).astype(np.float32)
        block[:, share:] = extra.astype(dtype)
        del extra
    mv = MatvecCounter(lambda v, s=sector: s.apply(
        np.ascontiguousarray(v.astype(np.complex128))) + shift * v)
    _, upper = spectral_bounds(lambda v: mv.one(v), dim)
    # Initial cut = top Ritz value of the seed. Accumulate X^H (H X) one column
    # at a time; column_stacking all `width` matvecs in complex128 would be a
    # full block in DOUBLE (~454 GB here) and OOMs against the fp32 block.
    from chebfsi_lowmem import _gram  # noqa
    k = block.shape[1]
    small = np.zeros((k, k), dtype=np.complex128)
    for j in range(k):
        hj = mv.one(block[:, j].astype(np.complex128))
        for row in range(0, dim, 1 << 20):
            piece = block[row:row + (1 << 20)].astype(np.complex128)
            small[:, j] += piece.conj().T @ hj[row:row + (1 << 20)]
    cut = float(np.linalg.eigvalsh(0.5 * (small + small.conj().T)).max())
    del small
    print(f"  upper={upper:.6g} initial cut={cut:.6g} rss={rss_gb():.1f} GB", flush=True)

    t0 = time.perf_counter()
    values, block = chebfsi_lowmem(mv, block, share, degree=args.degree,
                                   outer=args.outer, cut=cut, upper=upper,
                                   row_chunk=1 << 20, col_chunk=8)
    elapsed = time.perf_counter() - t0
    print(f"  chebfsi {elapsed/3600:.2f} h  matvecs={mv.count}  rss={rss_gb():.1f} GB",
          flush=True)

    # <ice_j | psi_m> = conj(amp_j) * block[row_j, m]
    projected = np.zeros((n_ice, block.shape[1]), dtype=np.complex128)
    projected[hit, :] = amp[hit].conj()[:, None] * block[rows, :].astype(np.complex128)
    out = os.path.join(args.outdir, f"sector{args.sector}.npz")
    np.savez_compressed(out, values=np.asarray(values, float), projected=projected,
                        share=share, width=width, dim=dim,
                        momentum=MOMENTUM[args.sector], matvecs=mv.count,
                        seconds=elapsed, dtype=str(np.dtype(dtype)))
    print(f"  wrote {out}  projected={projected.shape}", flush=True)
    print(json.dumps({"sector": args.sector, "momentum": MOMENTUM[args.sector],
                      "share": share, "width": width, "matvecs": mv.count,
                      "hours": elapsed / 3600, "peak_rss_gb": rss_gb()}), flush=True)


if __name__ == "__main__":
    main()
