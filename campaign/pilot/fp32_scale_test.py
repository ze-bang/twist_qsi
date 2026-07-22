"""Does fp32 STORAGE degrade Cholesky-QR2 as the dimension grows?

The block is stored fp32 but every Gram and solve accumulates in complex128, so
the error should come from per-element representation (~1e-7 relative) rather
than from summing N terms -- i.e. roughly dimension independent. That is
reasoning, not measurement, and the FCC-32 production config depends on it.

Isolates dimension at fixed width, so it needs no matvecs and no eigensolve.
"""
from __future__ import annotations
import sys, time
import numpy as np
sys.path.insert(0, "/lustre09/project/6003507/zhouzb79/twist_qsi/campaign/pilot")
from chebfsi_lowmem import cholesky_qr2, _gram

WIDTH = 64
rng = np.random.default_rng(0)

print(f"width={WIDTH}, Cholesky-QR2, Gram accumulated in complex128")
print(f"{'dim':>12} {'dtype':>8} {'GB':>7} {'||Q^H Q - I||':>16} {'time':>7}")
for dim in (10**6, 10**7, 75_146_310):
    for dtype, label in ((np.complex128, "fp64"), (np.complex64, "fp32")):
        # A deliberately ill-conditioned block: an ice gather is far from
        # random, and Cholesky-QR is exactly where conditioning bites.
        base = (rng.standard_normal((dim, WIDTH)) +
                1j * rng.standard_normal((dim, WIDTH))).astype(dtype)
        base[:, 1:] += 0.9 * base[:, :1]        # correlate columns
        block = np.ascontiguousarray(base)
        del base
        t0 = time.perf_counter()
        cholesky_qr2(block, 1 << 18)
        elapsed = time.perf_counter() - t0
        gram = _gram(block, 1 << 18)
        err = np.abs(gram - np.eye(WIDTH)).max()
        gb = block.nbytes / 1e9
        print(f"{dim:>12,} {label:>8} {gb:>7.1f} {err:>16.3e} {elapsed:>6.1f}s", flush=True)
        del block, gram
