"""Which high-symmetry points does the FCC-32 cluster actually reach?

The (Z_2)^3 translation group allows eight momenta; label each against the FCC
Brillouin-zone points so the DSSF panels can be named honestly.
"""
import itertools, sys
import numpy as np
sys.path.insert(0, "/lustre09/project/6003507/zhouzb79/twist_qsi/notes")
from recompute_finite_size_artifact import build_cluster, FCC_A

cl = build_cluster("fcc", (2, 2, 2))
A = np.asarray(FCC_A, dtype=float)           # primitive FCC vectors, a=1
L = cl.Lvecs                                 # supercell = 2 x primitive
B = 2.0 * np.pi * np.linalg.inv(L).T         # supercell reciprocal vectors
print("primitive FCC vectors:\n", A)
print("supercell reciprocal vectors (rows):\n", np.round(B, 4))

# conventional-cubic units: FCC high-symmetry points in 2pi/a
named = {
    "Gamma": np.array([0, 0, 0]),
    "X":     np.array([0, 0, 1]),
    "L":     np.array([0.5, 0.5, 0.5]),
    "W":     np.array([0.5, 0, 1]),
    "K":     np.array([0.75, 0.75, 0]),
}
recip_conv = 2 * np.pi * np.array([[ -1, 1, 1], [1, -1, 1], [1, 1, -1]])  # FCC->BCC recip

def classify(k):
    best, bestd = None, np.inf
    for name, point in named.items():
        target = 2 * np.pi * point
        for signs in itertools.product([1, -1], repeat=3):
            for perm in itertools.permutations(range(3)):
                cand = target[list(perm)] * np.array(signs)
                # equivalent modulo a reciprocal lattice vector of the FCC lattice
                diff = k - cand
                frac = np.linalg.solve(recip_conv.T, diff)
                d = np.linalg.norm(frac - np.rint(frac))
                if d < bestd:
                    bestd, best = d, name
    return best, bestd

print(f"\n{'sector':>7} {'(n1,n2,n3)':>12} {'k (2pi/a units)':>28} {'label':>7} {'resid':>9}")
counts = {}
for idx, n in enumerate(itertools.product((0, 1), repeat=3)):
    k = np.asarray(n, float) @ B
    label, resid = classify(k)
    counts[label] = counts.get(label, 0) + 1
    print(f"{idx:>7} {str(n):>12} {str(np.round(k/(2*np.pi), 3)):>28} {label:>7} {resid:>9.2e}")
print("\nmultiplicity:", counts)
