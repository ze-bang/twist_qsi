import itertools, numpy as np
ref = np.load("campaign/outputs/nonperturbative_cubic16_p0p046000.npz")
def storage_index(M):
    visited, reps, selfc = set(), [], []
    for idx in itertools.product(range(M), repeat=3):
        if idx in visited: continue
        neg = tuple((-v) % M for v in idx)
        visited.add(idx); visited.add(neg)
        (selfc if idx == neg else reps).append(idx if idx == neg else (idx, neg))
    pos = {idx: k for k, idx in enumerate(selfc)}
    for k, (idx, neg) in enumerate(reps):
        pos[idx] = len(selfc)+2*k; pos[neg] = len(selfc)+2*k+1
    return pos
for M, idx in ((3,(0,1,1)), (4,(1,1,2))):
    lex = list(itertools.product(range(M), repeat=3)).index(idx)
    sto = storage_index(M)[idx]
    print(f"M={M} {idx}: lexicographic slot={lex}  old-storage slot={sto}")
    if f"M{M}_character_indices" in ref.files:
        print(f"   npz records index at lex slot {lex} as "
              f"{tuple(ref[f'M{M}_character_indices'][lex])}, at old slot {sto} as "
              f"{tuple(ref[f'M{M}_character_indices'][sto])}")
    mine = np.load(f"campaign/pilot/h_mine_M{M}_{''.join(map(str,idx))}.npy")
    ops = ref[f"M{M}_operators_by_character"]
    print(f"   max|h_mine - ops[lex]| = {np.abs(mine-ops[lex]).max():.3e}")
    print(f"   max|h_mine - ops[old]| = {np.abs(mine-ops[sto]).max():.3e}")
