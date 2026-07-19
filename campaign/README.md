# Campaign outputs

`run_validation.py` is the authoritative local campaign.  It reconstructs
the cubic-16 and FCC-32 geometries, generates the order-two/order-three row
tables, performs zero-transport selection, combines the cubic clean band with
the exact 65,536-level microscopic spectrum, and evaluates the QMC metrics.

The exact cubic spectrum is cached because it is deterministic and moderately
expensive.  FCC-32 results produced here are explicitly labeled **order 3,
ice manifold**; they are not full-Hamiltonian ED.

The external QMC heat-capacity and entropy curves under `data/` are vector
extractions, not author raw data.  Their provenance and extraction script are
kept beside them.  QMC-SAC dynamics are a registered but pending gate.
