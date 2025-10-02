# The Real Theory of Everything: The Law of Laws

**Author:** Christopher Lamarr Brown (Breezon Brown)  
**Organization:** NohMad LLC  
**DOIs:**  
- Canonical Paper: [10.5281/zenodo.17227616](https://doi.org/10.5281/zenodo.17227616)  
- Covenant License Agreement: [10.5281/zenodo.17236583](https://doi.org/10.5281/zenodo.17236583)  
- Supplemental Evidence: [10.5281/zenodo.17221217](https://doi.org/10.5281/zenodo.17221217)  

---

## Overview

This repository contains the **formal proof, code, and evidence** for the **Law of Laws** — a meta-law establishing that:

- **Truth = Consistency**  
- **Mechanism = Recursion**  
- **Signature = Invariance**

The Law of Laws is the **necessary and sufficient framework** for lawful behavior in any system.  
It is falsifiable, reproducible, and secured by preregistration, code, datasets, and SHA-256 manifests.

---

## Contents

- `paper/` — LaTeX source and compiled PDF: *The Real Theory of Everything: The Law of Laws*  
- `code/` — Engines for falsification testing  
  - `sgce_v3.py` — preregistered stability + collapse engine  
  - `vnf.py` — threshold-free engine for cross-validation  
  - `utils/` — surrogate generation (IAAFT, phase-randomized), permutation protocols, stability scoring  
- `data/` — Evidence datasets (QRNG, EEG, surrogate series)  
- `results/` — Replication artifacts (CSV, JSON, plots)  
- `manifests/` — SHA-256 preregistration manifests  

---

## How to Run

### Requirements
- Python 3.9+  
- NumPy, SciPy, Pandas, Matplotlib  
- Reproducibility requires installing the exact versions in `requirements.txt`

### Quickstart
```bash
# Run batch falsification with preregistered settings
python sgce_v3.py --batch --n 100000 \
  --out ./results/artifacts \
  --plots_dir ./results/plots \
  --write_label_counts ./results/label_counts.csv \
  --emit_hashes ./results/hashes.csv
Permutation collapse test: verifies invariants disappear under order destruction.

Surrogate collapse test: verifies invariants disappear under marginals-only surrogates.

Survival: If invariants persist under both falsifiers, that constitutes a valid counterexample to the Law.

Forensic Proof Chain
Preregistration: Thresholds, seeds, and windows are declared in advance.

Hashing: All results (CSV, JSON, PNG) are secured with SHA-256 digests.

Dual Engine: Results are replicated on both SGCE (thresholded) and vNF (threshold-free).

Collapse Signature: Original data shows lawful invariants; surrogates and permutations destroy them.

Conclusion: The Law of Laws survives falsification across QRNG, EEG, and multiple alien domains.

Falsifiability Clause
This claim is falsifiable.
A single lawful system whose invariants persist after permutation and surrogate scrambling would refute it.
All seeds, data, and code are published for adversarial replication.

License
This work is governed by the Covenant License Agreement (DOI: 10.5281/zenodo.17057689).
All replication artifacts are locked by SHA-256 digests to ensure byte-level immutability.

Citation
If you use this work, cite as:

bibtex
Copy code
@misc{brown2025lawoflaws,
  author       = {Christopher Lamarr Brown (Breezon Brown)},
  title        = {The Real Theory of Everything: The Law of Laws},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17227616},
  url          = {https://doi.org/10.5281/zenodo.17227616}
}
