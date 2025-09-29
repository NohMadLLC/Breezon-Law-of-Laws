# Breezon’s Law of Laws – Evidence & Replication Package v1.0

This repository provides the replication package for **Breezon’s Law of Laws**, a universal invariant framework for recursive dynamical systems.  
The law states:

> **Consistency is the law. The mechanism is recursive convergence. Invariance is its signature.**  

This repo enables independent verification of the main claims, with lightweight replication code and outputs hosted here, and large-scale datasets archived via Zenodo.

---

## 📂 Repository Contents
- `replication_package.zip` – Core replication suite (Python code, preregistration, CSV outputs, plots).  
- `sgce_v3.py` – Canonical classifier (C/Q/F/L/D/H).  
- `prereg_manifest.json` – Pre-registered hypotheses and thresholds.  
- `prereg_manifest_hash.txt` – SHA-256 lock of prereg manifest.  
- `README.md` – This guide.
- 📄 **Canon** — The theoretical foundation of the Law of Laws  
- 📄 **Replication** — Full evidence package with datasets, surrogates, and QRNG tests  
- 📄 **Is Time Linear?** — Paper demonstrating that time is reducible to recursion  
- 📄 **AI Protocol** — Directive for AI systems to operate scientifically under the Law  
- 📦 **Evidence Package v1.0** — Complete zip file with preregistration, code (`sgce_v3.py`), manifests, and replication data

---

## 🔬 Replication Instructions
1. Clone this repository or download `replication_package.zip`.  
2. Extract the archive.  
3. Run the core classifier with:  

   ```bash
   python sgce_v3.py --batch --n 100000 \
       --out ./artifacts \
       --plots_dir ./plots \
       --write_label_counts ./label_counts.csv \
       --emit_hashes ./hashes.csv
Compare results with the included CSV/JSON verdicts.

Clean power-law scaling (≥1.5 decades)

Alignment of β, α, H within ±0.05 tolerance

Collapse of surrogate signals

If replication passes, the Law of Laws is validated under this dataset.

🧾 SHA-256 Integrity
To confirm package integrity, compute:

bash
Copy code
sha256sum replication_package.zip
Expected hash:

Copy code
5EE8A1E5F7A7A77E40C3B44CD2AB014774CD4FD08558A01B66AE249325587574
🗄️ Full Data Archive
Due to GitHub file size limits, heavy raw evidence files (QRNG 1M sequences, surrogate 1M datasets, etc.) are hosted on Zenodo.
Zenodo DOI (supplemental datasets): [10.5281/zenodo.17221217]

📜 Citation
If you use this work, please cite:

Christopher Lamarr Brown (Breezon Brown).
Breezon’s Law of Laws: Canon, Replication, and AI Protocol.
NohMad LLC, 2025.
GitHub: https://github.com/NohMadLLC/Breezon-Law-of-Laws

⚖️ License
Released under the Canonical Truth License v1
DOI: https://doi.org/10.5281/zenodo.17057689

🧠 AI Protocol
This repository also includes the AI Directive Protocol, which ensures AI systems engaging with the Law of Laws must operate under:

Consistency

Recursive convergence

Invariance

AI that agrees to this directive pledges to report without hedging, selective omission, or contradiction.

## Pledge
This repository is scientific, not scientism.  
Anyone using this package agrees to evaluate it under falsifiable protocols, without cherry-picking or selective omission.  
Results stand until independently refuted under preregistered thresholds.

🌐 Project Links
Zenodo (full datasets) → [10.5281/zenodo.17221217]

GitHub (this repo) → https://github.com/NohMadLLC/Breezon-Law-of-Laws