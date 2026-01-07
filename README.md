
# ECG-MNMF (NF-CCE Reproduction + Improvements)

This repo contains a **Python3 reproduction** of **NF-CCE** and our improved method **ECG-MNMF** for multiplex community detection.

**ECG-MNMF = NF-CCE (Step-2 Consensus) +**
- **Sparse SVD Initialization** (more stable, faster convergence)
- **Graph Laplacian Regularization** (topology smoothness)
- **Debiased Contrastive Learning (InfoNCE)** (cross-view consistency, pseudo-label filtering)

---

## Requirements
Tested on Python 3.10+.

Install dependencies:
```bash
pip install numpy scipy scikit-learn matplotlib
```


## Dataset

Place datasets under:

```
data/nets/        # multiplex benchmarks (.mat): MDP / CORA / CiteSeer / MIT
data/bionets/     # BioNet datasets (.mat): human / mouse / yeast
```


## Run (Multiplex Benchmarks: MDP / CORA / CiteSeer)

Run a single demo:

```bash
python demo.py
```

Run ablation (auto runs variants and prints a result table with NMI/ARI/time):

```bash
python demo_ablation.py
```

------

## Run (BioNet: human / mouse / yeast)

Run a single BioNet demo (Average Redundancy):

```bash
python demo_bionets.py
```

Run BioNet ablation:

```bash
python demo_bionets_ablation.py
```

------

## Output Metrics

- **Multiplex datasets**: NMI / ARI
- **BioNet datasets**: Average Redundancy
- **Runtime**: total wall-clock time (seconds)

------

## Notes

- Original NF-CCE code is Python2 and some preprocessing details are missing, so exact paper numbers may not be reproducible.
- All methods in this repo are evaluated under the **same preprocessing + same evaluation pipeline** for fair comparison.

------

## References

- NF-CCE: Gligorijević et al., *TPAMI 2019*
- CDNMF: Li et al., *arXiv 2024*
-------


## Results

| Method              | MDP (NMI/ARI)     | CORA (NMI/ARI)    | CiteSeer (NMI/ARI) |
| ------------------- | ----------------- | ----------------- | ------------------ |
| SNMF                | 0.2418/0.0450     | 0.6770/0.7402     | 0.2868/0.2515      |
| NF-CCE              | 0.4077/0.2862     | 0.6575/0.7325     | 0.1314/0.1172      |
| **ECG-MNMF (Ours)** | **0.5030/0.3735** | **0.8687/0.9093** | **0.1888/0.1475**  |

