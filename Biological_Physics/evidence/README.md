# Evidence

Validation chains, scripts, and supporting data for the biological physics framework.

## MCMC Analyses

### G-002 — H_min Architecture Floor Validation
- 8 architecture class parameters
- 5 independent chains, N_walkers = 64, N_burn = 1,000, N_prod = 10,000
- All R-hat < 1.001
- 8×10^5 total posterior samples
- Principal discovery: immune class correction 0.795 → 0.8389 ± 0.004 (6.44σ)

### G-008 — Cancer Floor Breach Validation
- 27/28 TCGA cancer types confirmed: A_tumor > A_normal
- n = 4,304 matched tumor-normal pairs
- Zero free parameters
- TGCT inversion (1/28): predicted as architectural inversion, confirmed

## Figures

| File | Description |
|------|-------------|
| `fig_thermodynamic_validation.py` | Reproduces the four-panel validation figure |
| `build_cell_architecture_card.py` | Generates the terminal class architecture card |

## Architecture Class H_min Registry (G-002 MCMC Posteriors)

| Class | H_min | R-hat | Notes |
|-------|-------|-------|-------|
| Pluripotent stem | 0.982166 | < 1.001 | hESC H1 reference |
| Adult tissue stem | 0.873718 | < 1.001 | HSC reference |
| Progenitor | 0.852216 | < 1.001 | CD34+ reference |
| Terminal/post-mitotic | 0.772837 | 0.9998 | Lowest floor; neurons |
| Cycling epithelial | 0.856055 | < 1.001 | Colonic mucosa |
| Immune/hematopoietic | 0.838889 | < 1.001 | **Corrected** from 0.795 |
| Secretory glandular | 0.843264 | < 1.001 | Breast tissue |
| Stromal | 0.862950 | < 1.001 | Fibroblast reference |

Global floor (H_min_global): **0.756500** — frontal cortex neuron, Lister 2013 / Roadmap E073
