# Data for the likelihoods of Koehn et al. (2025), arXiv:2402.04172

These tables are derived from the analysis directory of the paper. The tests in
`tests/test_inference/test_koehn2024_likelihoods.py` compare likelihood values obtained from them with the
per-EOS values of that analysis, and evaluating them on the 100 000 EOS of the paper reproduces its Table IV.

| File | Content | Source |
|---|---|---|
| `rex_PREX.npz`, `rex_CREX.npz`, `rex_dipole.npz` | log KDE on (E_sym, L_sym) | samples of Sec. III C of the paper; dipole KDE on 10 000 samples |
| `hic.npz` | skew-normal parameters and log sensitivity | Huth et al. (2022), Russotto et al. (2016) |
| `pqcd_miho.npz` | (n_H, p_H, weight) at mu_H = 2.6 GeV | Gorda et al. (2023), code of Komoltsev et al. |
| `pqcd_star.npz` | whitened log KDE of the conditioned high-density ensemble | Komoltsev et al. (2024), Zenodo 10592568 |
| `mr_4U_1702.npz` | M-R posterior grid, model D | Naettilae et al. (2017) |
| `mr_wCen.npz`, `mr_X5.npz` | M-R posterior grids, H atmosphere | Steiner et al. (2018) |
| `mr_SAX.npz` | log KDE on the (R, M) grid | Goodwin et al. (2019) |
| `mr_J1731.npz` | log KDE on the (R, M) grid | Doroshenko et al. (2022), Zenodo 6702215 |
| `mr_J0030_*.npz` | log KDE on the (R, M) grid | Riley et al. (2019), Zenodo 3386448; Miller et al. (2019), Zenodo 3473465 |
| `mr_J0740_*.npz` | log KDE on the (R, M) grid | Salmi et al. (2024), Zenodo 10519472; Dittmann et al. (2024), Zenodo 10215108 |
| `grb211211a.npz` | log KDE of the total ejecta mass | Kunert et al. (2024) |
| `postmerger_gw170817.npz` | binary masses and ejecta samples | joint GW170817 + AT2017gfo + GRB170817A inference of the paper |

The tables are built by `build_jester_tables.py` in the reproduction repository.
