# NICER Mass-Radius Posterior Samples

Mass-radius posteriors from NICER X-ray timing observations, extracted from Zenodo archives and saved as lightweight `.npz` files containing only mass, radius, and metadata.

## How to reproduce from scratch

```bash
uv run python NICER/download_nicer.py
```

The script (`download_nicer.py`) handles the full pipeline in one go:

1. **Download** — fetches Zenodo archives via `zenodo_get` into `NICER/zenodo_data/` (gitignored).
   Requires: `uv pip install zenodo-get`
2. **Extract** — parses raw text/tar.gz files and saves `.npz` outputs to this directory.
3. **Downsample** — reduces large posteriors to at most 100,000 samples (fixed seed 42).

To tweak behaviour, edit the constants at the top of `download_nicer.py`:

| Constant | Default | Effect |
|---|---|---|
| `IGNORE_CACHE` | `False` | Re-download/re-extract even if files exist |
| `DOWNLOAD_ZENODO` | `True` | Fetch raw archives from Zenodo |
| `EXTRACT_AMSTERDAM` | `True` | Extract Amsterdam tar.gz archives |
| `MAX_SAMPLES` | `100_000` | Downsample target (None = keep all) |

Zenodo record metadata lives in `zenodo_downloader.py`.

---

## File format

All `.npz` files contain:
- `radius` — equatorial circumferential radius in km
- `mass` — gravitational mass in solar masses
- `metadata` — dict with source, paper, Zenodo URL, hotspot model, etc.

```python
import numpy as np
data = np.load('filename.npz', allow_pickle=True)
radius = data['radius']        # km
mass   = data['mass']          # Msun
meta   = data['metadata'].item()
```

---

## PSR J0437−4715

### Amsterdam group — Choudhury et al. 2024 ([ApJL 971, L20](https://inspirehep.net/literature/2806113))
**Zenodo:** https://zenodo.org/records/13766753
**Data:** NICER-only
**Hotspot model:** CST+PDT (headline result, 3C50 background with AGN model)
**Source file:** `headline_result_samples_and_contours.tar.gz` → equal-weight samples

Files:
- `J04374715_amsterdam_CST_PDT_NICER_only_Choudhury2024.npz`

### Maryland group — Miller, Dittmann, Holt et al. 2026 ([ApJL 1000, L48](https://inspirehep.net/literature/3091115))
**Zenodo:** https://zenodo.org/records/17833896
**Data:** NICER-only
**Hotspot model:** 3-spot+GPL (headline result)
**Source file:** `J0437_NICER_RM.txt` — raw file stores importance weights alongside mass and radius, so the extraction script importance-resamples it to an equal-weight posterior before saving.

Files:
- `J04374715_maryland_3spotGPL_NICER_only_RM.npz`

---

## PSR J0614−3329

### Amsterdam group — Mauviard et al. 2025, "A NICER View of the 1.4 Msun Edge-on Pulsar PSR J0614-3329" ([ApJ 995, 60](https://inspirehep.net/literature/2936527))
**Zenodo:** https://zenodo.org/records/17380576
**Data:** NICER-only
**Hotspot model:** ST+PDT (headline result)
**Source file:** `Headline_Contours_and_Samples.tar.gz` → equal-weight samples

Files:
- `J06143329_amsterdam_ST_PDT_NICER_only_Mauviard2025.npz`

### Maryland group — Miller, Dittmann, Holt, et al. 2026, "The Radius of the Neutron Star PSR J0614-3329 from NICER Data" (arXiv:2609.00965)
**Zenodo:** https://zenodo.org/records/22131748
**Data:** NICER-only
**Hotspot model:** three circular spots (headline result)
**Source file:** `J0614_NICER_rm.txt` — raw file stores importance weights alongside mass and radius rather than equal-weight samples, so the extraction script importance-resamples it to an equal-weight posterior before saving.

Files:
- `J06143329_maryland_3circle_NICER_only_RM.npz`

---

## PSR J0030+0451

First millisecond pulsar observed by NICER with sufficient quality for mass-radius inference, analyzed independently by Maryland and Amsterdam groups, with the Amsterdam group having since updated its analysis as more NICER exposure accumulated.

### Amsterdam group — Kini, Mauviard, Salmi, et al. 2026 ("A NICER View of PSR J0030+0451: Updated Constraints from Six Years of NICER Observations", arXiv:2602.23743)
**Zenodo:** https://zenodo.org/records/18741942
**Data:** NICER+XMM (2017 Jul - 2023 Jan NICER exposure, ~50% more counts than the 2017-2018 dataset)
**Hotspot model:** PDT-U (Bayes-preferred over ST+PDT; only model released on Zenodo)
**Source file:** `equal_weight_samples_PDTU.txt` — already an equal-weight posterior (a separate `weighted_samples_PDTU.txt` with raw MultiNest weights is also on Zenodo but not used here)

Files:
- `J00300451_amsterdam_PDTU_NICERXMM_Kini2026.npz`

This is the **recommended/default Amsterdam dataset** for J0030+0451, used by the pretrained `amsterdam_pdtu` flow (see `jesterTOV/inference/flows/models/nicer_maf/J00300451/amsterdam_pdtu/`).

### Maryland group — Miller et al. 2019 ([ApJL 887, L24](https://inspirehep.net/literature/1770430))
**Zenodo:** https://zenodo.org/records/3473466

Two hotspot geometries × two prior variants:
- `J00300451_maryland_2spot_NICER_only_RM.npz`
- `J00300451_maryland_2spot_NICER_only_full.npz`
- `J00300451_maryland_3spot_NICER_only_RM.npz`
- `J00300451_maryland_3spot_NICER_only_full.npz`

"RM" = restricted-model prior; "full" = broader prior allowing more geometric freedom.

### Amsterdam group — Vinciguerra et al. 2023 ([ApJ 961, 62](https://inspirehep.net/literature/2689277))
**Zenodo:** https://zenodo.org/records/8239000
**Data:** NICER-only, four hotspot models — the paper's designated "reference run" for each (settings: SE 0.3/0.8, ET 0.1, LP 1e4, MM on)
**Source:** `updated_analyses_PSRJ0030_up_to_2018_NICER_data.tar.gz` (~7 GB full reproduction package) → `post_equal_weights.dat` for each model's reference run

Files:
- `J00300451_amsterdam_ST_U_NICER_only_Vinciguerra2023.npz`
- `J00300451_amsterdam_ST_PST_NICER_only_Vinciguerra2023.npz`
- `J00300451_amsterdam_ST_PDT_NICER_only_Vinciguerra2023.npz`
- `J00300451_amsterdam_PDT_U_NICER_only_Vinciguerra2023.npz`

The mass/radius column assignment and specific run/resume directory for each model were verified by matching the median and 68% credible interval of the extracted samples against the paper's Table `tab:compare_models` NICER-only row (all four matched to within Monte Carlo noise). Superseded for J0030 headline purposes by the Kini et al. 2026 dataset above (six years of NICER data instead of two), but useful for cross-checking the ST-U/ST+PST/ST+PDT/PDT-U model comparison discussed in the paper.

### Amsterdam group — Riley et al. 2019 ([ApJL 887, L21](https://inspirehep.net/literature/1770425))
**Zenodo:** https://zenodo.org/records/7096789

Five hotspot models, NICER-only:
- `J00300451_amsterdam_ST_S_NICER_only_Riley2019.npz`
- `J00300451_amsterdam_ST_U_NICER_only_Riley2019.npz`
- `J00300451_amsterdam_CDT_U_NICER_only_Riley2019.npz`
- `J00300451_amsterdam_ST_EST_NICER_only_Riley2019.npz`
- `J00300451_amsterdam_ST_PST_NICER_only_Riley2019.npz`

"ST" = symmetric spot; "U/S" = unrestricted/shared geometry; "CDT/EST/PST" = compound spot topologies. Superseded by the Kini et al. 2026 dataset above, which uses six years of NICER data instead of two.

---

## PSR J0740+6620

Massive millisecond pulsar providing high-density EOS constraints, analyzed by both groups.

### Maryland group — Miller et al. 2021 ([ApJL 918, L28](https://inspirehep.net/literature/1863305))
**Zenodo:** https://zenodo.org/records/4670689

Three dataset combinations × two prior variants:
- `J07406620_maryland_unknown_NICER_only_RM.npz`
- `J07406620_maryland_unknown_NICER_only_full.npz`
- `J07406620_maryland_unknown_NICERXMM_RM.npz`
- `J07406620_maryland_unknown_NICERXMM_full.npz`
- `J07406620_maryland_unknown_NICERXMM_relative_RM.npz`
- `J07406620_maryland_unknown_NICERXMM_relative_full.npz`

"NICERXMM" = joint NICER+XMM-Newton; "relative" = relative calibration between instruments.

### Amsterdam group — Salmi et al. 2024 (most recent, [ApJ 974, 294](https://inspirehep.net/literature/2800506))
**Zenodo:** https://zenodo.org/records/10519473

Equal-weight samples from NICER+XMM analysis, gamma hotspot model:
- `J07406620_amsterdam_gamma_NICERXMM_equal_weights_recent.npz`

### Amsterdam group — Riley et al. 2021 ([ApJL 918, L27](https://inspirehep.net/literature/1863307))
**Zenodo:** https://zenodo.org/records/7096886
**Data:** NICER+XMM, ST-U hotspot model
**Source:** `STU_NICERxXMM_FIH_run11.tar.gz` (MultiNest run archive, ~97 MB) → `post_equal_weights.dat`, already equal-weight

Files:
- `J07406620_amsterdam_STU_NICERXMM_Riley2021.npz`

Only this small run archive is fetched directly (not the full ~8 GB record, which also contains raw event lists, calibration products, and notebooks not needed for the M-R posterior). Mass/radius column assignment was verified against the paper's headline result (M = 2.072+0.067-0.066 Msun, R = 12.39+1.30-0.98 km). Superseded by the Salmi et al. 2024 dataset above.

## PSR J1614−2230

### Amsterdam group — Mauviard et al. 2026, "A NICER view of PSR J1614−2230: a massive and compact millisecond pulsar"
**Paper:** [arXiv:2609.00172](https://arxiv.org/abs/2609.00172)
**Zenodo:** https://zenodo.org/records/22163155
**Data:** NICER + XMM-Newton + Chandra
**Hotspot model:** ST-U (headline result)
**Source file:** `MR_samples_and_contours_J1614.tar.gz` (~9 MB) → `J1614_STU_40kLP_0p03SE_0p1ET_mrsamples_post_equal_weights`, already equal-weight (mass in Msun, then radius in km)

Files:
- `J16142230_amsterdam_STU_NICER_only_Mauviard2026.npz`

Column order was checked against the paper's headline result (M = 1.937 Msun, R_eq = 10.06 +1.25/−0.87 km).

## Pulsars without samples

- **PSR J2124−3358** (González-Caniulef et al. 2026, [arXiv:2607.03721](https://arxiv.org/abs/2607.03721)): at the time of writing the posterior samples are **not yet public**. The Zenodo record (https://zenodo.org/records/20640366) only contains plots and credible-region contours; the full reproduction files are to be released upon acceptance of the paper.
- **PSR J1231−1411** (Salmi et al. 2024, [ApJ 976, 58](https://inspirehep.net/literature/2831873), Zenodo https://zenodo.org/records/13358349): samples are public, but we know from private communication that they are probably not suitable for equation-of-state inference, so they are deliberately not used.
