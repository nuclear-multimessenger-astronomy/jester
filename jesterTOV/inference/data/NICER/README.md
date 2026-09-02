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

First millisecond pulsar observed by NICER with sufficient quality for mass-radius inference, analyzed independently by Maryland and Amsterdam groups.

### Maryland group — Miller et al. 2019 ([ApJL 887, L24](https://inspirehep.net/literature/1770430))
**Zenodo:** https://zenodo.org/records/3473466

Two hotspot geometries × two prior variants:
- `J00300451_maryland_2spot_NICER_only_RM.npz`
- `J00300451_maryland_2spot_NICER_only_full.npz`
- `J00300451_maryland_3spot_NICER_only_RM.npz`
- `J00300451_maryland_3spot_NICER_only_full.npz`

"RM" = restricted-model prior; "full" = broader prior allowing more geometric freedom.

### Amsterdam group — Riley et al. 2019 ([ApJL 887, L21](https://inspirehep.net/literature/1770425))
**Zenodo:** https://zenodo.org/records/7096789

Five hotspot models, NICER-only:
- `J00300451_amsterdam_ST_S_NICER_only_Riley2019.npz`
- `J00300451_amsterdam_ST_U_NICER_only_Riley2019.npz`
- `J00300451_amsterdam_CDT_U_NICER_only_Riley2019.npz`
- `J00300451_amsterdam_ST_EST_NICER_only_Riley2019.npz`
- `J00300451_amsterdam_ST_PST_NICER_only_Riley2019.npz`

Recommended model: ST+PST. "ST" = symmetric spot; "U/S" = unrestricted/shared geometry; "CDT/EST/PST" = compound spot topologies.

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
