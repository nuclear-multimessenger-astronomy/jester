# NICER Mass-Radius Posterior Samples

This directory contains extracted mass-radius posterior samples from NICER X-ray timing observations of millisecond pulsars. The data has been processed from Zenodo archives and converted to lightweight NumPy npz files containing only the mass and radius posteriors needed for equation of state inference.

**Note:** The original data samples are very heavy. Therefore we downsample to 100k samples where necessary. The scripts are stored here in case the NICER data has to be downloaded again from scratch. The script to downsample is in this subdir. 

TODO: investigate `unknown` in the filename. 

## PSR J0030+0451

PSR J0030+0451 was the first millisecond pulsar observed by NICER with sufficient quality for mass-radius inference. The observations were analyzed independently by two groups using different surface emission models.

### Maryland Group

The Maryland group analyzed J0030+0451 using models with multiple potentially overlapping circular hotspots (Miller et al. 2019, ApJL 887, L24). Their analysis considered two main geometric configurations:

The "2spot" model assumes two potentially overlapping circular hotspots on the neutron star surface. The "3spot" model extends this to three circular regions. Both models were run with two different priors on the geometric parameters, labeled "RM" (restricted model) and "full". The restricted model applies tighter priors on certain geometric parameters to avoid unphysical configurations, while the full model uses broader priors allowing more geometric freedom.

All Maryland J0030 analyses used NICER data only, without additional X-ray observatories. The posteriors contain approximately 400,000 samples for the 2spot models and 1,000,000 samples for the 3spot models.

Files:
- `J00300451_maryland_2spot_NICER_only_RM.npz`
- `J00300451_maryland_2spot_NICER_only_full.npz`
- `J00300451_maryland_3spot_NICER_only_RM.npz`
- `J00300451_maryland_3spot_NICER_only_full.npz`

### Amsterdam Group

The Amsterdam group analyzed J0030+0451 using physically-motivated surface emission models based on neutron star atmosphere theory. Their models are described using abbreviated names that indicate the hotspot geometry.

The Riley et al. 2019 analysis (ApJL 887, L21) used several geometric configurations. "ST" refers to a single circular hotspot (symmetric spot), while compound models like "ST+PST" indicate a primary symmetric spot plus a secondary partially-shadowed spot. The "U" suffix denotes that the two hotspots are unrestricted in their locations, while "S" indicates they share certain geometric parameters. "CDT" refers to a ceding-dominated topology where one spot partially overlaps another. "EST" indicates a configuration with superseding and ceding regions.

The recommended model from Riley et al. 2019 is ST+PST, which provided the tightest constraints on mass and radius. This model consists of a primary symmetric circular spot and a secondary spot with a more complex partially-shadowed topology.

Files from Riley et al. 2019 (NICER-only):
- `J00300451_amsterdam_ST_S_NICER_only_Riley2019.npz`
- `J00300451_amsterdam_ST_U_NICER_only_Riley2019.npz`
- `J00300451_amsterdam_CDT_U_NICER_only_Riley2019.npz`
- `J00300451_amsterdam_ST_EST_NICER_only_Riley2019.npz`
- `J00300451_amsterdam_ST_PST_NICER_only_Riley2019.npz`

The Vinciguerra et al. 2023 analysis updated the Riley et al. 2019 results using improved calibration and additional NICER data through 2018. They considered similar geometric models and also explored joint NICER+XMM-Newton analyses. These results are available in the larger Zenodo archive but have not yet been extracted to this directory.

## PSR J0740+6620

PSR J0740+6620 is a massive millisecond pulsar that provides important constraints on the equation of state at high densities. Like J0030, it was analyzed independently by Maryland and Amsterdam groups.

### Maryland Group

The Maryland group analyzed J0740+6620 using circular hotspot models similar to their J0030 analysis (Miller et al. 2021, ApJL 918, L28). Unlike J0030, the J0740 data was analyzed with three different datasets: NICER data alone, NICER combined with archival XMM-Newton data, and NICER combined with XMM-Newton using a relative calibration approach.

The "NICER-only" analyses use only NICER observations. The "NICER+XMM" analyses jointly fit NICER and XMM-Newton data using independent calibration uncertainties for each instrument. The "NICER+XMM-relative" analyses account for systematic calibration differences between the instruments through a relative calibration parameter. As with J0030, both RM (restricted model) and full prior variants were explored.

The Maryland J0740 files do not explicitly specify the hotspot geometry in the filename (labeled as "unknown" in our extraction) because the analysis focused primarily on mass-radius constraints rather than detailed surface mapping. The models used circular spots similar to the J0030 analysis.

Files:
- `J07406620_maryland_unknown_NICER_only_RM.npz`
- `J07406620_maryland_unknown_NICER_only_full.npz`
- `J07406620_maryland_unknown_NICERXMM_RM.npz`
- `J07406620_maryland_unknown_NICERXMM_full.npz`
- `J07406620_maryland_unknown_NICERXMM_relative_RM.npz`
- `J07406620_maryland_unknown_NICERXMM_relative_full.npz`

### Amsterdam Group

The Amsterdam group analyzed J0740+6620 using atmosphere-based models with careful background estimation (Salmi et al. 2022, ApJL 956, L4). Their analysis explored multiple background estimation techniques and data combinations.

The models are abbreviated as follows. "STU" indicates a symmetric spot unrestricted configuration, which was the primary model used. "STUa" denotes an antipodal configuration where the two hotspots are constrained to be on opposite sides of the neutron star. "STS" represents a symmetric configuration where both spots share identical properties.

The data combinations include "W21_SW" which refers to the Wolff et al. 2021 data with space weather background estimation, "3C50" which refers to data analyzed with the 3C50 background calibration source, and "3C50_XMM" which combines 3C50 NICER data with XMM-Newton observations. The recommended analysis from Salmi et al. 2022 for NICER-only constraints is the 3C50 dataset with the df3X background model (allowing background to vary from -3 sigma to unbounded above the nominal estimate).

The most recent Amsterdam analysis uses the "gamma" model designation, which likely refers to an updated geometric parameterization. The files labeled "recent" come from the latest Zenodo release and represent the current recommended Amsterdam constraints for J0740+6620.

Files from recent analysis (NICER+XMM):
- `J07406620_amsterdam_gamma_NICERXMM_equal_weights_recent.npz`

We extract only the equal-weight posterior samples, which are sufficient for equation of state inference. These samples have been resampled from the nested sampling output to have uniform weights, making them straightforward to use for analysis. The file contains approximately 380,000 independent samples providing excellent statistical coverage of the posterior distribution.

## File Format

All npz files contain three arrays:

The "radius" array contains equatorial circumferential radius samples in kilometers. The "mass" array contains gravitational mass samples in solar masses. The "metadata" dictionary contains information about the source, including the pulsar name, analysis group, paper reference, Zenodo record URL, hotspot model, data combination, and number of samples.

Files can be loaded in Python using:

```python
import numpy as np
data = np.load('filename.npz', allow_pickle=True)
radius = data['radius']  # km
mass = data['mass']      # Msun
metadata = data['metadata'].item()
```

## References

Maryland Group J0030+0451:
Miller et al. 2019, "PSR J0030+0451 Mass and Radius from NICER Data and Implications for the Properties of Neutron Star Matter", ApJL 887, L24

Amsterdam Group J0030+0451:
Riley et al. 2019, "A NICER View of PSR J0030+0451: Millisecond Pulsar Parameter Estimation", ApJL 887, L21
Vinciguerra et al. 2023, "An updated mass-radius analysis of the 2017-2018 NICER data set of PSR J0030+0451", arXiv:2308.09469

Maryland Group J0740+6620:
Miller et al. 2021, "The Radius of PSR J0740+6620 from NICER and XMM-Newton Data", ApJL 918, L28

Amsterdam Group J0740+6620:
Salmi et al. 2022, "The Radius of PSR J0740+6620 from NICER with NICER Background Estimates", ApJL 956, L4

## Data Sources

All data was downloaded from Zenodo and extracted using the `explore_and_extract_nicer.py` script in the parent directory. The original Zenodo records contain the full posterior samples for all model parameters, detailed analysis code, and figures from the publications. This directory contains only the mass-radius marginal posteriors.

Maryland J0030: https://zenodo.org/records/3473466
Amsterdam J0030 (Riley 2019): https://zenodo.org/records/3524457
Amsterdam J0030 (Vinciguerra 2023): https://zenodo.org/records/8239000
Maryland J0740: https://zenodo.org/records/4670689
Amsterdam J0740 (recent): https://zenodo.org/records/10519473
