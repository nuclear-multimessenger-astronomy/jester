# Data for chiral EFT likelihoods

## 2402.04172

This is the data taken from *From existing and new nuclear and astrophysical constraints to stringent limits on the equation of state of neutron-rich dense matter* by Hauke Koehn et al. The data consists of two datasets `low.dat` and `high.dat` that together comprise the uncertainty band from chiEFT. Each file contains 1650 rows with three columns: baryon number density (fm⁻³), pressure (MeV/fm³), and energy density (MeV/fm³). The density range spans from ~4×10⁻¹⁵ to 0.32 fm⁻³ (up to 2 nsat), where nuclear saturation density is taken as 0.16 fm⁻³. Only the first two columns (density and pressure) are used in the likelihood implementation.

- DOI: Phys.Rev.X 15 (2025) 2, 021014
- [Link](https://inspirehep.net/literature/2755989)