# JESTER Inference Examples

This directory contains example configurations for running Bayesian inference to constrain the neutron star equation of state (EOS) using various observational data.

## Available Examples

### 1. Prior-Only Sampling (`prior/`)
**Focus**: Sample from the prior without observational constraints

Uses a zero likelihood to sample purely from the prior distribution. Essential for understanding your prior assumptions, testing the inference setup, and comparing against data-constrained results.

**Use this example if you want to**:
- Test your prior specification
- Generate prior predictive samples
- Debug the inference pipeline without data
- Create a baseline for comparison with data-constrained inference

**Data required**: None (no data files needed)

[See full documentation →](prior/README.md)

---

FIXME: The other examples need to be implemented as well. 