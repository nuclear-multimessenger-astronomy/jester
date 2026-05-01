[![CI](https://github.com/nuclear-multimessenger-astronomy/jester/actions/workflows/ci.yml/badge.svg)](https://github.com/nuclear-multimessenger-astronomy/jester/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://nuclear-multimessenger-astronomy.github.io/jester/)
[![arXiv](https://img.shields.io/badge/arXiv-2504.15893-b31b1b.svg)](https://arxiv.org/abs/2504.15893)
[![Citations](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Finspirehep.net%2Fapi%2Fliterature%2F2915009&query=%24.metadata.citation_count&label=citations&color=blue)](https://inspirehep.net/literature/2915009)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nuclear-multimessenger-astronomy/jester/blob/main/examples/google_colab/GW170817_Google_Colab.ipynb)

# JESTER

*JAX-accelerated equation of state inference and TOV solvers*

`jester` performs fast and accurate Bayesian inference on the neutron star equation of state (EOS) using GPU-accelerated TOV solvers and samplers through `jax`.

> [!TIP]
> **The documentation is the best place to get started.**
> It covers installation, examples, a full Bayesian inference guide, and the API reference.
>
> **[Read the full documentation →](https://nuclear-multimessenger-astronomy.github.io/jester/)**

## Try it out

Try `jester` right away in your browser and infer the neutron star equation of state from GW170817 within 15 minutes — no installation required!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nuclear-multimessenger-astronomy/jester/blob/main/examples/google_colab/GW170817_Google_Colab.ipynb)

## What's in `jester`?

`jester` combines flexible EOS parametrizations with GPU-accelerated TOV solvers and modern Bayesian samplers. For full details on each component, see the [overview section of the documentation](https://nuclear-multimessenger-astronomy.github.io/jester/).

| Component | Description |
|---|---|
| **EOS models** | Metamodel, Metamodel + CSE, Spectral expansion |
| **TOV solvers** | General Relativity, Pressure anisotropy, Modified gravity (scalar-tensor) |
| **Likelihoods** | Nuclear experiments (chiEFT), Radio timing, NICER mass-radius, Gravitational waves |
| **Samplers** | Sequential Monte Carlo, Nested Sampling, FlowMC (normalizing flows) |

## Installation

`jester` depends on a [specific fork of blackjax](https://github.com/handley-lab/blackjax) for nested sampling support, which prevents publishing to PyPI. Install the latest version by cloning the repository:

```bash
git clone https://github.com/nuclear-multimessenger-astronomy/jester
cd jester
uv sync
```

Extra dependencies can be installed as follows:
```bash
uv sync --extra cuda12 # For GPU support (fast sampling)
uv sync --extra dev    # For developers (run tests, build docs)
```

To run Bayesian inference, make sure to install support for CUDA or upgrade `jax` according to [the `jax` documentation page](https://docs.jax.dev/en/latest/installation.html):
```bash
uv sync --extra cuda12
```

Having trouble? Check out the [FAQ](https://nuclear-multimessenger-astronomy.github.io/jester/developer_guide/faq.html).

## For developers

All development guidelines — including how to run tests, contribute code, add new EOS models, TOV solvers, or likelihoods, and how to write documentation — are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citing

If you use `jester` in your work, please cite the relevant papers according to which parts you are using.

See the [citing page in the documentation](https://nuclear-multimessenger-astronomy.github.io/jester/citing.html) for the full list of references.
