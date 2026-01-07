[![CI](https://github.com/nuclear-multimessenger-astronomy/jester/actions/workflows/ci.yml/badge.svg)](https://github.com/nuclear-multimessenger-astronomy/jester/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://nuclear-multimessenger-astronomy.github.io/jester/)
[![arXiv](https://img.shields.io/badge/arXiv-2504.15893-b31b1b.svg)](https://arxiv.org/abs/2504.15893)

# JESTER

JAX-accelerated equation of state inference and TOV solvers

`jester` is a package to perform inference on the equation of state (EOS) with Bayesian inference and accelerates the TOV solver calls and the entire sampling procedure by using GPU hardware through `jax`. 

Currently, `jester` supports the following EOS parametrizations:
- **Metamodel**: Taylor expansion of the energy density.
- **Metamodel+CSE**: Metamodel up to breakdown density (varied on-the-fly), and speed-of-sound extrapolation above the breakdown density parametrized by linear interpolation through a grid of speed of sound values.
- **Metamodel+peakCSE**: Metamodel up to breakdown density (varied on-the-fly), and speed-of-sound extrapolation above the breakdown density parametrized to have a Gaussian peak.

Moreover, the following samplers are supported:
- **Sequential Monte Carlo** (Recommended): Implemented with [`blackjax`](https://github.com/blackjax-devs/blackjax)
- **flowMC** ([GitHub](https://github.com/kazewong/flowMC)): Normalizing flow-enhanced MCMC sampling
- **Nested sampling**: Implemented in `blackjax` in [this specific fork](https://github.com/handley-lab/blackjax)

📚 **[Read the full documentation →](https://nuclear-multimessenger-astronomy.github.io/jester/)**

## Installation

The latest stable release version can be installed with `pip`:
```bash
pip install jesterTOV
```

To run Bayesian inference, make sure to install support for CUDA or upgrade `jax` according to [the `jax` documentation page](https://docs.jax.dev/en/latest/installation.html):
```bash
pip install -U "jax[cuda12]"
```

For developers, we recommend installing locally with `uv`:
```bash
git clone https://github.com/nuclear-multimessenger-astronomy/jester
cd jester
uv sync
```

Extra dependencies can be installed as follows:
```bash
uv sync --extra cuda12 # For GPU support
uv sync --extra docs   # To work on documentation locally
uv sync --extra tests  # To run tests locally 
```

## Examples

The `examples` folder shows how to use `jester`:
- `eos_tov`: Showing basic functionality to create an EOS from the different parametrizations supported in `jester`
- `kde_nf_validation`: Example usage of KDE and NF methods used for the NICER and GW likelihoods
- `inference`: Configuration files to run Bayesian inference on different likelihoods and with different samplers

To run the `inference` examples, navigate to the desired test case (organized as `<sampler>/<likelihood>`), and run
```bash
run_jester_inference config.yaml
```

Take a look at the `config.yaml` files, which contain all details for `jester` to execute the inference. Note that `jester` also needs a specified prior file. 

## Notes for developers

Building documentation locally:
```bash
# Install documentation dependencies
uv pip install -e ".[docs]"

# Build the documentation
uv run sphinx-build docs docs/_build/html

# Open in your browser
open docs/_build/html/index.html  # macOS
xdg-open docs/_build/html/index.html  # Linux
```

Running tests:
```bash
uv run pytest tests/
```

Code quality checks:
```bash
# Pre-commit checks
uv run pre-commit run --all-files

# Format and lint
uv run black .
uv run ruff check --fix .

# Type checking
uv run pyright                 # All files
uv run pyright jesterTOV/      # Specific directory
```

A CLAUDE.md file already exists in the repo for developers that want to use Claude Code. 

## Acknowledgements

If you use `jester` in your work, please cite our paper!
```
@article{Wouters:2025zju,
    author = "Wouters, Thibeau and Pang, Peter T. H. and Koehn, Hauke and Rose, Henrik and Somasundaram, Rahul and Tews, Ingo and Dietrich, Tim and Van Den Broeck, Chris",
    title = "{Leveraging differentiable programming in the inverse problem of neutron stars}",
    eprint = "2504.15893",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    reportNumber = "LA-UR-25-23486",
    doi = "10.1103/v2y8-kxvx",
    journal = "Phys. Rev. D",
    volume = "112",
    number = "4",
    pages = "043037",
    year = "2025"
}
```

If you use the `ptov.py` module, to enabble pressure anisotropy, please cite the following paper:
```
@article{Pang:2025fes,
    author = "Pang, Peter T. H. and Brown, Stephanie M. and Wouters, Thibeau and Van Den Broeck, Chris",
    title = "{Revealing tensions in neutron star observations with pressure anisotropy}",
    eprint = "2507.13039",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    month = "7",
    year = "2025"
}
```

Additionally, make sure to cite the following software papers which form the backbone of `jester`:
```bash 
# JAX software paper
@article{frostig2018compiling,
  title={Compiling machine learning programs via high-level tracing. Syst},
  author={Frostig, Roy and Johnson, MJ and Leary, Chris},
  journal={Mach. Learn},
  volume={4},
  number={9},
  year={2018}
}

# diffrax software paper
@misc{kidger2022neuraldifferentialequations,
      title={On Neural Differential Equations}, 
      author={Patrick Kidger},
      year={2022},
      eprint={2202.02435},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2202.02435}, 
}
```