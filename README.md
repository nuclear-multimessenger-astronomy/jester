[![CI](https://github.com/nuclear-multimessenger-astronomy/jester/actions/workflows/ci.yml/badge.svg)](https://github.com/nuclear-multimessenger-astronomy/jester/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://nuclear-multimessenger-astronomy.github.io/jester/)
[![arXiv](https://img.shields.io/badge/arXiv-2504.15893-b31b1b.svg)](https://arxiv.org/abs/2504.15893)

# JESTER

JAX-accelerated nuclear equation of state code and TOV solver -- with support for automatic differentiation!

## Installation

```bash
pip install jesterTOV
```

With optional dependencies:
```bash
pip install jesterTOV[examples]  # For running example notebooks
pip install jesterTOV[dev]       # For development (testing, pre-commit)
pip install jesterTOV[docs]      # For building documentation
```

Or install from source:
```bash
pip install git+https://github.com/nuclear-multimessenger-astronomy/jester
```

For GPU support:
```bash
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Documentation

📚 **[Read the full documentation →](https://nuclear-multimessenger-astronomy.github.io/jester/)**

### Building Documentation Locally

To build and view the documentation on your local machine:

```bash
# Install documentation dependencies
uv pip install -e ".[docs]"

# Build the documentation
uv run sphinx-build docs docs/_build/html

# Open in your browser
open docs/_build/html/index.html  # macOS
xdg-open docs/_build/html/index.html  # Linux
```

## Example notebooks

- `examples/eos_tov.ipynb`: Basic EOS and TOV solving
- `examples/automatic_differentiation.ipynb`: Gradient-based optimization

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
