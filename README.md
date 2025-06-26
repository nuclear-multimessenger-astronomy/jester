[![CI](https://github.com/nuclear-multimessenger-astronomy/jester/actions/workflows/ci.yml/badge.svg)](https://github.com/nuclear-multimessenger-astronomy/jester/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/nuclear-multimessenger-astronomy/jester/branch/main/graph/badge.svg)](https://codecov.io/gh/nuclear-multimessenger-astronomy/jester)
[![Documentation Status](https://readthedocs.org/projects/jester/badge/?version=latest)](https://jestertov.readthedocs.io/en/latest/)
[![arXiv](https://img.shields.io/badge/arXiv-2504.15893-b31b1b.svg)](https://arxiv.org/abs/2504.15893)

# JESTER

JAX-accelerated neutron star physics with automatic differentiation support.

## Installation

```bash
pip install git+https://github.com/nuclear-multimessenger-astronomy/jester
```

For GPU support:
```bash
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Quick Example

```python
import jesterTOV as jtov
eos = jtov.eos.MetaModel_EOS_model()
masses, radii = eos.M_R_curve()
```

## Documentation

📚 **[Read the full documentation →](https://jester.readthedocs.io/)**

## Examples

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
    month = "4",
    year = "2025"
}
```
