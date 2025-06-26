# JESTER - Jax-based EoS and Tov solvER

[![CI](https://github.com/nuclear-multimessenger-astronomy/jester/actions/workflows/ci.yml/badge.svg)](https://github.com/nuclear-multimessenger-astronomy/jester/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/nuclear-multimessenger-astronomy/jester/branch/main/graph/badge.svg)](https://codecov.io/gh/nuclear-multimessenger-astronomy/jester)

**J**ax-based **E**o**S** and **T**ov solv**ER** (JESTER) consists of a set of tools for solving the TOV equation with a given equation-of-state (EOS).

Since it is based on `jax`, `jester` can make use of hardware acceleration on GPUs and TPUs without any change of the source code! It can also be easily used together with other `jax`-based software, such as [Jim](https://github.com/kazewong/jim).

## Current EoS parameterizations supported
- MetaModel
- MetaModel with speed-of-sound extension scheme

## Installation

You may install the bleeding edge version by cloning this repo, or doing
```bash
pip install git+https://github.com/nuclear-multimessenger-astronomy/jester
```
Note: in order to use GPUs, you need to install a `jaxlib` version with CUDA support by running
```bash
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Examples

We currently provided two example notebooks in the `examples` folder:
- `eos_tov.ipynb`: A simple example of how to use `jester` to use the metamodel with speed-of-sound extension to generate an EOS from a set of parameters, and solve the TOV equations.
- `automatic_differentiation.ipynb`: A more advanced example of how to use `jester` to automatically differentiate the TOV equations with respect to the EOS parameters, and compute the derivatives of neutron star properties with respect to the EOS parameters to allow for gradient-based optimization on a few toy examples.

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
