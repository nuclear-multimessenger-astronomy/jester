# JESTER - Jax-based EoS and Tov solvER

**J**ax-based **E**o**S** and **T**ov solv**ER** (JESTER) consists of a set of tools for solving the TOV equation with a given equation-of-state (EOS).

Since it is based on JAX, JESTER makes use of hardware acceleration on GPUs and TPUs. It can also be easily used together with other JAX-based software, e.g., [Jim](https://github.com/kazewong/jim).

## Current EoS parameterizations supported
- MetaModel
- MetaModel with speed-of-sound extension

## Installation

You may install the bleeding edge version by cloning this repo, or doing
```bash
pip install git+https://github.com/nuclear-multimessenger-astronomy/jester
```
Note: in order to use GPUs, you need to install a `jaxlib` version with CUDA support by running
```bash
pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Acknowledgements

We'll put a link to the arXiv paper once it's up.