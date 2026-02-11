(citing)=
# Citing Jester

This is a guide on how to properly credit research works that introduced new features into the ``jester`` code.

_NOTE_: Papers introducing the physics underpinning the code are often described in the documentation of the respective class/functions: refer to the documentation of those classes for the correct citations.

## Jester methods papers

If you use the ``jester`` software in your research, please cite our methods paper

```bibtex
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

More specifically, if you use the TOV solver implemented in ``anistropy.py``, please cite
```bibtex
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

## Additional methods papers

Please consider citing the following papers as well:

``JAX`` paper

```bibtex
@article{frostig2018compiling,
  title={Compiling machine learning programs via high-level tracing. Syst},
  author={Frostig, Roy and Johnson, MJ and Leary, Chris},
  journal={Mach. Learn},
  volume={4},
  number={9},
  year={2018}
}
```

``diffrax`` methods paper (``JAX``-based numerical differential equation solvers), which is what our TOV solvers rely on:

```bibtex
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