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

## List of papers that used jester

Here, we showcase papers that made use of ``jester`` in their methodology. 
Did you write a paper using jester and would you like to show your work here? Open an issue or pull request!

Introducing a data-tempering sequential Monte Carlo for sequential hierarchical inference, applied to EOS inference with Einstein Telescope mock data:
```bibtex
@article{Wouters:2026nbh,
    author = "Wouters, Thibeau and Ng, Thomas C. K. and Koehn, Hauke and Gittins, Fabian and Pang, Peter T. H. and Dietrich, Tim and Van Den Broeck, Chris",
    title = "{Scalable and sequential inference of the neutron star equation of state with the Einstein Telescope}",
    eprint = "2610.07975",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    month = "10",
    year = "2026"
}
```

Constraining the EOS through direct-Urca cooling processes:
```bibtex
@article{Danarianto:2026ukh,
    author = "Danarianto, M. Dio and Grams, Guilherme and Wouters, Thibeau and Dietrich, Tim",
    title = "{Extending multi-messenger constraints on neutron star matter through the inclusion of direct Urca cooling}",
    eprint = "2609.30112",
    archivePrefix = "arXiv",
    primaryClass = "nucl-th",
    month = "9",
    year = "2026"
}]
```

Joint hierarchical inference on the EOS, the neutron star population, and cosmology with one year of multimessenger detections with the Einstein Telescope:
```bibtex
@article{Koehn:2026nin,
    author = "Koehn, Hauke and Wouters, Thibeau and Sadeh, Gilad and Pang, Peter T. H. and Bulla, Mattia and Van Den Broeck, Chris and Coughlin, Michael W. and Dietrich, Tim",
    title = "{Binary neutron stars in the next-generation era: Multi-messenger detection prospects and constraints on the equation of state, mass distribution, and cosmology}",
    eprint = "2607.28438",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    month = "7",
    year = "2026"
}
```

EOS posteriors from ``jester`` are used to build EOS-informed priors for GW data analysis:
```bibtex
@article{Wouters:2025csq,
    author = "Wouters, Thibeau and Pang, Peter T. H. and Dietrich, Tim and Van Den Broeck, Chris",
    title = "{Incorporating neutron star physics into gravitational wave inference with neural priors}",
    eprint = "2511.22987",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    month = "11",
    year = "2025"
}
```

Using ``jester`` to analyze potential implications for the EOS from a subthreshold BNS candidate, and simulating constraints derived from a similar event detected by third-generation instruments:
```bibtex
@article{Wouters:2025ull,
    author = "Wouters, Thibeau and Puecher, Anna and Pang, Peter T. H. and Dietrich, Tim",
    title = "{Analyzing GW231109{\_}235456 and understanding its potential implications for population studies, nuclear physics, and multi-messenger astronomy}",
    eprint = "2510.22290",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    month = "10",
    year = "2025"
}
```