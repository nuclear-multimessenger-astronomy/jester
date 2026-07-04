.. _eos-crust:

Crust models
============

The neutron star crust occupies the low-density outer layers of the star, from
the surface down to the crust-core transition density. 
The crust EOSs are then appended to the core EOS, for instance, by stitching the two together with a spline evaluated in the connection region between the crust and core. 
For the moment, ``jester`` only uses fixed crusts. 

Crust files can be provided in ``.npz`` file formats and can be loaded and used inside ``jester`` via the :class:`jesterTOV.eos.crust.Crust` class.
Check out the API for more information.

In ``jester``, we have three crust files derived from the SLy family of EOSs (BPS, DH and SLy), together with a larger set of crusts converted from the `nucleardatapy <https://github.com/jeromemargueron/nucleardatapy>`_ toolkit: twelve variants of the GMRS crust (Grams, Margueron, Somasundaram and Reddy, EPJA 58, 56, 2022) spanning a range of symmetry-energy slope parameters, and three MVCD crusts (Mondal, Viñas, Centelles and De, Phys. Rev. C 102, 015802, 2020). All of these are shown in the plot below.

.. plot:: overview/eos/crust_plot.py

   Pressure (top) and energy density (bottom) as a function of baryon number
   density for the built-in crust models.  Only the crust region is
   shown; no core model is appended.

The GMRS and MVCD crust files are not maintained by hand: they are regenerated from ``nucleardatapy`` by the conversion script at
``jesterTOV/crust_files/convert_nucleardatapy_crusts.py``. ``nucleardatapy`` is a large toolkit for nuclear data and meta-analyses and is
deliberately **not** a ``jester`` dependency; the script only needs a one-off install (see its module docstring for the exact command,
and the list of models it currently enables/excludes) to regenerate or extend the set of available crust files. Notably, the script uses
the pressure tabulated directly in the ``nucleardatapy`` data files rather than the toolkit's own spline-derived pressure, since the latter
was found to be non-monotonic and to deviate from the tabulated values by up to ~90% near the crust-core transition for at least one model
checked (GMRS-BSK14); models lacking a tabulated pressure column are excluded by default and can be enabled explicitly if needed.

Further resources
-----------------

* API reference: :class:`jesterTOV.eos.crust.Crust`
* `nucleardatapy <https://github.com/jeromemargueron/nucleardatapy>`_, the toolkit the GMRS/MVCD crust files were converted from
