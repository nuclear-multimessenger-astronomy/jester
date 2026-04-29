.. _eos-crust:

Crust models
============

The neutron star crust occupies the low-density outer layers of the star, from
the surface down to the crust-core transition density. 
The crust EOSs are then appended to the core EOS, for instance, by stitching the two together with a spline evaluated in the connection region between the crust and core. 
For the moment, ``jester`` only uses fixed crusts. 

Crust files can be provided in ``.npz`` file formats and can be loaded and used inside ``jester`` via the :class:`jesterTOV.eos.crust.Crust` class.
Check out the API for more information. 

In ``jester``, we currently have three very similar crust files derived from the SLy family of EOSs: BPS, DH and SLy, which are shown in the plot below.

.. plot:: overview/eos/crust_plot.py

   Pressure (top) and energy density (bottom) as a function of baryon number
   density for the three built-in crust models.  Only the crust region is
   shown; no core model is appended.



Further resources
-----------------

* API reference: :class:`jesterTOV.eos.crust.Crust`
