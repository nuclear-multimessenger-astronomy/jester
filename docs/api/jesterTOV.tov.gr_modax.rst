``jesterTOV.tov.gr_modax`` module
==================================

.. currentmodule:: jesterTOV.tov.gr_modax

General Relativity TOV solver on the modax GPU ODE kernels.

It integrates the same equations as :class:`~jesterTOV.tov.gr.GRTOVSolver` with the same tolerances, but all stars of
a batch run in one modax Tsit5 kernel, each with its own adaptive step sequence, and every EOS table is stored once and
shared by the stars of that EOS. Select it with ``backend: "modax"`` in the ``tov`` section of a configuration. It needs
a CUDA GPU and has no gradients.

Classes
-------

.. autosummary::
   :nosignatures:
   :toctree: _autosummary/

   GRTOVModaxSolver
