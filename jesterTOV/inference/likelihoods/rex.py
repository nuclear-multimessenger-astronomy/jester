# TODO: still needs to be implemented. For now, raises NotImplementedError

r"""
PREX and CREX neutron skin measurement constraints.

This module implements likelihood functions based on the PREX (Lead Radius
Experiment) and CREX (Calcium Radius Experiment) measurements of neutron skin
thickness in heavy nuclei. These experiments use parity-violating electron
scattering to measure the weak charge distribution, which is sensitive to the
neutron skin and thus constrains the symmetry energy parameters of the nuclear
equation of state.

The measurements are particularly sensitive to the symmetry energy E_sym and
its density slope L_sym, providing independent constraints that complement
astrophysical observations of neutron stars. The likelihood is implemented
using a kernel density estimate (KDE) of the experimental posterior in
(E_sym, L_sym) space.

References
----------
.. [1] Adhikari et al., "Precision Determination of the Neutral Weak Form Factor
   of Pb-208," Phys. Rev. Lett. 126, 172502 (2021).
.. [2] Adhikari et al., "Accurate Determination of the Neutron Skin Thickness of
   Ca-48 through Parity-Violation in Electron Scattering," Phys. Rev. Lett. 129,
   042501 (2022).
"""

from typing import Any

import jax.numpy as jnp
from jaxtyping import Array, Float

from jesterTOV.inference.base import LikelihoodBase


class REXLikelihood(LikelihoodBase):
    """Likelihood function for PREX or CREX neutron skin measurements.

    This likelihood constrains the symmetry energy parameters (E_sym, L_sym)
    using experimental measurements of neutron skin thickness from parity-violating
    electron scattering. The experimental data is represented as a kernel density
    estimate (KDE) of the posterior distribution in (E_sym, L_sym) parameter space,
    which is then evaluated for each candidate EOS.

    The neutron skin thickness is primarily sensitive to the pressure of neutron
    matter at subsaturation densities, which is controlled by E_sym (symmetry
    energy at saturation) and L_sym (its density slope). These experiments provide
    complementary information to astrophysical mass-radius measurements, as they
    probe different density regimes and physical processes.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment providing the constraint. Must be either "PREX"
        (Lead Radius Experiment, Pb-208) or "CREX" (Calcium Radius Experiment,
        Ca-48). The two experiments probe different mass regions and have
        different systematic uncertainties.
    posterior : Any
        Kernel density estimate of the experimental posterior distribution in
        (E_sym, L_sym) parameter space. This should be a callable object with
        a `logpdf` method that accepts a 2D array of shape (2, n_samples) and
        returns log-probability values. Typically constructed using
        scipy.stats.gaussian_kde or similar.

    Attributes
    ----------
    experiment_name : str
        The experiment name ("PREX" or "CREX")
    counter : int
        Internal counter tracking the number of likelihood evaluations.
        Used for debugging and performance monitoring.
    posterior : Any
        The KDE object representing the experimental posterior

    Raises
    ------
    AssertionError
        If experiment_name is not "PREX" or "CREX"

    Notes
    -----
    The likelihood evaluation extracts only the E_sym and L_sym parameters from
    the full parameter dictionary and evaluates the KDE at that point. Other
    nuclear parameters (K_sat, Q_sat, etc.) do not directly enter this likelihood,
    though they affect the overall EOS and may have indirect correlations.

    The KDE is evaluated in log-space to avoid numerical underflow for low-probability
    regions and to match the log-likelihood framework used throughout the inference.

    See Also
    --------
    ChiEFTLikelihood : Low-density constraints from chiral effective field theory

    Examples
    --------
    Create a PREX likelihood with a pre-computed KDE:

    >>> from scipy.stats import gaussian_kde
    >>> import numpy as np
    >>> # Load PREX posterior samples (example)
    >>> samples = np.load("prex_samples.npy")  # shape: (2, n_samples) for (E_sym, L_sym)
    >>> kde = gaussian_kde(samples)
    >>> from jesterTOV.inference.likelihoods import REXLikelihood
    >>> likelihood = REXLikelihood("PREX", kde)
    >>> params = {"E_sym": 32.0, "L_sym": 60.0}
    >>> log_like = likelihood.evaluate(params, data={})
    """

    experiment_name: str
    counter: int
    posterior: Any  # gaussian_kde type

    def __init__(
        self,
        experiment_name: str,
        posterior: Any,
    ) -> None:
        super().__init__()
        assert experiment_name in [
            "PREX",
            "CREX",
        ], "Only PREX and CREX are supported as experiment name arguments."
        self.experiment_name = experiment_name
        self.counter = 0
        self.posterior = posterior

    def evaluate(self, params: dict[str, Float | Array], data: dict[str, Any]) -> Float:
        """Evaluate the log-likelihood for PREX/CREX constraints.

        Parameters
        ----------
        params : dict[str, Float | Array]
            Dictionary containing EOS parameters. Required keys:
            - "E_sym" : Symmetry energy at saturation density (MeV)
            - "L_sym" : Slope of symmetry energy (MeV)
            Other parameters in the dict are ignored.
        data : dict[str, Any]
            Unused; included for API compatibility. All experimental data is
            encoded in the KDE provided during initialization.

        Returns
        -------
        Float
            Natural logarithm of the likelihood. This is the log-probability
            density of the (E_sym, L_sym) point under the experimental posterior
            KDE.

        Notes
        -----
        The method extracts E_sym and L_sym from the parameter dictionary,
        constructs a 2D array [E_sym, L_sym], and evaluates the KDE's logpdf
        method. The result is a scalar log-probability suitable for combination
        with other log-likelihoods in the inference pipeline.
        """
        log_likelihood_array = self.posterior.logpdf(
            jnp.array([params["E_sym"], params["L_sym"]])
        )
        log_likelihood = log_likelihood_array.at[0].get()
        return log_likelihood
