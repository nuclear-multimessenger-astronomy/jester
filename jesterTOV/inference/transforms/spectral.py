r"""
Spectral decomposition equation of state transform for neutron star inference.

This module implements the transform from spectral decomposition parameters
to observable neutron star properties by solving the TOV equations with the
spectral decomposition EOS parametrization. The spectral EOS follows Lindblom (2010)
PRD 82, 103011 and Lackey & Wade (2018) PRD 98, 063004, matching LALSuite's
implementation exactly.
"""

import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array, Float

from .base import JesterTransformBase
from jesterTOV.eos.spectral import SpectralDecomposition_EOS_model
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class SpectralTransform(JesterTransformBase):
    """Transform from spectral parameters to neutron star observables.

    This transform maps the 4-parameter spectral decomposition space to neutron star
    mass-radius-tidal deformability curves by constructing an equation of state and
    solving the Tolman-Oppenheimer-Volkoff equations. The spectral parametrization
    represents the adiabatic index as a polynomial in log-pressure space.

    The adiabatic index is parametrized as:
        log Γ(x) = γ₀ + γ₁·x + γ₂·x² + γ₃·x³
    where x = log(p/p₀) is the dimensionless log-pressure.

    Parameters
    ----------
    name_mapping : tuple[list[str], list[str]]
        Tuple of (input_names, output_names) defining the parameter transform.
        Input names should include the 4 spectral parameters; output names typically
        include mass, radius, and tidal deformability arrays.
    keep_names : list[str], optional
        Additional parameter names to preserve in the output dictionary alongside
        the transformed quantities. Default is to keep all input parameters.
    crust_name : str, optional
        Crust model to use ('SLy', 'DH', 'BPS'). Default 'SLy' for LALSuite compatibility.
    n_points_high : int, optional
        Number of high-density points to generate in spectral region. Default 500.
    **kwargs
        Additional arguments passed to JesterTransformBase.

    Attributes
    ----------
    eos : SpectralDecomposition_EOS_model
        The spectral decomposition EOS instance used for constructing p(n) curves

    Notes
    -----
    The 4 spectral parameters are:
    - gamma_0, gamma_1, gamma_2, gamma_3 : Coefficients of log Γ(x) polynomial expansion

    Typical parameter ranges from Lackey & Wade (2018):
    - γ₀ ∈ [0.2, 2.0]
    - γ₁ ∈ [-1.6, 1.7]
    - γ₂ ∈ [-0.6, 0.6]
    - γ₃ ∈ [-0.02, 0.02]

    The EOS enforces causality through parameter validation: Γ(x) must lie in [0.6, 4.5]
    for all pressures, which automatically ensures cs² ≤ 1.

    See Also
    --------
    MetaModelTransform : Nuclear empirical parameter (NEP) based EOS
    MetaModelCSETransform : Extended NEP with high-density CSE parametrization

    Examples
    --------
    Create a transform and apply it to spectral parameters:

    >>> from jesterTOV.inference.transforms import SpectralTransform
    >>> transform = SpectralTransform(
    ...     name_mapping=(
    ...         ["gamma_0", "gamma_1", "gamma_2", "gamma_3"],
    ...         ["masses_EOS", "radii_EOS", "Lambdas_EOS"]
    ...     ),
    ...     crust_name="SLy",
    ...     n_points_high=500
    ... )
    >>> params = {
    ...     "gamma_0": 0.5, "gamma_1": 0.2, "gamma_2": 0.0, "gamma_3": 0.0
    ... }
    >>> result = transform.forward(params)
    >>> print(result["masses_EOS"])  # Array of NS masses at different central pressures
    """

    def __init__(
        self,
        name_mapping: tuple[list[str], list[str]],
        keep_names: list[str] | None = None,
        crust_name: str = "SLy",
        n_points_high: int = 500,
        **kwargs,
    ):
        # Initialize base transform
        # Override crust_name with the one provided for spectral
        super().__init__(name_mapping, keep_names, crust_name=crust_name, **kwargs)

        # Create spectral decomposition EOS
        self.eos = SpectralDecomposition_EOS_model(
            crust_name=crust_name,
            n_points_high=n_points_high
        )

        # Set transform function
        self.transform_func = self.transform_func_spectral

    def get_eos_type(self) -> str:
        """Return the EOS parametrization identifier.

        Returns
        -------
        str
            The string "spectral" identifying this EOS parametrization.
            This is used for logging and output file organization.
        """
        return "spectral"

    def get_parameter_names(self) -> list[str]:
        """Return the list of spectral parameter names.

        Returns
        -------
        list[str]
            The 4 spectral parameter names: ["gamma_0", "gamma_1", "gamma_2", "gamma_3"].
            These define the polynomial expansion of log Γ(x) in dimensionless log-pressure.
        """
        return ["gamma_0", "gamma_1", "gamma_2", "gamma_3"]

    def _check_gamma_bounds(self, gamma: Float[Array, "4"]) -> Float:
        """
        Check for gamma parameter violations of LALSuite bounds.

        LALSuite requires Γ(x) ∈ [0.6, 4.5] for all x ∈ [0, xmax] to ensure
        physical validity and numerical stability. This method samples Γ(x)
        at 100 points and counts violations.

        Parameters
        ----------
        gamma : Float[Array, "4"]
            Spectral coefficients [γ₀, γ₁, γ₂, γ₃]

        Returns
        -------
        Float
            Number of points where Γ(x) violates bounds (0 = valid, >0 = violation)
            Returns a scalar for JAX compatibility
        """
        # Sample Γ(x) at 100 points as in LALSuite validation
        x_test = jnp.linspace(0.0, self.eos.xmax, 100)
        gamma_vals = vmap(lambda x: self.eos._adiabatic_index(x, gamma))(x_test)

        # Count violations: Γ < 0.6 or Γ > 4.5
        n_violations = jnp.sum((gamma_vals < 0.6) | (gamma_vals > 4.5))

        return n_violations

    def transform_func_spectral(self, params: dict[str, Float]) -> dict[str, Float]:
        """Compute neutron star observables from spectral decomposition parameters.

        This method constructs the equation of state from the spectral parameters,
        validates that the adiabatic index stays within physical bounds, and solves
        the TOV equations to obtain mass-radius-tidal deformability curves.

        Parameters
        ----------
        params : dict[str, Float]
            Dictionary containing the 4 spectral parameters (gamma_0, gamma_1, gamma_2,
            gamma_3) and any additional parameters specified in `keep_names`.

        Returns
        -------
        dict[str, Float]
            Dictionary containing EOS and TOV solution quantities:

            - logpc_EOS : Log10 of central pressures (geometric units)
            - masses_EOS : Neutron star masses at each central pressure (solar masses)
            - radii_EOS : Neutron star radii at each central pressure (km)
            - Lambdas_EOS : Dimensionless tidal deformabilities
            - n : Baryon number density grid (geometric units)
            - p : Pressure values on density grid (geometric units)
            - h : Specific enthalpy values (geometric units)
            - e : Energy density values (geometric units)
            - dloge_dlogp : Logarithmic derivative d(log e)/d(log p)
            - cs2 : Sound speed squared (computed from dloge_dlogp)
            - n_tov_failures : Number of TOV solver failures (scalar)
            - n_causality_violations : Number of causality violations (scalar)
            - n_stability_violations : Number of stability violations (scalar)
            - n_pressure_violations : Number of pressure violations (scalar)
            - n_gamma_violations : Number of Gamma bound violations (scalar): this is specific to spsectral EOS

        Notes
        -----
        The EOS construction automatically validates that Γ(x) ∈ [0.6, 4.5] for all
        pressures. If validation fails, a ValueError is raised. This ensures causality
        (cs² ≤ 1) and thermodynamic stability.
        """
        # Update with fixed parameters (currently empty)
        params.update(self.fixed_params)

        # Extract spectral parameters in correct order
        gamma = jnp.array([
            params["gamma_0"],
            params["gamma_1"],
            params["gamma_2"],
            params["gamma_3"]
        ])

        # Check gamma bounds BEFORE EOS construction (fast check)
        n_gamma_violations = self._check_gamma_bounds(gamma)

        # Create the EOS (validation happens inside construct_eos)
        ns, ps, hs, es, dloge_dlogps = self.eos.construct_eos(gamma)

        # TODO: double check this math, and perhaps document somewhere better
        # Compute sound speed squared from dloge_dlogp
        # cs² = dp/de = 1 / (de/dp) = 1 / Γ
        # But we have dloge_dlogp = d(log e)/d(log p) = (de/dp) * (p/e)
        # So: Γ = de/dp = dloge_dlogp * (e/p)
        # Therefore: cs² = 1/Γ = p / (e * dloge_dlogp)
        cs2 = ps / (es * dloge_dlogps)

        # Solve the TOV equations
        eos_tuple = (ns, ps, hs, es, dloge_dlogps, cs2)
        logpc_EOS, masses_EOS, radii_EOS, Lambdas_EOS = self._solve_tov(eos_tuple)

        # Create and return standardized output dictionary with gamma violations
        return self._create_return_dict(
            logpc_EOS,
            masses_EOS,
            radii_EOS,
            Lambdas_EOS,
            ns,
            ps,
            hs,
            es,
            dloge_dlogps,
            cs2,
            extra_constraints={"n_gamma_violations": n_gamma_violations},
        )
