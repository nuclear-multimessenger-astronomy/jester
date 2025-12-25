"""Constraint likelihood for enforcing EOS validity.

This module provides modular constraint checking functions and a likelihood
class that penalizes samples violating physical constraints.
"""

import jax.numpy as jnp

from jesterTOV.inference.base.likelihood import LikelihoodBase


def check_tov_validity(masses: jnp.ndarray, radii: jnp.ndarray, lambdas: jnp.ndarray) -> float:
    """
    Check if TOV integration succeeded by counting NaN values.

    Parameters
    ----------
    masses : jnp.ndarray
        Array of neutron star masses from TOV solution
    radii : jnp.ndarray
        Array of neutron star radii from TOV solution
    lambdas : jnp.ndarray
        Array of tidal deformabilities from TOV solution

    Returns
    -------
    float
        Number of NaN values found (0 = valid, >0 = invalid)
        Returns a scalar for JAX compatibility (avoids TracerBoolConversionError)
    """
    n_nan_masses = jnp.sum(jnp.isnan(masses))
    n_nan_radii = jnp.sum(jnp.isnan(radii))
    n_nan_lambdas = jnp.sum(jnp.isnan(lambdas))
    return n_nan_masses + n_nan_radii + n_nan_lambdas


def check_causality_violation(cs2: jnp.ndarray) -> float:
    """
    Check for causality violations (sound speed exceeds speed of light).

    Causality requires cs^2 <= 1 (in units where c=1).

    Parameters
    ----------
    cs2 : jnp.ndarray
        Array of squared sound speeds (cs^2/c^2)

    Returns
    -------
    float
        Number of points where cs^2 > 1 (0 = valid, >0 = violation)
        Returns a scalar for JAX compatibility
    """
    return jnp.sum(cs2 > 1.0)


def check_stability(cs2: jnp.ndarray) -> float:
    """
    Check for thermodynamic instability (negative sound speed squared).

    Stability requires cs^2 >= 0.

    Parameters
    ----------
    cs2 : jnp.ndarray
        Array of squared sound speeds (cs^2/c^2)

    Returns
    -------
    float
        Number of points where cs^2 < 0 (0 = valid, >0 = unstable)
        Returns a scalar for JAX compatibility
    """
    return jnp.sum(cs2 < 0.0)


def check_pressure_monotonicity(p: jnp.ndarray) -> float:
    """
    Check if pressure is monotonically increasing with density.

    Non-monotonic pressure indicates an unphysical EOS.

    Parameters
    ----------
    p : jnp.ndarray
        Array of pressure values (should be sorted by increasing density)

    Returns
    -------
    float
        Number of points where pressure decreases (0 = valid, >0 = non-monotonic)
        Returns a scalar for JAX compatibility
    """
    # Check if dp/dn > 0 (pressure increases with density)
    dp = jnp.diff(p)
    return jnp.sum(dp < 0.0)


def check_all_constraints(
    masses: jnp.ndarray,
    radii: jnp.ndarray,
    lambdas: jnp.ndarray,
    cs2: jnp.ndarray,
    p: jnp.ndarray,
) -> dict[str, float]:
    """
    Run all constraint checks and return violation counts.

    This is a convenience function that runs all individual checks
    and returns results in a dictionary.

    Parameters
    ----------
    masses : jnp.ndarray
        Neutron star masses from TOV
    radii : jnp.ndarray
        Neutron star radii from TOV
    lambdas : jnp.ndarray
        Tidal deformabilities from TOV
    cs2 : jnp.ndarray
        Squared sound speeds
    p : jnp.ndarray
        Pressure array (sorted by density)

    Returns
    -------
    dict[str, float]
        Dictionary with constraint violation counts:
        - 'n_tov_failures': Number of NaN in TOV solution
        - 'n_causality_violations': Number of cs^2 > 1 points
        - 'n_stability_violations': Number of cs^2 < 0 points
        - 'n_pressure_violations': Number of pressure decrease points

    Examples
    --------
    >>> constraints = check_all_constraints(masses, radii, lambdas, cs2, p)
    >>> if constraints['n_tov_failures'] > 0:
    ...     print("TOV integration failed!")
    """
    return {
        'n_tov_failures': check_tov_validity(masses, radii, lambdas),
        'n_causality_violations': check_causality_violation(cs2),
        'n_stability_violations': check_stability(cs2),
        'n_pressure_violations': check_pressure_monotonicity(p),
    }


class ConstraintLikelihood(LikelihoodBase):
    """
    Constraint likelihood for enforcing EOS physical validity.

    This likelihood reads constraint violation counts from the transform output
    and applies penalties using JAX-compatible operations (jnp.where).

    The transform must add violation counts to its output dictionary:
    - 'n_tov_failures': Number of NaN in TOV solution
    - 'n_causality_violations': Number of cs^2 > 1 points
    - 'n_stability_violations': Number of cs^2 < 0 points
    - 'n_pressure_violations': Number of non-monotonic pressure points

    Parameters
    ----------
    penalty_tov : float, optional
        Log likelihood penalty for TOV integration failure (default: -1e10)
    penalty_causality : float, optional
        Log likelihood penalty for causality violation (default: -1e10)
    penalty_stability : float, optional
        Log likelihood penalty for thermodynamic instability (default: -1e5)
    penalty_pressure : float, optional
        Log likelihood penalty for non-monotonic pressure (default: -1e5)

    Examples
    --------
    >>> # In config.yaml
    >>> likelihoods:
    >>>   - type: "constraints"
    >>>     enabled: true
    >>>     parameters:
    >>>       penalty_tov: -1.0e10
    >>>       penalty_causality: -1.0e10
    """

    def __init__(
        self,
        penalty_tov: float = -1e10,
        penalty_causality: float = -1e10,
        penalty_stability: float = -1e5,
        penalty_pressure: float = -1e5,
    ):
        super().__init__()
        self.penalty_tov = penalty_tov
        self.penalty_causality = penalty_causality
        self.penalty_stability = penalty_stability
        self.penalty_pressure = penalty_pressure

    def evaluate(self, params: dict[str, float], data: dict) -> float:
        """
        Evaluate constraint log likelihood.

        Returns 0.0 if all constraints satisfied, applies penalties otherwise.
        Uses jnp.where for JAX compatibility (no Python if-statements).

        Parameters
        ----------
        params : dict[str, float]
            Must contain constraint violation counts from transform:
            - 'n_tov_failures'
            - 'n_causality_violations'
            - 'n_stability_violations'
            - 'n_pressure_violations'
        data : dict
            Not used (constraints are in params from transform)

        Returns
        -------
        float
            Sum of penalties (0.0 if valid, large negative if invalid)
        """
        # Get violation counts from transform output (default to 0 if not present)
        n_tov_failures = params.get('n_tov_failures', 0.0)
        n_causality_violations = params.get('n_causality_violations', 0.0)
        n_stability_violations = params.get('n_stability_violations', 0.0)
        n_pressure_violations = params.get('n_pressure_violations', 0.0)

        # Apply penalties using jnp.where (JAX-compatible, no branching)
        # If count > 0, apply penalty, otherwise 0.0
        penalty_tov = jnp.where(n_tov_failures > 0, self.penalty_tov, 0.0)
        penalty_caus = jnp.where(n_causality_violations > 0, self.penalty_causality, 0.0)
        penalty_stab = jnp.where(n_stability_violations > 0, self.penalty_stability, 0.0)
        penalty_press = jnp.where(n_pressure_violations > 0, self.penalty_pressure, 0.0)

        # Sum all penalties
        log_likelihood = penalty_tov + penalty_caus + penalty_stab + penalty_press

        return log_likelihood
