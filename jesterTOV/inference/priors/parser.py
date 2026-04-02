r"""Parser for .prior specification files in bilby-style Python format."""

from dataclasses import dataclass
from pathlib import Path
from typing import Union, Any, Dict
from jesterTOV.inference.base import (
    CombinePrior,
    Prior,
    UniformPrior,
    MultivariateGaussianPrior,
)
from jesterTOV.inference.base.prior import Fixed


@dataclass
class ParsedPrior:
    """Result of parsing a ``.prior`` file.

    Parameters
    ----------
    prior : CombinePrior
        Combined prior containing only the sampled (non-fixed) parameters.
    fixed_params : dict[str, float]
        Mapping of parameter name to fixed value for parameters declared with
        ``Fixed(...)`` in the prior file.
    """

    prior: CombinePrior
    fixed_params: dict[str, float]


def parse_prior_file(
    prior_file: Union[str, Path],
    nb_CSE: int = 0,
) -> ParsedPrior:
    """Parse .prior file (Python format) and return a :class:`ParsedPrior`.

    The prior file should contain Python variable assignments in bilby-style format:

    .. code-block:: python

        K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
        Q_sat = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
        nbreak = UniformPrior(0.16, 0.32, parameter_names=["nbreak"])

        # Pin a parameter to a fixed value (not sampled):
        lambda_BL = Fixed(0.0, parameter_names=["lambda_BL"])

    Parameters declared with :class:`~jesterTOV.inference.base.prior.Fixed` are
    collected into :attr:`ParsedPrior.fixed_params` and excluded from the
    sampling prior.

    The parser will automatically:

    - Include all NEP parameters (``_sat`` and ``_sym`` parameters)
    - Include ``nbreak`` only if ``nb_CSE > 0``
    - Add CSE grid parameters (``n_CSE_i_u``, ``cs2_CSE_i``) if ``nb_CSE > 0``,
      **unless** those parameters are already defined in the prior file — either
      as :class:`~jesterTOV.inference.base.prior.Fixed` entries (pinned to a
      constant) or as custom sampled priors with user-specified bounds.  This
      allows partial or full fixing of the CSE grid while leaving other CSE
      parameters free.

    Parameters
    ----------
    prior_file : str or Path
        Path to .prior specification file (Python format)
    nb_CSE : int, optional
        Number of CSE parameters (0 for MetaModel only)

    Returns
    -------
    ParsedPrior
        Parsed prior with sampled :class:`CombinePrior` and ``fixed_params`` dict.

    Raises
    ------
    FileNotFoundError
        If prior file does not exist
    ValueError
        If prior file format is invalid or no priors found

    Examples
    --------
    >>> result = parse_prior_file("nep_standard.prior", nb_CSE=8)
    >>> print(result.prior.n_dim)  # Number of sampled dimensions
    25  # 8 NEP + 1 nbreak + 8*2 CSE grid params
    >>> print(result.fixed_params)  # Any fixed parameters
    {}
    """
    prior_file = Path(prior_file)

    if not prior_file.exists():
        raise FileNotFoundError(f"Prior specification file not found: {prior_file}")

    # Read the prior file
    with open(prior_file, "r") as f:
        prior_code = f.read()

    # Create execution namespace with required imports only
    namespace: dict[str, Any] = {
        "UniformPrior": UniformPrior,
        "MultivariateGaussianPrior": MultivariateGaussianPrior,
        "Fixed": Fixed,
    }

    # Execute the prior file to populate the namespace
    try:
        exec(prior_code, namespace)
    except Exception as e:
        raise ValueError(f"Error executing prior file {prior_file}: {e}") from e

    # Extract all Prior objects from the namespace
    excluded_keys = {
        "__builtins__",
        "UniformPrior",
        "MultivariateGaussianPrior",
        "Fixed",
    }
    all_priors: Dict[str, Prior] = {}

    for key, value in namespace.items():
        if key not in excluded_keys and isinstance(value, Prior):
            all_priors[key] = value

    # Separate Fixed parameters from sampled priors
    fixed_params: dict[str, float] = {}
    sampled_priors: Dict[str, Prior] = {}

    for param_name, prior in all_priors.items():
        if isinstance(prior, Fixed):
            fixed_params[prior.parameter_names[0]] = prior.value
        else:
            sampled_priors[param_name] = prior

    # Filter sampled priors based on configuration
    prior_list = []

    for param_name, prior in sampled_priors.items():
        # Always include NEP parameters (_sat and _sym)
        if param_name.endswith("_sat") or param_name.endswith("_sym"):
            prior_list.append(prior)
        # Include nbreak only if nb_CSE > 0
        elif param_name == "nbreak":
            if nb_CSE > 0:
                prior_list.append(prior)
        else:
            # Include any other parameters not handled by special cases
            prior_list.append(prior)

    # Add CSE grid parameters programmatically if nb_CSE > 0.
    # Skip any parameter that the user has already provided — either as a
    # Fixed entry (already in fixed_params) or as a custom sampled prior
    # (already added to prior_list via the loop above).
    if nb_CSE > 0:
        for i in range(nb_CSE):
            for param_name in [f"n_CSE_{i}_u", f"cs2_CSE_{i}"]:
                if param_name not in fixed_params and param_name not in sampled_priors:
                    prior_list.append(
                        UniformPrior(0.0, 1.0, parameter_names=[param_name])
                    )

        # Final cs2 parameter for the grid point at nmax
        final_cs2_name = f"cs2_CSE_{nb_CSE}"
        if final_cs2_name not in fixed_params and final_cs2_name not in sampled_priors:
            prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[final_cs2_name]))

    if len(prior_list) == 0:
        raise ValueError(
            f"No sampled priors found in {prior_file}. "
            "Check file format and ensure at least one Prior object is defined. "
            "Note: Fixed parameters do not count as sampled priors."
        )

    return ParsedPrior(prior=CombinePrior(prior_list), fixed_params=fixed_params)
