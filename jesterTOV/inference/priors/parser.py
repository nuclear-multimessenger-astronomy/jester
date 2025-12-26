r"""Parser for .prior specification files in bilby-style Python format."""

from pathlib import Path
from typing import Union, Any
from jesterTOV.inference.base import CombinePrior, Prior, UniformPrior


def parse_prior_file(
    prior_file: Union[str, Path],
    nb_CSE: int = 0,
) -> CombinePrior:
    """Parse .prior file (Python format) and return CombinePrior object.

    The prior file should contain Python variable assignments in bilby-style format:

        K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
        Q_sat = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])
        nbreak = UniformPrior(0.16, 0.32, parameter_names=["nbreak"])

    The parser will automatically:
    - Include all NEP parameters (_sat and _sym parameters)
    - Include nbreak only if nb_CSE > 0
    - Add CSE grid parameters (n_CSE_i_u, cs2_CSE_i) automatically if nb_CSE > 0

    Parameters
    ----------
    prior_file : str or Path
        Path to .prior specification file (Python format)
    nb_CSE : int, optional
        Number of CSE parameters (0 for MetaModel only)

    Returns
    -------
    CombinePrior
        Combined prior object ready for sampling

    Raises
    ------
    FileNotFoundError
        If prior file does not exist
    ValueError
        If prior file format is invalid or no priors found

    Examples
    --------
    >>> prior = parse_prior_file("nep_standard.prior", nb_CSE=8)
    >>> print(prior.n_dim)  # Number of dimensions
    25  # 8 NEP + 1 nbreak + 8*2 CSE grid params

    >>> prior = parse_prior_file("nep_standard.prior", nb_CSE=0)
    >>> print(prior.n_dim)
    8  # 8 NEP parameters only
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
    }

    # Execute the prior file to populate the namespace
    try:
        exec(prior_code, namespace)
    except Exception as e:
        raise ValueError(
            f"Error executing prior file {prior_file}: {e}"
        ) from e

    # Extract all Prior objects from the namespace
    excluded_keys = {"__builtins__", "UniformPrior"}
    all_priors = {}

    for key, value in namespace.items():
        if key not in excluded_keys and isinstance(value, Prior):
            all_priors[key] = value

    # Filter priors based on configuration
    prior_list = []

    for param_name, prior in all_priors.items():
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

    # Add CSE grid parameters programmatically if nb_CSE > 0
    if nb_CSE > 0:
        for i in range(nb_CSE):
            # Add n_CSE_i_u parameters (uniform [0, 1])
            prior_list.append(
                UniformPrior(0.0, 1.0, parameter_names=[f"n_CSE_{i}_u"])
            )
            # Add cs2_CSE_i parameters (uniform [0, 1])
            prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{i}"]))

        # Add final cs2 parameter for the grid point at nmax
        # This gives us nb_CSE grid points + 1 final point at nmax
        prior_list.append(UniformPrior(0.0, 1.0, parameter_names=[f"cs2_CSE_{nb_CSE}"]))

    if len(prior_list) == 0:
        raise ValueError(
            f"No priors found in {prior_file}. "
            "Check file format and ensure at least one Prior object is defined."
        )

    return CombinePrior(prior_list)
