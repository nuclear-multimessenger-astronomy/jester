r"""Factory for creating transforms from configuration."""

from ..config.schema import TransformConfig
from .base import JesterTransformBase
from .metamodel import MetaModelTransform
from .metamodel_cse import MetaModelCSETransform


def create_transform(
    config: TransformConfig,
    name_mapping: tuple[list[str], list[str]] | None = None,
    keep_names: list[str] | None = None,
) -> JesterTransformBase:
    """Create transform from configuration.

    Parameters
    ----------
    config : TransformConfig
        Transform configuration object
    name_mapping : tuple[list[str], list[str]], optional
        Tuple of (input_names, output_names) for the transform.
        If None, will be constructed from config and prior.
    keep_names : list[str], optional
        Parameter names to keep in transformed output.
        By default, all input parameters are kept.

    Returns
    -------
    JesterTransformBase
        Configured transform instance

    Raises
    ------
    ValueError
        If transform type is unknown

    Examples
    --------
    >>> from jesterTOV.inference.config import TransformConfig
    >>> config = TransformConfig(type="metamodel", ndat_metamodel=100)
    >>> transform = create_transform(config)
    >>> print(transform.get_eos_type())
    'MM'
    """
    # Validate transform type early
    if config.type not in ("metamodel", "metamodel_cse"):
        raise ValueError(f"Unknown transform type: {config.type}")

    # Common keyword arguments for all transforms
    common_kwargs = {
        "ndat_metamodel": config.ndat_metamodel,
        "nmax_nsat": config.nmax_nsat,
        "min_nsat_TOV": config.min_nsat_TOV,
        "ndat_TOV": config.ndat_TOV,
        "nb_masses": config.nb_masses,
        "crust_name": config.crust_name,
        "keep_names": keep_names,
    }

    # Default name mapping if not provided
    # This will be properly set up when we integrate with the sampler
    if name_mapping is None:
        # Create default mapping based on transform type
        if config.type == "metamodel":
            input_names = [
                "K_sat",
                "Q_sat",
                "Z_sat",
                "E_sym",
                "L_sym",
                "K_sym",
                "Q_sym",
                "Z_sym",
            ]
        else:  # metamodel_cse (already validated above)
            input_names = [
                "K_sat",
                "Q_sat",
                "Z_sat",
                "E_sym",
                "L_sym",
                "K_sym",
                "Q_sym",
                "Z_sym",
                "nbreak",
            ]
            # Add CSE grid parameters
            for i in range(config.nb_CSE):
                input_names.extend([f"n_CSE_{i}_u", f"cs2_CSE_{i}"])
            # Add final cs2 value at nmax
            input_names.append(f"cs2_CSE_{config.nb_CSE}")

        output_names = ["logpc_EOS", "masses_EOS", "radii_EOS", "Lambdas_EOS"]
        name_mapping = (input_names, output_names)

    # Create transform based on type (type already validated)
    if config.type == "metamodel":
        return MetaModelTransform(name_mapping=name_mapping, **common_kwargs)
    else:  # metamodel_cse
        return MetaModelCSETransform(
            name_mapping=name_mapping, nb_CSE=config.nb_CSE, **common_kwargs
        )


def get_transform_input_names(config: TransformConfig) -> list[str]:
    """Get list of input parameter names for a given transform config.

    Parameters
    ----------
    config : TransformConfig
        Transform configuration

    Returns
    -------
    list[str]
        List of parameter names expected by this transform

    Examples
    --------
    >>> config = TransformConfig(type="metamodel_cse", nb_CSE=8)
    >>> names = get_transform_input_names(config)
    >>> print(len(names))
    26  # 8 NEP + 1 nbreak + 8*2 CSE grid + 1 final cs2
    """
    # Validate transform type early
    if config.type not in ("metamodel", "metamodel_cse"):
        raise ValueError(f"Unknown transform type: {config.type}")

    if config.type == "metamodel":
        return [
            "K_sat",
            "Q_sat",
            "Z_sat",
            "E_sym",
            "L_sym",
            "K_sym",
            "Q_sym",
            "Z_sym",
        ]
    else:  # metamodel_cse (already validated above)
        names = [
            "K_sat",
            "Q_sat",
            "Z_sat",
            "E_sym",
            "L_sym",
            "K_sym",
            "Q_sym",
            "Z_sym",
            "nbreak",
        ]
        # Add CSE grid parameters
        for i in range(config.nb_CSE):
            names.extend([f"n_CSE_{i}_u", f"cs2_CSE_{i}"])
        # Add final cs2 value at nmax
        names.append(f"cs2_CSE_{config.nb_CSE}")
        return names
