r"""Factory functions for creating likelihoods from configuration"""

from pathlib import Path

from ..config.schema import (
    LikelihoodConfig,
    GWLikelihoodConfig,
    GWResampledLikelihoodConfig,
    NICERLikelihoodConfig,
    NICERKDELikelihoodConfig,
    RadioLikelihoodConfig,
    ChiEFTLikelihoodConfig,
    EOSConstraintsLikelihoodConfig,
    TOVConstraintsLikelihoodConfig,
    GammaConstraintsLikelihoodConfig,
    DeprecatedConstraintsLikelihoodConfig,
    REXLikelihoodConfig,
    ZeroLikelihoodConfig,
)
from .combined import CombinedLikelihood, ZeroLikelihood
from .gw import GWLikelihood, GWLikelihoodResampled
from .nicer import NICERLikelihood, NICERKDELikelihood
from .radio import RadioTimingLikelihood
from .chieft import ChiEFTLikelihood
from .constraints import (
    ConstraintEOSLikelihood,
    ConstraintTOVLikelihood,
    ConstraintGammaLikelihood,
)
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")

# Preset flow model directories for GW events with trained flows
# Paths are relative to jesterTOV/inference/ directory
GW_EVENT_PRESETS = {
    "GW170817": "flows/models/gw_maf/gw170817/gw170817_xp_nrtv3",
    "GW190425": "flows/models/gw_maf/gw190425/gw190425_xp_nrtv3",
}


def get_gw_model_dir(event_name: str, model_dir: str | None) -> str:
    """
    Get model directory for GW event, using presets if path is not provided.

    Parameters
    ----------
    event_name : str
        Name of the GW event (case-insensitive)
    model_dir : str | None
        User-provided model directory, or None/empty string to use preset

    Returns
    -------
    str
        Absolute path to model directory

    Raises
    ------
    ValueError
        If model_dir is not provided and event is not in presets
    """
    # Normalize event name to uppercase for preset lookup
    event_name_upper = event_name.upper()

    # If model_dir is provided and not empty, use it directly
    if model_dir:
        return str(Path(model_dir).resolve())

    # Check if event is in presets
    if event_name_upper not in GW_EVENT_PRESETS:
        raise ValueError(
            f"No model_dir provided for event '{event_name}' and event is not in presets. "
            f"Available presets: {list(GW_EVENT_PRESETS.keys())}. "
            f"Please provide model_dir explicitly in the configuration."
        )

    # Get preset path and convert to absolute
    preset_path = GW_EVENT_PRESETS[event_name_upper]
    # Resolve relative to jesterTOV/inference directory
    inference_dir = Path(__file__).parent.parent
    model_dir_abs = (inference_dir / preset_path).resolve()

    # Log warning that we're using default path
    logger.warning(
        f"No model_dir provided for event '{event_name}'. "
        f"Using default preset path: {model_dir_abs}"
    )

    return str(model_dir_abs)


def create_likelihood(
    config: LikelihoodConfig,
):
    """
    Create likelihood from configuration

    Parameters
    ----------
    config : LikelihoodConfig
        Likelihood configuration (discriminated union)

    Returns
    -------
    LikelihoodBase or None
        Configured likelihood instance, or None if disabled
    """
    if not config.enabled:
        return None

    # Type narrowing with match statement
    match config:
        case GWLikelihoodConfig() | GWResampledLikelihoodConfig():
            # GW likelihoods are handled specially in create_combined_likelihood
            # This function should not be called directly for GW type
            raise RuntimeError(
                "GW likelihoods should be created via create_combined_likelihood, "
                "not create_likelihood directly"
            )

        case NICERLikelihoodConfig() | NICERKDELikelihoodConfig():
            # NICER likelihoods are handled specially in create_combined_likelihood
            # This function should not be called directly for NICER types
            raise RuntimeError(
                "NICER likelihoods should be created via create_combined_likelihood, "
                "not create_likelihood directly"
            )

        case RadioLikelihoodConfig():
            # Radio timing likelihoods are handled specially in create_combined_likelihood
            # This function should not be called directly for radio type
            raise RuntimeError(
                "Radio timing likelihoods should be created via create_combined_likelihood, "
                "not create_likelihood directly"
            )

        case ChiEFTLikelihoodConfig():
            return ChiEFTLikelihood(
                low_filename=config.low_filename,
                high_filename=config.high_filename,
                nb_n=config.nb_n,
            )

        case REXLikelihoodConfig():
            # FIXME: Implement load_rex_posterior(experiment_name) -> gaussian_kde
            # This should load PREX/CREX posterior KDE from data files
            # For now, raise NotImplementedError
            raise NotImplementedError(
                f"REX likelihood data loading not implemented. "
                f"Need to implement load_rex_posterior('{config.experiment_name}') -> gaussian_kde"
            )

        case EOSConstraintsLikelihoodConfig():
            return ConstraintEOSLikelihood(
                penalty_causality=config.penalty_causality,
                penalty_stability=config.penalty_stability,
                penalty_pressure=config.penalty_pressure,
            )

        case TOVConstraintsLikelihoodConfig():
            return ConstraintTOVLikelihood(
                penalty_tov=config.penalty_tov,
            )

        case GammaConstraintsLikelihoodConfig():
            return ConstraintGammaLikelihood(
                penalty_gamma=config.penalty_gamma,
            )

        case DeprecatedConstraintsLikelihoodConfig():
            # Handle deprecated combined constraints
            logger.warning(
                "Using deprecated 'constraints' likelihood type. "
                "Please use 'constraints_eos' + 'constraints_tov' instead."
            )
            # Create combined likelihood with both EOS and TOV constraints
            eos_constraint = ConstraintEOSLikelihood(
                penalty_causality=config.penalty_causality,
                penalty_stability=config.penalty_stability,
                penalty_pressure=config.penalty_pressure,
            )
            tov_constraint = ConstraintTOVLikelihood(
                penalty_tov=config.penalty_tov,
            )
            return CombinedLikelihood([eos_constraint, tov_constraint])

        case ZeroLikelihoodConfig():
            return ZeroLikelihood()

        case _:
            raise ValueError(f"Unknown likelihood type: {config.type}")


def create_combined_likelihood(
    likelihood_configs: list[LikelihoodConfig],
):
    """
    Create combined likelihood from list of configs

    Parameters
    ----------
    likelihood_configs : list[LikelihoodConfig]
        List of likelihood configurations

    Returns
    -------
    LikelihoodBase
        Combined likelihood or single likelihood

    Raises
    ------
    ValueError
        If no likelihoods are enabled
    """
    likelihoods = []

    for config in likelihood_configs:
        if not config.enabled:
            continue

        # Use match statement for type narrowing
        match config:
            # Special handling for GW likelihoods (presampled): create one likelihood per event
            case GWLikelihoodConfig():
                # Create one GWLikelihood (presampled) per event
                for event in config.events:
                    # Get model directory (use preset if not provided)
                    model_dir = get_gw_model_dir(
                        event_name=event["name"], model_dir=event.get("model_dir")
                    )

                    gw_likelihood = GWLikelihood(
                        event_name=event["name"],
                        model_dir=model_dir,
                        penalty_value=config.penalty_value,
                        N_masses_evaluation=config.N_masses_evaluation,
                        N_masses_batch_size=config.N_masses_batch_size,
                        seed=config.seed,
                    )
                    likelihoods.append(gw_likelihood)

            # Special handling for GW likelihoods with resampling: create one likelihood per event
            case GWResampledLikelihoodConfig():
                # Create one GWLikelihoodResampled per event
                for event in config.events:
                    # Get model directory (use preset if not provided)
                    model_dir = get_gw_model_dir(
                        event_name=event["name"], model_dir=event.get("model_dir")
                    )

                    gw_likelihood = GWLikelihoodResampled(
                        event_name=event["name"],
                        model_dir=model_dir,
                        penalty_value=config.penalty_value,
                        N_masses_evaluation=config.N_masses_evaluation,
                        N_masses_batch_size=config.N_masses_batch_size,
                    )
                    likelihoods.append(gw_likelihood)

            # Special handling for NICER likelihoods (flow-based): create one likelihood per pulsar
            case NICERLikelihoodConfig():
                # Create one NICERLikelihood (flow-based) per pulsar
                for pulsar in config.pulsars:
                    nicer_likelihood = NICERLikelihood(
                        psr_name=pulsar["name"],
                        amsterdam_model_dir=pulsar.get("amsterdam_model_dir"),
                        maryland_model_dir=pulsar.get("maryland_model_dir"),
                        N_masses_evaluation=config.N_masses_evaluation,
                        N_masses_batch_size=config.N_masses_batch_size,
                    )
                    likelihoods.append(nicer_likelihood)

            # Special handling for NICER KDE likelihoods (legacy): create one likelihood per pulsar
            case NICERKDELikelihoodConfig():
                # Create one NICERKDELikelihood (KDE-based) per pulsar
                for pulsar in config.pulsars:
                    nicer_kde_likelihood = NICERKDELikelihood(
                        psr_name=pulsar["name"],
                        amsterdam_samples_file=pulsar["amsterdam_samples_file"],
                        maryland_samples_file=pulsar["maryland_samples_file"],
                        N_masses_evaluation=config.N_masses_evaluation,
                        N_masses_batch_size=config.N_masses_batch_size,
                    )
                    likelihoods.append(nicer_kde_likelihood)

            # Special handling for radio timing likelihoods: create one likelihood per pulsar
            case RadioLikelihoodConfig():
                # Create one RadioTimingLikelihood per pulsar
                for pulsar in config.pulsars:
                    # Type checker needs help since dict values are str | float
                    psr_name = pulsar["name"]
                    assert isinstance(psr_name, str), "name must be a string"
                    mass_mean = pulsar["mass_mean"]
                    assert isinstance(
                        mass_mean, (int, float)
                    ), "mass_mean must be a number"
                    mass_std = pulsar["mass_std"]
                    assert isinstance(
                        mass_std, (int, float)
                    ), "mass_std must be a number"

                    radio_likelihood = RadioTimingLikelihood(
                        psr_name=psr_name,
                        mean=float(mass_mean),
                        std=float(mass_std),
                        penalty_value=config.penalty_value,
                    )
                    likelihoods.append(radio_likelihood)

            case _:
                # For other likelihoods, use standard creation
                likelihood = create_likelihood(config)
                if likelihood is not None:
                    likelihoods.append(likelihood)

    if len(likelihoods) == 0:
        raise ValueError("No likelihoods enabled in configuration")
    elif len(likelihoods) == 1:
        return likelihoods[0]
    else:
        return CombinedLikelihood(likelihoods)
