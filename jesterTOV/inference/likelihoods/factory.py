"""Factory functions for creating likelihoods from configuration"""

from ..config.schema import LikelihoodConfig
# FIXME: DataLoader removed - need to implement data loading functions
from .combined import CombinedLikelihood, ZeroLikelihood
from .gw import GWLikelihood
from .nicer import NICERLikelihood, NICERLikelihood_with_masses
from .radio import RadioTimingLikelihood
from .chieft import ChiEFTLikelihood
from .rex import REXLikelihood


def create_likelihood(
    config: LikelihoodConfig,
    data_loader=None,
):
    """
    Create likelihood from configuration

    Parameters
    ----------
    config : LikelihoodConfig
        Likelihood configuration
    data_loader : None
        DEPRECATED - data loading will be handled differently

    Returns
    -------
    LikelihoodBase or None
        Configured likelihood instance, or None if disabled
    """
    if not config.enabled:
        return None

    params = config.parameters

    if config.type == "gw":
        # GW likelihoods are handled specially in create_combined_likelihood
        # This function should not be called directly for GW type
        raise RuntimeError(
            "GW likelihoods should be created via create_combined_likelihood, "
            "not create_likelihood directly"
        )

    elif config.type == "nicer":
        targets = params.get("targets", ["J0030", "J0740"])
        analysis_groups = params.get("analysis_groups", ["amsterdam", "maryland"])

        # FIXME: Implement load_nicer_kde(psr_name, analysis_group, n_samples) -> gaussian_kde
        # This should:
        # 1. Load NICER posterior samples from data files
        # 2. Construct KDE from the samples
        # 3. Handle caching to avoid re-creating KDEs
        # For now, raise NotImplementedError
        raise NotImplementedError(
            f"NICER likelihood data loading not implemented. "
            f"Need to implement load_nicer_kde() for targets: {targets}, groups: {analysis_groups}"
        )

    elif config.type == "radio":
        return RadioTimingLikelihood(
            psr_name=params.get("psr_name", "J0740+6620"),
            mean=params.get("mass_mean", 2.08),
            std=params.get("mass_std", 0.07),
            nb_masses=params.get("nb_masses", 100),
        )

    elif config.type == "chieft":
        # FIXME: Implement load_chieft_bands() -> (n_low, p_low, n_high, p_high)
        # This should load ChiEFT pressure-density bands from data files
        # For now, raise NotImplementedError
        raise NotImplementedError(
            "ChiEFT likelihood data loading not implemented. "
            "Need to implement load_chieft_bands() -> (n_low, p_low, n_high, p_high)"
        )

    elif config.type == "rex":
        experiment_name = params.get("experiment_name", "PREX")

        # FIXME: Implement load_rex_posterior(experiment_name) -> gaussian_kde
        # This should load PREX/CREX posterior KDE from data files
        # For now, raise NotImplementedError
        raise NotImplementedError(
            f"REX likelihood data loading not implemented. "
            f"Need to implement load_rex_posterior('{experiment_name}') -> gaussian_kde"
        )

    elif config.type == "zero":
        return ZeroLikelihood()

    else:
        raise ValueError(f"Unknown likelihood type: {config.type}")


def create_combined_likelihood(
    likelihood_configs: list[LikelihoodConfig],
    data_loader=None,
):
    """
    Create combined likelihood from list of configs

    Parameters
    ----------
    likelihood_configs : list[LikelihoodConfig]
        List of likelihood configurations
    data_loader : None
        DEPRECATED - data loading will be handled differently

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

        # Special handling for GW likelihoods: create one likelihood per event
        if config.type == "gw":
            params = config.parameters
            events = params["events"]  # Required, validated by schema
            penalty_value = params.get("penalty_value", -99999.0)
            N_masses_evaluation = params.get("N_masses_evaluation", 20)
            N_masses_batch_size = params.get("N_masses_batch_size", 10)

            # Create one GWLikelihood per event
            for event in events:
                gw_likelihood = GWLikelihood(
                    event_name=event["name"],
                    model_dir=event["model_dir"],
                    penalty_value=penalty_value,
                    N_masses_evaluation=N_masses_evaluation,
                    N_masses_batch_size=N_masses_batch_size,
                )
                likelihoods.append(gw_likelihood)
        else:
            # For non-GW likelihoods, use standard creation
            likelihood = create_likelihood(config, data_loader)
            if likelihood is not None:
                likelihoods.append(likelihood)

    if len(likelihoods) == 0:
        raise ValueError("No likelihoods enabled in configuration")
    elif len(likelihoods) == 1:
        return likelihoods[0]
    else:
        return CombinedLikelihood(likelihoods)
