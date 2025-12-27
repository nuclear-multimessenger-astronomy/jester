"""Data loading and path management for jesterTOV inference"""

# FIXME: DataLoader class was removed. Need to implement data loading functionality:
# - load_nicer_kde(psr_name, analysis_group, n_samples) -> gaussian_kde
# - load_chieft_bands() -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
# - load_rex_posterior(experiment_name) -> gaussian_kde
# - load_gw_nf_model(event_name, model_path) -> normalizing flow model
# These should be implemented as standalone functions with lazy loading/caching

__all__ = []
