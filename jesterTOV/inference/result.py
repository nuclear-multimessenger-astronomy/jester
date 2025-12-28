"""Unified HDF5 storage for JESTER inference results.

This module provides the InferenceResult class for storing and loading
inference results from all sampler types (FlowMC, BlackJAX SMC, BlackJAX NS-AW).
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Literal

import h5py
import numpy as np
from jaxtyping import Array

from .config.schema import InferenceConfig
from .samplers.jester_sampler import JesterSampler
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")

SamplerType = Literal["flowmc", "blackjax_smc", "blackjax_ns_aw"]


class InferenceResult:
    """Unified HDF5-based storage for JESTER inference results.

    This class provides a clean interface for saving and loading inference results
    from all sampler types. It stores:
    - Posterior samples (parameters + derived EOS quantities + sampler-specific data)
    - Metadata (sampler configuration + run statistics)
    - Histories (time-series data for diagnostics)

    Attributes
    ----------
    sampler_type : SamplerType
        Sampler backend type ("flowmc", "blackjax_smc", or "blackjax_ns_aw")
    posterior : Dict[str, np.ndarray]
        All posterior samples including parameters, derived quantities, and sampler-specific data
    metadata : Dict[str, Any]
        Sampler configuration and run statistics
    histories : Dict[str, np.ndarray] | None
        Time-series histories (ESS, acceptance, etc.) for diagnostics

    Examples
    --------
    Creating a result from a sampler:

    >>> result = InferenceResult.from_sampler(sampler, config, runtime=3600.0)
    >>> result.add_derived_eos(eos_quantities)
    >>> result.save("output/results.h5")

    Loading a result:

    >>> result = InferenceResult.load("output/results.h5")
    >>> print(result.summary())
    >>> masses = result.posterior["masses_EOS"]
    """

    def __init__(
        self,
        sampler_type: SamplerType,
        posterior: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
        histories: Dict[str, np.ndarray] | None = None,
    ):
        """Initialize InferenceResult.

        Parameters
        ----------
        sampler_type : SamplerType
            Sampler backend type
        posterior : Dict[str, np.ndarray]
            Posterior samples (parameters + derived + sampler-specific)
        metadata : Dict[str, Any]
            Sampler configuration and run statistics
        histories : Dict[str, np.ndarray] | None, optional
            Time-series histories for diagnostics
        """
        self.sampler_type = sampler_type
        self.posterior = posterior
        self.metadata = metadata
        self.histories = histories

    @classmethod
    def from_sampler(
        cls,
        sampler: JesterSampler,
        config: InferenceConfig,
        runtime: float,
        training: bool = False,
    ) -> "InferenceResult":
        """Create InferenceResult from sampler output.

        Parameters
        ----------
        sampler : JesterSampler
            Sampler instance after sampling is complete
        config : InferenceConfig
            Configuration used for inference
        runtime : float
            Total runtime in seconds
        training : bool, optional
            Whether to extract training samples (FlowMC only), by default False

        Returns
        -------
        InferenceResult
            Result object with samples and metadata
        """
        # Detect sampler type
        sampler_class_name = sampler.__class__.__name__
        if "FlowMC" in sampler_class_name:
            sampler_type = "flowmc"
        elif "SMC" in sampler_class_name:
            sampler_type = "blackjax_smc"
        elif "NS" in sampler_class_name or "NestedSampling" in sampler_class_name:
            sampler_type = "blackjax_ns_aw"
        else:
            # Fallback to config
            sampler_type = config.sampler.type  # type: ignore[assignment]

        logger.info(f"Extracting results from {sampler_type} sampler")

        # Get samples and log_prob from sampler
        samples = sampler.get_samples(training=training)
        log_prob = sampler.get_log_prob(training=training)
        n_samples = sampler.get_n_samples(training=training)

        # Convert JAX arrays to NumPy and separate parameter types
        posterior: Dict[str, np.ndarray] = {}
        sampler_specific: Dict[str, np.ndarray] = {}

        # Special fields that are sampler-specific, not parameters
        sampler_specific_keys = {'weights', 'ess', 'logL', 'logL_birth'}

        for key, value in samples.items():
            np_value = np.array(value)  # Convert JAX → NumPy
            if key in sampler_specific_keys:
                sampler_specific[key] = np_value
            else:
                posterior[key] = np_value

        # Add log_prob to posterior (common to all samplers)
        posterior['log_prob'] = np.array(log_prob)

        # Serialize config to JSON
        config_dict = config.model_dump()
        config_json = json.dumps(config_dict, indent=2)

        # Build metadata
        metadata: Dict[str, Any] = {
            'sampler': sampler_type,
            'runtime_seconds': float(runtime),
            'n_samples': int(n_samples),
            'seed': int(config.seed),
            'creation_timestamp': datetime.now().isoformat(),
            'config_json': config_json,
        }

        # Extract sampler-specific metadata and histories
        histories: Dict[str, np.ndarray] | None = None

        if sampler_type == "flowmc":
            # FlowMC: Get metadata from sampler state
            state = sampler.sampler.get_sampler_state(training=training)  # type: ignore[union-attr]

            # Add FlowMC-specific metadata
            flowmc_config = config.sampler  # type: ignore[attr-defined]
            metadata.update({
                'n_chains': int(flowmc_config.n_chains),  # type: ignore[attr-defined]
                'n_loop_training': int(flowmc_config.n_loop_training),  # type: ignore[attr-defined]
                'n_loop_production': int(flowmc_config.n_loop_production),  # type: ignore[attr-defined]
                'n_local_steps': int(flowmc_config.n_local_steps),  # type: ignore[attr-defined]
                'n_global_steps': int(flowmc_config.n_global_steps),  # type: ignore[attr-defined]
                'n_epochs': int(flowmc_config.n_epochs),  # type: ignore[attr-defined]
                'learning_rate': float(flowmc_config.learning_rate),  # type: ignore[attr-defined]
                'train_thinning': int(flowmc_config.train_thinning),  # type: ignore[attr-defined]
                'output_thinning': int(flowmc_config.output_thinning),  # type: ignore[attr-defined]
            })

            # Extract histories
            if 'local_accs' in state:
                histories = {
                    'local_accs': np.array(state['local_accs']),
                    'global_accs': np.array(state['global_accs']),
                    'loss_vals': np.array(state['loss_vals']),
                }

        elif sampler_type == "blackjax_smc":
            # SMC: Get metadata from sampler.metadata dict
            smc_metadata = sampler.metadata  # type: ignore[attr-defined]

            # Add SMC-specific metadata
            metadata.update({
                'kernel_type': str(smc_metadata['kernel_type']),
                'n_particles': int(smc_metadata['n_particles']),
                'n_mcmc_steps': int(smc_metadata['n_mcmc_steps']),
                'target_ess': float(smc_metadata['target_ess']),
                'annealing_steps': int(smc_metadata['annealing_steps']),
                'final_ess': float(smc_metadata['final_ess']),
                'final_ess_percent': float(smc_metadata['final_ess_percent']),
                'mean_ess': float(smc_metadata['mean_ess']),
                'min_ess': float(smc_metadata['min_ess']),
                'mean_acceptance': float(smc_metadata['mean_acceptance']),
                'logZ': float(smc_metadata['logZ']),
                'logZ_err': float(smc_metadata['logZ_err']),
            })

            # Extract histories
            histories = {
                'lmbda_history': np.array(smc_metadata['lmbda_history']),
                'ess_history': np.array(smc_metadata['ess_history']),
                'acceptance_history': np.array(smc_metadata['acceptance_history']),
            }

        elif sampler_type == "blackjax_ns_aw":
            # NS-AW: Get metadata from sampler.metadata dict
            ns_metadata = sampler.metadata  # type: ignore[attr-defined]

            # Add NS-AW-specific metadata
            metadata.update({
                'n_live': int(ns_metadata['n_live']),
                'n_delete': int(ns_metadata['n_delete']),
                'n_delete_frac': float(ns_metadata['n_delete_frac']),
                'n_target': int(ns_metadata['n_target']),
                'max_mcmc': int(ns_metadata['max_mcmc']),
                'max_proposals': int(ns_metadata['max_proposals']),
                'termination_dlogz': float(ns_metadata['termination_dlogz']),
                'n_iterations': int(ns_metadata['n_iterations']),
                'n_likelihood_evaluations': int(ns_metadata['n_likelihood_evaluations']),
                'logZ': float(ns_metadata['logZ']),
                'logZ_err': float(ns_metadata['logZ_err']),
            })

            # Optional anesthetic evidence (only add if both are present)
            if 'logZ_anesthetic' in ns_metadata and 'logZ_err_anesthetic' in ns_metadata:
                metadata['logZ_anesthetic'] = float(ns_metadata['logZ_anesthetic'])
                metadata['logZ_err_anesthetic'] = float(ns_metadata['logZ_err_anesthetic'])

        # Store sampler-specific data if present
        if sampler_specific:
            posterior['_sampler_specific'] = sampler_specific  # type: ignore[assignment]

        return cls(
            sampler_type=sampler_type,  # type: ignore[arg-type]
            posterior=posterior,
            metadata=metadata,
            histories=histories,
        )

    def add_derived_eos(self, eos_dict: Dict[str, Array]) -> None:
        """Add derived EOS quantities to posterior.

        This should be called after TOV solver generates M-R-Lambda curves.

        Parameters
        ----------
        eos_dict : Dict[str, Array]
            Dictionary of derived EOS quantities (masses_EOS, radii_EOS, Lambdas_EOS, n, p, e, cs2, etc.)
        """
        logger.info("Adding derived EOS quantities to posterior")

        # Convert JAX arrays to NumPy and add to posterior
        for key, value in eos_dict.items():
            self.posterior[key] = np.array(value)

        logger.info(f"Added {len(eos_dict)} derived EOS quantities")

    def save(self, filepath: str | Path) -> None:
        """Save to HDF5 file.

        Parameters
        ----------
        filepath : str | Path
            Path to output HDF5 file

        Raises
        ------
        OSError
            If file cannot be written
        """
        filepath = Path(filepath)
        logger.info(f"Saving inference results to {filepath}")

        # Create output directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(filepath, 'w') as f:
            # Create /posterior group
            posterior_grp = f.create_group('posterior')

            # Separate parameters from derived quantities
            # Heuristic: derived quantities have specific names
            derived_keys = {'masses_EOS', 'radii_EOS', 'Lambdas_EOS', 'n', 'p', 'e', 'cs2'}
            sampler_specific_keys = {'weights', 'ess', 'logL', 'logL_birth'}

            # Get sampler-specific data if present
            sampler_specific_data = self.posterior.pop('_sampler_specific', {})
            if isinstance(sampler_specific_data, np.ndarray):
                sampler_specific_data = {}

            # Create subgroups
            params_grp = posterior_grp.create_group('parameters')
            derived_grp = posterior_grp.create_group('derived_eos')
            sampler_grp = posterior_grp.create_group('sampler_specific')

            # Distribute datasets to appropriate groups
            for key, value in self.posterior.items():
                if key == 'log_prob':
                    # log_prob goes directly in /posterior
                    posterior_grp.create_dataset('log_prob', data=value)
                elif key in derived_keys:
                    derived_grp.create_dataset(key, data=value)
                elif key in sampler_specific_keys:
                    sampler_grp.create_dataset(key, data=value)
                else:
                    # Assume it's a parameter
                    params_grp.create_dataset(key, data=value)

            # Add sampler-specific data from the dict
            for key, value in sampler_specific_data.items():  # type: ignore[union-attr]
                sampler_grp.create_dataset(key, data=value)

            # Create /metadata group
            metadata_grp = f.create_group('metadata')

            # Store config as JSON dataset
            config_json = self.metadata.pop('config_json', '{}')
            metadata_grp.create_dataset('config', data=config_json)

            # Store all other metadata as HDF5 attributes
            for key, value in self.metadata.items():
                # HDF5 attributes must be scalars or small arrays
                if isinstance(value, (int, float, str, bool)):
                    metadata_grp.attrs[key] = value
                elif isinstance(value, np.ndarray) and value.size < 10:
                    metadata_grp.attrs[key] = value
                else:
                    # Skip large arrays (should be in histories)
                    logger.warning(f"Skipping large metadata field: {key}")

            # Create /histories group if applicable
            if self.histories is not None and len(self.histories) > 0:
                histories_grp = f.create_group('histories')
                for key, value in self.histories.items():
                    histories_grp.create_dataset(key, data=value)

        logger.info(f"Successfully saved results to {filepath}")

    @classmethod
    def load(cls, filepath: str | Path) -> "InferenceResult":
        """Load from HDF5 file.

        Parameters
        ----------
        filepath : str | Path
            Path to HDF5 file

        Returns
        -------
        InferenceResult
            Loaded result object

        Raises
        ------
        FileNotFoundError
            If file does not exist
        OSError
            If file cannot be read
        """
        filepath = Path(filepath)
        logger.info(f"Loading inference results from {filepath}")

        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")

        with h5py.File(filepath, 'r') as f:
            # Load posterior
            posterior: Dict[str, np.ndarray] = {}

            # Load from parameters subgroup
            if 'posterior/parameters' in f:
                for key in f['posterior/parameters'].keys():  # type: ignore[union-attr]
                    posterior[key] = f['posterior/parameters'][key][:]  # type: ignore[index]

            # Load from derived_eos subgroup
            if 'posterior/derived_eos' in f:
                for key in f['posterior/derived_eos'].keys():  # type: ignore[union-attr]
                    posterior[key] = f['posterior/derived_eos'][key][:]  # type: ignore[index]

            # Load from sampler_specific subgroup
            if 'posterior/sampler_specific' in f:
                sampler_specific = {}
                for key in f['posterior/sampler_specific'].keys():  # type: ignore[union-attr]
                    dataset = f['posterior/sampler_specific'][key]  # type: ignore[index]
                    # Handle scalar vs array datasets
                    if dataset.shape == ():  # type: ignore[union-attr]
                        # Scalar dataset - use [()] instead of [:]
                        sampler_specific[key] = dataset[()]  # type: ignore[index]
                    else:
                        # Array dataset - use [:]
                        sampler_specific[key] = dataset[:]  # type: ignore[index]
                if sampler_specific:
                    posterior['_sampler_specific'] = sampler_specific  # type: ignore[assignment]

            # Load log_prob
            if 'posterior/log_prob' in f:
                posterior['log_prob'] = f['posterior/log_prob'][:]  # type: ignore[index]

            # Load metadata
            metadata: Dict[str, Any] = {}

            # Load config JSON
            if 'metadata/config' in f:
                config_json = f['metadata/config'][()]  # type: ignore[index]
                if isinstance(config_json, bytes):
                    config_json = config_json.decode('utf-8')
                metadata['config_json'] = config_json

            # Load attributes
            if 'metadata' in f:
                for key, value in f['metadata'].attrs.items():
                    metadata[key] = value

            # Load histories
            histories: Dict[str, np.ndarray] | None = None
            if 'histories' in f:
                histories = {}
                for key in f['histories'].keys():  # type: ignore[union-attr]
                    histories[key] = f['histories'][key][:]  # type: ignore[index]

            # Get sampler type from metadata
            sampler_type = metadata.get('sampler', 'unknown')

        logger.info(f"Successfully loaded {sampler_type} results")

        return cls(
            sampler_type=sampler_type,  # type: ignore[arg-type]
            posterior=posterior,
            metadata=metadata,
            histories=histories,
        )

    @property
    def config_dict(self) -> Dict[str, Any]:
        """Deserialize configuration JSON to dictionary.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary
        """
        config_json = self.metadata.get('config_json', '{}')
        return json.loads(config_json)

    def summary(self) -> str:
        """Generate human-readable summary of results.

        Returns
        -------
        str
            Formatted summary string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("JESTER Inference Results Summary")
        lines.append("=" * 60)

        # Basic info
        lines.append(f"Sampler: {self.sampler_type}")
        lines.append(f"Creation time: {self.metadata.get('creation_timestamp', 'unknown')}")
        lines.append(f"Runtime: {self.metadata.get('runtime_seconds', 0):.1f} seconds")
        lines.append(f"Random seed: {self.metadata.get('seed', 'unknown')}")
        lines.append(f"Number of samples: {self.metadata.get('n_samples', 0)}")

        # Sampler-specific info
        if self.sampler_type == "flowmc":
            lines.append(f"\nFlowMC Configuration:")
            lines.append(f"  Chains: {self.metadata.get('n_chains', '?')}")
            lines.append(f"  Training loops: {self.metadata.get('n_loop_training', '?')}")
            lines.append(f"  Production loops: {self.metadata.get('n_loop_production', '?')}")

        elif self.sampler_type == "blackjax_smc":
            lines.append(f"\nBlackJAX SMC Configuration:")
            lines.append(f"  Kernel type: {self.metadata.get('kernel_type', '?')}")
            lines.append(f"  Particles: {self.metadata.get('n_particles', '?')}")
            lines.append(f"  Annealing steps: {self.metadata.get('annealing_steps', '?')}")
            lines.append(f"  Final ESS: {self.metadata.get('final_ess_percent', '?'):.1f}%")
            lines.append(f"  Mean acceptance: {self.metadata.get('mean_acceptance', '?'):.3f}")
            if 'logZ' in self.metadata:
                lines.append(f"  Evidence: log(Z) = {self.metadata.get('logZ', 0):.2f}")

        elif self.sampler_type == "blackjax_ns_aw":
            lines.append(f"\nBlackJAX Nested Sampling Configuration:")
            lines.append(f"  Live points: {self.metadata.get('n_live', '?')}")
            lines.append(f"  Iterations: {self.metadata.get('n_iterations', '?')}")
            lines.append(f"  Evidence: log(Z) = {self.metadata.get('logZ', 0):.2f} ± {self.metadata.get('logZ_err', 0):.2f}")

        # Posterior info
        # Extract parameter keys (excluding special fields)
        param_keys = [k for k in self.posterior.keys()
                      if k not in {'log_prob', 'masses_EOS', 'radii_EOS', 'Lambdas_EOS',
                                   'n', 'p', 'e', 'cs2', '_sampler_specific'}]

        lines.append(f"\nPosterior Samples:")
        lines.append(f"  Parameters: {len(param_keys)} ({', '.join(param_keys[:5])}{'...' if len(param_keys) > 5 else ''})")

        # Check for derived EOS quantities
        has_derived = 'masses_EOS' in self.posterior
        lines.append(f"  Derived EOS quantities: {'Yes' if has_derived else 'No'}")

        lines.append("=" * 60)

        return "\n".join(lines)
