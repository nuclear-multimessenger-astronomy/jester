r"""Aspire posterior-reuse sampler for jesterTOV.

This module integrates the `aspire` library (Accelerated Sequential Posterior
Inference via REuse) as a jester sampler backend. It trains a normalizing flow
on samples from a previous jester run, then draws a new posterior under an
updated likelihood without re-running the full EOS+TOV pipeline.

The bridge uses JAX throughout (flowjax backend) for consistency with the
rest of jester.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PRNGKeyArray

from jesterTOV.logging_config import get_logger

from ..base import LikelihoodBase, Prior, BijectiveTransform, NtoMTransform
from ..base.prior import UniformPrior, CombinePrior
from ..config.schemas.samplers import AspireSamplerConfig
from .jester_sampler import JesterSampler, SamplerOutput

logger = get_logger("jester")


def _extract_prior_bounds(
    prior: Prior,
) -> dict[str, tuple[float, float]] | None:
    """Extract parameter bounds from a jester prior.

    Parameters
    ----------
    prior : Prior
        jester prior, typically a :class:`CombinePrior`.

    Returns
    -------
    dict[str, tuple[float, float]] | None
        Mapping from parameter name to ``(xmin, xmax)`` for each
        :class:`UniformPrior` component. Returns ``None`` if no bounded
        priors are found (e.g. a pure Gaussian prior).
    """
    if not isinstance(prior, CombinePrior):
        return None

    bounds: dict[str, tuple[float, float]] = {}
    for p in prior.base_prior:
        if isinstance(p, UniformPrior):
            name = p.parameter_names[0]
            bounds[name] = (float(p.xmin), float(p.xmax))

    return bounds if bounds else None


class AspireSampler(JesterSampler):
    r"""Posterior-reuse sampler using the aspire normalizing flow library.

    Workflow
    --------
    1. Load upstream jester samples from HDF5 (prior or previous posterior).
    2. Resample if SMC importance weights are present to produce an
       unweighted training set.
    3. Build JAX-native :func:`log_likelihood` and :func:`log_prior`
       callables via :func:`jax.vmap` over the jester likelihood/prior.
    4. Train a normalizing flow (flowjax backend) on the upstream samples.
    5. Draw a new posterior under the updated likelihood using aspire SMC.
    6. Return results as :class:`~jesterTOV.inference.samplers.SamplerOutput`.

    Parameters
    ----------
    likelihood : LikelihoodBase
        New likelihood to evaluate (the one the posterior is updated *to*).
    prior : Prior
        Prior distribution shared between upstream and current run.
    sample_transforms : list[BijectiveTransform] | None
        Bijective transforms applied during sampling (with Jacobians).
    likelihood_transforms : list[NtoMTransform] | None
        N-to-M transforms applied before likelihood evaluation (e.g. JesterTransform).
    config : AspireSamplerConfig
        Sampler configuration.
    seed : int
        Random seed.
    """

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Prior,
        sample_transforms: list[BijectiveTransform] | None,
        likelihood_transforms: list[NtoMTransform] | None,
        config: AspireSamplerConfig,
        seed: int,
    ) -> None:
        super().__init__(likelihood, prior, sample_transforms, likelihood_transforms)
        self.config = config
        self.seed = seed
        self._output: SamplerOutput | None = None

    # ------------------------------------------------------------------
    # Core sampling
    # ------------------------------------------------------------------

    def sample(self, key: PRNGKeyArray) -> None:
        """Train a normalizing flow on upstream samples and draw the new posterior.

        Parameters
        ----------
        key : PRNGKeyArray
            JAX random key (passed for API compatibility; aspire uses its
            own random state internally).
        """
        from ..result import InferenceResult

        try:
            from aspire import Aspire
            from aspire.samples import Samples as AspireSamples
        except ImportError as exc:
            raise ImportError(
                "The 'aspire' package is required for AspireSampler. "
                "Install it from /Users/Woute029/Documents/Code/projects/31_jester_aspire/aspire/"
            ) from exc

        # Bridge aspire's logger to jester's handler so SMC iteration logs appear.
        aspire_root_logger = logging.getLogger("aspire")
        aspire_root_logger.setLevel(logging.INFO)
        for _handler in logger.handlers:
            if _handler not in aspire_root_logger.handlers:
                aspire_root_logger.addHandler(_handler)
        aspire_root_logger.propagate = False

        logger.info("=" * 60)
        logger.info("AspireSampler: starting")
        logger.info(f"  upstream_result_path : {self.config.upstream_result_path}")
        logger.info(f"  n_resample           : {self.config.n_resample}")
        logger.info(f"  flow backend         : flowjax")
        logger.info(f"  flow n_epochs        : {self.config.n_epochs}")
        logger.info(f"  flow batch_size      : {self.config.batch_size}")
        logger.info(f"  flow learning_rate   : {self.config.lr}")
        logger.info(f"  aspire sampler       : {self.config.aspire_sampler}")
        logger.info(f"  n_samples            : {self.config.n_samples}")
        logger.info("=" * 60)
        t0_total = time.time()

        # ------------------------------------------------------------------
        # 1. Obtain training samples from upstream result or prior
        # ------------------------------------------------------------------
        param_names = self.prior.parameter_names
        logger.info(f"Parameters ({len(param_names)}): {param_names}")

        if self.config.upstream_result_path is not None:
            # Load samples from a previous jester run
            upstream = InferenceResult.load(self.config.upstream_result_path)
            x_np = np.column_stack(
                [np.asarray(upstream.posterior[p]) for p in param_names]
            )
            logger.info(f"Upstream samples shape: {x_np.shape}")

            # Resample if SMC importance weights are present
            sampler_specific = upstream.posterior.get("_sampler_specific", {})
            if isinstance(sampler_specific, dict) and "weights" in sampler_specific:
                w = np.asarray(sampler_specific["weights"], dtype=np.float64)
                w = w / w.sum()
                rng = np.random.default_rng(self.seed)
                n_resample = min(self.config.n_resample, len(w))
                idx = rng.choice(len(w), size=n_resample, p=w, replace=True)
                x_np = x_np[idx]
                logger.info(
                    f"Resampled {n_resample} training samples from {len(w)} "
                    "weighted upstream particles"
                )
            else:
                n_resample = min(self.config.n_resample, len(x_np))
                x_np = x_np[:n_resample]
                logger.info(f"Using {n_resample} upstream samples for flow training")
        else:
            # No upstream result — draw fresh samples from the prior
            logger.info(
                f"No upstream_result_path set — sampling {self.config.n_resample} "
                "points from the prior to seed the normalizing flow"
            )
            key, key_prior = jax.random.split(key)
            prior_dict = self.prior.sample(key_prior, self.config.n_resample)
            x_np = np.column_stack(
                [np.asarray(prior_dict[p]) for p in param_names]
            )
            logger.info(f"Prior samples shape: {x_np.shape}")

        # ------------------------------------------------------------------
        # 3. Build JAX-native aspire callables via vmap
        # ------------------------------------------------------------------
        likelihood = self.likelihood
        prior = self.prior
        likelihood_transforms = self.likelihood_transforms

        def _log_likelihood_single(x_row: Array) -> Array:
            pdict = dict(zip(param_names, x_row))
            for t in likelihood_transforms:
                pdict = t.forward(pdict)
            return likelihood.evaluate(pdict)

        def _log_prior_single(x_row: Array) -> Array:
            pdict = dict(zip(param_names, x_row))
            return prior.log_prob(pdict)

        log_likelihood_vmap = jax.vmap(_log_likelihood_single)
        log_prior_vmap = jax.vmap(_log_prior_single)

        def aspire_log_likelihood(samples: Any) -> Array:
            return log_likelihood_vmap(samples.x)

        def aspire_log_prior(samples: Any) -> Array:
            return log_prior_vmap(samples.x)

        # ------------------------------------------------------------------
        # 4. Instantiate aspire with JAX (flowjax) backend
        # ------------------------------------------------------------------
        prior_bounds = _extract_prior_bounds(prior)

        aspire_obj = Aspire(
            log_likelihood=aspire_log_likelihood,
            log_prior=aspire_log_prior,
            dims=len(param_names),
            parameters=param_names,
            prior_bounds=prior_bounds,
            bounded_to_unbounded=prior_bounds is not None,
            flow_backend="flowjax",
            xp=jnp,
            dtype=jnp.float64,  # match jester's 64-bit precision
        )

        # ------------------------------------------------------------------
        # 5. Train flow on upstream samples
        # ------------------------------------------------------------------
        logger.info(
            f"Training normalizing flow: {len(x_np)} samples, "
            f"{self.config.n_epochs} max epochs, "
            f"batch_size={self.config.batch_size}, lr={self.config.lr}"
        )
        training_samples = AspireSamples(
            x=jnp.array(x_np, dtype=jnp.float64),
            parameters=param_names,
        )
        t0_flow = time.time()
        # flowjax backend uses max_epochs / learning_rate (not n_epochs / lr)
        flow_history = aspire_obj.fit(
            training_samples,
            max_epochs=self.config.n_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.lr,
        )
        dt_flow = time.time() - t0_flow
        # Log training loss summary if history is available
        if hasattr(flow_history, "training_loss") and flow_history.training_loss:
            train_losses = flow_history.training_loss
            logger.info(
                f"Flow training complete in {dt_flow:.1f}s "
                f"({len(train_losses)} epochs) — "
                f"train loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f}"
            )
        else:
            logger.info(f"Flow training complete in {dt_flow:.1f}s")

        # ------------------------------------------------------------------
        # 6. Draw new posterior samples
        # ------------------------------------------------------------------
        logger.info(
            f"Sampling posterior via aspire '{self.config.aspire_sampler}' "
            f"({self.config.n_samples} samples) — SMC iterations will be logged below..."
        )
        t0_smc = time.time()
        posterior_aspire = aspire_obj.sample_posterior(
            n_samples=self.config.n_samples,
            sampler=self.config.aspire_sampler,
        )
        dt_smc = time.time() - t0_smc
        log_evidence_msg = ""
        if hasattr(posterior_aspire, "log_evidence") and posterior_aspire.log_evidence is not None:
            log_ev = float(posterior_aspire.log_evidence)
            log_evidence_msg = f", log Z = {log_ev:.3f}"
            if hasattr(posterior_aspire, "log_evidence_error") and posterior_aspire.log_evidence_error is not None:
                log_evidence_msg += f" ± {float(posterior_aspire.log_evidence_error):.3f}"
        logger.info(f"Posterior sampling complete in {dt_smc:.1f}s{log_evidence_msg}")

        # ------------------------------------------------------------------
        # 7. Convert to SamplerOutput
        # ------------------------------------------------------------------
        x_out = np.asarray(posterior_aspire.x)  # (n_samples, n_dims)
        samples_dict: dict[str, Array] = {
            name: jnp.array(x_out[:, i]) for i, name in enumerate(param_names)
        }

        # Compute log-posterior for output (likelihood + prior)
        log_prob = aspire_log_likelihood(posterior_aspire) + aspire_log_prior(
            posterior_aspire
        )

        metadata: dict[str, Any] = {}
        # Attach aspire importance weights if present (importance sampling)
        if hasattr(posterior_aspire, "log_w") and posterior_aspire.log_w is not None:
            metadata["log_w"] = jnp.array(np.asarray(posterior_aspire.log_w))

        self._output = SamplerOutput(
            samples=samples_dict,
            log_prob=log_prob,
            metadata=metadata,
        )
        dt_total = time.time() - t0_total
        logger.info(
            f"AspireSampler complete — {self.config.n_samples} posterior samples "
            f"in {dt_total:.1f}s total "
            f"(flow: {dt_flow:.1f}s, SMC: {dt_smc:.1f}s)"
        )

    # ------------------------------------------------------------------
    # JesterSampler interface
    # ------------------------------------------------------------------

    def get_sampler_output(self) -> SamplerOutput:
        if self._output is None:
            raise RuntimeError("Call sample() before accessing results")
        return self._output

    def get_samples(self) -> dict[str, Array]:
        return self.get_sampler_output().samples

    def get_log_prob(self) -> Array:
        return self.get_sampler_output().log_prob

    def get_n_samples(self) -> int:
        return int(jnp.shape(self.get_sampler_output().log_prob)[0])

    def print_summary(self, transform: bool = True) -> None:
        if self._output is None:
            logger.info("AspireSampler: no results yet (call sample() first)")
            return
        n = self.get_n_samples()
        logger.info(f"AspireSampler summary: {n} posterior samples")
        log_prob = np.asarray(self._output.log_prob)
        logger.info(
            f"  log_prob: min={log_prob.min():.2f}, "
            f"mean={log_prob.mean():.2f}, max={log_prob.max():.2f}"
        )
