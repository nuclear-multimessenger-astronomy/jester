r"""Post-hoc diagnostics for neural variational inference.

``importance_diagnostics`` ports ``gwax/gwax/variational.py``'s ``importance``/``_importance``
self-normalized importance sampling diagnostics (effective sample size, log-evidence estimate)
to jester's API, and adds a Pareto-:math:`\hat{k}` check via ``arviz.psislw`` (Pareto-smoothed
importance sampling) — a standard convergence diagnostic for how well a trained flow
approximates its target (:math:`\hat{k} < 0.7` is the usual "good enough" threshold).

``compare_posteriors`` is a lightweight two-sample comparison (per-parameter mean/std ratio
+ a KS test) used throughout the milestone validation scripts in
``neural_bayesian_updates_jester/`` to check NVI posteriors against SMC baselines.
"""

from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import Float, PRNGKeyArray
from scipy.stats import ks_2samp

from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.inference.base.prior import Prior
from jesterTOV.inference.base.transform import NtoMTransform
from jesterTOV.inference.variational.targets import (
    make_flow_prior_target,
    make_prior_likelihood_target,
)


def _importance_stats(log_weights: Float[Array, " n"]) -> dict[str, float]:
    """Effective-sample-size and log-evidence estimate from unnormalized log weights.

    Ports ``gwax.variational._importance``.
    """
    n = log_weights.size
    log_evidence = jax.nn.logsumexp(log_weights) - jnp.log(n)
    log_sq_mean = 2 * log_evidence
    log_mean_sq = jax.nn.logsumexp(2 * log_weights) - jnp.log(n)
    efficiency = jnp.exp(log_sq_mean - log_mean_sq)
    ess = efficiency * n
    log_evidence_variance = 1 / ess - 1 / n
    log_evidence_sigma = log_evidence_variance**0.5
    return dict(
        ess=float(ess),
        efficiency=float(efficiency),
        log_evidence=float(log_evidence),
        log_evidence_sigma=float(log_evidence_sigma),
    )


def importance_diagnostics(
    key: PRNGKeyArray,
    flow: Any,
    likelihood: LikelihoodBase,
    likelihood_transforms: Sequence[NtoMTransform],
    parameter_names: Sequence[str],
    *,
    prior: Prior | None = None,
    prev_flow: Any | None = None,
    n_samples: int = 20_000,
) -> dict[str, Any]:
    """Self-normalized importance-sampling diagnostics for a trained flow vs. its target.

    Draws ``n_samples`` from ``flow``, evaluates importance weights against the target
    (``prior x likelihood``, or the sequential-update target if ``prev_flow`` is given
    instead of ``prior``), and reports effective sample size, efficiency, a self-normalized
    log-evidence estimate, and the Pareto-:math:`\\hat{k}` shape parameter from
    Pareto-smoothed importance sampling (``arviz.psislw``) — the standard PSIS convergence
    diagnostic (:math:`\\hat{k} < 0.7` indicates the flow is a good enough proposal).

    Exactly one of ``prior``/``prev_flow`` must be given, matching
    :func:`jesterTOV.inference.variational.trainer.train_vi`'s two training modes.
    """
    if (prior is None) == (prev_flow is None):
        raise ValueError("Exactly one of `prior` or `prev_flow` must be given.")

    raw_flow = flow.flow if hasattr(flow, "flow") else flow

    if prior is not None:
        log_target_fn = make_prior_likelihood_target(
            prior, likelihood, likelihood_transforms, parameter_names
        )
    else:
        log_target_fn = make_flow_prior_target(
            prev_flow, likelihood, likelihood_transforms, parameter_names
        )

    samples, log_flows = raw_flow.sample_and_log_prob(key, (n_samples,))
    log_targets = log_target_fn(samples)
    log_weights = log_targets - log_flows

    import arviz as az

    _, khat = az.psislw(np.asarray(log_weights))

    return dict(
        samples=samples,
        log_weights=log_weights,
        khat=float(khat),
        **_importance_stats(log_weights),
    )


def compare_posteriors(
    samples_a: Mapping[str, Array | np.ndarray],
    samples_b: Mapping[str, Array | np.ndarray],
    names: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Per-parameter comparison of two sets of posterior samples.

    For each parameter in ``names``, reports the mean difference in units of the pooled
    standard deviation, the ratio of standard deviations, and a two-sample Kolmogorov-Smirnov
    test (statistic + p-value) between ``samples_a[name]`` and ``samples_b[name]``.

    Used to compare NVI posteriors against SMC baselines (and, in Milestone 0, two
    independent SMC runs of the same target against each other, to fix tolerance
    thresholds before any NVI code is exercised).
    """
    report: dict[str, dict[str, float]] = {}
    for name in names:
        a = np.asarray(samples_a[name])
        b = np.asarray(samples_b[name])
        mean_a, mean_b = a.mean(), b.mean()
        std_a, std_b = a.std(), b.std()
        pooled_std = float(np.sqrt(0.5 * (std_a**2 + std_b**2)))
        ks_result = ks_2samp(a, b)
        # scipy's stub exposes ks_2samp's return as a bare generic tuple; the runtime
        # values are always floats (statistic, pvalue).
        ks_stat = float(ks_result[0])  # type: ignore[arg-type]
        ks_pvalue = float(ks_result[1])  # type: ignore[arg-type]
        report[name] = dict(
            mean_a=float(mean_a),
            mean_b=float(mean_b),
            mean_diff_over_pooled_std=float((mean_a - mean_b) / pooled_std)
            if pooled_std > 0
            else float("nan"),
            std_a=float(std_a),
            std_b=float(std_b),
            std_ratio=float(std_a / std_b) if std_b > 0 else float("nan"),
            ks_statistic=ks_stat,
            ks_pvalue=ks_pvalue,
        )
    return report
