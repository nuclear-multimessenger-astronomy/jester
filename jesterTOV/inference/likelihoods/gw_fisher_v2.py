r"""GW Fisher-forecast likelihood, v2: direct 4D (m1, m2, LambdaTilde, deltaLambdaTilde)
comparison, replacing v1's (LambdaTilde, q)-marginal-with-q-quadrature construction.

Motivation
----------
:class:`~jesterTOV.inference.likelihoods.gw_fisher.GWFisherLikelihood` (v1) was found
to have a structural bias: because it assumes ``Cov(LambdaTilde, q) = 0`` (the only
option available, since gwfast's exported Fisher errors are marginalized/diagonal --
see the investigation this class grew out of), its per-source marginalized likelihood
analytically rewards candidate EOS curves for having as *flat* a
:math:`\tilde{\Lambda}(q)` dependence as possible, independent of the true correlation
or the true EOS's actual slope. Confirmed empirically: gradient ascent on the real v1
likelihood, started exactly at the point that reproduces the true injected EOS almost
perfectly, climbs steadily *away* from the truth.

This class does not fix that (the zero-correlation assumption is still the only option
without a genuine covariance/Fisher-matrix export from gwfast -- see the module
docstring's "Data files" section), but it removes a *different*, independently-found
problem with the natural first attempt at extending v1 to more dimensions: inverting
the observed :math:`(\tilde{\Lambda}, \delta\tilde{\Lambda})` measurement to
:math:`(\Lambda_1, \Lambda_2)` via the closed-form Wade et al. formula is numerically
dangerous for realistic Fisher-error magnitudes -- :math:`\delta\tilde{\Lambda}` is
very weakly measured for most sources (often consistent with zero), and the inversion's
denominator has a genuine singularity as :math:`q \to 1` (equal mass) that Monte Carlo
sampling routinely lands near. Empirically this produced sample fractions well above
50% landing at unphysical (negative) :math:`\Lambda_1`/:math:`\Lambda_2` for many
sources, occasional numerical blow-ups to absurd magnitudes even among the
"physical" (positive) samples, and a measurable bias in the surviving sample's mean
even after discarding the negative tail.

**This class avoids the inversion entirely** by never representing the observed data
as :math:`(\Lambda_1, \Lambda_2)`. Instead, it keeps the data in its native
:math:`(m_1, m_2, \tilde{\Lambda}, \delta\tilde{\Lambda})` representation (exactly the
quantities gwfast actually reports Fisher errors for) and forward-transforms the
*candidate EOS's* prediction -- :math:`\Lambda_{1,X}(m_1)`, :math:`\Lambda_{2,X}(m_2)`
are already deterministic, one-to-one, always-non-negative functions of mass for any
single EOS, and the forward Wade et al. map to
:math:`(\tilde{\Lambda}_X, \delta\tilde{\Lambda}_X)` involves no division that can
vanish. This mirrors :class:`~jesterTOV.inference.likelihoods.gw.GWLikelihood`'s
convention for real events (pre-sample once at construction from the observational
model, evaluate the joint density with the EOS's own prediction substituted in,
Monte-Carlo-average via ``logsumexp(...) - log(N)``) with a per-source diagonal
Gaussian standing in for a trained normalizing flow.

Design: correlation-readiness
------------------------------
Each source's Gaussian is built as a full 4x4 mean/covariance pair, not four separate
1D distributions, even though the covariance is diagonal today (no cross terms are
available anywhere in the current data pipeline -- see the module docstring's "Data
files" section). ``jax.scipy.stats.multivariate_normal.logpdf`` already accepts a
fully general covariance matrix, so if a future gwfast export ever includes genuine
cross-covariance terms (e.g. ``cov_LambdaTilde_m1_src``, ``cov_LambdaTilde_m2_src``),
the *only* change needed is filling in the corresponding off-diagonal entries of
``covs`` in :func:`_build_source_gaussians_4d` below -- ``evaluate()`` and the mass
pre-sampling both already operate on the general 4x4 matrix and require no changes.

Data files
----------
Same two HDF5 files as v1 (see that module's docstring for the general format), with
one additional requirement: ``gwfast_result_file`` must include an ``err_deltaLambda``
column. This is only present for ``2Lmis``/``2Lpar`` detector configurations, not
``Delta`` (see ``et-bgr-jester/datasets/CLAUDE.md``) -- constructing this class against
a ``Delta`` result file raises a clear ``KeyError`` naming the missing column, rather
than silently falling back to a lower-dimensional construction.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal
from jaxtyping import Array, Float

from jesterTOV import utils
from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.inference.likelihoods.gw_fisher import (
    _load_injection_catalog,
    _positivity_quality_mask,
    _read_hdf5_datasets,
    _validate_component_mass_ordering,
)
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")

# v1's _GWFAST_RESULT_KEYS plus err_deltaLambda -- everything this likelihood needs
# from the gwfast result file.
_GWFAST_RESULT_KEYS_V2: tuple[str, ...] = (
    "err_LambdaTilde",
    "err_deltaLambda",
    "err_m1_src",
    "err_m2_src",
    "idx_det_in_cat",
    "snrs",
)


def _load_gwfast_result_v2(path: str | Path) -> dict[str, np.ndarray]:
    """Load the gwfast Fisher-forecast result HDF5 file, including err_deltaLambda.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If ``err_deltaLambda`` (or any other required key) is missing -- this is the
        clear, fail-fast signal that a ``Delta``-config result file was passed to a
        likelihood that requires ``2Lmis``/``2Lpar``-style data.
    """
    return _read_hdf5_datasets(path, _GWFAST_RESULT_KEYS_V2)


def _build_source_gaussians_4d(
    m1_true: np.ndarray,
    m2_true: np.ndarray,
    err_m1: np.ndarray,
    err_m2: np.ndarray,
    lambda_tilde_true: np.ndarray,
    err_lambda_tilde: np.ndarray,
    delta_lambda_true: np.ndarray,
    err_delta_lambda: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Per-source 4D Gaussian in :math:`(m_1, m_2, \tilde{\Lambda}, \delta\tilde{\Lambda})`.

    All four quantities are independent 1-sigma Fisher errors reported directly by
    gwfast -- no inversion, no derived-quantity error propagation (contrast with v1's
    :math:`q`, which requires propagating ``err_m1_src``/``err_m2_src`` through
    :math:`q = m_2/m_1`). The covariance is diagonal because that is genuinely all the
    current data pipeline provides (see the module docstring) -- built as a full 4x4
    matrix, not four independent 1D distributions, so a future non-zero cross term
    only requires changing this function, not :meth:`GWFisherLikelihoodV2.evaluate`.

    Parameters
    ----------
    m1_true, m2_true : np.ndarray, shape (n_sources,)
        True (injected) source-frame component masses, ``m1_true >= m2_true``.
    err_m1, err_m2 : np.ndarray, shape (n_sources,)
        1-sigma Fisher errors on the component masses.
    lambda_tilde_true, delta_lambda_true : np.ndarray, shape (n_sources,)
        True (injected) effective and antisymmetric tidal terms.
    err_lambda_tilde, err_delta_lambda : np.ndarray, shape (n_sources,)
        1-sigma Fisher errors on the effective and antisymmetric tidal terms.

    Returns
    -------
    means : np.ndarray, shape (n_sources, 4)
        Per-source ``[m1_true, m2_true, lambda_tilde_true, delta_lambda_true]``.
    covs : np.ndarray, shape (n_sources, 4, 4)
        Per-source diagonal covariance in the same ordering.
    """
    n_sources = m1_true.shape[0]
    means = np.stack([m1_true, m2_true, lambda_tilde_true, delta_lambda_true], axis=-1)
    covs = np.zeros((n_sources, 4, 4))
    covs[:, 0, 0] = err_m1**2
    covs[:, 1, 1] = err_m2**2
    covs[:, 2, 2] = err_lambda_tilde**2
    covs[:, 3, 3] = err_delta_lambda**2
    return means, covs


# Physical clipping bounds for sampled component masses. Needed because
# err_m1_src/err_m2_src (the whole source-frame mass uncertainty, which -- see
# GWFisherLikelihoodV2's module docstring and the investigation this class grew out
# of -- inherits nearly all of its size from the redshift/luminosity-distance
# degeneracy, not from the intrinsic (very well measured) detector-frame chirp mass)
# is occasionally enormous relative to the mass itself: some sources in the real
# SFHo/2Lmis SNR>=30 catalog have sigma_m1 more than 100x m1_true. Sampling a plain,
# unbounded Gaussian for such a source routinely produces "masses" of hundreds of
# solar masses or negative -- physically meaningless, but not automatically excluded
# by anything downstream (jnp.interp just clamps to the EOS table's edge value, so
# these don't crash or NaN, they silently inject noise that a gradient-based fit can
# and does exploit). Clipping to a generous but physical NS mass window turns this
# from "silently wrong" into "safely uninformative" for those samples, without
# needing per-source data-quality cuts.
_MIN_SAMPLED_MASS: float = 0.5  # Msun -- well below any realistic NS mass
_MAX_SAMPLED_MASS: float = 3.5  # Msun -- well above any realistic NS mass (not BH)


def _sample_masses(
    means: np.ndarray,
    covs: np.ndarray,
    n_mass_samples: int,
    seed: int,
    min_mass: float = _MIN_SAMPLED_MASS,
    max_mass: float = _MAX_SAMPLED_MASS,
) -> np.ndarray:
    r"""Pre-sample :math:`(m_1, m_2)` once per source from the 4D Gaussian's own
    :math:`(m_1, m_2)` marginal (the leading 2x2 block of each source's covariance),
    clip to a physical mass window, then enforce :math:`m_1 \geq m_2` by swapping any
    sample that violates it (order matters: clip first, since clipping can itself
    change which of a pair is larger for samples that were wildly out of range).

    Draws from the full 2x2 mass block (via a batched Cholesky factorization), not two
    independent 1D Gaussians, so this is already correct if a future data source adds
    a genuine ``Cov(m1, m2)`` term -- consistent with :func:`_build_source_gaussians_4d`
    building a general (currently diagonal) covariance rather than four separate 1D
    distributions.

    Parameters
    ----------
    means : np.ndarray, shape (n_sources, 4)
    covs : np.ndarray, shape (n_sources, 4, 4)
    n_mass_samples : int
    seed : int
    min_mass, max_mass : float, optional
        Physical clipping window for sampled component masses (default: 0.5-3.5
        Msun -- see the module-level constants' docstring above for why this is
        needed). Sources with extremely large ``err_m1_src``/``err_m2_src`` will have
        many samples piled up at these bounds rather than spread over an
        astrophysically meaningless range; this makes those samples uninformative
        (roughly constant contribution regardless of the candidate EOS) rather than
        actively distorting the fit.

    Returns
    -------
    np.ndarray, shape (n_sources, n_mass_samples, 2)
        Pre-sampled ``[m1, m2]`` pairs per source, ``min_mass <= m2 <= m1 <= max_mass``.
    """
    rng = np.random.default_rng(seed)
    n_sources = means.shape[0]
    mass_cov = covs[:, :2, :2]
    L = np.linalg.cholesky(mass_cov)  # shape (n_sources, 2, 2), batched
    z = rng.standard_normal((n_sources, n_mass_samples, 2))
    mass_samples = means[:, None, :2] + np.einsum("sij,skj->ski", L, z)
    mass_samples = np.clip(mass_samples, min_mass, max_mass)

    m1_samples, m2_samples = mass_samples[..., 0], mass_samples[..., 1]
    swap = m2_samples > m1_samples
    m1_fixed = np.where(swap, m2_samples, m1_samples)
    m2_fixed = np.where(swap, m1_samples, m2_samples)
    return np.stack([m1_fixed, m2_fixed], axis=-1)


class GWFisherLikelihoodV2(LikelihoodBase):
    r"""EOS likelihood from gwfast Fisher-forecast BNS sources via direct
    :math:`(m_1, m_2, \tilde{\Lambda}, \delta\tilde{\Lambda})` comparison.

    See the module docstring for the full motivation and design (in particular, why
    this avoids inverting to :math:`(\Lambda_1, \Lambda_2)`, and why the covariance is
    built as a general 4x4 matrix despite being diagonal today).

    Unlike v1's fixed mass-ratio quadrature grid, this class pre-samples
    :math:`(m_1, m_2)` pairs per source from their own Fisher-measured Gaussian
    (mirroring :class:`~jesterTOV.inference.likelihoods.gw.GWLikelihood`'s treatment of
    real events: pre-sample once at construction from the observational model,
    Monte-Carlo-average the log-likelihood over those fixed samples at evaluate time)
    rather than deterministic trapezoidal quadrature over a shared grid -- so, unlike
    v1, this class's likelihood value has residual Monte Carlo noise as a function of
    ``n_mass_samples`` (increase it to reduce that noise, at the cost of proportionally
    more likelihood evaluations per source).

    Parameters
    ----------
    gwfast_result_file : str
        Path to the gwfast Fisher-forecast result HDF5 file. Must include
        ``err_deltaLambda`` (``2Lmis``/``2Lpar`` configs only, not ``Delta``).
    injection_catalog_file : str
        Path to the matching injection catalog HDF5 file.
    snr_threshold : float, optional
        Additional SNR cut on top of whatever detection threshold is already baked
        into ``gwfast_result_file`` (default: ``0.0``, i.e. no extra cut).
    quality_cut_n_sigma : float, optional
        Data-quality cut (default: ``0.0``, **disabled**, matching v1): exclude
        sources where ``m1_src``, ``m2_src``, or ``LambdaTilde`` isn't
        significantly positive relative to its own Fisher error -- see
        :func:`~jesterTOV.inference.likelihoods.gw_fisher._positivity_quality_mask`.
        Not applied to ``deltaLambda``, which is physically allowed to be negative.
        **Disabled by default**: even a strict 3-sigma cut removes ~99% of detected
        SFHo sources at SNR>=30 in the real catalog this was tested against -- this
        is the norm for this data, not a rare outlier, so enabling it is a
        deliberate choice that trades away most of the sample. Check ``n_sources``
        after construction if you do enable it.
    penalty_value : float, optional
        Log-likelihood penalty applied when a sampled component mass exceeds
        :math:`M_{\rm TOV}` of the candidate EOS (default: ``0.0``, i.e. no penalty).
    n_mass_samples : int, optional
        Number of pre-sampled ``(m1, m2)`` pairs per source (default: ``500``,
        matching :class:`~jesterTOV.inference.likelihoods.gw.GWLikelihood`'s default).
        Larger values reduce Monte Carlo estimator noise at the cost of proportionally
        more likelihood evaluations.
    source_batch_size : int, optional
        Batch size for ``jax.lax.map`` over sources (default: ``1``, a plain scan --
        keeps memory flat under the outer particle ``vmap`` used by e.g. the SMC
        sampler), mirroring v1's ``source_batch_size``.
    mass_batch_size : int, optional
        Batch size for ``jax.lax.map`` over the pre-sampled mass pairs (default: ``1``),
        mirroring v1's ``q_batch_size`` (renamed since there is no more mass-ratio
        grid) and :class:`~jesterTOV.inference.likelihoods.gw.GWLikelihood`'s
        ``N_masses_batch_size``.
    seed : int, optional
        Random seed for mass pre-sampling (default: ``42``). Fixed seed ensures
        reproducibility and deterministic likelihood evaluation across sampler steps.

    Attributes
    ----------
    n_sources : int
        Number of sources retained after the SNR cut.
    """

    gwfast_result_file: str
    injection_catalog_file: str
    snr_threshold: float
    penalty_value: float
    n_mass_samples: int
    source_batch_size: int
    mass_batch_size: int
    seed: int
    n_sources: int

    def __init__(
        self,
        gwfast_result_file: str,
        injection_catalog_file: str,
        snr_threshold: float = 0.0,
        penalty_value: float = 0.0,
        n_mass_samples: int = 500,
        source_batch_size: int = 1,
        mass_batch_size: int = 1,
        seed: int = 42,
        quality_cut_n_sigma: float = 0.0,
    ) -> None:
        super().__init__()

        self.gwfast_result_file = gwfast_result_file
        self.injection_catalog_file = injection_catalog_file
        self.snr_threshold = snr_threshold
        self.penalty_value = penalty_value
        self.n_mass_samples = n_mass_samples
        self.source_batch_size = source_batch_size
        self.mass_batch_size = mass_batch_size
        self.seed = seed
        self.quality_cut_n_sigma = quality_cut_n_sigma

        logger.info(
            f"[v2] Loading gwfast Fisher-forecast data from {gwfast_result_file} "
            f"and {injection_catalog_file}"
        )
        result = _load_gwfast_result_v2(gwfast_result_file)
        injections = _load_injection_catalog(injection_catalog_file)
        _validate_component_mass_ordering(
            injections["m1_src"], injections["m2_src"], injection_catalog_file
        )

        idx = result["idx_det_in_cat"].astype(int)
        n_detected = idx.shape[0]
        m1_true = injections["m1_src"][idx]
        m2_true = injections["m2_src"][idx]
        lambda_tilde_true = injections["LambdaTilde"][idx]
        eta_true = injections["eta"][idx]
        lambda1_true = injections["Lambda1"][idx]
        lambda2_true = injections["Lambda2"][idx]
        delta_lambda_true = np.asarray(
            utils.delta_lambda_tilde_from_lambda1_lambda2(
                jnp.asarray(lambda1_true), jnp.asarray(lambda2_true), jnp.asarray(eta_true)
            )
        )
        snrs = result["snrs"][idx]

        mask_snr = snrs >= snr_threshold
        if not np.any(mask_snr):
            raise ValueError(
                f"0 of {n_detected} detected sources in {gwfast_result_file} have "
                f"SNR >= {snr_threshold}; max SNR in file is {float(np.max(snrs))}."
            )
        n_excluded_snr = n_detected - int(np.sum(mask_snr))
        if n_excluded_snr > 0:
            logger.info(
                f"[v2] SNR cut: excluding {n_excluded_snr}/{n_detected} sources with "
                f"SNR < {snr_threshold}"
            )

        # Data-quality cut: m1_src, m2_src, and LambdaTilde are physically constrained
        # to be positive -- exclude sources whose Fisher error is so large the value
        # isn't significantly positive (see _positivity_quality_mask's docstring in
        # gw_fisher.py). Deliberately NOT applied to deltaLambda, which is physically
        # allowed to be negative (antisymmetric tidal term).
        mask_quality = (
            _positivity_quality_mask(m1_true, result["err_m1_src"], quality_cut_n_sigma, "m1_src")
            & _positivity_quality_mask(m2_true, result["err_m2_src"], quality_cut_n_sigma, "m2_src")
            & _positivity_quality_mask(
                lambda_tilde_true, result["err_LambdaTilde"], quality_cut_n_sigma, "LambdaTilde"
            )
        )
        mask = mask_snr & mask_quality
        if not np.any(mask):
            raise ValueError(
                f"0 of {n_detected} detected sources in {gwfast_result_file} survive "
                f"the combined SNR (>= {snr_threshold}) and {quality_cut_n_sigma}-sigma "
                "positivity data-quality cuts."
            )

        means, covs = _build_source_gaussians_4d(
            m1_true[mask],
            m2_true[mask],
            result["err_m1_src"][mask],
            result["err_m2_src"][mask],
            lambda_tilde_true[mask],
            result["err_LambdaTilde"][mask],
            delta_lambda_true[mask],
            result["err_deltaLambda"][mask],
        )

        self.n_sources = int(np.sum(mask))
        self._means: Float[Array, "n_sources 4"] = jnp.asarray(means)
        self._covs: Float[Array, "n_sources 4 4"] = jnp.asarray(covs)

        logger.info(
            f"[v2] Pre-sampling {n_mass_samples} (m1,m2) pairs per source, seed={seed}"
        )
        mass_samples = _sample_masses(means, covs, n_mass_samples, seed)
        self._mass_samples: Float[Array, "n_sources n_mass_samples 2"] = jnp.asarray(
            mass_samples
        )

        logger.info(
            f"GWFisherLikelihoodV2: {n_detected} detected sources, "
            f"{self.n_sources} retained after snr_threshold={snr_threshold} and "
            f"{quality_cut_n_sigma}-sigma positivity data-quality cuts "
            f"({n_detected - self.n_sources} excluded total), "
            f"n_mass_samples={n_mass_samples}, "
            f"source_batch_size={source_batch_size}, mass_batch_size={mass_batch_size}"
        )

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate summed log likelihood over all retained sources for given EOS parameters.

        For each source, Monte-Carlo-averages the source's 4D Gaussian density,
        evaluated at ``(m1, m2, LambdaTilde_X(m1,m2), deltaLambdaTilde_X(m1,m2))`` for
        each pre-sampled ``(m1, m2)`` pair, then sums the resulting per-source
        log-likelihoods.

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain:
            - 'masses_EOS': Array of neutron star masses from EOS
            - 'Lambdas_EOS': Array of tidal deformabilities from EOS

        Returns
        -------
        Float
            Summed log likelihood over all retained sources.
        """
        masses_EOS: Float[Array, " n_points"] = params["masses_EOS"]
        Lambdas_EOS: Float[Array, " n_points"] = params["Lambdas_EOS"]
        mtov: Float = jnp.max(masses_EOS)
        n_mass_samples = self.n_mass_samples

        def process_one_source(carry) -> Float:
            mean, cov, mass_samples = carry

            def process_one_sample(mass_pair: Float[Array, " 2"]) -> Float:
                m1, m2 = mass_pair[0], mass_pair[1]
                lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right=1.0)
                lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right=1.0)
                eta = utils.symmetric_mass_ratio_from_mass_ratio(m2 / m1)
                lambda_tilde = utils.lambda_tilde_from_lambda1_lambda2(
                    lambda_1, lambda_2, eta
                )
                delta_lambda = utils.delta_lambda_tilde_from_lambda1_lambda2(
                    lambda_1, lambda_2, eta
                )
                point = jnp.array([m1, m2, lambda_tilde, delta_lambda])
                logpdf = multivariate_normal.logpdf(point, mean, cov)
                penalty_m1 = jnp.where(m1 > mtov, self.penalty_value, 0.0)
                penalty_m2 = jnp.where(m2 > mtov, self.penalty_value, 0.0)
                return logpdf + penalty_m1 + penalty_m2

            all_logprobs = jax.lax.map(
                process_one_sample, mass_samples, batch_size=self.mass_batch_size
            )
            return logsumexp(all_logprobs) - jnp.log(n_mass_samples)

        per_source_loglike = jax.lax.map(
            process_one_source,
            (self._means, self._covs, self._mass_samples),
            batch_size=self.source_batch_size,
        )
        return jnp.sum(per_source_loglike)
