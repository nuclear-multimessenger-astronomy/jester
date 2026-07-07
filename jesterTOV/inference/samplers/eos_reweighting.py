r"""EOS reweighting sampler for jesterTOV.

Evaluates jester's GPU-accelerated likelihoods on a discrete set of
tabulated EOS curves (M, :math:`\Lambda`, R tables) rather than sampling a
parametric EOS model.  Returns the marginal log-likelihood per EOS and the
Bayesian evidence :math:`\log Z`.

Only ``masses`` is strictly required in the EOS files: ``lambdas`` and
``radii`` are each optional, but at least one must be present (a set with
only ``lambdas`` supports GW-only reweighting, a set with only ``radii``
supports NICER-only reweighting). Which curve(s) are actually required is
determined by the enabled likelihoods and validated at config-parsing time
in :class:`~jesterTOV.inference.config.schemas.eos_reweighting.EOSReweightingInferenceConfig`.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import scipy.special
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .jester_sampler import JesterSampler, SamplerOutput
from ..base import LikelihoodBase
from ..config.schemas.samplers import EOSReweightingConfig
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


def resample_eos_posterior(
    masses: Float[Array, "N L"] | Float[np.ndarray, "N L"],
    lambdas: Float[Array, "N L"] | Float[np.ndarray, "N L"] | None,
    radii: Float[Array, "N L"] | Float[np.ndarray, "N L"] | None,
    posterior_weights: Float[Array, " N"] | Float[np.ndarray, " N"],
    n_samples: int | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    r"""Resample a tabulated EOS set into equal-weight posterior samples.

    This is a small, standalone utility that turns the *discrete* weighted
    posterior produced by :meth:`EOSReweightingSampler.sample` (one weight
    per input EOS curve) into an equal-weight set of posterior draws, by
    drawing curves with replacement according to ``posterior_weights``. It
    only needs the gridded curves and weights, so it works equally well on
    a fresh set of EOS curves that were reweighted outside of
    :class:`EOSReweightingSampler` (e.g. with a custom likelihood), as long
    as ``masses``/``lambdas``/``radii`` share a common mass grid — see
    :meth:`EOSReweightingSampler.load_and_grid` for a helper that builds
    such a grid from a list of NPZ files.

    Parameters
    ----------
    masses :
        Common-grid EOS curve of shape ``[N, L]``, e.g. as returned by
        :meth:`EOSReweightingSampler.load_and_grid`.
    lambdas, radii :
        Common-grid EOS curves of shape ``[N, L]``, or ``None`` if that
        curve was not available in the EOS set (e.g. a GW-only set has no
        ``radii``). The corresponding ``*_EOS`` key is omitted from the
        returned dict when ``None``.
    posterior_weights :
        Normalised posterior weight per EOS (sums to 1), e.g.
        ``sample_output.metadata["evidence"]["posterior_weights"]``.
    n_samples :
        Number of samples to draw. ``None`` (default) uses the Kish
        effective sample size :math:`N_\mathrm{eff} = 1/\sum_a (w^{(a)})^2`,
        rounded to the nearest integer.
    seed :
        Seed for the NumPy random generator used to draw the resample.

    Returns
    -------
    dict
        - ``eos_index`` : index into the input arrays for each draw, shape ``[n_samples]``
        - ``masses_EOS`` : resampled curve, shape ``[n_samples, L]``
        - ``Lambdas_EOS``, ``radii_EOS`` : resampled curves, shape ``[n_samples, L]``,
          present only if the corresponding input was not ``None``
    """
    weights = np.asarray(posterior_weights, dtype=np.float64)
    weights = weights / weights.sum()

    if n_samples is None:
        n_eff = 1.0 / np.sum(weights**2)
        n_samples = max(1, int(round(n_eff)))

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(weights), size=n_samples, replace=True, p=weights)

    result: dict[str, np.ndarray] = {
        "eos_index": idx,
        "masses_EOS": np.asarray(masses)[idx],
    }
    if lambdas is not None:
        result["Lambdas_EOS"] = np.asarray(lambdas)[idx]
    if radii is not None:
        result["radii_EOS"] = np.asarray(radii)[idx]
    return result


class EOSReweightingSampler(JesterSampler):
    r"""Sampler that reweights a discrete EOS set by jester likelihoods.

    Unlike parametric samplers implemented in ``jester``, this class does not sample a prior.
    Instead, it receives a fixed set of tabulated EOS curves and evaluates the
    combined likelihood on each one.  The result is a discrete posterior
    over the EOS set together with the Bayesian evidence :math:`\log Z`.

    Parameters
    ----------
    likelihood : LikelihoodBase
        Combined likelihood (GW, NICER, radio, ...) created by
        :func:`~jesterTOV.inference.likelihoods.factory.create_combined_likelihood`.
    prior : object, optional
        Accepted for API compatibility with :class:`JesterSampler`; however, ignored for the reweighting.
    sample_transforms : list, optional
        Accepted for API compatibility; however, ignored for the reweighting.
    likelihood_transforms : list, optional
        Accepted for API compatibility; however, ignored for the reweighting.
    config : EOSReweightingConfig
        Sampler configuration including EOS file paths and grid settings.
    seed : int, optional
        Accepted for API compatibility; however, ignored for the reweighting.
    """

    def __init__(
        self,
        likelihood: LikelihoodBase,
        prior: Any = None,
        sample_transforms: list | None = None,
        likelihood_transforms: list | None = None,
        config: EOSReweightingConfig | None = None,
        seed: int = 0,
    ) -> None:
        if config is None:
            raise ValueError("EOSReweightingSampler requires a config argument")
        # NOTE: Do NOT call super().__init__() — JesterSampler assumes a parametric prior.
        # prior, sample_transforms, likelihood_transforms, seed are unused (API compat).
        _ = prior, sample_transforms, likelihood_transforms, seed
        self.likelihood = likelihood  # type: ignore[assignment]
        self.config = config

    #: Absolute cap (:math:`M_\odot`) applied to the common mass grid upper
    #: bound in :meth:`load_and_grid` when ``m_max`` is not given explicitly.
    DEFAULT_M_MAX_CAP: float = 3.0

    @staticmethod
    def _regrid(
        mass_grid: np.ndarray,
        masses_list: list[np.ndarray],
        lambdas_list: list[np.ndarray] | None,
        radii_list: list[np.ndarray] | None,
        m_tov_list: list[float],
    ) -> tuple[
        Float[Array, "N L"], Float[Array, "N L"] | None, Float[Array, "N L"] | None
    ]:
        r"""Interpolate a list of ragged (M, Lambda, R) curves onto a shared mass grid.

        Values above each curve's own :math:`M_\mathrm{TOV}` are set to zero.
        ``lambdas_list``/``radii_list`` may be ``None`` if that curve is not
        available in the EOS set, in which case the corresponding output is
        also ``None``.
        """
        lam_interp_list: list[np.ndarray] | None = (
            [] if lambdas_list is not None else None
        )
        rad_interp_list: list[np.ndarray] | None = (
            [] if radii_list is not None else None
        )
        mass_interp_list: list[np.ndarray] = []

        for i, (m_i, m_tov_i) in enumerate(zip(masses_list, m_tov_list)):
            # Clamp the per-curve mass grid at this curve's own M_TOV above
            # that point, instead of leaving the shared grid's (identical
            # across curves) values in place. All likelihoods derive M_TOV
            # via `jnp.max(masses_EOS)`; if every curve carried the same
            # broadcast grid, jnp.max would return the same (grid-capped)
            # value for every EOS regardless of its true M_TOV, making the
            # likelihoods blind to M_TOV differences between EOS curves.
            above = mass_grid > m_tov_i
            mass_g = mass_grid.copy()
            mass_g[above] = m_tov_i
            mass_interp_list.append(mass_g)

            if lambdas_list is not None and lam_interp_list is not None:
                lam_g = np.interp(mass_grid, m_i, lambdas_list[i], left=0.0, right=0.0)
                # Zero out above M_TOV (interp already handles right=0.0, but be explicit)
                lam_g[above] = 0.0
                lam_interp_list.append(lam_g)

            if radii_list is not None and rad_interp_list is not None:
                rad_g = np.interp(mass_grid, m_i, radii_list[i], left=0.0, right=0.0)
                rad_g[above] = 0.0
                rad_interp_list.append(rad_g)

        all_masses = jnp.array(np.stack(mass_interp_list))
        all_lambdas = (
            jnp.array(np.stack(lam_interp_list))
            if lam_interp_list is not None
            else None
        )
        all_radii = (
            jnp.array(np.stack(rad_interp_list))
            if rad_interp_list is not None
            else None
        )
        return all_masses, all_lambdas, all_radii

    def load_and_grid(
        self,
        paths: list[str],
        n_grid: int,
        m_min: float,
        m_max: float | None,
    ) -> tuple[
        tuple[
            Float[Array, "N L"], Float[Array, "N L"] | None, Float[Array, "N L"] | None
        ],
        tuple[
            Float[Array, "N L"], Float[Array, "N L"] | None, Float[Array, "N L"] | None
        ],
    ]:
        r"""Load EOS files and resample all curves onto two common mass grids.

        Two grids are built from the same underlying curves: a *likelihood*
        grid (bounded by ``m_min``/``m_max``, as configured) used to evaluate
        the likelihoods, and a *full* grid that always spans the EOS set's
        own natural range — from :math:`\min(M)` across all curves up to
        :math:`\max(M_\mathrm{TOV})` — regardless of ``m_min``/``m_max``.
        Those two bounds only restrict what jester's likelihoods see; the
        full-range curves are what gets resampled into the posterior
        (:meth:`sample`) for downstream plotting/output, so posterior
        M-R/M-Lambda curves and the saved result are never truncated by the
        likelihood grid's bounds.

        Parameters
        ----------
        paths :
            Paths to NPZ files.  Each file must contain a ``masses`` key,
            plus at least one of ``lambdas`` or ``radii`` (1-D or 2-D
            arrays); the absent one is skipped, so only likelihoods that
            don't need it may be used with such an EOS set (see
            :class:`~jesterTOV.inference.config.schemas.eos_reweighting.EOSReweightingInferenceConfig`
            for the enabled-likelihood cross-check). All files passed here
            must provide the same set of keys. For a file with N curves the
            arrays must be shaped ``[N, n_points]``, with matching shapes
            across keys — ragged/heterogeneous curves within a single file
            are not supported and raise :class:`ValueError`. Curves may
            differ in length *between* files, since each curve is resampled
            onto the common mass grid independently.
        n_grid :
            Number of mass grid points (used for both grids).
        m_min :
            Lower bound of the *likelihood* grid in :math:`M_\odot`.
        m_max :
            Upper bound of the *likelihood* grid in :math:`M_\odot`.
            If ``None``, then use :math:`\max(M_\mathrm{TOV})` across all curves in
            ``paths``, capped at :attr:`DEFAULT_M_MAX_CAP` :math:`M_\odot`
            (a warning is logged if the cap is applied, since likelihoods
            will not be evaluated above it).

        Returns
        -------
        likelihood_grid : tuple of Float[Array, "N L"] | None
            ``(masses, lambdas, radii)`` on the ``[m_min, m_max]``-bounded
            grid used for likelihood evaluation. ``lambdas``/``radii`` are
            ``None`` if not present in the EOS set.
        full_grid : tuple of Float[Array, "N L"] | None
            ``(masses, lambdas, radii)`` on the EOS set's natural
            ``[min(M), max(M_TOV)]`` grid, used for posterior resampling —
            unaffected by ``m_min``/``m_max``. ``lambdas``/``radii`` are
            ``None`` if not present in the EOS set.
        """
        masses_list: list[np.ndarray] = []
        lambdas_list: list[np.ndarray] = []
        radii_list: list[np.ndarray] = []
        m_tov_list: list[float] = []
        n_total = 0
        n_discarded = 0
        has_lambdas: bool | None = None
        has_radii: bool | None = None

        for path in paths:
            data = np.load(path)
            if "masses" not in data:
                raise ValueError(f"EOS file '{path}' is missing the 'masses' key.")
            m = data["masses"].astype(np.float64)

            file_has_lambdas = "lambdas" in data
            file_has_radii = "radii" in data
            if has_lambdas is None:
                has_lambdas, has_radii = file_has_lambdas, file_has_radii
                if not has_lambdas and not has_radii:
                    raise ValueError(
                        f"EOS file '{path}' must contain at least one of 'lambdas' "
                        "or 'radii' (in addition to 'masses')."
                    )
            elif file_has_lambdas != has_lambdas or file_has_radii != has_radii:
                raise ValueError(
                    f"EOS file '{path}' provides 'lambdas'={file_has_lambdas}, "
                    f"'radii'={file_has_radii}, which differs from the preceding "
                    f"files ('lambdas'={has_lambdas}, 'radii'={has_radii}). All EOS "
                    "files passed to load_and_grid must provide the same curves."
                )

            lam = data["lambdas"].astype(np.float64) if has_lambdas else None
            rad = data["radii"].astype(np.float64) if has_radii else None

            # Normalise to shape [N_file, n_points]
            if m.ndim == 1:
                m = m[None, :]
                if lam is not None:
                    lam = lam[None, :]
                if rad is not None:
                    rad = rad[None, :]

            if (lam is not None and m.shape != lam.shape) or (
                rad is not None and m.shape != rad.shape
            ):
                raise ValueError(
                    f"EOS file '{path}' has mismatched shapes: "
                    f"masses{m.shape}, "
                    f"lambdas{lam.shape if lam is not None else None}, "
                    f"radii{rad.shape if rad is not None else None}. "
                    "Ragged/heterogeneous curves are not supported within a single "
                    "file — 'masses', 'lambdas', and 'radii' must all be [N, n_points] "
                    "(or [n_points] for a single curve)."
                )

            for i in range(m.shape[0]):
                n_total += 1
                m_i = m[i]
                lam_i = lam[i] if lam is not None else None
                rad_i = rad[i] if rad is not None else None
                is_bad = np.any(np.isnan(m_i)) or np.any(m_i < 0)
                if lam_i is not None:
                    is_bad = is_bad or np.any(np.isnan(lam_i)) or np.any(lam_i < 0)
                if rad_i is not None:
                    is_bad = is_bad or np.any(np.isnan(rad_i)) or np.any(rad_i < 0)
                if is_bad:
                    n_discarded += 1
                    continue

                # `_regrid` uses np.interp, which requires the mass array to
                # be monotonically increasing; sort defensively rather than
                # trusting the NPZ file's row order.
                order = np.argsort(m_i)
                m_i = m_i[order]
                if lam_i is not None:
                    lam_i = lam_i[order]
                if rad_i is not None:
                    rad_i = rad_i[order]

                masses_list.append(m_i)
                if lam_i is not None:
                    lambdas_list.append(lam_i)
                if rad_i is not None:
                    radii_list.append(rad_i)

                # Prefer radii (matches historical behaviour); fall back to
                # lambdas when radii is not part of this EOS set. One of the
                # two is guaranteed non-None by the has_lambdas/has_radii
                # check above.
                nonzero_source = rad_i if rad_i is not None else lam_i
                assert nonzero_source is not None
                nonzero = np.nonzero(nonzero_source > 0)[0]
                m_tov = (
                    float(m_i[nonzero[-1]]) if len(nonzero) > 0 else float(np.max(m_i))
                )
                m_tov_list.append(m_tov)

        if n_discarded > 0:
            logger.warning(
                f"Discarded {n_discarded}/{n_total} "
                f"({100.0 * n_discarded / n_total:.1f}%) non-physical EOS curves "
                "(NaN or negative mass/radius/lambda values)."
            )
        if len(masses_list) == 0:
            raise ValueError(
                "All EOS curves were discarded as non-physical "
                "(NaN or negative mass/radius/lambda values)."
            )

        lambdas_list_or_none = lambdas_list if has_lambdas else None
        radii_list_or_none = radii_list if has_radii else None

        m_tov_arr = np.asarray(m_tov_list)
        max_m_tov = float(np.max(m_tov_arr))
        min_m_full = float(np.min([m_i[0] for m_i in masses_list]))

        # Likelihood grid upper bound (may be capped)
        if m_max is None:
            m_cap = self.DEFAULT_M_MAX_CAP
            if max_m_tov > m_cap:
                n_above = int(np.sum(m_tov_arr > m_cap))
                logger.warning(
                    f"Maximum M_TOV across the EOS set is {max_m_tov:.3f} M_sun, "
                    f"which exceeds {m_cap:.1f} M_sun. Capping the *likelihood* mass "
                    f"grid at {m_cap:.1f} M_sun instead. {n_above}/{len(m_tov_arr)} EOS "
                    f"curves have M_TOV above {m_cap:.1f} M_sun; likelihoods will not "
                    "be evaluated above it. The posterior curves used for plotting "
                    "are not affected by this cap."
                )
                m_max_likelihood = m_cap
            else:
                m_max_likelihood = max_m_tov
        else:
            m_max_likelihood = m_max

        likelihood_mass_grid = np.linspace(m_min, m_max_likelihood, n_grid)
        likelihood_grid = self._regrid(
            likelihood_mass_grid,
            masses_list,
            lambdas_list_or_none,
            radii_list_or_none,
            m_tov_list,
        )

        if m_min <= min_m_full and m_max_likelihood >= max_m_tov:
            full_grid = likelihood_grid
        else:
            full_mass_grid = np.linspace(min_m_full, max_m_tov, n_grid)
            full_grid = self._regrid(
                full_mass_grid,
                masses_list,
                lambdas_list_or_none,
                radii_list_or_none,
                m_tov_list,
            )

        return likelihood_grid, full_grid

    def make_eos_fn(self) -> Callable[[dict[str, Array]], Array]:
        r"""Build a single-EOS log-likelihood callable for use with :func:`jax.lax.map`.

        Returns
        -------
        Callable
            ``f(curve) → scalar log-likelihood``, where ``curve`` is a dict
            with a ``"masses"`` key and, if available for this EOS set,
            ``"lambdas"`` and/or ``"radii"`` keys.
        """
        likelihood = self.likelihood

        def f(curve: dict[str, Array]) -> Array:
            params: dict[str, Array] = {"masses_EOS": curve["masses"]}
            if "lambdas" in curve:
                params["Lambdas_EOS"] = curve[
                    "lambdas"
                ]  # capital L — matches gw.py:303–304
            if "radii" in curve:
                params["radii_EOS"] = curve["radii"]
            return likelihood.evaluate(params)

        return f

    def evaluate_batch(
        self,
        f: Callable[[dict[str, Array]], Array],
        all_masses: Float[Array, "N L"],
        all_lambdas: Float[Array, "N L"] | None,
        all_radii: Float[Array, "N L"] | None,
    ) -> Float[Array, " N"]:
        r"""Evaluate *f* on all N EOS curves using :func:`jax.lax.map`.

        Splits the work into batches of ``config.batch_size`` curves and logs
        throughput/ETA after each batch is processed. The per-batch
        ``jax.lax.map`` call is wrapped in :func:`jax.jit` so that batches
        sharing the same shape (all but typically the last one) reuse a
        single compiled executable instead of retracing on every iteration
        of the Python loop.

        Parameters
        ----------
        f :
            Single-EOS evaluator returned by :meth:`make_eos_fn`.
        all_masses :
            Stacked JAX array of shape ``[N, L]``.
        all_lambdas, all_radii :
            Stacked JAX arrays of shape ``[N, L]``, or ``None`` if that
            curve is not present in the EOS set (omitted from the dict
            passed to ``f`` in that case).

        Returns
        -------
        Float[Array, " N"]
            Log-likelihoods per EOS.
        """
        all_batches_time_start = time.monotonic()

        N = all_masses.shape[0]  # number of EOSs to process
        batch_size = self.config.batch_size

        # NOTE: it is a bit awkward that it seems batching is done twice
        # (Python loop + lax.map's own batch_size). However, using jax.vmap
        # here turned out to be a bit slower, so we keep this implementation.
        # `bs` is static since jax.lax.map requires a concrete Python int.
        jitted_map = jax.jit(
            lambda stacked, bs: jax.lax.map(f, stacked, batch_size=bs),
            static_argnums=1,
        )

        # Initialize everything for storing the results of the inference
        results: list[Array] = []
        start_time = time.monotonic()
        processed = 0

        # Loop over the batches
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            stacked: dict[str, Array] = {"masses": all_masses[start:end]}
            if all_lambdas is not None:
                stacked["lambdas"] = all_lambdas[start:end]
            if all_radii is not None:
                stacked["radii"] = all_radii[start:end]
            current_bs = min(batch_size, end - start)

            batch_result: Float[Array, " _"] = jitted_map(stacked, current_bs)

            results.append(batch_result)
            processed = end

            elapsed = time.monotonic() - start_time
            fraction = processed / N
            eta = elapsed / fraction * (1.0 - fraction) if fraction > 0 else 0.0
            logger.info(
                f"EOS reweighting: {processed}/{N} EOS "
                f"({fraction * 100:.0f}%) | "
                f"elapsed {elapsed:.1f}s | ETA {eta:.1f}s"
            )

        log_likelihoods: Float[Array, " N"] = jnp.concatenate(results)

        all_batches_time_end = time.monotonic()
        logger.info(
            f"EOS reweighting: all EOS processed in {all_batches_time_end - all_batches_time_start:.1f}s"
        )

        return log_likelihoods

    def compute_evidence(self, log_likelihoods: Float[Array, " N"]) -> dict[str, Any]:
        r"""Compute Bayesian evidence and effective sample size from log-likelihoods.

        Same computation as ``lwp.utils.utils.estimate_evidence`` (with its
        default uniform ``prior``), just carried out in log space for
        numerical stability. Writing :math:`w^{(a)} = \exp(\text{log\_likelihoods}_a)`
        for the raw per-EOS likelihood and :math:`p^{(a)} = w^{(a)} / \sum_b w^{(b)}`
        for the normalised posterior weight, ``lwp`` computes

        .. math::
            Z = \frac{1}{N}\sum_{a=1}^{N} w^{(a)} \,, \qquad
            Z_2 = \frac{1}{N}\sum_{a=1}^{N} \left(w^{(a)}\right)^2 \,, \qquad
            \delta Z = \sqrt{\frac{Z_2 - Z^2}{N}}\,,

        where :math:`\delta Z` is the standard error of the mean over the
        ``N`` weights, and reports the relative uncertainty :math:`\delta Z / Z`
        after propagating it to log space via the delta method
        (:math:`\sigma_{\log Z} \approx \delta Z / Z`). Using
        :math:`N_\mathrm{eff} = Z^2/Z_2\cdot N = 1/\sum_a (p^{(a)})^2` (the Kish
        effective sample size, ``lwp.stats.stats.nkde``) this simplifies to

        .. math::
            \log Z = \mathrm{logsumexp}(\text{log\_likelihoods}) - \log N \,, \qquad
            \sigma_{\log Z} = \sqrt{\frac{1}{N_\mathrm{eff}} - \frac{1}{N}}\,.

        Parameters
        ----------
        log_likelihoods :
            Per-EOS log-likelihoods of shape ``[N]``.

        Returns
        -------
        dict with keys ``log_Z``, ``log_Z_std``, ``N_eff``,
        ``N_eff_fraction``, ``posterior_weights``.
        """
        log_likelihoods_np = np.asarray(log_likelihoods)
        N = int(log_likelihoods_np.shape[0])
        lse = scipy.special.logsumexp(log_likelihoods_np)
        log_Z = float(lse - np.log(N))

        posterior_weights = np.exp(log_likelihoods_np - lse)

        # Kish effective sample size (lwp.stats.stats.nkde): 1 / sum(p_i^2).
        N_eff = float(1.0 / np.sum(posterior_weights**2))
        log_Z_std = float(np.sqrt(max(0.0, 1.0 / N_eff - 1.0 / N)))

        return {
            "log_Z": log_Z,
            "log_Z_std": log_Z_std,
            "N_eff": N_eff,
            "N_eff_fraction": N_eff / N,
            "posterior_weights": posterior_weights,
        }

    def sample(self, key: PRNGKeyArray) -> SamplerOutput:  # type: ignore[override]
        r"""Evaluate all EOS curves and return evidence + posterior samples.

        After computing the per-EOS evidence and posterior weights, this
        also draws :attr:`~jesterTOV.inference.config.schemas.samplers.EOSReweightingConfig.n_resample`
        (default: :math:`N_\mathrm{eff}`) equal-weight posterior samples via
        :func:`resample_eos_posterior`, so that downstream postprocessing
        (e.g. :func:`~jesterTOV.inference.postprocessing.postprocessing.generate_eos_reweighting_plots`)
        can plot mass-radius/mass-Lambda curves directly without having to
        deal with the unequal per-EOS weights itself. The resampled curves
        are drawn from the *full*, uncapped mass grid (see
        :meth:`load_and_grid`), so plots always show each EOS out to its true
        :math:`M_\mathrm{TOV}` even if the likelihood grid was capped.

        Parameters
        ----------
        key :
            JAX random key. Only its bits are used (to seed the NumPy RNG
            for resampling); the likelihood evaluation itself is
            deterministic.

        Returns
        -------
        SamplerOutput
            - ``samples["eos_index"]`` : integer index per input EOS (length N)
            - ``samples["log_likelihood"]`` : log-likelihood per input EOS (length N)
            - ``samples["posterior_weight"]`` : normalised posterior weight per input EOS (length N)
            - ``samples["masses_EOS"]`` : resampled equal-weight posterior curve (length N_resampled)
            - ``samples["Lambdas_EOS"]``, ``samples["radii_EOS"]`` : resampled equal-weight
              posterior curves (length N_resampled), present only if that curve is
              available in the EOS set
            - ``samples["resampled_eos_index"]`` : index into the input EOS set for each resampled draw
            - ``samples["resampled_log_likelihood"]`` : log-likelihood of each resampled draw
            - ``log_prob`` : same as ``samples["log_likelihood"]`` (length N)
            - ``metadata["evidence"]`` : evidence dict (log_Z, log_Z_std, N_eff, …)
            - ``metadata["N_eos"]`` : total number of input EOS curves
            - ``metadata["N_resampled"]`` : number of resampled posterior draws
        """
        f = self.make_eos_fn()

        logger.info(f"Loading EOS file: {self.config.eos_file}")
        (all_masses, all_lambdas, all_radii), (
            full_masses,
            full_lambdas,
            full_radii,
        ) = self.load_and_grid(
            [self.config.eos_file],
            self.config.n_grid,
            self.config.m_min,
            self.config.m_max,
        )
        N = int(all_masses.shape[0])
        logger.info(
            f"EOS set: {N} curves evaluated on"
            f"[{self.config.m_min:.2f}, {float(all_masses[0, -1]):.3f}] M_sun; "
        )

        logger.info("Evaluating likelihoods on EOS set...")
        log_likelihoods = self.evaluate_batch(f, all_masses, all_lambdas, all_radii)
        ev = self.compute_evidence(log_likelihoods)
        logger.info(
            f"log Z = {ev['log_Z']:.3f} ± {ev['log_Z_std']:.3f}  "
            f"N_eff = {ev['N_eff']:.1f} ({ev['N_eff_fraction']*100:.1f}%)"
        )

        resample_seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
        resampled = resample_eos_posterior(
            full_masses,
            full_lambdas,
            full_radii,
            ev["posterior_weights"],
            n_samples=self.config.n_resample,
            seed=resample_seed,
        )
        n_resampled = int(resampled["eos_index"].shape[0])
        resampled_log_likelihood = np.asarray(log_likelihoods)[resampled["eos_index"]]
        logger.info(
            f"Resampled {n_resampled} equal-weight posterior EOS draws "
            f"(N_eff = {ev['N_eff']:.1f})"
        )

        samples: dict[str, Array] = {
            "eos_index": jnp.arange(N),
            "log_likelihood": log_likelihoods,
            "posterior_weight": ev["posterior_weights"],
            "masses_EOS": jnp.array(resampled["masses_EOS"]),
            "resampled_eos_index": jnp.array(resampled["eos_index"]),
            "resampled_log_likelihood": jnp.array(resampled_log_likelihood),
        }
        if "Lambdas_EOS" in resampled:
            samples["Lambdas_EOS"] = jnp.array(resampled["Lambdas_EOS"])
        if "radii_EOS" in resampled:
            samples["radii_EOS"] = jnp.array(resampled["radii_EOS"])
        metadata: dict[str, Any] = {
            "evidence": ev,
            "N_eos": N,
            "N_resampled": n_resampled,
        }

        return SamplerOutput(
            samples=samples,
            log_prob=log_likelihoods,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # The following methods are not applicable for this sampler type.    #
    # They raise informative errors rather than NotImplementedError so   #
    # callers get a clear message.                                       #
    # ------------------------------------------------------------------ #

    def get_samples(self) -> dict[str, Array]:
        raise RuntimeError(
            "EOSReweightingSampler: use sample() to obtain a SamplerOutput directly"
        )

    def get_log_prob(self) -> Array:
        raise RuntimeError(
            "EOSReweightingSampler: use sample() to obtain a SamplerOutput directly"
        )

    def get_n_samples(self) -> int:
        raise RuntimeError(
            "EOSReweightingSampler: use sample() to obtain a SamplerOutput directly"
        )

    def get_sampler_output(self) -> SamplerOutput:
        raise RuntimeError(
            "EOSReweightingSampler: use sample() to obtain a SamplerOutput directly"
        )
