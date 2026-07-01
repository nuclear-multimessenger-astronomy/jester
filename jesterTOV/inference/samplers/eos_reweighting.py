r"""EOS reweighting sampler for jesterTOV.

Evaluates jester's GPU-accelerated likelihoods on a discrete set of
tabulated EOS curves (M, :math:`\Lambda`, R tables) rather than sampling a
parametric EOS model.  Returns the marginal log-likelihood per EOS and the
Bayesian evidence :math:`\log Z`.
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

    def load_and_grid(
        self,
        paths: list[str],
        n_grid: int,
        m_min: float,
        m_max: float | None,
    ) -> tuple[Float[Array, "N L"], Float[Array, "N L"], Float[Array, "N L"]]:
        r"""Load EOS files and resample all curves onto a common mass grid.

        Parameters
        ----------
        paths :
            Paths to NPZ files.  Each file must contain keys ``masses``,
            ``lambdas``, and ``radii`` (1-D or 2-D arrays).  For a file
            with N curves the arrays must be shaped ``[N, n_points]``,
            with matching shapes across the three keys — ragged/heterogeneous
            curves within a single file are not supported and raise
            :class:`ValueError`.  Curves may differ in length *between*
            files, since each curve is resampled onto the common mass
            grid independently.
        n_grid :
            Number of mass grid points.
        m_min :
            Lower bound of the common mass grid in :math:`M_\odot`.
        m_max :
            Upper bound in :math:`M_\odot`.  ``None`` → use
            :math:`\max(M_\mathrm{TOV})` across all curves in ``paths``,
            capped at 3.0 :math:`M_\odot` (a warning is logged if the cap
            is applied, since likelihoods will not be evaluated above it).

        Returns
        -------
        all_masses : Float[Array, "N L"]
            Common mass grid repeated for each EOS (all rows identical).
        all_lambdas : Float[Array, "N L"]
            Tidal deformability on the common grid; zero above :math:`M_\mathrm{TOV}`.
        all_radii : Float[Array, "N L"]
            Radius in km on the common grid; zero above :math:`M_\mathrm{TOV}`.
        """
        masses_list: list[np.ndarray] = []
        lambdas_list: list[np.ndarray] = []
        radii_list: list[np.ndarray] = []
        m_tov_list: list[float] = []

        for path in paths:
            data = np.load(path)
            m = data["masses"].astype(np.float64)
            lam = data["lambdas"].astype(np.float64)
            if "radii" not in data:
                raise ValueError(
                    f"EOS file '{path}' is missing the 'radii' key. "
                    "All EOS files must contain 'masses', 'lambdas', and 'radii' arrays."
                )
            rad = data["radii"].astype(np.float64)

            # Normalise to shape [N_file, n_points]
            if m.ndim == 1:
                m = m[None, :]
                lam = lam[None, :]
                rad = rad[None, :]

            if m.shape != lam.shape or m.shape != rad.shape:
                raise ValueError(
                    f"EOS file '{path}' has mismatched shapes: "
                    f"masses{m.shape}, lambdas{lam.shape}, radii{rad.shape}. "
                    "Ragged/heterogeneous curves are not supported within a single "
                    "file — 'masses', 'lambdas', and 'radii' must all be [N, n_points] "
                    "(or [n_points] for a single curve)."
                )

            for i in range(m.shape[0]):
                masses_list.append(m[i])
                lambdas_list.append(lam[i])
                radii_list.append(rad[i])
                m_tov_list.append(float(np.max(m[i])))

        # Common grid upper bound
        if m_max is None:
            m_tov_arr = np.asarray(m_tov_list)
            max_m_tov = float(np.max(m_tov_arr))
            if max_m_tov > 3.0:
                n_above = int(np.sum(m_tov_arr > 3.0))
                logger.warning(
                    f"Maximum M_TOV across the EOS set is {max_m_tov:.3f} M_sun, "
                    f"which exceeds 3.0 M_sun. Capping the common mass grid at "
                    f"3.0 M_sun instead. {n_above}/{len(m_tov_arr)} EOS curves have "
                    "M_TOV above 3.0 M_sun; the interpolated grid (and any "
                    "likelihoods evaluated on it) will not extend to their true "
                    "M_TOV. Please double-check that this is acceptable for your use case."
                )
                m_max_grid = 3.0
            else:
                m_max_grid = max_m_tov
        else:
            m_max_grid = m_max
        mass_grid = np.linspace(m_min, m_max_grid, n_grid)

        lam_interp_list: list[np.ndarray] = []
        rad_interp_list: list[np.ndarray] = []

        for i, (m_i, lam_i, rad_i, m_tov_i) in enumerate(
            zip(masses_list, lambdas_list, radii_list, m_tov_list)
        ):
            lam_g = np.interp(mass_grid, m_i, lam_i, left=0.0, right=0.0)
            rad_g = np.interp(mass_grid, m_i, rad_i, left=0.0, right=0.0)
            # Zero out above M_TOV (interp already handles right=0.0, but be explicit)
            above = mass_grid > m_tov_i
            lam_g[above] = 0.0
            rad_g[above] = 0.0
            lam_interp_list.append(lam_g)
            rad_interp_list.append(rad_g)

        N = len(masses_list)
        all_masses = jnp.array(np.broadcast_to(mass_grid, (N, n_grid)))
        all_lambdas = jnp.array(np.stack(lam_interp_list))
        all_radii = jnp.array(np.stack(rad_interp_list))
        return all_masses, all_lambdas, all_radii

    def make_eos_fn(self) -> Callable[[tuple[Array, Array, Array]], Array]:
        r"""Build a single-EOS log-likelihood callable for use with :func:`jax.lax.map`.

        Returns
        -------
        Callable
            ``f((masses, lambdas, radii)) → scalar log-likelihood``
        """
        likelihood = self.likelihood

        def f(args: tuple[Array, Array, Array]) -> Array:
            masses, lambdas, radii = args
            params = {
                "masses_EOS": masses,
                "Lambdas_EOS": lambdas,  # capital L — matches gw.py:303–304
                "radii_EOS": radii,
            }
            return likelihood.evaluate(params)

        return f

    def evaluate_batch(
        self,
        f: Callable[[tuple[Array, Array, Array]], Array],
        all_masses: Float[Array, "N L"],
        all_lambdas: Float[Array, "N L"],
        all_radii: Float[Array, "N L"],
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
        all_masses, all_lambdas, all_radii :
            Stacked JAX arrays of shape ``[N, L]``.

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
            stacked = (
                all_masses[start:end],
                all_lambdas[start:end],
                all_radii[start:end],
            )
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

    def sample(self, key: PRNGKeyArray) -> SamplerOutput:  # type: ignore[override]  # key unused — no stochastic component
        r"""Evaluate all EOS curves and return evidence + posterior weights.

        Parameters
        ----------
        key :
            JAX random key (not used; accepted for API compatibility).

        Returns
        -------
        SamplerOutput
            - ``samples["eos_index"]`` : integer index per EOS
            - ``samples["log_likelihood"]`` : log-likelihood per EOS
            - ``samples["posterior_weight"]`` : normalised posterior weight per EOS
            - ``log_prob`` : same as ``samples["log_likelihood"]``
            - ``metadata["evidence"]`` : evidence dict (log_Z, log_Z_std, N_eff, …)
            - ``metadata["N_eos"]`` : total number of EOS curves
        """
        f = self.make_eos_fn()

        logger.info(f"Loading EOS file: {self.config.eos_file}")
        all_masses, all_lambdas, all_radii = self.load_and_grid(
            [self.config.eos_file],
            self.config.n_grid,
            self.config.m_min,
            self.config.m_max,
        )
        N = int(all_masses.shape[0])
        logger.info(
            f"EOS set: {N} curves on "
            f"[{self.config.m_min:.2f}, {float(all_masses[0, -1]):.3f}] M_sun "
            f"({self.config.n_grid} grid points)"
        )

        logger.info("Evaluating likelihoods on EOS set...")
        log_likelihoods = self.evaluate_batch(f, all_masses, all_lambdas, all_radii)
        ev = self.compute_evidence(log_likelihoods)
        logger.info(
            f"log Z = {ev['log_Z']:.3f} ± {ev['log_Z_std']:.3f}  "
            f"N_eff = {ev['N_eff']:.1f} ({ev['N_eff_fraction']*100:.1f}%)"
        )

        samples: dict[str, Array] = {
            "eos_index": jnp.arange(N),
            "log_likelihood": log_likelihoods,
            "posterior_weight": ev["posterior_weights"],
        }
        metadata: dict[str, Any] = {
            "evidence": ev,
            "N_eos": N,
        }

        return SamplerOutput(
            samples=samples,
            log_prob=log_likelihoods,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # The following methods are not applicable for this sampler type.      #
    # They raise informative errors rather than NotImplementedError so     #
    # callers get a clear message.                                         #
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
