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

    Unlike parametric samplers, this class does not sample a prior — it
    receives a fixed set of tabulated EOS curves and evaluates the
    combined likelihood on each one.  The result is a discrete posterior
    over the EOS set together with the Bayesian evidence :math:`\log Z`.

    Parameters
    ----------
    likelihood : LikelihoodBase
        Combined likelihood (GW, NICER, radio, …) created by
        :func:`~jesterTOV.inference.likelihoods.factory.create_combined_likelihood`.
    prior : object, optional
        Accepted for API compatibility with :class:`JesterSampler`; ignored.
    sample_transforms : list, optional
        Accepted for API compatibility; ignored.
    likelihood_transforms : list, optional
        Accepted for API compatibility; ignored.
    config : EOSReweightingConfig
        Sampler configuration including EOS file paths and grid settings.
    seed : int, optional
        Accepted for API compatibility; ignored (no stochastic component).
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
        # Do NOT call super().__init__() — JesterSampler assumes a parametric prior.
        # prior, sample_transforms, likelihood_transforms, seed are unused (API compat).
        _ = prior, sample_transforms, likelihood_transforms, seed
        self.likelihood = likelihood  # type: ignore[assignment]
        self.config = config

    def _load_and_grid(
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
            with N curves the arrays should be shaped ``[N, n_points]``.
        n_grid :
            Number of mass grid points.
        m_min :
            Lower bound of the common mass grid in :math:`M_\odot`.
        m_max :
            Upper bound in :math:`M_\odot`.  ``None`` → use
            :math:`\min(M_\mathrm{TOV})` across all curves in ``paths``.

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

            for i in range(m.shape[0]):
                masses_list.append(m[i])
                lambdas_list.append(lam[i])
                radii_list.append(rad[i])
                m_tov_list.append(float(np.max(m[i])))

        # Common grid upper bound
        m_max_grid = float(np.min(m_tov_list)) if m_max is None else m_max
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

    def _make_eos_fn(self) -> Callable[[tuple[Array, Array, Array]], Array]:
        r"""Build a single-EOS log-likelihood callable for use with :func:`jax.lax.map`.

        Returns
        -------
        Callable
            ``f((masses, lambdas, radii)) → scalar log-weight``
        """
        likelihood = self.likelihood

        def f(args: tuple[Array, Array, Array]) -> Array:
            masses, lambdas, radii = args
            params = {
                "masses_EOS": masses,
                "Lambdas_EOS": lambdas,   # capital L — matches gw.py:303–304
                "radii_EOS": radii,
            }
            return likelihood.evaluate(params)

        return f

    def _evaluate_batch(
        self,
        f: Callable[[tuple[Array, Array, Array]], Array],
        all_masses: Float[Array, "N L"],
        all_lambdas: Float[Array, "N L"],
        all_radii: Float[Array, "N L"],
    ) -> Float[Array, " N"]:
        r"""Evaluate *f* on all N EOS curves using :func:`jax.lax.map`.

        Splits the work into progress chunks (controlled by
        ``config.progress_interval``) so that throughput and ETA can be logged
        at regular intervals.  Within each chunk, auto-tunes ``batch_size``:
        starts at ``config.batch_size`` (or chunk size if ``None``), halves
        on OOM, and reuses the found size for subsequent chunks.

        Parameters
        ----------
        f :
            Single-EOS evaluator returned by :meth:`_make_eos_fn`.
        all_masses, all_lambdas, all_radii :
            Stacked JAX arrays of shape ``[N, L]``.

        Returns
        -------
        Float[Array, " N"]
            Log-likelihood per EOS.
        """
        N = all_masses.shape[0]
        progress_interval = self.config.progress_interval

        # Determine Python-level chunk size for progress reporting.
        if progress_interval > 0.0 and N > 1:
            n_chunks = max(1, round(1.0 / progress_interval))
            chunk_size = max(1, (N + n_chunks - 1) // n_chunks)
        else:
            chunk_size = N  # single chunk — no intermediate progress

        # JAX-internal batch size (memory management); will be updated on OOM.
        found_batch_size: int = (
            self.config.batch_size if self.config.batch_size is not None else chunk_size
        )

        results: list[Array] = []
        start_time = time.monotonic()
        processed = 0

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            stacked = (all_masses[start:end], all_lambdas[start:end], all_radii[start:end])
            current_bs = min(found_batch_size, end - start)
            chunk_result: Float[Array, " _"] = jnp.empty(0)  # overwritten in the loop below

            while current_bs >= 1:
                try:
                    chunk_result = jax.lax.map(
                        f, stacked, batch_size=current_bs
                    )
                    jax.block_until_ready(chunk_result)
                    if current_bs != found_batch_size:
                        logger.info(f"Auto-reduced batch_size to {current_bs}")
                    found_batch_size = current_bs
                    break
                except Exception as exc:  # noqa: BLE001
                    if current_bs == 1:
                        raise RuntimeError(
                            f"EOS evaluation failed even at batch_size=1: {exc}"
                        ) from exc
                    prev = current_bs
                    current_bs = max(1, current_bs // 2)
                    logger.warning(
                        f"batch_size={prev} failed ({type(exc).__name__}); "
                        f"retrying with batch_size={current_bs}"
                    )

            results.append(chunk_result)
            processed = end

            if progress_interval > 0.0:
                elapsed = time.monotonic() - start_time
                fraction = processed / N
                eta = elapsed / fraction * (1.0 - fraction) if fraction > 0 else 0.0
                logger.info(
                    f"EOS reweighting: {processed}/{N} EOS "
                    f"({fraction * 100:.0f}%) | "
                    f"elapsed {elapsed:.1f}s | ETA {eta:.1f}s"
                )

        log_weights: Float[Array, " N"] = jnp.concatenate(results)
        return log_weights

    def _bootstrap_log_Z(self, log_weights: Float[Array, " N"]) -> float:
        r"""Estimate :math:`\log Z` uncertainty via bootstrap resampling.

        Parameters
        ----------
        log_weights :
            Per-EOS log-likelihoods.

        Returns
        -------
        float
            Standard deviation of :math:`\log Z` over bootstrap resamples.
        """
        N = log_weights.shape[0]
        lw_np = np.array(log_weights)
        rng = np.random.default_rng(42)
        log_z_boots: list[float] = []
        for _ in range(self.config.n_bootstrap):
            idx = rng.integers(0, N, size=N)
            boot = lw_np[idx]
            log_z_boots.append(float(np.logaddexp.reduce(boot) - np.log(N)))
        return float(np.std(log_z_boots))

    def _compute_evidence(
        self, log_weights: Float[Array, " N"]
    ) -> dict[str, Any]:
        r"""Compute Bayesian evidence and effective sample size from log-weights.

        Parameters
        ----------
        log_weights :
            Per-EOS log-likelihoods of shape ``[N]``.

        Returns
        -------
        dict with keys ``log_Z``, ``log_Z_std``, ``N_eff``,
        ``N_eff_fraction``, ``posterior_weights``.
        """
        N = int(log_weights.shape[0])
        log_Z = float(jax.scipy.special.logsumexp(log_weights) - jnp.log(N))
        # log ESS = 2·logsumexp(w) − logsumexp(2w)  (Kish formula in log space)
        log_ESS = float(
            2.0 * jax.scipy.special.logsumexp(log_weights)
            - jax.scipy.special.logsumexp(2.0 * log_weights)
        )
        N_eff = float(jnp.exp(log_ESS))
        posterior_weights = jnp.exp(
            log_weights - jax.scipy.special.logsumexp(log_weights)
        )
        log_Z_std = self._bootstrap_log_Z(log_weights)
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
            - ``samples["log_weight"]`` : log-likelihood per EOS
            - ``samples["posterior_weight"]`` : normalised posterior weight per EOS
            - ``log_prob`` : same as ``samples["log_weight"]``
            - ``metadata["evidence"]`` : evidence dict (log_Z, log_Z_std, N_eff, …)
            - ``metadata["N_eos"]`` : total number of EOS curves
        """
        f = self._make_eos_fn()

        logger.info(f"Loading EOS file: {self.config.eos_file}")
        all_masses, all_lambdas, all_radii = self._load_and_grid(
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
        log_weights = self._evaluate_batch(f, all_masses, all_lambdas, all_radii)
        ev = self._compute_evidence(log_weights)
        logger.info(
            f"log Z = {ev['log_Z']:.3f} ± {ev['log_Z_std']:.3f}  "
            f"N_eff = {ev['N_eff']:.1f} ({ev['N_eff_fraction']*100:.1f}%)"
        )

        samples: dict[str, Array] = {
            "eos_index": jnp.arange(N),
            "log_weight": log_weights,
            "posterior_weight": ev["posterior_weights"],
        }
        metadata: dict[str, Any] = {
            "evidence": ev,
            "N_eos": N,
        }

        return SamplerOutput(
            samples=samples,
            log_prob=log_weights,
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
