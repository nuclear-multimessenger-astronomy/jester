r"""Gravitational-wave Fisher-forecast likelihood for simulated BNS populations.

This likelihood consumes `gwfast <https://github.com/CosmoStatGW/gwfast>`_
Fisher-matrix forecasts for a simulated population of binary-neutron-star (BNS)
sources -- e.g. a mock Einstein Telescope catalog -- rather than a trained
normalizing-flow posterior for a single real event (contrast with
:class:`~jesterTOV.inference.likelihoods.gw.GWLikelihood`). Each source only has
marginalized (diagonal, uncorrelated) 1-sigma Fisher errors, not a sampled posterior.

Formalism (arXiv:2410.08008, Eq. 6 and Appendix B, which in turn cites Abbott et al.
2019, PRX 9, 011001 for the origin of using a 2D :math:`P(\tilde{\Lambda}, q)`
posterior): for a source with fixed source-frame chirp mass :math:`m_c` and observed
density :math:`P(\tilde{\Lambda}, q)`,

.. math::
    \log \mathcal{L}_{\rm source}(X) = \log \int_{q_{\rm min}}^{q_{\rm max}}
        P\big(\tilde{\Lambda}_X(q, m_c),\, q\big)\, dq

where, for each trial mass ratio :math:`q`, the component masses
:math:`m_1(m_c, q) \geq m_2(m_c, q)` are recovered by inverting the chirp-mass/mass-ratio
relation, :math:`\Lambda_1 = \Lambda_X(m_1)` and :math:`\Lambda_2 = \Lambda_X(m_2)` come
directly off the candidate EOS's own (deterministic, one-to-one) mass-tidal-deformability
curve, and :math:`\tilde{\Lambda}_X(q)` is the standard mass-weighted combination. This
sidesteps inverting the observation into :math:`(\Lambda_1, \Lambda_2)` directly (which is
underdetermined without a Fisher error on :math:`\delta\tilde{\Lambda}`): instead, for each
candidate EOS we predict what :math:`\tilde{\Lambda}(q)` should look like and test that
prediction against the observed/fitted :math:`P(\tilde{\Lambda}, q)`. The total
log-likelihood sums over all included sources (independent events).

Per-source Gaussian fit
------------------------
Each detected source's :math:`P(\tilde{\Lambda}, q)` is approximated as a 2D Gaussian,
built analytically (delta-method / linear error propagation) from the source's Fisher
errors -- see :func:`_fit_source_gaussians`. gwfast fits :math:`\tilde{\Lambda}` directly
as its own waveform parameter (so ``err_LambdaTilde`` is available natively), but not
:math:`q` (which must be derived from ``err_m1_src``/``err_m2_src``). Because gwfast only
stores *marginalized* (diagonal) Fisher errors -- no cross term between
``err_LambdaTilde`` and ``err_m1_src``/``err_m2_src`` is ever stored -- the cross-covariance
:math:`\mathrm{Cov}(\tilde{\Lambda}, q)` is exactly zero under that same diagonal
assumption, not just approximately zero, which makes the fit fully closed-form (no
random sampling, no extra hyperparameters).

Data files
----------
Two HDF5 files are required (see `et-bgr-jester/datasets/scripts/load_gwfast_posteriors.py`
for the reference loading conventions this mirrors -- that script lives in a sibling,
unpackaged repository and is not imported here):

- ``gwfast_result_file``: one row per *detected* source, with ``err_LambdaTilde``,
  ``err_m1_src``, ``err_m2_src`` (1-sigma marginalized Fisher errors),
  ``idx_det_in_cat`` (index of each detected source into the injection catalog), and
  ``snrs`` (SNR of every *injected* source, detected or not -- indexed via
  ``idx_det_in_cat``).
- ``injection_catalog_file``: true/injected values for every injected source, with
  ``m1_src``, ``m2_src`` (source-frame component masses), ``Mc`` (detector-frame chirp
  mass), ``z`` (redshift), ``Lambda1``, ``Lambda2``, ``eta``.

**Assumed convention:** ``m1_src`` is the heavier component (:math:`q = m_2/m_1 \leq 1`,
matching the LVK/bilby convention). This was inferred from consistent usage in
`et-bgr-jester`'s scripts but was not independently re-verified against gwfast's own
source; :func:`_validate_component_mass_ordering` defends this assumption at
construction time with a clear error if it's ever violated.
"""

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal
from jaxtyping import Array, Float

from jesterTOV import utils
from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")

# Only the subset of each gwfast/injection-catalog HDF5 file this likelihood needs.
_GWFAST_RESULT_KEYS: tuple[str, ...] = (
    "err_LambdaTilde",
    "err_m1_src",
    "err_m2_src",
    "idx_det_in_cat",
    "snrs",
)
_INJECTION_CATALOG_KEYS: tuple[str, ...] = (
    "m1_src",
    "m2_src",
    "Mc",
    "z",
    "Lambda1",
    "Lambda2",
    "eta",
)


def _read_hdf5_datasets(path: str | Path, keys: Iterable[str]) -> dict[str, np.ndarray]:
    """Read a fixed set of named datasets from an HDF5 file into numpy arrays.

    Parameters
    ----------
    path : str | Path
        Path to the HDF5 file.
    keys : Iterable[str]
        Dataset names to read.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from dataset name to its contents.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If any of ``keys`` is missing from the file, naming the file and the missing
        key(s) explicitly.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with h5py.File(path, "r") as f:
        available = set(f.keys())
        missing = sorted(set(keys) - available)
        if missing:
            raise KeyError(
                f"{path} is missing required dataset(s): {missing}. "
                f"Present datasets: {sorted(available)}"
            )
        data: dict[str, np.ndarray] = {}
        for key in keys:
            dataset = f[key]
            assert isinstance(
                dataset, h5py.Dataset
            ), f"{path}: expected '{key}' to be an HDF5 dataset, got {type(dataset)}"
            data[key] = np.asarray(dataset[()])
        return data


def _load_gwfast_result(path: str | Path) -> dict[str, np.ndarray]:
    """Load the gwfast Fisher-forecast result HDF5 file (per-detected-source data)."""
    return _read_hdf5_datasets(path, _GWFAST_RESULT_KEYS)


def _load_injection_catalog(path: str | Path) -> dict[str, np.ndarray]:
    """Load the injection catalog HDF5 file (true values for every injected source).

    Derives ``Mc_src`` (source-frame chirp mass) and ``LambdaTilde`` (true effective
    tidal deformability, via :func:`jesterTOV.utils.lambda_tilde_from_lambda1_lambda2`
    -- the same implementation :meth:`GWFisherLikelihood.evaluate` uses, so there is
    only one copy of the formula to maintain).
    """
    data = _read_hdf5_datasets(path, _INJECTION_CATALOG_KEYS)
    data["Mc_src"] = data["Mc"] / (1.0 + data["z"])
    data["LambdaTilde"] = np.asarray(
        utils.lambda_tilde_from_lambda1_lambda2(
            jnp.asarray(data["Lambda1"]),
            jnp.asarray(data["Lambda2"]),
            jnp.asarray(data["eta"]),
        )
    )
    return data


def _validate_component_mass_ordering(
    m1_src: np.ndarray, m2_src: np.ndarray, catalog_file: str | Path
) -> None:
    """Raise ``ValueError`` if ``m1_src < m2_src`` for any injected source.

    ``GWFisherLikelihood`` assumes ``m1_src`` is always the heavier component so that
    ``Lambda_1``/``Lambda_2`` are labelled consistently with the
    :func:`~jesterTOV.utils.lambda_tilde_from_lambda1_lambda2` formula. This check runs
    on the *full* raw catalog, before any detected-source or SNR filtering.
    """
    violations = np.flatnonzero(m1_src < m2_src)
    if violations.size:
        i = int(violations[0])
        raise ValueError(
            f"{catalog_file}: expected m1_src >= m2_src (m1_src assumed to be the "
            f"heavier component) for every injected source, but {violations.size} "
            f"row(s) violate this. First offending row is index {i}: "
            f"m1_src={m1_src[i]!r}, m2_src={m2_src[i]!r}."
        )


def _build_q_grid(
    q_min: float, q_max: float, dq: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build a fixed mass-ratio quadrature grid and matching trapezoidal weights.

    Uses ``np.linspace(q_min, q_max, n_q)`` with a concrete Python int ``n_q`` computed
    once here -- not ``np.arange(q_min, q_max, dq)``, whose element count is fragile
    under floating-point rounding of ``(q_max - q_min) / dq`` and can silently differ by
    one between otherwise-identical runs/platforms (which would change a JIT-traced
    array's static shape).

    Parameters
    ----------
    q_min, q_max : float
        Integration bounds.
    dq : float
        Requested grid spacing. The actual spacing used,
        ``(q_max - q_min) / (n_q - 1)``, equals ``dq`` only when it evenly divides
        ``q_max - q_min``.

    Returns
    -------
    q_grid : np.ndarray, shape (n_q,)
        Evenly spaced quadrature points, ``q_grid[0] == q_min``, ``q_grid[-1] == q_max``.
    trapz_weights : np.ndarray, shape (n_q,)
        Composite trapezoidal-rule weights (``dq_actual`` in the interior, half that at
        both endpoints), such that
        ``jax.scipy.special.logsumexp(logpdf_grid, b=trapz_weights)`` equals the log of
        the trapezoidal estimate of :math:`\\int \\exp(\\mathrm{logpdf}(q))\\, dq`.
    """
    n_q = max(int(round((q_max - q_min) / dq)) + 1, 2)
    q_grid = np.linspace(q_min, q_max, n_q)
    dq_actual = (q_max - q_min) / (n_q - 1)
    if abs(dq_actual - dq) > 0.05 * dq:
        logger.warning(
            f"Requested dq={dq} does not evenly divide q_max-q_min={q_max - q_min}; "
            f"using dq_actual={dq_actual} ({n_q} grid points) instead."
        )
    trapz_weights = np.full(n_q, dq_actual)
    trapz_weights[0] *= 0.5
    trapz_weights[-1] *= 0.5
    return q_grid, trapz_weights


def _fit_source_gaussians(
    m1_true: np.ndarray,
    m2_true: np.ndarray,
    err_m1: np.ndarray,
    err_m2: np.ndarray,
    lambda_tilde_true: np.ndarray,
    err_lambda_tilde: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Analytic (delta-method) Gaussian fit to :math:`P(\tilde{\Lambda}, q)` per source.

    Fully vectorized over all sources at once (plain numpy, never JAX-traced -- this
    only ever runs once, at construction time).

    :math:`\tilde{\Lambda}`'s variance is exact (:math:`\mathrm{Var}(\tilde{\Lambda}) =
    \mathrm{err\_LambdaTilde}^2`, since it's a direct gwfast Fisher parameter).
    :math:`q = m_2/m_1`'s variance follows standard linear error propagation on the two
    (assumed independent) component-mass Fisher errors:

    .. math::
        \frac{\partial q}{\partial m_1} = -\frac{q}{m_1}, \qquad
        \frac{\partial q}{\partial m_2} = \frac{1}{m_1} \qquad \Longrightarrow \qquad
        \mathrm{Var}(q) \approx \left(\frac{q}{m_1}\right)^2 \mathrm{Var}(m_1)
            + \left(\frac{1}{m_1}\right)^2 \mathrm{Var}(m_2)

    The cross-covariance :math:`\mathrm{Cov}(\tilde{\Lambda}, q)` is **exactly** zero,
    not merely approximately so: gwfast only stores marginalized (diagonal) Fisher
    errors, so ``err_LambdaTilde`` is statistically independent of
    ``err_m1_src``/``err_m2_src`` by construction, and a deterministic function of
    variables independent of :math:`\tilde{\Lambda}` (here, :math:`q = m_2/m_1`) is
    itself independent of :math:`\tilde{\Lambda}` -- no linearization is needed for that
    part of the argument, unlike the :math:`\mathrm{Var}(q)` approximation above.

    Parameters
    ----------
    m1_true, m2_true : np.ndarray, shape (n_sources,)
        True (injected) source-frame component masses, ``m1_true >= m2_true``.
    err_m1, err_m2 : np.ndarray, shape (n_sources,)
        1-sigma Fisher errors on the component masses.
    lambda_tilde_true : np.ndarray, shape (n_sources,)
        True (injected) effective tidal deformability.
    err_lambda_tilde : np.ndarray, shape (n_sources,)
        1-sigma Fisher error on the effective tidal deformability.

    Returns
    -------
    means : np.ndarray, shape (n_sources, 2)
        Per-source ``[lambda_tilde_true, q_true]``.
    covs : np.ndarray, shape (n_sources, 2, 2)
        Per-source diagonal covariance in the same ``[lambda_tilde, q]`` ordering.
    """
    q_true = m2_true / m1_true
    var_lambda_tilde = err_lambda_tilde**2
    var_q = (m2_true / m1_true**2) ** 2 * err_m1**2 + (1.0 / m1_true) ** 2 * err_m2**2

    n_sources = m1_true.shape[0]
    means = np.stack([lambda_tilde_true, q_true], axis=-1)
    covs = np.zeros((n_sources, 2, 2))
    covs[:, 0, 0] = var_lambda_tilde
    covs[:, 1, 1] = var_q
    return means, covs


class GWFisherLikelihood(LikelihoodBase):
    r"""EOS likelihood from gwfast Fisher-forecast BNS sources via :math:`\tilde{\Lambda}`-``q`` marginalization.

    See the module docstring for the full formalism, data-file requirements, and the
    per-source Gaussian-fit derivation. Unlike
    :class:`~jesterTOV.inference.likelihoods.gw.GWLikelihood`/
    :class:`~jesterTOV.inference.likelihoods.gw.StackedGWLikelihood` (which handle 1-2
    real events, each with a trained normalizing-flow posterior), this class is
    designed to scale to hundreds-to-thousands of simulated sources: it loads both
    input files, filters by SNR, and stacks every source's data itself -- there is no
    architecture-compatibility concern like ``StackedGWLikelihood``'s
    ``_flow_architecture_signature`` check, since a ``(mean, cov, Mc_src)`` triple
    trivially stacks across any number of sources.

    Evaluation performs deterministic trapezoidal quadrature over a fixed mass-ratio
    grid (built once at construction from ``q_min``/``q_max``/``dq``), not Monte Carlo
    averaging over pre-sampled points like ``GWLikelihood``/``MockMassRadiusLikelihood``
    -- so there is no random seed and no sampling noise at evaluation time.

    Like every other likelihood in this module, this class has no internal
    ``@jax.jit``: it relies on the outer sampler (SMC/FlowMC) to JIT-compile the full
    log-posterior as one unit.

    Parameters
    ----------
    gwfast_result_file : str
        Path to the gwfast Fisher-forecast result HDF5 file.
    injection_catalog_file : str
        Path to the matching injection catalog HDF5 file.
    q_min, q_max : float
        Mass-ratio quadrature bounds, :math:`0 < q_{\rm min} < q_{\rm max} \leq 1`.
    dq : float
        Requested mass-ratio grid spacing (see :func:`_build_q_grid`).
    snr_threshold : float, optional
        Additional SNR cut on top of whatever detection threshold is already baked
        into ``gwfast_result_file`` (default: ``0.0``, i.e. no extra cut).
    penalty_value : float, optional
        Log-likelihood penalty applied when a trial component mass exceeds
        :math:`M_{\rm TOV}` of the candidate EOS (default: ``0.0``, i.e. no penalty).
    source_batch_size : int, optional
        Batch size for ``jax.lax.map`` over sources (default: ``1``, a plain scan --
        keeps memory flat under the outer particle ``vmap`` used by e.g. the SMC
        sampler). This is the knob that matters most for this class: unlike
        ``GWLikelihood``/``StackedGWLikelihood`` (1-2 events), an ET-scale population
        can put hundreds-to-thousands of sources through this axis.
    q_batch_size : int, optional
        Batch size for ``jax.lax.map`` over the mass-ratio grid (default: ``1``),
        mirroring ``N_masses_batch_size`` elsewhere for interface consistency; less
        critical than ``source_batch_size`` since the grid size is a small,
        user-chosen quadrature-accuracy parameter rather than something that scales
        with the dataset.

    Attributes
    ----------
    n_sources : int
        Number of sources retained after the SNR cut.
    """

    gwfast_result_file: str
    injection_catalog_file: str
    q_min: float
    q_max: float
    dq: float
    snr_threshold: float
    penalty_value: float
    source_batch_size: int
    q_batch_size: int
    n_sources: int

    def __init__(
        self,
        gwfast_result_file: str,
        injection_catalog_file: str,
        q_min: float,
        q_max: float,
        dq: float,
        snr_threshold: float = 0.0,
        penalty_value: float = 0.0,
        source_batch_size: int = 1,
        q_batch_size: int = 1,
    ) -> None:
        super().__init__()

        if not (0.0 < q_min < q_max <= 1.0):
            raise ValueError(
                f"Require 0 < q_min < q_max <= 1, got q_min={q_min}, q_max={q_max}"
            )
        if dq <= 0.0:
            raise ValueError(f"dq must be positive, got dq={dq}")

        self.gwfast_result_file = gwfast_result_file
        self.injection_catalog_file = injection_catalog_file
        self.q_min = q_min
        self.q_max = q_max
        self.dq = dq
        self.snr_threshold = snr_threshold
        self.penalty_value = penalty_value
        self.source_batch_size = source_batch_size
        self.q_batch_size = q_batch_size

        logger.info(
            f"Loading gwfast Fisher-forecast data from {gwfast_result_file} "
            f"and {injection_catalog_file}"
        )
        result = _load_gwfast_result(gwfast_result_file)
        injections = _load_injection_catalog(injection_catalog_file)
        _validate_component_mass_ordering(
            injections["m1_src"], injections["m2_src"], injection_catalog_file
        )

        idx = result["idx_det_in_cat"].astype(int)
        n_detected = idx.shape[0]
        m1_true = injections["m1_src"][idx]
        m2_true = injections["m2_src"][idx]
        mc_src_true = injections["Mc_src"][idx]
        lambda_tilde_true = injections["LambdaTilde"][idx]
        snrs = result["snrs"][idx]

        mask = snrs >= snr_threshold
        if not np.any(mask):
            raise ValueError(
                f"0 of {n_detected} detected sources in {gwfast_result_file} have "
                f"SNR >= {snr_threshold}; max SNR in file is {float(np.max(snrs))}."
            )

        means, covs = _fit_source_gaussians(
            m1_true[mask],
            m2_true[mask],
            result["err_m1_src"][mask],
            result["err_m2_src"][mask],
            lambda_tilde_true[mask],
            result["err_LambdaTilde"][mask],
        )

        self.n_sources = int(np.sum(mask))
        self._mc_src: Float[Array, " n_sources"] = jnp.asarray(mc_src_true[mask])
        self._means: Float[Array, "n_sources 2"] = jnp.asarray(means)
        self._covs: Float[Array, "n_sources 2 2"] = jnp.asarray(covs)

        q_grid, trapz_weights = _build_q_grid(q_min, q_max, dq)
        self._q_grid: Float[Array, " n_q"] = jnp.asarray(q_grid)
        self._trapz_weights: Float[Array, " n_q"] = jnp.asarray(trapz_weights)

        logger.info(
            f"GWFisherLikelihood: {n_detected} detected sources, "
            f"{self.n_sources} retained after snr_threshold={snr_threshold}, "
            f"q-grid has {q_grid.shape[0]} points over [{q_min}, {q_max}], "
            f"source_batch_size={source_batch_size}, q_batch_size={q_batch_size}"
        )

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate summed log likelihood over all retained sources for given EOS parameters.

        For each source, integrates the fitted :math:`P(\\tilde{\\Lambda}, q)` Gaussian
        along the candidate EOS's predicted :math:`\\tilde{\\Lambda}_X(q)` curve via
        trapezoidal quadrature over the fixed mass-ratio grid, then sums the resulting
        per-source log-likelihoods.

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

        def process_one_source(carry) -> Float:
            mc_src, mean, cov = carry

            def process_one_q(q: Float) -> Float:
                m1, m2 = utils.component_masses_from_chirp_mass_and_mass_ratio(
                    mc_src, q
                )
                lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right=1.0)
                lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right=1.0)
                eta = utils.symmetric_mass_ratio_from_mass_ratio(q)
                lambda_tilde = utils.lambda_tilde_from_lambda1_lambda2(
                    lambda_1, lambda_2, eta
                )
                logpdf = multivariate_normal.logpdf(
                    jnp.array([lambda_tilde, q]), mean, cov
                )
                penalty_m1 = jnp.where(m1 > mtov, self.penalty_value, 0.0)
                penalty_m2 = jnp.where(m2 > mtov, self.penalty_value, 0.0)
                return logpdf + penalty_m1 + penalty_m2

            # Deterministic trapezoidal quadrature over the fixed q-grid (not a Monte
            # Carlo average) -- b=trapz_weights makes this a stable
            # log(sum(weight * exp(logpdf))), i.e. log of the trapezoidal integral.
            logpdf_grid = jax.lax.map(
                process_one_q, self._q_grid, batch_size=self.q_batch_size
            )
            return logsumexp(logpdf_grid, b=self._trapz_weights)

        per_source_loglike = jax.lax.map(
            process_one_source,
            (self._mc_src, self._means, self._covs),
            batch_size=self.source_batch_size,
        )
        return jnp.sum(per_source_loglike)
