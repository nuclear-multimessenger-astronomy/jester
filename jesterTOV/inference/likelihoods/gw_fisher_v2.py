r"""GW Fisher-forecast likelihood using gwfast's full per-source Fisher covariance.

Each detected source's Fisher information is provided as a full 4x4 covariance
matrix over :math:`(\mathcal{M}_{c,\rm src}, \eta, \tilde{\Lambda}, \delta\tilde{\Lambda})`
(source-frame chirp mass, symmetric mass ratio, effective and antisymmetric tidal
terms), including the genuine cross-correlations between all four quantities -- not
just their marginal (diagonal) errors. The basis ordering within the 4x4 matrix is
given by the file's own ``covariance_keys`` group and is read dynamically rather than
assumed.

For a candidate EOS, each pre-sampled mass pair :math:`(m_1, m_2)` (drawn once at
construction from the source's own correlated :math:`(\mathcal{M}_{c,\rm src}, \eta)`
sub-block) determines :math:`\Lambda_{1,X}(m_1)`, :math:`\Lambda_{2,X}(m_2)`
deterministically from the EOS's mass-tidal-deformability curve, which forward-transform
to :math:`\tilde{\Lambda}_X`, :math:`\delta\tilde{\Lambda}_X`. The full 4D Gaussian
density is evaluated at the point implied by that mass hypothesis and the EOS's tidal
prediction, Monte-Carlo-averaged over the pre-sampled mass pairs
(``logsumexp(...) - log(N)``), then summed over sources.

Data file
---------
A single HDF5 file per detector/EOS/SNR-threshold combination, with:

- ``covariance``: shape ``(4, 4, n_sources)``, the per-source Fisher covariance matrix.
- ``covariance_keys``: group of four scalar datasets (``Mc_src``, ``eta``,
  ``LambdaTilde``, ``deltaLambda``) giving each quantity's index into the leading two
  axes of ``covariance``.
- ``detected_event_parameters``: group of per-source true/injected values, including
  ``m1_src``, ``m2_src``, ``Lambda1``, ``Lambda2``, ``Mc``, ``z``, ``eta``.
- ``idx_det_in_cat``: index of each detected source into the full injected population
  (used only to look up that source's SNR).
- ``snrs``: SNR of every injected source (detected or not), indexed via
  ``idx_det_in_cat``.

No separate injection catalog file is needed: every detected source's true parameters
are already included alongside its Fisher covariance.
"""

from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from jax.scipy.stats import multivariate_normal
from jaxtyping import Array, Float

from jesterTOV import utils
from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.inference.likelihoods.gw_fisher import (
    _positivity_quality_mask,
    _validate_component_mass_ordering,
)
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")

_REQUIRED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "covariance",
        "covariance_keys",
        "detected_event_parameters",
        "idx_det_in_cat",
        "snrs",
    }
)
_REQUIRED_COVARIANCE_BASIS_KEYS: frozenset[str] = frozenset(
    {"Mc_src", "eta", "LambdaTilde", "deltaLambda"}
)
_REQUIRED_DETECTED_PARAM_KEYS: frozenset[str] = frozenset(
    {"m1_src", "m2_src", "Lambda1", "Lambda2", "Mc", "z", "eta"}
)


def _dataset(group: h5py.Group, key: str, path: str | Path) -> h5py.Dataset:
    """Fetch ``group[key]``, asserting it is a concrete HDF5 dataset (not a group or
    named datatype) -- narrows h5py's union return type for pyright.
    """
    obj = group[key]
    assert isinstance(obj, h5py.Dataset), (
        f"{path}: expected '{key}' to be an HDF5 dataset, got {type(obj)}"
    )
    return obj


def _load_gwfast_covariance_result(path: str | Path) -> dict:
    """Load a gwfast Fisher-covariance result HDF5 file.

    Parameters
    ----------
    path : str | Path
        Path to the HDF5 file.

    Returns
    -------
    dict
        ``covariance`` (``np.ndarray``, shape ``(n_sources, 4, 4)``, source axis
        moved to the front), ``key_order`` (``dict[str, int]`` mapping each of
        ``Mc_src``/``eta``/``LambdaTilde``/``deltaLambda`` to its index in the 4x4
        matrix), ``detected_params`` (``dict[str, np.ndarray]``, one entry per key in
        :data:`_REQUIRED_DETECTED_PARAM_KEYS`), and ``snrs`` (``np.ndarray``, shape
        ``(n_sources,)``, already indexed via ``idx_det_in_cat``).

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If any required top-level dataset/group, covariance-basis key, or detected-
        event parameter is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with h5py.File(path, "r") as f:
        missing_top = _REQUIRED_TOP_LEVEL_KEYS - set(f.keys())
        if missing_top:
            raise KeyError(
                f"{path} is missing required dataset(s)/group(s): {sorted(missing_top)}. "
                f"Present: {sorted(f.keys())}"
            )

        covariance_keys_group = f["covariance_keys"]
        assert isinstance(covariance_keys_group, h5py.Group), (
            f"{path}: expected 'covariance_keys' to be an HDF5 group"
        )
        key_order = {
            key: int(np.asarray(_dataset(covariance_keys_group, key, path)[()]))
            for key in covariance_keys_group.keys()  # noqa: SIM118 -- h5py.Group, not a dict
        }
        missing_basis = _REQUIRED_COVARIANCE_BASIS_KEYS - set(key_order.keys())
        if missing_basis:
            raise KeyError(
                f"{path}: covariance_keys is missing required entries: {sorted(missing_basis)}"
            )

        dep_group = f["detected_event_parameters"]
        assert isinstance(dep_group, h5py.Group), (
            f"{path}: expected 'detected_event_parameters' to be an HDF5 group"
        )
        missing_dep = _REQUIRED_DETECTED_PARAM_KEYS - set(dep_group.keys())
        if missing_dep:
            raise KeyError(
                f"{path}: detected_event_parameters is missing required dataset(s): "
                f"{sorted(missing_dep)}"
            )

        covariance = np.moveaxis(np.asarray(_dataset(f, "covariance", path)[()]), -1, 0)
        detected_params = {
            key: np.asarray(_dataset(dep_group, key, path)[()])
            for key in _REQUIRED_DETECTED_PARAM_KEYS
        }
        idx_det_in_cat = np.asarray(_dataset(f, "idx_det_in_cat", path)[()]).astype(int)
        snrs = np.asarray(_dataset(f, "snrs", path)[()])[idx_det_in_cat]

    return {
        "covariance": covariance,
        "key_order": key_order,
        "detected_params": detected_params,
        "snrs": snrs,
    }


def _build_means(
    detected_params: dict[str, np.ndarray], key_order: dict[str, int]
) -> np.ndarray:
    r"""Per-source true mean vector in the file's own
    :math:`(\mathcal{M}_{c,\rm src}, \eta, \tilde{\Lambda}, \delta\tilde{\Lambda})`
    basis ordering.

    :math:`\eta` is taken directly from ``detected_params`` (already a native gwfast
    fit parameter); :math:`\tilde{\Lambda}` and :math:`\delta\tilde{\Lambda}` are
    derived from ``Lambda1``/``Lambda2``/``eta`` via the same forward formulas
    :meth:`GWFisherLikelihoodV2.evaluate` uses for a candidate EOS.

    Parameters
    ----------
    detected_params : dict[str, np.ndarray]
        Per-source true values, see :data:`_REQUIRED_DETECTED_PARAM_KEYS`.
    key_order : dict[str, int]
        Basis ordering, see :func:`_load_gwfast_covariance_result`.

    Returns
    -------
    np.ndarray, shape (n_sources, 4)
    """
    eta_true = detected_params["eta"]
    lambda1_true = detected_params["Lambda1"]
    lambda2_true = detected_params["Lambda2"]
    mc_src_true = detected_params["Mc"] / (1.0 + detected_params["z"])
    lambda_tilde_true = np.asarray(
        utils.lambda_tilde_from_lambda1_lambda2(
            jnp.asarray(lambda1_true), jnp.asarray(lambda2_true), jnp.asarray(eta_true)
        )
    )
    delta_lambda_true = np.asarray(
        utils.delta_lambda_tilde_from_lambda1_lambda2(
            jnp.asarray(lambda1_true), jnp.asarray(lambda2_true), jnp.asarray(eta_true)
        )
    )
    n_sources = eta_true.shape[0]
    means = np.zeros((n_sources, 4))
    means[:, key_order["Mc_src"]] = mc_src_true
    means[:, key_order["eta"]] = eta_true
    means[:, key_order["LambdaTilde"]] = lambda_tilde_true
    means[:, key_order["deltaLambda"]] = delta_lambda_true
    return means


# Physical clipping bounds applied to sampled component masses, after converting a
# sampled (Mc_src, eta) draw to (m1, m2) -- a purely numerical safety net for the EOS
# table lookup, not part of the statistical model (the Gaussian is still evaluated
# against the raw sampled Mc_src/eta, see _sample_masses).
_MIN_SAMPLED_MASS: float = 0.5  # Msun
_MAX_SAMPLED_MASS: float = 3.5  # Msun
_MIN_ETA: float = 1e-6
_MAX_ETA: float = 0.25 - _MIN_ETA


def _minimum_variance_quality_mask(
    variances: np.ndarray, min_variance: float, name: str
) -> np.ndarray:
    r"""Data-quality cut for a source whose Fisher variance on some quantity is
    implausibly, artificially tiny -- the opposite failure mode from
    :func:`~jesterTOV.inference.likelihoods.gw_fisher._positivity_quality_mask`'s
    "error too large" case.

    Motivation: near the symmetric-mass-ratio boundary (:math:`\eta \to 0.25`, i.e.
    :math:`m_1 \to m_2`), the linearized Fisher-matrix approximation is known to become
    unreliable (a standard, documented pathology of Fisher-matrix parameter estimation
    near near-degenerate points, e.g. Vallisneri 2008) -- waveform derivatives w.r.t.
    different intrinsic parameters become nearly degenerate there, and a covariance
    matrix (the Fisher matrix's inverse) approaching singularity does not uniformly
    blow up: it develops both an enormous eigenvalue (the near-degenerate combined
    direction) *and* a spuriously tiny eigenvalue (the orthogonal combination), even
    though the *true* physical information about that combination has not actually
    improved. This is a linearization artifact, not real measurement precision.

    Empirically confirmed on the real SFHo/Delta SNR>=12 covariance data: sources
    within :math:`10^{-6}` of the :math:`\eta=0.25` boundary have a median
    ``Var(eta)`` five orders of magnitude below the population median (`4.4e-9` vs.
    `4.4e-4`), with the worst cases down to `~1e-12` -- i.e. `eta` reported as known to
    aphysically high precision. Because ``GWFisherLikelihoodV2`` samples `(Mc_src,eta)`
    from this same covariance and then evaluates the full 4D Gaussian at the
    EOS-substituted point, an artificially tiny ``Var(eta)`` turns that source's
    contribution into a near-delta-function constraint: any candidate EOS whose
    predicted point deviates from the mean by even a small amount (inevitable once many
    other sources are also being satisfied simultaneously) incurs an enormous
    log-likelihood penalty. Stacked over the ~100-600 such sources present in a
    full SNR>=12 catalog (a small fraction of sources, but a large absolute count now
    that thousands of sources are retained), this creates a badly-conditioned,
    needle-like posterior that a fixed-size random-walk step cannot navigate --
    diagnosed as the root cause of the SMC acceptance collapse this cut was added to
    fix (see ``et-bgr-jester/runs/debug/FINDINGS.md``, Part 5).

    Parameters
    ----------
    variances : np.ndarray
        Per-source Fisher variances (diagonal covariance entries) for one quantity.
    min_variance : float
        Minimum variance a source must have to be retained. ``min_variance <= 0``
        disables the cut entirely (returns an all-``True`` mask).
    name : str
        Human-readable name of the quantity, for the log message.

    Returns
    -------
    np.ndarray
        Boolean mask, ``True`` for sources passing the cut. Logs the number and
        fraction excluded (if any) via the module logger.
    """
    if min_variance <= 0:
        return np.ones_like(variances, dtype=bool)
    good = variances >= min_variance
    n_bad = int(np.sum(~good))
    if n_bad > 0:
        logger.info(
            f"Data quality cut: excluding {n_bad}/{len(variances)} sources "
            f"({100 * n_bad / len(variances):.1f}%) where Var({name}) < {min_variance:.3e} "
            "-- Fisher covariance implausibly over-precise, likely a linearization "
            "artifact near a parameter-degeneracy boundary (see docstring)."
        )
    return good


def _sample_masses(
    means: np.ndarray,
    covs: np.ndarray,
    key_order: dict[str, int],
    n_mass_samples: int,
    seed: int,
    pool_size: int = 2000,
    max_rounds: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Pre-sample :math:`(m_1, m_2)` pairs per source from the source's own correlated
    :math:`(\mathcal{M}_{c,\rm src}, \eta)` sub-block (the real Fisher covariance, not
    an independent-Gaussian approximation), then convert to component masses.

    Uses **rejection sampling**, not clipping: a draw is accepted only if its raw
    :math:`\eta` is in the valid domain :math:`(0,0.25)` *and* its implied
    :math:`(m_1,m_2)` falls in the physical window
    ``[_MIN_SAMPLED_MASS, _MAX_SAMPLED_MASS]``. This matters because ``evaluate()``
    later re-derives ``(Mc_src, eta)`` from whatever ``(m1,m2)`` ends up stored here and
    evaluates the source's 4D Gaussian *at that recomputed point* -- for a *clipped*
    draw, the recomputed ``(Mc_src,eta)`` no longer equals the ``(Mc_src,eta)`` actually
    drawn from the source's own covariance, silently invalidating the "evaluate at the
    point you sampled" identity the Monte Carlo estimator's correctness depends on (see
    ``et-bgr-jester/runs/debug/FINDINGS.md``, Part 6 #35). Rejection sampling never
    evaluates at a point other than one that was actually drawn, so no such drift is
    possible by construction.

    Poorly-measured sources (huge ``Var(Mc_src)``/``Var(eta)``) may have low acceptance
    probability and end up with fewer than ``n_mass_samples`` accepted draws -- this is
    honest (more Monte Carlo noise for that source, not a silent bias) rather than the
    previous behaviour of padding with a systematically-wrong point. Sampling proceeds
    in rounds of ``pool_size`` draws, each round only re-sampling sources still short of
    ``n_mass_samples`` accepted draws (bounds memory: a single ``pool_size`` large enough
    to satisfy the *worst* source for the *whole* population is infeasible -- e.g.
    ``pool_size=30_000`` for ~14,000 SNR>=12 sources exhausts available memory).

    Parameters
    ----------
    means : np.ndarray, shape (n_sources, 4)
    covs : np.ndarray, shape (n_sources, 4, 4)
    key_order : dict[str, int]
    n_mass_samples : int
    seed : int
    pool_size : int, optional
        Number of raw ``(Mc_src,eta)`` draws per source per round (default 2000).
    max_rounds : int, optional
        Maximum number of resampling rounds for sources still short of
        ``n_mass_samples`` accepted draws (default 6). Empirically (real SFHo/Delta
        SNR>=12 data, the hardest production case): round-by-round shortfall count
        4424 -> 2446 -> 1605 -> 1179 -> 937 -> 766 (of 14,470 sources), ~5.7s total,
        zero sources left with 0 accepted draws.

    Returns
    -------
    mass_samples : np.ndarray, shape (n_sources, n_mass_samples, 2)
        Accepted ``[m1, m2]`` pairs. Slots beyond a source's accepted count are filled
        with a physically-valid dummy value (``1.0`` Msun for both components) --
        ``evaluate()`` masks these out via ``n_accepted``, never mistaking them for real
        constraints.
    n_accepted : np.ndarray, shape (n_sources,), dtype int
        Number of accepted draws per source (``<= n_mass_samples``). The true Monte
        Carlo sample count to normalize by in ``evaluate()`` -- NOT ``n_mass_samples``.
    """
    rng = np.random.default_rng(seed)
    n_sources = means.shape[0]
    idx = [key_order["Mc_src"], key_order["eta"]]
    sub_mean = means[:, idx]
    sub_cov = covs[:, idx][:, :, idx]
    L = np.linalg.cholesky(sub_cov)

    m1_out = np.ones((n_sources, n_mass_samples))
    m2_out = np.ones((n_sources, n_mass_samples))
    n_accepted = np.zeros(n_sources, dtype=int)
    pending = np.arange(n_sources)

    for _ in range(max_rounds):
        if len(pending) == 0:
            break
        z = rng.standard_normal((len(pending), pool_size, 2))
        raw = sub_mean[pending, None, :] + np.einsum("sij,skj->ski", L[pending], z)
        mc_src_raw, eta_raw = raw[..., 0], raw[..., 1]

        valid_eta = (eta_raw > _MIN_ETA) & (eta_raw < _MAX_ETA) & (mc_src_raw > 1e-6)
        eta_safe = np.where(valid_eta, eta_raw, 0.1)
        mc_src_safe = np.where(valid_eta, mc_src_raw, 1.0)
        q = np.asarray(
            utils.mass_ratio_from_symmetric_mass_ratio(jnp.asarray(eta_safe))
        )
        m1, m2 = utils.component_masses_from_chirp_mass_and_mass_ratio(
            jnp.asarray(mc_src_safe), jnp.asarray(q)
        )
        m1, m2 = np.asarray(m1), np.asarray(m2)
        valid = (
            valid_eta
            & (m1 >= _MIN_SAMPLED_MASS)
            & (m1 <= _MAX_SAMPLED_MASS)
            & (m2 >= _MIN_SAMPLED_MASS)
            & (m2 <= _MAX_SAMPLED_MASS)
        )

        still_pending = []
        for local_i, s in enumerate(pending):
            need = n_mass_samples - n_accepted[s]
            good = np.nonzero(valid[local_i])[0][:need]
            n_new = len(good)
            if n_new > 0:
                sl = slice(n_accepted[s], n_accepted[s] + n_new)
                m1_out[s, sl] = m1[local_i, good]
                m2_out[s, sl] = m2[local_i, good]
                n_accepted[s] += n_new
            if n_accepted[s] < n_mass_samples:
                still_pending.append(s)
        pending = np.asarray(still_pending, dtype=int)

    n_short = len(pending)
    n_zero = int(np.sum(n_accepted == 0))
    if n_zero > 0:
        raise ValueError(
            f"{n_zero}/{n_sources} sources have ZERO accepted mass samples after "
            f"{max_rounds} rounds of {pool_size} draws each -- their Fisher covariance "
            "puts essentially no probability in the physical mass window "
            f"[{_MIN_SAMPLED_MASS}, {_MAX_SAMPLED_MASS}] Msun. Increase pool_size/"
            "max_rounds, or apply a data-quality cut (quality_cut_n_sigma/"
            "min_eta_variance) to exclude these sources."
        )
    if n_short > 0:
        logger.info(
            f"Mass rejection sampling: {n_short}/{n_sources} sources ({100 * n_short / n_sources:.1f}%) "
            f"did not reach the full {n_mass_samples} accepted draws after {max_rounds} rounds "
            f"(min accepted: {int(n_accepted[n_accepted > 0].min()) if n_short < n_sources or n_zero == 0 else 'n/a'}) "
            "-- these sources' log-likelihood contribution uses a smaller, honest Monte "
            "Carlo sample count instead of a silently-biased one."
        )
    return np.stack([m1_out, m2_out], axis=-1), n_accepted


class GWFisherLikelihoodV2(LikelihoodBase):
    r"""EOS likelihood from gwfast Fisher-forecast BNS sources using the full
    per-source Fisher covariance matrix.

    See the module docstring for the full data-file format and evaluation scheme.

    Parameters
    ----------
    gwfast_result_file : str
        Path to the gwfast Fisher-covariance result HDF5 file.
    snr_threshold : float, optional
        Additional SNR cut on top of whatever detection threshold is already baked
        into ``gwfast_result_file`` (default: ``0.0``, i.e. no extra cut).
    quality_cut_n_sigma : float, optional
        Data-quality cut (default: ``0.0``, disabled): exclude sources where
        ``Mc_src``, ``eta``, or ``LambdaTilde`` isn't significantly positive relative
        to its own Fisher error -- see
        :func:`~jesterTOV.inference.likelihoods.gw_fisher._positivity_quality_mask`.
        Not applied to ``deltaLambda``, which is physically allowed to be negative.
    min_eta_variance : float, optional
        Data-quality cut (default: ``0.0``, disabled): exclude sources whose Fisher
        ``Var(eta)`` is below this floor -- see
        :func:`_minimum_variance_quality_mask` for the mechanism this guards against
        (Fisher-matrix linearization breakdown near the :math:`\eta=0.25` equal-mass
        boundary, which can report an artificially over-precise ``eta``, turning that
        source into a near-delta-function constraint that can stall an MCMC sampler).
    penalty_value : float, optional
        Log-likelihood penalty applied when a sampled component mass exceeds
        :math:`M_{\rm TOV}` of the candidate EOS (default: ``0.0``, i.e. no penalty).
    n_mass_samples : int, optional
        Number of pre-sampled ``(m1, m2)`` pairs per source (default: ``500``).
        Larger values reduce Monte Carlo estimator noise at the cost of proportionally
        more likelihood evaluations. Sampled via rejection sampling (see
        :func:`_sample_masses`) -- poorly-measured sources may end up with fewer than
        this many accepted draws; ``evaluate()`` accounts for this per source rather
        than assuming every source reaches the full count.
    mass_rejection_pool_size : int, optional
        Raw ``(Mc_src,eta)`` draws per source per rejection-sampling round
        (default: ``2000``). See :func:`_sample_masses`.
    mass_rejection_max_rounds : int, optional
        Maximum rejection-sampling rounds for sources still short of
        ``n_mass_samples`` accepted draws (default: ``6``). See :func:`_sample_masses`.
    source_batch_size : int, optional
        Batch size for ``jax.lax.map`` over sources (default: ``1``, a plain scan --
        keeps memory flat under the outer particle ``vmap`` used by e.g. the SMC
        sampler).
    mass_batch_size : int, optional
        Batch size for ``jax.lax.map`` over the pre-sampled mass pairs (default: ``1``).
    seed : int, optional
        Random seed for mass pre-sampling (default: ``42``). Fixed seed ensures
        reproducibility and deterministic likelihood evaluation across sampler steps.

    Attributes
    ----------
    n_sources : int
        Number of sources retained after the SNR and data-quality cuts.
    """

    gwfast_result_file: str
    snr_threshold: float
    penalty_value: float
    n_mass_samples: int
    mass_rejection_pool_size: int
    mass_rejection_max_rounds: int
    source_batch_size: int
    mass_batch_size: int
    seed: int
    quality_cut_n_sigma: float
    min_eta_variance: float
    n_sources: int

    def __init__(
        self,
        gwfast_result_file: str,
        snr_threshold: float = 0.0,
        min_eta_variance: float = 0.0,
        penalty_value: float = 0.0,
        n_mass_samples: int = 500,
        mass_rejection_pool_size: int = 2000,
        mass_rejection_max_rounds: int = 6,
        source_batch_size: int = 1,
        mass_batch_size: int = 1,
        seed: int = 42,
        quality_cut_n_sigma: float = 0.0,
    ) -> None:
        super().__init__()

        self.gwfast_result_file = gwfast_result_file
        self.snr_threshold = snr_threshold
        self.min_eta_variance = min_eta_variance
        self.penalty_value = penalty_value
        self.n_mass_samples = n_mass_samples
        self.mass_rejection_pool_size = mass_rejection_pool_size
        self.mass_rejection_max_rounds = mass_rejection_max_rounds
        self.source_batch_size = source_batch_size
        self.mass_batch_size = mass_batch_size
        self.seed = seed
        self.quality_cut_n_sigma = quality_cut_n_sigma

        logger.info(f"Loading gwfast Fisher-covariance data from {gwfast_result_file}")
        data = _load_gwfast_covariance_result(gwfast_result_file)
        key_order = data["key_order"]
        detected_params = data["detected_params"]
        covariance = data["covariance"]
        snrs = data["snrs"]

        _validate_component_mass_ordering(
            detected_params["m1_src"], detected_params["m2_src"], gwfast_result_file
        )

        n_detected = detected_params["m1_src"].shape[0]
        mask_snr = snrs >= snr_threshold
        if not np.any(mask_snr):
            raise ValueError(
                f"0 of {n_detected} detected sources in {gwfast_result_file} have "
                f"SNR >= {snr_threshold}; max SNR in file is {float(np.max(snrs))}."
            )
        n_excluded_snr = n_detected - int(np.sum(mask_snr))
        if n_excluded_snr > 0:
            logger.info(
                f"SNR cut: excluding {n_excluded_snr}/{n_detected} sources with "
                f"SNR < {snr_threshold}"
            )

        means = _build_means(detected_params, key_order)
        i_mc, i_eta, i_lt = (
            key_order["Mc_src"],
            key_order["eta"],
            key_order["LambdaTilde"],
        )
        err_mc_src = np.sqrt(covariance[:, i_mc, i_mc])
        err_eta = np.sqrt(covariance[:, i_eta, i_eta])
        err_lambda_tilde = np.sqrt(covariance[:, i_lt, i_lt])
        mask_quality = (
            _positivity_quality_mask(
                means[:, i_mc], err_mc_src, quality_cut_n_sigma, "Mc_src"
            )
            & _positivity_quality_mask(
                means[:, i_eta], err_eta, quality_cut_n_sigma, "eta"
            )
            & _positivity_quality_mask(
                means[:, i_lt], err_lambda_tilde, quality_cut_n_sigma, "LambdaTilde"
            )
            & _minimum_variance_quality_mask(
                covariance[:, i_eta, i_eta], min_eta_variance, "eta"
            )
        )
        mask = mask_snr & mask_quality
        if not np.any(mask):
            raise ValueError(
                f"0 of {n_detected} detected sources in {gwfast_result_file} survive "
                f"the combined SNR (>= {snr_threshold}), {quality_cut_n_sigma}-sigma "
                f"positivity, and min_eta_variance={min_eta_variance:.3e} data-quality "
                "cuts."
            )

        self.n_sources = int(np.sum(mask))
        self._i_mc = i_mc
        self._i_eta = i_eta
        self._i_lt = i_lt
        self._i_dlt = key_order["deltaLambda"]
        self._means: Float[Array, "n_sources 4"] = jnp.asarray(means[mask])
        self._covs: Float[Array, "n_sources 4 4"] = jnp.asarray(covariance[mask])

        logger.info(
            f"Pre-sampling {n_mass_samples} (m1,m2) pairs per source, seed={seed}"
        )
        mass_samples, n_accepted = _sample_masses(
            means[mask],
            covariance[mask],
            key_order,
            n_mass_samples,
            seed,
            pool_size=mass_rejection_pool_size,
            max_rounds=mass_rejection_max_rounds,
        )
        self._mass_samples: Float[Array, "n_sources n_mass_samples 2"] = jnp.asarray(
            mass_samples
        )
        self._n_accepted: Float[Array, " n_sources"] = jnp.asarray(n_accepted)

        logger.info(
            f"GWFisherLikelihoodV2: {n_detected} detected sources, "
            f"{self.n_sources} retained after snr_threshold={snr_threshold}, "
            f"{quality_cut_n_sigma}-sigma positivity, and min_eta_variance="
            f"{min_eta_variance:.3e} data-quality cuts "
            f"({n_detected - self.n_sources} excluded total), "
            f"n_mass_samples={n_mass_samples} (rejection-sampled, pool_size="
            f"{mass_rejection_pool_size}, max_rounds={mass_rejection_max_rounds}), "
            f"source_batch_size={source_batch_size}, mass_batch_size={mass_batch_size}"
        )

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """
        Evaluate summed log likelihood over all retained sources for given EOS parameters.

        For each source, Monte-Carlo-averages the source's full 4D Gaussian density
        (real Fisher covariance, including cross-correlations), evaluated at
        ``(Mc_src, eta, LambdaTilde_X, deltaLambdaTilde_X)`` for each pre-sampled
        ``(m1, m2)`` pair, then sums the resulting per-source log-likelihoods.

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
        slot_idx = jnp.arange(n_mass_samples)
        i_mc, i_eta, i_lt, i_dlt = self._i_mc, self._i_eta, self._i_lt, self._i_dlt

        def process_one_source(carry) -> Float:
            mean, cov, mass_samples, n_accepted = carry

            def process_one_sample(mass_pair: Float[Array, " 2"]) -> Float:
                m1, m2 = mass_pair[0], mass_pair[1]
                lambda_1 = jnp.interp(m1, masses_EOS, Lambdas_EOS, right=1.0)
                lambda_2 = jnp.interp(m2, masses_EOS, Lambdas_EOS, right=1.0)
                eta = utils.symmetric_mass_ratio_from_mass_ratio(m2 / m1)
                mc_src = utils.chirp_mass_from_component_masses(m1, m2)
                lambda_tilde = utils.lambda_tilde_from_lambda1_lambda2(
                    lambda_1, lambda_2, eta
                )
                delta_lambda = utils.delta_lambda_tilde_from_lambda1_lambda2(
                    lambda_1, lambda_2, eta
                )
                point = (
                    jnp.zeros(4)
                    .at[i_mc]
                    .set(mc_src)
                    .at[i_eta]
                    .set(eta)
                    .at[i_lt]
                    .set(lambda_tilde)
                    .at[i_dlt]
                    .set(delta_lambda)
                )
                logpdf = multivariate_normal.logpdf(point, mean, cov)
                penalty_m1 = jnp.where(m1 > mtov, self.penalty_value, 0.0)
                penalty_m2 = jnp.where(m2 > mtov, self.penalty_value, 0.0)
                return logpdf + penalty_m1 + penalty_m2

            all_logprobs = jax.lax.map(
                process_one_sample, mass_samples, batch_size=self.mass_batch_size
            )
            # Slots beyond this source's accepted count hold a dummy-valid mass pair
            # (see _sample_masses) -- mask them out rather than let them silently
            # contribute a meaningless logpdf, and normalize by the true accepted
            # count (n_accepted), not the nominal n_mass_samples: a source that only
            # produced e.g. 65/200 accepted draws (poorly-measured source, low
            # acceptance probability) is a smaller, honest Monte Carlo average, not
            # padded out to look like a full-precision one.
            all_logprobs = jnp.where(slot_idx < n_accepted, all_logprobs, -jnp.inf)
            return logsumexp(all_logprobs) - jnp.log(n_accepted)

        per_source_loglike = jax.lax.map(
            process_one_source,
            (self._means, self._covs, self._mass_samples, self._n_accepted),
            batch_size=self.source_batch_size,
        )
        return jnp.sum(per_source_loglike)
