r"""Memory-efficient per-NS anisotropy likelihood wrapper."""

import jax.numpy as jnp
from jaxtyping import Array, Float

from jesterTOV.tov.base import TOVSolverBase
from jesterTOV.tov.data_classes import EOSData
from jesterTOV.inference.base.likelihood import LikelihoodBase
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class IndividualGammaLikelihood(LikelihoodBase):
    r"""Evaluates per-NS likelihoods with individually solved TOV families.

    For each NS source the TOV equations are solved once with that source's
    own anisotropy parameters.  The resulting family is immediately consumed
    by the inner likelihood, so XLA can reuse the same memory buffer
    sequentially.

    GW events are handled as pairs: two families (mass1, mass2) are computed
    and injected as ``masses_EOS`` / ``Lambdas_EOS`` (primary) and
    ``masses_EOS_2`` / ``Lambdas_EOS_2`` (secondary) before calling the
    inner GW likelihood.

    The two ``for`` loops in :meth:`evaluate` are over static Python lists
    and are unrolled at JIT trace time — there is no Python-level branching
    at runtime.

    Parameters
    ----------
    ns_sources : list
        List of :class:`~jesterTOV.inference.run_inference.NSSource` instances.
    ns_source_to_likelihood : dict[str, LikelihoodBase]
        Maps each ``ns_key`` (radio/NICER) or ``event_name`` (GW) to its
        inner likelihood object.
    tov_solver : TOVSolverBase
        TOV solver instance shared across all NS sources.
    sampled_aniso_params : list[str]
        Names of anisotropy parameters that are sampled (not fixed).
    fixed_aniso_params : dict[str, float]
        Anisotropy parameters pinned to constant values.
    ndat_TOV : int
        Number of central pressure points for the M-R-Λ family.
    min_nsat_TOV : float
        Minimum density for TOV integration in units of nsat.
    """

    def __init__(
        self,
        ns_sources: list,
        ns_source_to_likelihood: dict[str, LikelihoodBase],
        tov_solver: TOVSolverBase,
        sampled_aniso_params: list[str],
        fixed_aniso_params: dict[str, float],
        ndat_TOV: int = 100,
        min_nsat_TOV: float = 0.75,
    ) -> None:
        super().__init__()
        self.tov_solver = tov_solver
        self.sampled_aniso_params = sampled_aniso_params
        self.fixed_aniso_params = fixed_aniso_params
        self.ndat_TOV = ndat_TOV
        self.min_nsat_TOV = min_nsat_TOV

        # Pre-split NS sources into radio/NICER vs. GW at init.
        self._radio_nicer: list[tuple[str, LikelihoodBase]] = []
        seen_gw: set[str] = set()
        self._gw_events: list[tuple[str, LikelihoodBase]] = []

        for src in ns_sources:
            if src.lk_type in ("radio", "nicer", "mock_mr"):
                lk = ns_source_to_likelihood[src.ns_key]
                self._radio_nicer.append((src.ns_key, lk))
            elif src.lk_type == "gw" and src.event_name not in seen_gw:
                seen_gw.add(src.event_name)
                lk = ns_source_to_likelihood[src.event_name]
                self._gw_events.append((src.event_name, lk))

    def _reconstruct_eos_data(self, params: dict) -> EOSData:
        return EOSData(
            ns=params["n"],
            ps=params["p"],
            hs=params["h"],
            es=params["e"],
            dloge_dlogps=params["dloge_dlogp"],
            cs2=params["cs2"],
        )

    def _build_tov_params(self, params: dict, ns_key: str) -> dict:
        tov: dict = {p: params[f"{p}_{ns_key}"] for p in self.sampled_aniso_params}
        tov.update(self.fixed_aniso_params)
        return tov

    def evaluate(self, params: dict[str, Float | Array]) -> Float:
        """Evaluate the summed log-likelihood over all NS sources.

        Parameters
        ----------
        params : dict[str, Float | Array]
            Must contain the EOS arrays (``n``, ``p``, ``h``, ``e``,
            ``dloge_dlogp``, ``cs2``) produced by :class:`JesterTransform`,
            plus one per-NS anisotropy value for each sampled parameter
            (e.g. ``lambda_DY_J0030``).

        Returns
        -------
        Float
            Sum of all per-NS log-likelihoods.
        """
        eos_data = self._reconstruct_eos_data(params)
        total: Float = jnp.array(0.0)

        for ns_key, lk in self._radio_nicer:
            family = self.tov_solver.construct_family(
                eos_data,
                ndat=self.ndat_TOV,
                min_nsat=self.min_nsat_TOV,
                tov_params=self._build_tov_params(params, ns_key),
            )
            inner_params = {
                **params,
                "masses_EOS": family.masses,
                "radii_EOS": family.radii,
                "Lambdas_EOS": family.lambdas,
            }
            total = total + lk.evaluate(inner_params)

        for event_name, lk in self._gw_events:
            family_1 = self.tov_solver.construct_family(
                eos_data,
                ndat=self.ndat_TOV,
                min_nsat=self.min_nsat_TOV,
                tov_params=self._build_tov_params(params, f"{event_name}_mass1"),
            )
            family_2 = self.tov_solver.construct_family(
                eos_data,
                ndat=self.ndat_TOV,
                min_nsat=self.min_nsat_TOV,
                tov_params=self._build_tov_params(params, f"{event_name}_mass2"),
            )
            inner_params = {
                **params,
                "masses_EOS": family_1.masses,
                "Lambdas_EOS": family_1.lambdas,
                "masses_EOS_2": family_2.masses,
                "Lambdas_EOS_2": family_2.lambdas,
            }
            total = total + lk.evaluate(inner_params)

        return total
