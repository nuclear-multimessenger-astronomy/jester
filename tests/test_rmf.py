"""Tests for the RMF (relativistic mean-field) equation of state.

Reference parameter set matches CompactObject's own example (FSU2-like nucleonic RMF),
see CompactObject/Test_Case/test_EOSgenerators.ipynb and rmf_eos/data/reference_rmf_theta.csv
in the companion validation repository. Numerical parity against the reference
implementation is validated separately in rmf_eos/scripts/ (outside this repo), since that
requires importing CompactObject; these tests check jester-side correctness only
(shapes, causality, monotonicity, jit/vmap compatibility, and parameter validation).
"""

import math

import jax
import jax.numpy as jnp
import pytest

from jesterTOV.eos.rmf import RMF_EOS_model
from jesterTOV.tov.gr import GRTOVSolver


@pytest.fixture
def rmf_params() -> dict[str, float]:
    """FSU2-like reference parameter set from CompactObject's own example notebook."""
    return {
        "m_sig": 495.0 / 197.33,
        "m_w": 3.96544,
        "m_rho": 3.86662,
        "g_sigma": math.sqrt(107.5751),
        "g_omega": math.sqrt(182.3949),
        "g_rho": math.sqrt(206.4260),
        "kappa": 3.09114168 / 197.33,
        "lambda_0": -0.00168015405,
        "zeta": 0.024,
        "Lambda_w": 0.045,
    }


class TestRMFEOSModel:
    def test_get_required_parameters(self):
        model = RMF_EOS_model()
        params = model.get_required_parameters()
        assert set(params) == {
            "m_sig",
            "m_w",
            "m_rho",
            "g_sigma",
            "g_omega",
            "g_rho",
            "kappa",
            "lambda_0",
            "zeta",
            "Lambda_w",
        }

    def test_construct_eos_shapes_and_causality(self, rmf_params):
        model = RMF_EOS_model()
        eos_data = model.construct_eos(rmf_params)

        n = len(eos_data.ns)
        assert n > 0
        assert len(eos_data.ps) == n
        assert len(eos_data.hs) == n
        assert len(eos_data.es) == n
        assert len(eos_data.dloge_dlogps) == n
        assert len(eos_data.cs2) == n

        assert jnp.all(jnp.isfinite(eos_data.ns))
        assert jnp.all(jnp.isfinite(eos_data.ps))
        assert jnp.all(jnp.isfinite(eos_data.es))
        assert jnp.all(jnp.isfinite(eos_data.cs2))

        assert jnp.all(eos_data.ns > 0)
        assert jnp.all(eos_data.ps > 0)
        assert jnp.all(eos_data.es > 0)
        assert jnp.all(eos_data.cs2 >= 0.0)
        assert jnp.all(eos_data.cs2 <= 1.0)

        assert jnp.all(jnp.diff(eos_data.ns) > 0), "n must be monotonically increasing"
        assert jnp.all(jnp.diff(eos_data.es) > 0), "e must be monotonically increasing"
        assert jnp.all(jnp.diff(eos_data.ps) > 0), "p must be monotonically increasing"

        assert eos_data.extra_constraints is not None
        assert int(eos_data.extra_constraints["n_root_solver_failures"]) == 0

    def test_construct_eos_missing_parameter_raises(self, rmf_params):
        model = RMF_EOS_model()
        incomplete = dict(rmf_params)
        del incomplete["kappa"]
        with pytest.raises(KeyError):
            model.construct_eos(incomplete)

    def test_construct_eos_jit_compatible(self, rmf_params):
        model = RMF_EOS_model()
        construct = jax.jit(model.construct_eos)
        eos_data = construct(rmf_params)
        assert jnp.all(jnp.isfinite(eos_data.es))

    def test_construct_eos_vmap_compatible(self, rmf_params):
        model = RMF_EOS_model()
        batch_params = {
            k: jnp.array([v, v * 1.01, v * 0.99]) for k, v in rmf_params.items()
        }
        construct = jax.jit(jax.vmap(model.construct_eos))
        eos_data = construct(batch_params)
        assert eos_data.ns.shape[0] == 3
        assert jnp.all(jnp.isfinite(eos_data.es))
        assert jnp.all(eos_data.extra_constraints["n_root_solver_failures"] == 0)

    def test_construct_family_smoke(self, rmf_params):
        """End-to-end smoke test: RMF EOS plugs into the TOV solver and produces a
        valid mass-radius-tidal family."""
        model = RMF_EOS_model()
        eos_data = model.construct_eos(rmf_params)

        solver = GRTOVSolver()
        family = solver.construct_family(eos_data, ndat=20, min_nsat=0.75)

        assert jnp.all(jnp.isfinite(family.masses))
        assert jnp.all(jnp.isfinite(family.radii))
        assert jnp.all(jnp.isfinite(family.lambdas))
        assert (
            jnp.max(family.masses) > 0.5
        ), "should produce at least sub-solar-mass stars"
