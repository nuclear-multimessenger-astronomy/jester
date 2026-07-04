"""Tests for the MIT bag model equation of state (self-bound strange quark matter).

Reference EOS: p = (e - 4B)/3, see CompactObject/EOSgenerators/MITbag_EOS.py and
Test_Case/test_Bayesian_inference_MITbag_EOS.ipynb (B sampled in [20, 100] MeV/fm^3).
"""

import jax
import jax.numpy as jnp
import pytest

from jesterTOV.eos.mit_bag import MITBag_EOS_model
from jesterTOV.tov.gr import GRTOVSolver


@pytest.fixture
def mit_bag_params() -> dict[str, float]:
    return {"B": 60.0}


class TestMITBagEOSModel:
    def test_get_required_parameters(self):
        model = MITBag_EOS_model()
        assert model.get_required_parameters() == ["B"]

    def test_construct_eos_shapes_and_causality(self, mit_bag_params):
        model = MITBag_EOS_model()
        eos_data = model.construct_eos(mit_bag_params)

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
        assert jnp.all(eos_data.ps >= 0)
        assert jnp.all(eos_data.es > 0)
        assert jnp.allclose(eos_data.cs2, 1.0 / 3.0)

        assert jnp.all(jnp.diff(eos_data.ns) > 0), "n must be monotonically increasing"
        assert jnp.all(jnp.diff(eos_data.es) > 0), "e must be monotonically increasing"
        assert jnp.all(jnp.diff(eos_data.ps) > 0), "p must be monotonically increasing"

    def test_pressure_energy_relation(self, mit_bag_params):
        """p = (e - 4B)/3 should hold exactly on the underlying (nuclear-unit) grid,
        with e starting a hair above 4B (see module docstring: the exact surface
        point p=0 is avoided since downstream TOV code interpolates in log(p))."""
        from jesterTOV import utils

        model = MITBag_EOS_model()
        B = mit_bag_params["B"]

        eos_data = model.construct_eos(mit_bag_params)
        p_nuclear = eos_data.ps / utils.MeV_fm_inv3_to_geometric
        e_nuclear = eos_data.es / utils.MeV_fm_inv3_to_geometric

        assert jnp.allclose(p_nuclear, (e_nuclear - 4.0 * B) / 3.0, rtol=1e-8)
        assert jnp.isclose(e_nuclear[0], 4.0 * B, rtol=1e-6)
        assert p_nuclear[0] > 0.0

    def test_construct_eos_missing_parameter_raises(self):
        model = MITBag_EOS_model()
        with pytest.raises(KeyError):
            model.construct_eos({})

    def test_construct_eos_jit_compatible(self, mit_bag_params):
        model = MITBag_EOS_model()
        construct = jax.jit(model.construct_eos)
        eos_data = construct(mit_bag_params)
        assert jnp.all(jnp.isfinite(eos_data.es))

    def test_construct_eos_vmap_compatible(self, mit_bag_params):
        model = MITBag_EOS_model()
        batch_params = {
            k: jnp.array([v, v * 1.1, v * 0.9]) for k, v in mit_bag_params.items()
        }
        construct = jax.jit(jax.vmap(model.construct_eos))
        eos_data = construct(batch_params)
        assert eos_data.ns.shape[0] == 3
        assert jnp.all(jnp.isfinite(eos_data.es))

    def test_hs_positive_eager_and_jit_match(self):
        """Regression test for a compilation-path-dependent NaN-poisoning bug.

        The pseudo-enthalpy hs[0] should be a tiny positive placeholder (see
        construct_eos), but was originally implemented as `1e-30 + log(ratio)` where
        `ratio` is exactly 1 in exact arithmetic. Under jax.jit, floating-point
        non-associativity between how `(e - B)` and `(e_min - B)` get computed made
        `ratio` land a few ULPs below 1, giving `log(ratio)` a small *negative* value
        that swamped the `1e-30` additive guard -- jester's `hs > 0` validity check
        then poisoned the entire EOS table with NaN for a compilation-path-dependent
        fraction of `B` draws (see rmf_eos/FINDINGS.md). Fixed with jnp.maximum
        instead of addition; this test scans many B values in both eager and jit
        execution to guard against regressions.
        """
        model = MITBag_EOS_model()
        construct_jit = jax.jit(model.construct_eos)

        for B in jnp.linspace(20.0, 100.0, 50):
            eos_eager = model.construct_eos({"B": B})
            eos_jit = construct_jit({"B": B})

            assert jnp.all(eos_eager.hs > 0), f"eager hs<=0 at B={B}"
            assert jnp.all(eos_jit.hs > 0), f"jit hs<=0 at B={B}"
            assert not jnp.any(jnp.isnan(eos_jit.es)), f"jit NaN es at B={B}"
            assert not jnp.any(jnp.isnan(eos_jit.ps)), f"jit NaN ps at B={B}"

    def test_construct_family_smoke(self, mit_bag_params):
        """End-to-end smoke test: MIT bag EOS plugs into the TOV solver and produces
        a valid mass-radius-tidal family.

        Self-bound quark matter has no crust: the tabulated density floor is set by
        the bag constant itself (n ~ 4B/m_N), not by a small fraction of nuclear
        saturation density -- `min_nsat` must be large enough that `min_nsat * 0.16
        fm^-3` falls inside the table, matching the reference implementation's own
        density range 20-100 MeV/fm^3 for B (see test_Bayesian_inference_MITbag_EOS.ipynb).
        """
        model = MITBag_EOS_model()
        eos_data = model.construct_eos(mit_bag_params)

        solver = GRTOVSolver()
        family = solver.construct_family(eos_data, ndat=20, min_nsat=2.0)

        assert jnp.all(jnp.isfinite(family.masses))
        assert jnp.all(jnp.isfinite(family.radii))
        assert jnp.all(jnp.isfinite(family.lambdas))
        assert jnp.max(family.masses) > 0.5, "should produce at least sub-solar-mass stars"
