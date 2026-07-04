"""Tests for the strangeon matter equation of state (self-bound strangeon stars).

Reference parameter set matches CompactObject's own example, see
CompactObject/Test_Case/test_Bayesian_inference_Strangeon_EOS.ipynb (Nq=18 fixed,
epsilon in [10, 170] MeV, ns in [0.17, 0.36] fm^-3).
"""

import jax
import jax.numpy as jnp
import pytest

from jesterTOV.eos.strangeon import StrangeonEOS_model
from jesterTOV.tov.gr import GRTOVSolver


@pytest.fixture
def strangeon_params() -> dict[str, float]:
    return {"epsilon": 40.0, "ns": 0.3}


class TestStrangeonEOSModel:
    def test_get_required_parameters(self):
        model = StrangeonEOS_model()
        assert set(model.get_required_parameters()) == {"epsilon", "ns"}

    def test_construct_eos_shapes_and_causality(self, strangeon_params):
        model = StrangeonEOS_model()
        eos_data = model.construct_eos(strangeon_params)

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

    def test_baryon_density_conversion(self, strangeon_params):
        """jester's ns field must be Nq/3 times the strangeon number density used
        internally (see module docstring re: ns vs n baryon/strangeon distinction)."""
        from jesterTOV import utils

        Nq = 18
        model = StrangeonEOS_model(Nq=Nq)
        eos_data = model.construct_eos(strangeon_params)

        ns_param = strangeon_params["ns"]
        n_min_strangeon = 3.0 * ns_param / Nq
        n_min_baryon_expected = n_min_strangeon * (Nq / 3.0)

        assert jnp.allclose(
            eos_data.ns[0] / utils.fm_inv3_to_geometric,
            n_min_baryon_expected,
            rtol=1e-6,
        )

    def test_construct_eos_missing_parameter_raises(self, strangeon_params):
        model = StrangeonEOS_model()
        incomplete = dict(strangeon_params)
        del incomplete["ns"]
        with pytest.raises(KeyError):
            model.construct_eos(incomplete)

    def test_construct_eos_jit_compatible(self, strangeon_params):
        model = StrangeonEOS_model()
        construct = jax.jit(model.construct_eos)
        eos_data = construct(strangeon_params)
        assert jnp.all(jnp.isfinite(eos_data.es))

    def test_construct_eos_vmap_compatible(self, strangeon_params):
        model = StrangeonEOS_model()
        batch_params = {
            k: jnp.array([v, v * 1.05, v * 0.95]) for k, v in strangeon_params.items()
        }
        construct = jax.jit(jax.vmap(model.construct_eos))
        eos_data = construct(batch_params)
        assert eos_data.ns.shape[0] == 3
        assert jnp.all(jnp.isfinite(eos_data.es))

    def test_no_nan_near_surface_across_prior_range(self):
        """Regression test for catastrophic cancellation at the surface density.

        p(n) is the difference of two closely-balanced terms designed to cancel
        exactly to zero at n=n_min (the surface density). Evaluating exactly at that
        root is subject to catastrophic cancellation: the residual floating-point
        noise can land on either side of zero, and a spuriously negative p[0]
        poisons this class's log(p)-based derivatives (dloge_dlogps, hence cs2) with
        NaN for some (epsilon, ns) draws (found via a real posterior sample, see
        rmf_eos/FINDINGS.md). Fixed by starting the density grid a hair above
        n_min; this test scans a grid over the full prior box as a regression guard.
        """
        for epsilon in jnp.linspace(10.0, 170.0, 10):
            for ns in jnp.linspace(0.17, 0.36, 10):
                model = StrangeonEOS_model()
                eos_data = model.construct_eos({"epsilon": epsilon, "ns": ns})

                assert jnp.all(eos_data.ps > 0), f"ps<=0 at epsilon={epsilon}, ns={ns}"
                assert jnp.all(eos_data.hs > 0), f"hs<=0 at epsilon={epsilon}, ns={ns}"
                assert not jnp.any(
                    jnp.isnan(eos_data.cs2)
                ), f"NaN cs2 at epsilon={epsilon}, ns={ns}"
                assert not jnp.any(
                    jnp.isnan(eos_data.dloge_dlogps)
                ), f"NaN dloge_dlogps at epsilon={epsilon}, ns={ns}"

    def test_construct_family_smoke(self, strangeon_params):
        """End-to-end smoke test: strangeon EOS plugs into the TOV solver and
        produces a valid mass-radius-tidal family.

        Self-bound strangeon matter has no crust: the tabulated density floor is
        the surface baryon density `ns` itself, which already exceeds nuclear
        saturation density in the reference implementation's own prior range
        (0.17-0.36 fm^-3) -- `min_nsat` must be large enough that `min_nsat * 0.16
        fm^-3` falls inside the table.
        """
        model = StrangeonEOS_model()
        eos_data = model.construct_eos(strangeon_params)

        solver = GRTOVSolver()
        family = solver.construct_family(eos_data, ndat=20, min_nsat=2.5)

        assert jnp.all(jnp.isfinite(family.masses))
        assert jnp.all(jnp.isfinite(family.radii))
        assert jnp.all(jnp.isfinite(family.lambdas))
        assert (
            jnp.max(family.masses) > 0.1
        ), "should produce at least some finite-mass stars"
