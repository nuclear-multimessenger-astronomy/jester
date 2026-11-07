"""Tests of the modax backend of the GR TOV solver against the diffrax solver.

The right-hand side test runs anywhere, since the device function is plain Python.
The solver tests need a CUDA GPU and the optional modax dependency, and skip otherwise.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jesterTOV.eos.metamodel import MetaModel_with_CSE_EOS_model
from jesterTOV.tov.gr import GRTOVSolver, _tov_ode
from jesterTOV.tov.gr_modax import (
    ERROR_WEIGHTS,
    N_COLUMNS,
    N_HEADER,
    N_STATE,
    make_tov_rhs,
    pack_table,
)

jax.config.update("jax_enable_x64", True)

NB_CSE = 8


@pytest.fixture(scope="module")
def eos_model():
    return MetaModel_with_CSE_EOS_model(
        nmax_nsat=25.0, ndat_metamodel=100, ndat_CSE=100, nb_CSE=NB_CSE
    )


def _params(rng_seed: int) -> dict:
    """A draw from the prior used in the production runs, away from its edges."""
    rng = np.random.default_rng(rng_seed)
    params = {
        "E_sat": -16.0,
        "K_sat": rng.uniform(180.0, 280.0),
        "Q_sat": rng.uniform(-300.0, 300.0),
        "Z_sat": rng.uniform(-500.0, 500.0),
        "E_sym": rng.uniform(30.0, 36.0),
        "L_sym": rng.uniform(40.0, 90.0),
        "K_sym": rng.uniform(-200.0, 100.0),
        "Q_sym": rng.uniform(-300.0, 500.0),
        "Z_sym": rng.uniform(-500.0, 500.0),
        "nbreak": rng.uniform(0.18, 0.3),
    }
    for i in range(NB_CSE):
        params[f"n_CSE_{i}_u"] = rng.uniform(0.0, 1.0)
        params[f"cs2_CSE_{i}"] = rng.uniform(0.2, 0.9)
    params[f"cs2_CSE_{NB_CSE}"] = rng.uniform(0.2, 0.9)
    return params


def _usable_eos(eos_model, seed_start: int = 0):
    """First draw with a physical EOS, so the tests exercise a real family."""
    for seed in range(seed_start, seed_start + 200):
        eos = eos_model.construct_eos(_params(seed))
        if (
            bool(jnp.all(eos.cs2 > 0))
            and bool(jnp.all(eos.cs2 <= 1))
            and bool(jnp.all(eos.ps > 0))
        ):
            return eos
    raise RuntimeError("no physical EOS among the draws")


def _bisection_tov_rhs(n_table):
    """Frozen copy of the right-hand side with the original while-loop bisection, the reference for the search."""
    import math

    stride, offset, last, four_pi = N_COLUMNS, N_HEADER, n_table - 1, 4.0 * math.pi

    def tov_rhs(y, s, p):
        h0 = y[4]
        if h0 == 0.0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        r, m, H, b = y[0], y[1], y[2], y[3]
        h = h0 * (1.0 - s)
        h_first = p[offset + 3]
        if h < h_first:
            h = h_first
        lo, hi = 0, last
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if p[offset + stride * mid + 3] <= h:
                lo = mid
            else:
                hi = mid
        ilo = offset + stride * lo
        ihi = offset + stride * hi
        frac_log = (math.log(h) - p[ilo + 0]) / (p[ihi + 0] - p[ilo + 0])
        frac = (h - p[ilo + 3]) / (p[ihi + 3] - p[ilo + 3])
        e = math.exp(p[ilo + 1] + frac_log * (p[ihi + 1] - p[ilo + 1]))
        pres = math.exp(p[ilo + 2] + frac_log * (p[ihi + 2] - p[ilo + 2]))
        dloge = p[ilo + 4] + frac * (p[ihi + 4] - p[ilo + 4])
        dedp = e / pres * dloge
        A = 1.0 / (1.0 - 2.0 * m / r)
        C1 = 2.0 / r + A * (2.0 * m / (r * r) + four_pi * r * (pres - e))
        term = 2.0 * (m + four_pi * r * r * r * pres) / (r * (r - 2.0 * m))
        C0 = A * (-6.0 / (r * r) + four_pi * (e + pres) * dedp + four_pi * (5.0 * e + 9.0 * pres)) - term * term
        drdh = -r * (r - 2.0 * m) / (m + four_pi * r * r * r * pres)
        dmdh = four_pi * r * r * e * drdh
        dHdh = b * drdh
        dbdh = -(C0 * H + C1 * b) * drdh
        return -h0 * drdh, -h0 * dmdh, -h0 * dHdh, -h0 * dbdh, 0.0

    return tov_rhs


class TestRightHandSide:
    """The device function is plain Python, so it can be compared with `_tov_ode` here."""

    def test_matches_tov_ode(self, eos_model):
        eos = _usable_eos(eos_model)
        n_table = eos.hs.shape[0]
        rhs = make_tov_rhs(n_table)
        eos_dict = {
            "p": eos.ps,
            "h": eos.hs,
            "e": eos.es,
            "dloge_dlogp": eos.dloge_dlogps,
        }
        hmax = float(eos.hs[-1])
        # One row per EOS: the table and nothing else. The start enthalpy h0 of the star is the
        # fifth state component.
        row = np.asarray(pack_table(eos))
        assert row.shape[0] == N_HEADER + N_COLUMNS * n_table
        # An enthalpy inside the table, one below its first node, and the surface.
        for frac in [0.9, 0.5, 0.123, 0.01, 1e-3, 1e-5, 0.0]:
            h0 = hmax * 0.97
            h = hmax * frac
            s = 1.0 - h / h0
            y = (1.0e4 * (1.1 - frac), 1.5e3 * (1.1 - frac), 3.0e7, 2.0e3)
            expected = jax.tree_util.tree_map(
                float, _tov_ode(jnp.asarray(h), y, eos_dict)
            )
            got = rhs(y + (h0,), s, row)
            assert len(got) == N_STATE
            # dy/ds = -h0 dy/dh, and h0 itself is constant
            for g, e in zip(got[:4], expected):
                assert g == pytest.approx(-h0 * e, rel=1e-11)
            assert got[4] == 0.0

    def test_bracket_search_matches_the_reference_bisection(self, eos_model):
        """The node search must return exactly the bracket of the plain bisection it replaced, for every enthalpy:
        inside the table, on nodes, below the first node (clamped) and at or above the last node."""
        from jesterTOV.tov.gr_modax import _bracket_lo

        eos = _usable_eos(eos_model)
        hs = np.asarray(eos.hs)
        last = len(hs) - 1

        def reference(h):  # the bisection of the first modax version
            lo, hi = 0, last
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if hs[mid] <= h:
                    lo = mid
                else:
                    hi = mid
            return lo

        rng = np.random.default_rng(1)
        candidates = np.concatenate(
            [
                hs,  # exactly on every node
                np.nextafter(hs, -np.inf),
                np.nextafter(hs, np.inf),
                rng.uniform(hs[0], hs[-1], 5000),
                np.exp(rng.uniform(np.log(hs[0]), np.log(hs[-1]), 5000)),  # the crust spans ~30 decades
                [hs[-1] * 1.5, hs[-1] * 10],
            ]
        )
        search = _bracket_lo(len(hs))
        for h in np.maximum(candidates, hs[0]):  # the RHS clamps h to the first node
            assert search(hs, h) == reference(h), h

    def test_rhs_is_bit_identical_to_the_bisection_version(self, eos_model):
        """Only the node search changed, so the right-hand side must return exactly the same numbers as before."""
        eos = _usable_eos(eos_model)
        n_table = eos.hs.shape[0]
        row = np.asarray(pack_table(eos))
        new = make_tov_rhs(n_table)
        old = _bisection_tov_rhs(n_table)
        hs = np.asarray(eos.hs)
        rng = np.random.default_rng(2)
        for _ in range(3000):
            h0 = rng.uniform(hs[0], hs[-1])
            s = rng.choice([rng.uniform(0.0, 1.0), 0.0, 1.0, 1.0 - hs[0] / h0])
            y = (rng.uniform(10.0, 2e4), rng.uniform(1.0, 3e3), rng.uniform(1.0, 1e8), rng.uniform(-1e4, 1e4), h0)
            assert new(y, s, row) == old(y, s, row)

    def test_rhs_is_bit_identical_across_eos_and_the_crust(self, eos_model):
        """As above, for more EOS and with h drawn log-uniformly, so the crust nodes are exercised too."""
        rng = np.random.default_rng(5)
        for seed in range(0, 200, 40):
            eos = _usable_eos(eos_model, seed)
            n_table = eos.hs.shape[0]
            row = np.asarray(pack_table(eos))
            new, old = make_tov_rhs(n_table), _bisection_tov_rhs(n_table)
            hs = np.asarray(eos.hs)
            for _ in range(1000):
                h0 = rng.uniform(hs[0], hs[-1])
                h = np.exp(rng.uniform(np.log(hs[0]), np.log(h0)))
                s = 1.0 - h / h0
                y = (rng.uniform(10.0, 2e4), rng.uniform(1.0, 3e3), rng.uniform(1.0, 1e8), rng.uniform(-1e4, 1e4), h0)
                assert new(y, s, row) == old(y, s, row)

    def test_zero_when_h0_is_zero(self, eos_model):
        """An invalid star is given h0 = 0, which must stop it before the (possibly NaN) table is read."""
        eos = _usable_eos(eos_model)
        rhs = make_tov_rhs(eos.hs.shape[0])
        row = np.full_like(np.asarray(pack_table(eos)), np.nan)
        assert rhs((1.0, 1.0, 1.0, 1.0, 0.0), 0.5, row) == (0.0,) * N_STATE

    def test_error_weights_keep_the_four_component_norm(self):
        """modax divides the error norm by the number of components, so the four physical weights
        are scaled by sqrt(5/4) and the h0 component gets weight zero."""
        w = np.asarray(ERROR_WEIGHTS)
        assert w.shape == (N_STATE,)
        assert w[4] == 0.0
        assert np.sum(w[:4] ** 2) / N_STATE == pytest.approx(1.0)



@pytest.fixture(scope="module")
def modax_solver():
    pytest.importorskip("modax")
    from numba_cuda_mlir import cuda

    if not cuda.is_available():
        pytest.skip("modax needs a CUDA GPU")
    from jesterTOV.tov.gr_modax import GRTOVModaxSolver

    return GRTOVModaxSolver()


class TestSolver:
    # The two solvers differ in method (Tsit5, Dopri5) at rtol 1e-5, so they agree at this level and not to round-off.
    REL_TOL = 3e-4

    def test_family_matches_diffrax(self, eos_model, modax_solver):
        eos = _usable_eos(eos_model)
        ref = GRTOVSolver().construct_family(eos, ndat=100, min_nsat=0.75)
        got = modax_solver.construct_family(eos, ndat=100, min_nsat=0.75)
        assert float(ref.masses[-1]) > 1.0
        np.testing.assert_allclose(got.masses, ref.masses, rtol=self.REL_TOL)
        np.testing.assert_allclose(got.radii, ref.radii, rtol=self.REL_TOL)

    def test_lambda_error_comparable_to_diffrax(self, eos_model, modax_solver):
        """At rtol 1e-5 both solvers carry an error of order 1e-3 in Lambda, largest at low mass where
        k2 cancels strongly, so modax is judged by its distance to a tight reference next to that of diffrax."""
        from diffrax import Dopri5, ODETerm, PIDController, SaveAt, diffeqsolve

        from jesterTOV.tov.gr import _calc_k2, _initial_conditions

        eos = _usable_eos(eos_model)
        pcs = jnp.logspace(
            jnp.log10(GRTOVSolver()._get_pc_min(eos, 0.75)),
            jnp.log10(GRTOVSolver()._get_pc_max(eos)),
            100,
        )

        def tight(pc):
            args = {
                "p": eos.ps,
                "h": eos.hs,
                "e": eos.es,
                "dloge_dlogp": eos.dloge_dlogps,
            }
            h0, dh, y0 = _initial_conditions(eos, pc)
            sol = diffeqsolve(
                ODETerm(_tov_ode),
                Dopri5(scan_kind="bounded"),
                t0=h0,
                t1=0,
                dt0=dh,
                y0=y0,
                args=args,
                saveat=SaveAt(t1=True),
                stepsize_controller=PIDController(rtol=1e-10, atol=1e-10),
                throw=False,
                max_steps=20000,
            )
            R, M, H, b = (x[-1] for x in sol.ys)
            return _calc_k2(R, M, H, b)

        ref = jax.vmap(tight)(pcs)
        dfx = jax.vmap(lambda pc: GRTOVSolver().solve(eos, pc).k2)(pcs)
        _, _, got = modax_solver._solve_many(eos, pcs)
        err_got = np.abs(np.asarray(got) / np.asarray(ref) - 1)
        err_dfx = np.abs(np.asarray(dfx) / np.asarray(ref) - 1)
        assert np.median(err_got) < 3 * np.median(err_dfx)
        assert err_got.max() < 3 * err_dfx.max()
        assert err_got.max() < 0.02



    def test_vmap_matches_single_solves(self, eos_model, modax_solver):
        """A vmap over EOS has to lower to one launch and return the same as separate calls."""
        eos_list = [_usable_eos(eos_model, 10 * i) for i in range(3)]
        batched = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *eos_list)

        def family(e):
            return modax_solver.construct_family(e, ndat=50, min_nsat=0.75)

        got = jax.jit(jax.vmap(family))(batched)
        assert got.masses.shape == (3, 50)
        # jit and eager round the inputs differently, and the step controller amplifies that, so these
        # paths agree at the level of the tolerances. Bit-identity holds for identical inputs, tested below.
        for i, e in enumerate(eos_list):
            single = family(e)
            np.testing.assert_allclose(got.masses[i], single.masses, rtol=self.REL_TOL)
            np.testing.assert_allclose(got.radii[i], single.radii, rtol=self.REL_TOL)

    def test_eos_independent_of_its_neighbours(self, eos_model, modax_solver):
        """The stars of one EOS must give bit-identical results alone, next to another EOS and in another order."""
        from jesterTOV.tov.gr import _initial_conditions
        from jesterTOV.tov.gr_modax import N_COLUMNS, _solve_flat

        ndat = 50

        def inputs(eos):
            pcs = jnp.logspace(
                jnp.log10(modax_solver._get_pc_min(eos, 0.75)),
                jnp.log10(modax_solver._get_pc_max(eos)),
                ndat,
            )
            h0, _, state = jax.vmap(lambda pc: _initial_conditions(eos, pc))(pcs)
            return jnp.stack(state + (h0,), axis=1), pack_table(eos)[None, :]

        def launch(y0, tables):
            return np.asarray(_solve_flat(make_tov_rhs, N_COLUMNS, ndat, y0, tables))

        (y_a, t_a), (y_b, t_b) = inputs(_usable_eos(eos_model, 0)), inputs(
            _usable_eos(eos_model, 10)
        )
        together = launch(jnp.concatenate([y_a, y_b]), jnp.concatenate([t_a, t_b]))
        assert np.array_equal(together[ndat:], launch(y_b, t_b))
        swapped = launch(jnp.concatenate([y_b, y_a]), jnp.concatenate([t_b, t_a]))
        assert np.array_equal(swapped[:ndat], together[ndat:])
        assert np.array_equal(swapped[ndat:], together[:ndat])

    def test_lax_map_with_batches(self, eos_model, modax_solver):
        """`jax.lax.map(batch_size=...)` is how jester evaluates particles after sampling."""
        eos_list = [_usable_eos(eos_model, 10 * i) for i in range(5)]
        batched = jax.tree_util.tree_map(lambda *x: jnp.stack(x), *eos_list)

        def family(e):
            return modax_solver.construct_family(e, ndat=50, min_nsat=0.75).masses

        got = jax.lax.map(family, batched, batch_size=2)
        assert got.shape == (5, 50)
        np.testing.assert_allclose(got[4], family(eos_list[4]), rtol=self.REL_TOL)

    def test_invalid_eos_gives_nan_and_does_not_disturb_neighbours(
        self, eos_model, modax_solver
    ):
        """An unphysical EOS is poisoned with NaN by construct_family and must not spin to max_steps."""
        good = _usable_eos(eos_model)
        bad = good._replace(cs2=good.cs2.at[50].set(1.5))
        batched = jax.tree_util.tree_map(lambda *x: jnp.stack(x), good, bad)

        def family(e):
            return modax_solver.construct_family(e, ndat=50, min_nsat=0.75)

        got = jax.vmap(family)(batched)
        assert bool(jnp.all(jnp.isnan(got.masses[1])))
        assert bool(jnp.all(jnp.isfinite(got.masses[0])))
        np.testing.assert_allclose(
            got.masses[0], family(good).masses, rtol=self.REL_TOL
        )

    def test_transform_config_selects_backend(self):
        from jesterTOV.inference.config.schema import GRTOVConfig

        assert GRTOVConfig().backend == "diffrax"
        assert GRTOVConfig(backend="modax").backend == "modax"
        with pytest.raises(ValueError):
            GRTOVConfig(backend="other")
