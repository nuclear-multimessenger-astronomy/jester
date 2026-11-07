r"""Accuracy of the GR TOV solvers against LALSuite, ported from the jester review (``04_tov``).

Two checks, for every TOV backend available (diffrax always, modax with a CUDA GPU):

1. **LALSuite EOS tables.** The tables in ``jesterTOV/tabulated_eos/lalsuite`` store LAL's own M-R-:math:`\Lambda` curves
   (``generate_lalsuite_eos.py``), so no lalsuite is needed. Each table is solved at the review's central pressures
   (``PC_SI``), with :math:`d\ln\varepsilon/d\ln p` clipped to [0, 5] as in the review's "sanitize" run (LAL's spline
   derivative spikes near crust-core transitions), and R(M), :math:`\Lambda(M)` are compared with LAL's curve on a common
   mass grid (relative RMS in percent, stable branch, M >= 0.75 Msun).
2. **jester EOS through LALSuite** (slow, needs lalsuite). Metamodel+CSE EOS drawn from the prior are written as LAL
   tables (DH crust + jester core), and both codes solve them at the same central pressures.

A few EOS are known outliers against LAL (MS1, MS1B and SLY230A here; the review found the same kind with spiky
:math:`d\ln\varepsilon/d\ln p`), so the tests require the bulk to agree tightly and allow a few outliers within loose bounds.
Measured with diffrax on the 12 tables: median RMS 0.0012 % in R and 0.12 % in :math:`\Lambda`; modax gives the same.
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jesterTOV
from jesterTOV import utils
from jesterTOV.tov.data_classes import EOSData
from jesterTOV.tov.gr import GRTOVSolver

jax.config.update("jax_enable_x64", True)

LAL_DIR = Path(jesterTOV.__file__).parent / "tabulated_eos" / "lalsuite"
LAL_EOS_NAMES = sorted(p.stem for p in LAL_DIR.glob("*.npz"))
PC_SI = np.logspace(np.log10(2e33), np.log10(1e35), 100)  # central pressures of the review [Pa]
M_MIN = 0.75  # [Msun] only stars above this mass are compared, as in the review
DLOGE_DLOGP_CLIP_HI = 5.0
N_INTERP = 500


def _modax_available() -> bool:
    try:
        import modax.tsit5  # type: ignore[import-not-found]  # noqa: F401
        from numba_cuda_mlir import cuda  # type: ignore[import-not-found]
    except ImportError:
        return False
    return bool(cuda.is_available())


BACKENDS = ["diffrax"] + (["modax"] if _modax_available() else [])


@pytest.fixture(scope="module", params=BACKENDS)
def solve_many(request):
    """``(eos_data, pcs) -> (M, R, k2)`` in geometric units, for one backend."""
    if request.param == "modax":
        from jesterTOV.tov.gr_modax import GRTOVModaxSolver

        solver = GRTOVModaxSolver()
        return jax.jit(lambda eos, pcs: solver._solve_many(eos, pcs))
    solver = GRTOVSolver()

    def diffrax_many(eos, pcs):
        sol = jax.vmap(lambda pc: solver.solve(eos, pc, {}))(pcs)
        return sol.M, sol.R, sol.k2

    return jax.jit(diffrax_many)


def _observables(M, R, k2):
    M, R, k2 = (np.asarray(x) for x in (M, R, k2))
    return M / utils.solar_mass_in_meter, R / 1e3, 2.0 / 3.0 * k2 * (M / R) ** -5


def _stable_branch(m, *arrays):
    """Stars above M_MIN, up to the first decrease of the mass with central pressure."""
    m = np.where(np.isfinite(m) & (m >= M_MIN), m, np.nan)
    for i in range(1, len(m)):
        if np.isfinite(m[i]) and np.isfinite(m[i - 1]) and m[i] < m[i - 1]:
            m = m.copy()
            m[i:] = np.nan
            break
    ok = np.isfinite(m)
    return (m[ok],) + tuple(np.asarray(a)[ok] for a in arrays)


def _rms_mass_interp(m_ref, v_ref, m_tst, v_tst) -> float:
    """Relative RMS difference [%] of v(M) on the overlap of the two mass ranges, as in the review."""
    lo = max(m_ref.min(), m_tst.min()) * 1.01
    hi = min(m_ref.max(), m_tst.max()) * 0.99
    if lo >= hi:
        return float("nan")
    grid = np.linspace(lo, hi, N_INTERP)
    ref = np.interp(grid, m_ref, v_ref)
    return float(np.sqrt(np.mean(((np.interp(grid, m_tst, v_tst) - ref) / ref * 100.0) ** 2)))


def _load_lal_table(name):
    d = np.load(LAL_DIR / f"{name}.npz")
    eos = EOSData(
        ns=jnp.array(d["n"]),
        ps=jnp.array(d["p"]),
        hs=jnp.array(d["h"]),
        es=jnp.array(d["e"]),
        dloge_dlogps=jnp.array(np.clip(d["dloge_dlogp"], 0.0, DLOGE_DLOGP_CLIP_HI)),
        cs2=jnp.array(d["cs2"]),
    )
    return eos, d["masses_EOS"], d["radii_EOS"], d["Lambda_EOS"]


@pytest.fixture(scope="module")
def lal_table_errors(solve_many):
    """RMS [%] of R and Lambda against LAL's own curve, per table."""
    pcs = jnp.asarray(PC_SI * utils.pressure_SI_to_geometric)
    errors = {}
    for name in LAL_EOS_NAMES:
        eos, m_lal, r_lal, l_lal = _load_lal_table(name)
        m_lal, r_lal, l_lal = _stable_branch(m_lal, r_lal, l_lal)
        m, r, lam = _stable_branch(*_observables(*solve_many(eos, pcs)))
        errors[name] = {
            "n_stars": len(m),
            "R": _rms_mass_interp(m_lal, r_lal, m, r),
            "Lambda": _rms_mass_interp(m_lal, l_lal, m, lam),
        }
    return errors


class TestLALSuiteTables:
    def test_tables_are_present(self):
        assert len(LAL_EOS_NAMES) >= 10

    def test_every_table_gives_a_family(self, lal_table_errors):
        for name, err in lal_table_errors.items():
            assert err["n_stars"] >= 20, (name, err)
            assert np.isfinite(err["R"]) and np.isfinite(err["Lambda"]), (name, err)

    def test_bulk_matches_lal(self, lal_table_errors):
        r = np.array([e["R"] for e in lal_table_errors.values()])
        lam = np.array([e["Lambda"] for e in lal_table_errors.values()])
        assert np.median(r) < 0.005, lal_table_errors
        assert np.median(lam) < 0.3, lal_table_errors

    def test_few_outliers_and_none_far_off(self, lal_table_errors):
        outliers = [n for n, e in lal_table_errors.items() if e["R"] > 0.02 or e["Lambda"] > 1.0]
        assert len(outliers) <= 3, (outliers, lal_table_errors)
        for name, err in lal_table_errors.items():
            assert err["R"] < 0.5 and err["Lambda"] < 5.0, (name, err)


# ---------------------------------------------------------------------------------------------------------------------
# 2. jester metamodel+CSE EOS, solved by LALSuite and jester from the same table
# ---------------------------------------------------------------------------------------------------------------------

N_JESTER_EOS = 20


def _jester_eos_tables():
    """Draws of metamodel+CSE EOS with a star above 1 Msun, and (ps, es) tables with the full DH crust prepended."""
    from jesterTOV.eos.metamodel.metamodel_CSE import MetaModel_with_CSE_EOS_model
    from jesterTOV.inference.base.prior import CombinePrior, Prior, UniformPrior

    model = MetaModel_with_CSE_EOS_model(
        ndat_metamodel=200, nmax_nsat=25.0, nb_CSE=8, crust_name="DH", max_nbreak_nsat=2.0
    )
    priors: list[Prior] = [
        UniformPrior(150.0, 300.0, ["K_sat"]),
        UniformPrior(-500.0, 1100.0, ["Q_sat"]),
        UniformPrior(-2500.0, 1500.0, ["Z_sat"]),
        UniformPrior(28.0, 45.0, ["E_sym"]),
        UniformPrior(10.0, 200.0, ["L_sym"]),
        UniformPrior(-300.0, 100.0, ["K_sym"]),
        UniformPrior(-800.0, 800.0, ["Q_sym"]),
        UniformPrior(-2500.0, 1500.0, ["Z_sym"]),
        UniformPrior(0.16, 0.32, ["nbreak"]),
    ]
    for i in range(8):
        priors += [UniformPrior(0.0, 1.0, [f"n_CSE_{i}_u"]), UniformPrior(0.0, 1.0, [f"cs2_CSE_{i}"])]
    priors.append(UniformPrior(0.0, 1.0, ["cs2_CSE_8"]))
    prior = CombinePrior(priors)

    crust_ps = np.asarray(model.metamodel.ps_crust) * utils.MeV_fm_inv3_to_geometric
    crust_es = np.asarray(model.metamodel.es_crust) * utils.MeV_fm_inv3_to_geometric
    check = GRTOVSolver()
    tables = []
    key = jax.random.PRNGKey(42)
    for _ in range(200):
        if len(tables) == N_JESTER_EOS:
            break
        key, sub = jax.random.split(key)
        params = {k: float(v[0]) for k, v in prior.sample(sub, 1).items()}
        eos = model.construct_eos({**params, "E_sat": -16.0})
        ps, es = np.asarray(eos.ps), np.asarray(eos.es)
        if not (np.all(np.isfinite(ps)) and np.all(ps > 0) and np.all(np.diff(ps) > 0)):
            continue
        masses = [
            float(check.solve(eos, pc * utils.pressure_SI_to_geometric, {}).M / utils.solar_mass_in_meter)
            for pc in (1e34, 3e34, 8e34)
        ]
        if not any(np.isfinite(m) and m >= 1.0 for m in masses):
            continue
        above = ps > crust_ps.max()
        ps_full = np.concatenate([crust_ps, ps[above]])
        es_full = np.concatenate([crust_es, es[above]])
        keep = np.concatenate([[True], np.diff(ps_full) > 0])
        ps_full, es_full = ps_full[keep], es_full[keep]
        # jester's own table from the same (p, e) points, with h and dlog(e)/dlog(p) as interpolate_eos builds them
        ns, ps_j, hs_j, es_j, dl_j = model.interpolate_eos(
            jnp.zeros(len(ps_full)),
            jnp.asarray(ps_full / utils.MeV_fm_inv3_to_geometric),
            jnp.asarray(es_full / utils.MeV_fm_inv3_to_geometric),
        )
        jester_eos = EOSData(ns=ns, ps=ps_j, hs=hs_j, es=es_j, dloge_dlogps=dl_j, cs2=jnp.zeros(len(ps_j)))
        tables.append((ps_full, es_full, jester_eos))
    return tables


@pytest.mark.slow
def test_jester_eos_match_lalsuite_in_bulk(solve_many):
    lal = pytest.importorskip("lal")
    lalsimulation = pytest.importorskip("lalsimulation")
    tables = _jester_eos_tables()
    assert len(tables) == N_JESTER_EOS
    pcs = jnp.asarray(PC_SI * utils.pressure_SI_to_geometric)
    r_err, l_err = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for i, (ps_full, es_full, jester_eos) in enumerate(tables):
            path = Path(tmp) / f"eos_{i}.dat"
            np.savetxt(path, np.column_stack([ps_full, es_full]), fmt="%.15e")
            lal_eos = lalsimulation.SimNeutronStarEOSFromFile(str(path))
            lal_family = np.full((len(PC_SI), 3), np.nan)  # M [Msun], R [km], Lambda
            for j, pc in enumerate(PC_SI):
                try:
                    r_m, m_kg, k2 = lalsimulation.SimNeutronStarTOVODEIntegrate(pc, lal_eos)
                except Exception:  # noqa: BLE001 - LAL raises generic errors past its table
                    continue
                c = m_kg * lal.MRSUN_SI / lal.MSUN_SI / r_m
                lal_family[j] = (m_kg / lal.MSUN_SI, r_m / 1e3, 2.0 / 3.0 * k2 / c**5)
            m_l, r_l, l_l = _stable_branch(*lal_family.T)
            m_j, r_j, l_j = _stable_branch(*_observables(*solve_many(jester_eos, pcs)))
            if len(m_l) < 5 or len(m_j) < 5:
                continue
            r_err.append(_rms_mass_interp(m_l, r_l, m_j, r_j))
            l_err.append(_rms_mass_interp(m_l, l_l, m_j, l_j))
    r_err, l_err = np.array(r_err), np.array(l_err)
    assert len(r_err) >= 0.9 * N_JESTER_EOS
    # The review (200 EOS) found medians of 0.07 % (R) and ~1 % (Lambda), with rare outliers up to tens of percent.
    assert np.nanmedian(r_err) < 0.15, r_err
    assert np.nanmedian(l_err) < 2.0, l_err
    assert np.mean(l_err < 3.5) >= 0.8, l_err
