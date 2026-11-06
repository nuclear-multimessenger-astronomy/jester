"""Cross-check jester likelihoods against the per-EOS values of Koehn et al. (2025).

Each test feeds Koehn's own EOS and M-R tables into a jester likelihood and compares log L with the
values stored by the original analysis (up to one constant, since those files are normalised over
the EOS set). This is an input/output pair from an independent code, not a self-consistency test.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jesterTOV import utils
from jesterTOV.inference.likelihoods.chieft import ChiEFTLikelihood
from jesterTOV.inference.likelihoods.hic import HICLikelihood
from jesterTOV.inference.likelihoods.mr_table import load_koehn2024_mr_likelihood
from jesterTOV.inference.likelihoods.pqcd import PQCDLikelihood, PQCDStarLikelihood
from jesterTOV.inference.likelihoods.radio import BlackWidowLikelihood, RadioTimingLikelihood
from jesterTOV.inference.likelihoods.rex import REXLikelihood, load_rex_posterior

DATA = np.load(Path(__file__).parent / "data" / "koehn2024_subset.npz")
N_EOS = len(DATA["ids"])
MARGIN = 1e-6  # EOS below this fraction of the largest weight do not enter the comparison


def eos_params(k: int) -> dict:
    """Build the dictionary the jester transform would return from Koehn's tables for EOS k."""
    keep_macro = ~np.isnan(DATA["M"][k])
    keep_micro = ~np.isnan(DATA["n"][k])
    nbreak, K_sat, Q_sat, Z_sat, E_sym, L_sym, K_sym, Q_sym, Z_sym = DATA["params"][k]
    return {
        "masses_EOS": jnp.array(DATA["M"][k][keep_macro]),
        "radii_EOS": jnp.array(DATA["R"][k][keep_macro]),
        "logpc_EOS": jnp.log10(jnp.array(DATA["pc"][k][keep_macro]) * utils.MeV_fm_inv3_to_geometric),
        "n": jnp.array(DATA["n"][k][keep_micro]) * utils.fm_inv3_to_geometric,
        "p": jnp.array(DATA["p"][k][keep_micro]) * utils.MeV_fm_inv3_to_geometric,
        "e": jnp.array(DATA["e"][k][keep_micro]) * utils.MeV_fm_inv3_to_geometric,
        "nbreak": float(nbreak),
        "E_sym": float(E_sym),
        "L_sym": float(L_sym),
    }


ALL_PARAMS = [eos_params(k) for k in range(N_EOS)]


def deviation(likelihood, name: str) -> np.ndarray:
    """Offset-free |log L_jester - log L_Koehn| for the EOS that carry weight."""
    logl = np.array([float(likelihood.evaluate(p)) for p in ALL_PARAMS])
    ref = DATA[f"logL_{name}"]
    good = np.isfinite(ref) & (ref > ref[np.isfinite(ref)].max() + np.log(MARGIN))
    diff = logl[good] - ref[good]
    return np.abs(diff - np.median(diff))


@pytest.mark.parametrize("experiment", ["PREX", "CREX"])
def test_rex_matches_koehn(experiment):
    likelihood = REXLikelihood(experiment, load_rex_posterior(experiment))
    assert np.percentile(deviation(likelihood, experiment), 95) < 0.01


def test_dipole_matches_koehn():
    # The stored file comes from a KDE on an unseeded random subset of 10000 samples, so the
    # comparison cannot be exact and only the bulk is compared.
    likelihood = REXLikelihood("dipole", load_rex_posterior("dipole"))
    assert np.median(deviation(likelihood, "dipole")) < 0.15


def test_hic_matches_koehn():
    assert np.percentile(deviation(HICLikelihood(), "HIC"), 95) < 0.03


def test_pqcd_matches_koehn():
    assert deviation(PQCDLikelihood(), "pQCD").max() < 1e-6


def test_pqcd_star_matches_koehn():
    assert np.percentile(deviation(PQCDStarLikelihood(), "pQCD_star"), 95) < 0.05


@pytest.mark.parametrize("name,mean,std", [("J0740", 2.08, 0.07), ("J1614", 1.94, 0.03), ("J0348", 2.01, 0.04)])
def test_radio_matches_koehn(name, mean, std):
    likelihood = RadioTimingLikelihood(name, mean, std, m_min=0.0)
    assert deviation(likelihood, f"{name}_radio").max() < 1e-6


def test_chieft_matches_koehn_with_in_band_score():
    # Koehn's file integrates a step function (+1 in band) with Simpson weights on 48 EOS nodes. Koehn's own
    # formula on a fine grid deviates from the file by 0.11 at the 99th percentile, which sets this tolerance.
    likelihood = ChiEFTLikelihood(nb_n=400, log_f_in_band=1.0)
    assert np.percentile(deviation(likelihood, "chiEFT"), 95) < 0.1


def test_chieft_default_is_unchanged():
    """Without the option the score inside the band stays 0, as stated in the paper."""
    params = ALL_PARAMS[0]
    assert float(ChiEFTLikelihood().evaluate(params)) == pytest.approx(
        float(ChiEFTLikelihood(log_f_in_band=0.0).evaluate(params))
    )


@pytest.mark.parametrize(
    "name,tol",
    [("qLMXBs", 0.05), ("SAX", 0.05), ("J1731", 0.05), ("J0030", 0.05), ("J0740", 0.05)],
)
def test_mr_tables_match_koehn(name, tol):
    assert np.percentile(deviation(load_koehn2024_mr_likelihood(name), name), 95) < tol


def test_burster_matches_koehn_for_eos_with_weight():
    # The grid of 4U 1702-429 ends abruptly at 0.807 Msun and Koehn integrates it with Simpson weights. EOS whose
    # curve crosses that edge with tiny L differ by up to 1 in log L, also between scipy versions of his own formula.
    dev = deviation(load_koehn2024_mr_likelihood("4U_1702"), "4U_1702")
    assert np.median(dev) < 0.01
    ref = DATA["logL_4U_1702"]
    good = np.isfinite(ref) & (ref > ref[np.isfinite(ref)].max() + np.log(MARGIN))
    assert dev[ref[good] > ref[good].max() - np.log(100)].max() < 0.05


def test_black_widow_fit_close_to_koehn():
    # Koehn uses RNS for M_rot(707 Hz); jester uses the fit from the same paper, Eq. (Mrot_relation).
    likelihood = BlackWidowLikelihood()
    dev = deviation(likelihood, "J0952")
    assert np.median(dev) < 0.1


def test_black_widow_mass_limit_is_above_mtov():
    likelihood = BlackWidowLikelihood()
    params = ALL_PARAMS[np.argmax(DATA["logL_J0952"])]
    mtov = float(jnp.max(params["masses_EOS"]))
    assert mtov < float(likelihood.rotating_mass_limit(params)) < 1.2 * mtov


def test_grb_ejecta_fits_match_nmma():
    """Reference values computed with the formulas in nmma/core/conversion.py (BNSEjectaFitting)."""
    from jesterTOV.inference.likelihoods.grb import dynamical_ejecta_mass, log10_disk_mass

    m1, m2, c1, c2 = 1.5, 1.3, 0.18, 0.16
    expected_dyn = 1e-3 * (
        m1 * (-9.3335 / c1 + 114.17 * (m2 / m1) ** 1.5465 - 337.56 * c1)
        + m2 * (-9.3335 / c2 + 114.17 * (m1 / m2) ** 1.5465 - 337.56 * c2)
    )
    assert float(dynamical_ejecta_mass(m1, m2, c1, c2)) == pytest.approx(max(expected_dyn, 0.0))

    mtov, r16, total, q = 2.2, 12.0 / 1.476625038050125, 2.8, 1.3 / 1.5
    threshold = (-3.606 * mtov / r16 + 2.38) * mtov
    xi = 0.5 * np.tanh(5.879 * (q - 0.886))
    a, b = -1.725 - 2.337 * xi, -0.564 - 0.437 * xi
    expected_disk = max(a * (1 + b * np.tanh((0.958 - total / threshold) / 0.057)), -3.0)
    assert float(log10_disk_mass(total, q, mtov, r16)) == pytest.approx(expected_disk)


@pytest.mark.parametrize(
    "config",
    [
        {"type": "rex", "experiment_name": "dipole"},
        {"type": "hic"},
        {"type": "pqcd"},
        {"type": "pqcd", "marginalise_extensions": True},
        {"type": "black_widow"},
        {"type": "mr_table", "name": "J0740"},
        {"type": "grb_ejecta", "n_samples": 100},
    ],
)
def test_factory_builds_a_finite_likelihood(config):
    from pydantic import TypeAdapter

    from jesterTOV.inference.config.schema import LikelihoodConfig
    from jesterTOV.inference.likelihoods.factory import create_likelihood

    likelihood = create_likelihood(TypeAdapter(LikelihoodConfig).validate_python(config))
    assert np.isfinite(float(likelihood.evaluate(ALL_PARAMS[0])))


def test_postmerger_uses_tov_threshold_consistently():
    from jesterTOV.inference.likelihoods.postmerger import masses_of_central_pressure

    k = int(np.argmax(np.nanmax(DATA["M"], axis=1) > 2.0))
    params = ALL_PARAMS[k]
    table_p = params["p"]
    e = params["e"]
    h = utils.cumtrapz(table_p / (e + table_p), jnp.log(table_p))
    loge, logp = jnp.log(e), jnp.log(table_p)
    dlog = jnp.gradient(loge) / jnp.gradient(logp)
    eos = {"n": params["n"], "p": table_p, "h": h, "e": e, "dloge_dlogp": dlog}
    pc = jnp.power(10.0, params["logpc_EOS"][jnp.argmax(params["masses_EOS"])])
    gravitational, baryonic = masses_of_central_pressure(pc, eos)
    # Binding energy of a massive star is known to be between 5 and 25 percent of its gravitational mass.
    assert 0.05 < float(baryonic - gravitational) / float(gravitational) < 0.25
