"""Test configuration for JESTER test suite."""

import pytest
import jax.numpy as jnp
from jesterTOV import utils
from jesterTOV.tov.data_classes import EOSData


@pytest.fixture
def sample_density_arrays():
    """Sample density, pressure, and energy arrays for testing."""
    n = jnp.linspace(0.1, 1.0, 50)  # fm^-3
    p = n**1.5 * 20.0  # MeV/fm^3 - simple polytropic relation
    e = n * 100.0 + p  # MeV/fm^3 - simple energy density
    return n, p, e


@pytest.fixture
def sample_eos_dict():
    """Sample EOS dictionary for TOV solving tests using realistic polytropic EOS."""
    # Create realistic polytropic EOS: P ∝ ρ^Γ with Γ ≈ 2-3 for neutron star matter
    n = jnp.linspace(0.1, 2.0, 50)  # fm^-3, density range
    p = 15.0 * (n / 0.16) ** 2.2  # MeV/fm^3, polytropic with realistic stiffness
    e = n * 939.0 + p  # MeV/fm^3, rest mass energy + pressure contribution

    # Convert to geometric units
    p_geo = p * utils.MeV_fm_inv3_to_geometric
    e_geo = e * utils.MeV_fm_inv3_to_geometric

    # Calculate enthalpy and dloge_dlogp
    h = utils.cumtrapz(p_geo / (e_geo + p_geo), jnp.log(p_geo))
    dloge_dlogp = jnp.diff(jnp.log(e)) / jnp.diff(jnp.log(p))
    dloge_dlogp = jnp.concatenate([jnp.array([dloge_dlogp[0]]), dloge_dlogp])
    dedp = e_geo / p_geo * dloge_dlogp
    cs2 = 1.0 / dedp
    eos_dict = {"p": p_geo, "h": h, "e": e_geo, "dloge_dlogp": dloge_dlogp, "cs2": cs2}
    return eos_dict


@pytest.fixture
def sample_eos_data():
    """Sample EOSData for TOV solver tests (uses same data as sample_eos_dict)."""
    # Create realistic polytropic EOS
    n = jnp.linspace(0.1, 2.0, 50)  # fm^-3, density range
    p = 15.0 * (n / 0.16) ** 2.2  # MeV/fm^3
    e = n * 939.0 + p  # MeV/fm^3

    # Convert to geometric units
    ns = n * utils.fm_inv3_to_geometric
    ps = p * utils.MeV_fm_inv3_to_geometric
    es = e * utils.MeV_fm_inv3_to_geometric

    # Calculate auxiliary quantities
    hs = utils.cumtrapz(ps / (es + ps), jnp.log(ps))
    dloge_dlogps = jnp.diff(jnp.log(e)) / jnp.diff(jnp.log(p))
    dloge_dlogps = jnp.concatenate([jnp.array([dloge_dlogps[0]]), dloge_dlogps])
    dedps = es / ps * dloge_dlogps
    cs2 = 1.0 / dedps

    return EOSData(ns=ns, ps=ps, hs=hs, es=es, dloge_dlogps=dloge_dlogps, cs2=cs2)


@pytest.fixture
def metamodel_params():
    """Sample MetaModel parameters for testing."""
    return {
        "kappas": (0.1, 0.05, 0.01, 0.15, 0.08, 0.02),
        "v_nq": [1.0, 2.0, 0.5, 0.8, 0.3],
        "b_sat": 17.0,
        "b_sym": 25.0,
        "nsat": 0.16,
        "nmin_MM_nsat": 0.12 / 0.16,
        "nmax_nsat": 2.0,
        "ndat": 100,
        "crust_name": "DH",
        "max_n_crust_nsat": 0.5,
        "ndat_spline": 10,
    }


@pytest.fixture
def nep_dict():
    """Sample NEP dictionary for EOS construction."""
    return {
        "E_sat": -16.0,
        "K_sat": 220.0,
        "Q_sat": 0.0,
        "Z_sat": 0.0,
        "E_sym": 32.0,
        "L_sym": 90.0,
        "K_sym": 0.0,
        "Q_sym": 0.0,
        "Z_sym": 0.0,
    }
