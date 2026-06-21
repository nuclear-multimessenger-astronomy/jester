"""Tests for per-NS individual anisotropy (individual-gamma) mode.

Covers:
  - NSSource / collect_ns_sources utility
  - Prior expansion logic
  - JesterTransform parameter names in individual mode
  - JesterTransform.forward() does NOT produce per-NS family arrays
  - IndividualGammaLikelihood.evaluate() wiring (radio/NICER and GW)
  - GWLikelihood secondary-family path
  - End-to-end SMC-RW run with individual=true and radio pulsars
"""

from __future__ import annotations

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _nep_params() -> dict[str, float]:
    return {
        "E_sat": -16.0,
        "K_sat": 240.0,
        "Q_sat": 400.0,
        "Z_sat": 0.0,
        "E_sym": 31.7,
        "L_sym": 58.7,
        "K_sym": -100.0,
        "Q_sym": 0.0,
        "Z_sym": 0.0,
    }


def _nep_priors():
    from jesterTOV.inference.base.prior import UniformPrior

    names = [
        "E_sat",
        "K_sat",
        "Q_sat",
        "Z_sat",
        "E_sym",
        "L_sym",
        "K_sym",
        "Q_sym",
        "Z_sym",
    ]
    bounds = [
        (-16.1, -15.9),
        (150.0, 300.0),
        (-500.0, 1100.0),
        (-2500.0, 1500.0),
        (28.0, 45.0),
        (10.0, 200.0),
        (-400.0, 200.0),
        (-1000.0, 1500.0),
        (-2000.0, 1500.0),
    ]
    return [
        UniformPrior(lo, hi, parameter_names=[n]) for n, (lo, hi) in zip(names, bounds)
    ]


def _eos_config():
    from jesterTOV.inference.config.schema import MetamodelEOSConfig

    return MetamodelEOSConfig(type="metamodel", ndat_metamodel=100, nmax_nsat=3.0)


def _aniso_config(individual: bool = False):
    from jesterTOV.inference.config.schema import AnisotropyTOVConfig

    return AnisotropyTOVConfig(type="anisotropy", individual=individual)


# ---------------------------------------------------------------------------
# 1. collect_ns_sources — radio only
# ---------------------------------------------------------------------------


class TestCollectNSSources:
    def _radio_config(self, psrs):
        cfg = MagicMock()
        cfg.likelihoods = [
            MagicMock(
                enabled=True,
                pulsars=[{"name": p, "mass_mean": 2.0, "mass_std": 0.05} for p in psrs],
            )
        ]
        # Make the mock instance-check pass for RadioLikelihoodConfig
        from jesterTOV.inference.config.schema import RadioLikelihoodConfig

        cfg.likelihoods[0].__class__ = RadioLikelihoodConfig
        return cfg

    def test_radio_only(self):
        from jesterTOV.inference.run_inference import collect_ns_sources, NSSource
        from jesterTOV.inference.config.schema import RadioLikelihoodConfig

        lk = MagicMock(spec=RadioLikelihoodConfig)
        lk.enabled = True
        lk.pulsars = [
            {"name": "J0740", "mass_mean": 2.08, "mass_std": 0.07},
            {"name": "J1614", "mass_mean": 1.928, "mass_std": 0.017},
        ]

        cfg = MagicMock()
        cfg.likelihoods = [lk]

        sources = collect_ns_sources(cfg)
        assert len(sources) == 2
        assert sources[0] == NSSource(
            ns_key="J0740", lk_type="radio", event_name="J0740"
        )
        assert sources[1] == NSSource(
            ns_key="J1614", lk_type="radio", event_name="J1614"
        )

    def test_gw_event(self):
        from jesterTOV.inference.run_inference import collect_ns_sources, NSSource
        from jesterTOV.inference.config.schema import GWLikelihoodConfig, GWEventConfig

        event = GWEventConfig(name="GW170817")
        lk = MagicMock(spec=GWLikelihoodConfig)
        lk.enabled = True
        lk.events = [event]

        cfg = MagicMock()
        cfg.likelihoods = [lk]

        sources = collect_ns_sources(cfg)
        assert len(sources) == 2
        assert sources[0] == NSSource(
            ns_key="GW170817_mass1", lk_type="gw", event_name="GW170817"
        )
        assert sources[1] == NSSource(
            ns_key="GW170817_mass2", lk_type="gw", event_name="GW170817"
        )

    def test_disabled_likelihood_skipped(self):
        from jesterTOV.inference.run_inference import collect_ns_sources
        from jesterTOV.inference.config.schema import RadioLikelihoodConfig

        lk = MagicMock(spec=RadioLikelihoodConfig)
        lk.enabled = False
        lk.pulsars = [{"name": "J0740", "mass_mean": 2.08, "mass_std": 0.07}]

        cfg = MagicMock()
        cfg.likelihoods = [lk]

        sources = collect_ns_sources(cfg)
        assert sources == []


# ---------------------------------------------------------------------------
# 2. Prior expansion
# ---------------------------------------------------------------------------


class TestPriorExpansion:
    def test_expansion_replaces_base_param(self):
        from jesterTOV.inference.base.prior import UniformPrior, CombinePrior
        from jesterTOV.inference.run_inference import NSSource

        base_priors = _nep_priors() + [
            UniformPrior(0.0, 0.5, parameter_names=["lambda_DY"])
        ]
        prior = CombinePrior(base_priors)  # type: ignore[arg-type]

        ns_sources = [
            NSSource(ns_key="J0030", lk_type="nicer", event_name="J0030"),
            NSSource(ns_key="J0740", lk_type="nicer", event_name="J0740"),
        ]

        # Simulate what setup_prior does for individual mode
        from jesterTOV.tov.anisotropy import AnisotropyTOVSolver

        aniso_required = AnisotropyTOVSolver().get_required_parameters()
        sampled_aniso_names = [
            p
            for p in aniso_required
            if any(p in pr.parameter_names for pr in prior.base_prior)
        ]

        base_aniso_priors: dict = {}
        retained: list = []
        for pr in prior.base_prior:
            if pr.parameter_names[0] in sampled_aniso_names:
                base_aniso_priors[pr.parameter_names[0]] = pr
            else:
                retained.append(pr)

        expanded: list = []
        for p_name, base_pr in base_aniso_priors.items():
            for ns in ns_sources:
                expanded.append(base_pr.copy_with_name(f"{p_name}_{ns.ns_key}"))

        new_prior = CombinePrior(retained + expanded)  # type: ignore[arg-type]

        assert "lambda_DY" not in new_prior.parameter_names
        assert "lambda_DY_J0030" in new_prior.parameter_names
        assert "lambda_DY_J0740" in new_prior.parameter_names
        assert "E_sat" in new_prior.parameter_names

    def test_copy_with_name_preserves_bounds(self):
        from jesterTOV.inference.base.prior import UniformPrior

        base = UniformPrior(-0.5, 1.0, parameter_names=["lambda_DY"])
        copy = base.copy_with_name("lambda_DY_J0030")

        assert copy.xmin == -0.5
        assert copy.xmax == 1.0
        assert copy.parameter_names == ["lambda_DY_J0030"]


# ---------------------------------------------------------------------------
# 3. JesterTransform parameter names in individual mode
# ---------------------------------------------------------------------------


class TestTransformIndividualMode:
    def test_parameter_names_individual(self):
        from jesterTOV.inference.transforms import JesterTransform

        transform = JesterTransform.from_config(
            eos_config=_eos_config(),
            tov_config=_aniso_config(),
            individual_ns_sources=["J0030", "J0740"],
            fixed_params={"lambda_BL": 0.0, "lambda_HB": 1.0},
        )
        names = transform.get_parameter_names()
        assert "lambda_DY_J0030" in names
        assert "lambda_DY_J0740" in names
        assert "lambda_DY" not in names
        assert "lambda_BL" not in names
        assert "lambda_HB" not in names
        for nep in ["E_sat", "K_sat", "Q_sat"]:
            assert nep in names

    def test_parameter_names_shared(self):
        from jesterTOV.inference.transforms import JesterTransform

        transform = JesterTransform.from_config(
            eos_config=_eos_config(),
            tov_config=_aniso_config(),
        )
        names = transform.get_parameter_names()
        assert "lambda_DY" in names
        assert "lambda_DY_J0030" not in names

    def test_forward_no_per_ns_family_arrays(self):
        """In individual mode, forward() must NOT add masses_EOS_J0030 etc."""
        from jesterTOV.inference.transforms import JesterTransform

        ns_sources = ["J0030", "J0740"]
        transform = JesterTransform.from_config(
            eos_config=_eos_config(),
            tov_config=_aniso_config(),
            individual_ns_sources=ns_sources,
            fixed_params={"lambda_BL": 0.0, "lambda_HB": 1.0},
        )

        params = {**_nep_params(), "lambda_DY_J0030": 0.1, "lambda_DY_J0740": 0.2}
        result = transform.forward(params)

        # Reference family is present
        assert "masses_EOS" in result
        assert "radii_EOS" in result
        assert "Lambdas_EOS" in result
        # Per-NS family arrays must NOT be present
        for ns in ns_sources:
            assert f"masses_EOS_{ns}" not in result
        # Per-NS gamma values must be preserved
        assert "lambda_DY_J0030" in result
        assert "lambda_DY_J0740" in result


# ---------------------------------------------------------------------------
# 4. IndividualGammaLikelihood — radio/NICER wiring
# ---------------------------------------------------------------------------


class TestIndividualGammaLikelihoodRadio:
    def _make_eos_data(self):
        from jesterTOV.eos.metamodel import MetaModel_EOS_model

        return MetaModel_EOS_model().construct_eos(_nep_params())

    def _make_family(self, eos_data, tov_params):
        from jesterTOV.tov.anisotropy import AnisotropyTOVSolver

        return AnisotropyTOVSolver().construct_family(
            eos_data, ndat=50, min_nsat=0.75, tov_params=tov_params
        )

    def test_construct_family_called_once_per_radio_pulsar(self):
        from jesterTOV.inference.likelihoods.individual_gamma import (
            IndividualGammaLikelihood,
        )
        from jesterTOV.inference.run_inference import NSSource
        from jesterTOV.tov.anisotropy import AnisotropyTOVSolver

        eos_data = self._make_eos_data()
        gr_params = {"lambda_BL": 0.0, "lambda_DY": 0.0, "lambda_HB": 1.0}
        family = self._make_family(eos_data, gr_params)

        mock_tov = MagicMock(spec=AnisotropyTOVSolver)
        mock_tov.construct_family.return_value = family

        mock_lk_J0030 = MagicMock()
        mock_lk_J0030.evaluate.return_value = jnp.array(-1.0)
        mock_lk_J0740 = MagicMock()
        mock_lk_J0740.evaluate.return_value = jnp.array(-2.0)

        ns_sources = [
            NSSource(ns_key="J0030", lk_type="radio", event_name="J0030"),
            NSSource(ns_key="J0740", lk_type="radio", event_name="J0740"),
        ]
        lk = IndividualGammaLikelihood(
            ns_sources=ns_sources,
            ns_source_to_likelihood={
                "J0030": mock_lk_J0030,
                "J0740": mock_lk_J0740,
            },
            tov_solver=mock_tov,
            sampled_aniso_params=["lambda_DY"],
            fixed_aniso_params={"lambda_BL": 0.0, "lambda_HB": 1.0},
            ndat_TOV=50,
            min_nsat_TOV=0.75,
        )

        params = {
            "n": eos_data.ns,
            "p": eos_data.ps,
            "h": eos_data.hs,
            "e": eos_data.es,
            "dloge_dlogp": eos_data.dloge_dlogps,
            "cs2": eos_data.cs2,
            "lambda_DY_J0030": 0.1,
            "lambda_DY_J0740": 0.2,
        }
        result = lk.evaluate(params)

        assert mock_tov.construct_family.call_count == 2
        # First call should use J0030's lambda_DY = 0.1
        call_args_0 = mock_tov.construct_family.call_args_list[0]
        assert call_args_0.kwargs["tov_params"]["lambda_DY"] == pytest.approx(0.1)
        call_args_1 = mock_tov.construct_family.call_args_list[1]
        assert call_args_1.kwargs["tov_params"]["lambda_DY"] == pytest.approx(0.2)
        assert float(result) == pytest.approx(-3.0)

    def test_construct_family_twice_for_gw(self):
        from jesterTOV.inference.likelihoods.individual_gamma import (
            IndividualGammaLikelihood,
        )
        from jesterTOV.inference.run_inference import NSSource
        from jesterTOV.tov.anisotropy import AnisotropyTOVSolver

        eos_data = self._make_eos_data()
        gr_params = {"lambda_BL": 0.0, "lambda_DY": 0.0, "lambda_HB": 1.0}
        family = self._make_family(eos_data, gr_params)

        mock_tov = MagicMock(spec=AnisotropyTOVSolver)
        mock_tov.construct_family.return_value = family

        mock_lk_gw = MagicMock()
        mock_lk_gw.evaluate.return_value = jnp.array(-5.0)

        ns_sources = [
            NSSource(ns_key="GW170817_mass1", lk_type="gw", event_name="GW170817"),
            NSSource(ns_key="GW170817_mass2", lk_type="gw", event_name="GW170817"),
        ]
        lk = IndividualGammaLikelihood(
            ns_sources=ns_sources,
            ns_source_to_likelihood={"GW170817": mock_lk_gw},
            tov_solver=mock_tov,
            sampled_aniso_params=["lambda_DY"],
            fixed_aniso_params={"lambda_BL": 0.0, "lambda_HB": 1.0},
            ndat_TOV=50,
            min_nsat_TOV=0.75,
        )

        params = {
            "n": eos_data.ns,
            "p": eos_data.ps,
            "h": eos_data.hs,
            "e": eos_data.es,
            "dloge_dlogp": eos_data.dloge_dlogps,
            "cs2": eos_data.cs2,
            "lambda_DY_GW170817_mass1": 0.05,
            "lambda_DY_GW170817_mass2": 0.08,
        }
        lk.evaluate(params)

        # Two family solves: one for mass1, one for mass2
        assert mock_tov.construct_family.call_count == 2
        # Check that the inner GW likelihood received masses_EOS_2
        inner_call_kwargs = mock_lk_gw.evaluate.call_args[0][0]
        assert "masses_EOS_2" in inner_call_kwargs
        assert "Lambdas_EOS_2" in inner_call_kwargs


# ---------------------------------------------------------------------------
# 5. GWLikelihood secondary-family path (functional diff test)
# ---------------------------------------------------------------------------


class TestGWLikelihoodSecondaryFamily:
    """Verify that GWLikelihood.evaluate() uses the secondary family for m2.

    We build a minimal JAX-compatible flow stub whose log_prob is sensitive to
    lambda_2 (the 4th element of the (m1, m2, l1, l2) input vector), then
    check that the likelihood value changes when a different secondary family
    is injected.
    """

    @staticmethod
    def _make_stub_likelihood(mass_samples: jnp.ndarray) -> "GWLikelihood":  # type: ignore[name-defined]
        """Build a GWLikelihood backed by a stub flow: log_prob(x) = -sum(x^2)."""
        import equinox as eqx
        from jesterTOV.inference.likelihoods.gw import GWLikelihood

        # Build a minimal flow whose log_prob is differentiable and lambda-sensitive
        class _StubFlow(eqx.Module):  # type: ignore[misc]
            def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
                return -jnp.sum(x**2)

            def sample(self, key: jnp.ndarray, shape: tuple) -> jnp.ndarray:
                return mass_samples

        gw_lk = object.__new__(GWLikelihood)
        gw_lk.flow = _StubFlow()
        gw_lk.penalty_value = 0.0
        gw_lk.N_masses_evaluation = len(mass_samples)
        gw_lk.N_masses_batch_size = len(mass_samples)
        gw_lk.event_name = "GW170817"
        gw_lk.fixed_mass_samples = mass_samples
        return gw_lk

    def test_secondary_family_changes_result(self):
        """Injecting a different secondary Lambda family changes the log-likelihood."""
        mass_samples = jnp.array([[1.1, 1.0], [1.2, 1.05]])
        gw_lk = self._make_stub_likelihood(mass_samples)

        masses = jnp.linspace(0.5, 2.5, 50)
        lambdas_primary = jnp.full(50, 400.0)
        lambdas_secondary = jnp.full(50, 100.0)  # distinctly different

        # Shared-gamma: both m1 and m2 use the primary family
        result_shared = gw_lk.evaluate(
            {"masses_EOS": masses, "Lambdas_EOS": lambdas_primary}
        )
        # Individual mode: m2 uses a different secondary family
        result_individual = gw_lk.evaluate(
            {
                "masses_EOS": masses,
                "Lambdas_EOS": lambdas_primary,
                "masses_EOS_2": masses,
                "Lambdas_EOS_2": lambdas_secondary,
            }
        )

        # The log-likelihoods must differ (secondary lambda is much smaller → smaller l2^2 penalty)
        assert abs(float(result_shared) - float(result_individual)) > 0.1

    def test_secondary_keys_absent_matches_same_family_both(self):
        """Without secondary keys, passing the same family as both primary and
        secondary must give the same result as the shared-gamma path."""
        mass_samples = jnp.array([[1.1, 1.0], [1.2, 1.05]])
        gw_lk = self._make_stub_likelihood(mass_samples)

        masses = jnp.linspace(0.5, 2.5, 50)
        lambdas = jnp.full(50, 400.0)

        result_shared = gw_lk.evaluate({"masses_EOS": masses, "Lambdas_EOS": lambdas})
        result_explicit_same = gw_lk.evaluate(
            {
                "masses_EOS": masses,
                "Lambdas_EOS": lambdas,
                "masses_EOS_2": masses,
                "Lambdas_EOS_2": lambdas,
            }
        )

        assert abs(float(result_shared) - float(result_explicit_same)) < 1e-6


# ---------------------------------------------------------------------------
# 6. End-to-end SMC-RW smoke test with individual=true and radio pulsars
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.e2e
def test_e2e_individual_gamma_smc(tmp_path):
    """Smoke test: tiny SMC run with individual=true and two radio pulsars.

    Verifies the full pipeline (prior expansion → transform → IndividualGammaLikelihood
    → sampler) produces finite log-probabilities with no NaNs.
    """
    import jax

    jax.config.update("jax_enable_x64", True)

    from jesterTOV.inference.config.schema import InferenceConfig
    from jesterTOV.inference.run_inference import (
        setup_prior,
        setup_transform,
        setup_likelihood,
        determine_keep_names,
    )
    from jesterTOV.inference.samplers import create_sampler

    prior_content = (
        'E_sat = UniformPrior(-16.1, -15.9, parameter_names=["E_sat"])\n'
        'K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])\n'
        'Q_sat = UniformPrior(-500.0, 1100.0, parameter_names=["Q_sat"])\n'
        'Z_sat = UniformPrior(-2500.0, 1500.0, parameter_names=["Z_sat"])\n'
        'E_sym = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])\n'
        'L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])\n'
        'K_sym = UniformPrior(-400.0, 200.0, parameter_names=["K_sym"])\n'
        'Q_sym = UniformPrior(-1000.0, 1500.0, parameter_names=["Q_sym"])\n'
        'Z_sym = UniformPrior(-2000.0, 1500.0, parameter_names=["Z_sym"])\n'
        'lambda_BL = Fixed(0.0, parameter_names=["lambda_BL"])\n'
        'lambda_HB = Fixed(1.0, parameter_names=["lambda_HB"])\n'
        'lambda_DY = UniformPrior(0.0, 0.5, parameter_names=["lambda_DY"])\n'
    )
    prior_file = tmp_path / "individual_gamma.prior"
    prior_file.write_text(prior_content)

    config_dict = {
        "seed": 0,
        "dry_run": False,
        "validate_only": False,
        "eos": {
            "type": "metamodel",
            "ndat_metamodel": 30,
            "nmax_nsat": 2.0,
            "crust_name": "DH",
        },
        "tov": {
            "type": "anisotropy",
            "individual": True,
            "ndat_TOV": 30,
            "min_nsat_TOV": 0.75,
            "nb_masses": 20,
        },
        "prior": {"specification_file": str(prior_file)},
        "likelihoods": [
            {"type": "constraints_eos", "enabled": True},
            {
                "type": "radio",
                "enabled": True,
                "pulsars": [
                    {"name": "J1614-2230", "mass_mean": 1.937, "mass_std": 0.014},
                    {"name": "J0740+6620", "mass_mean": 2.08, "mass_std": 0.07},
                ],
            },
        ],
        "sampler": {
            "type": "smc-rw",
            "n_particles": 5,
            "n_mcmc_steps": 1,
            "target_ess": 0.9,
            "random_walk_sigma": 0.1,
            "output_dir": str(tmp_path / "outdir"),
            "n_eos_samples": 5,
        },
        "postprocessing": {"enabled": False},
    }

    config = InferenceConfig(**config_dict)
    prior, fixed_params = setup_prior(config)

    # After expansion, per-NS lambda_DY params must be present
    assert "lambda_DY_J1614-2230" in prior.parameter_names
    assert "lambda_DY_J0740+6620" in prior.parameter_names
    assert "lambda_DY" not in prior.parameter_names

    keep_names = determine_keep_names(config, prior, fixed_params)
    transform = setup_transform(
        config, prior=prior, keep_names=keep_names, fixed_params=fixed_params
    )
    likelihood = setup_likelihood(config, transform)

    sampler = create_sampler(
        config=config.sampler,
        prior=prior,
        likelihood=likelihood,
        likelihood_transforms=[transform],
        seed=config.seed,
    )

    key = jax.random.PRNGKey(config.seed)
    sampler.sample(key)

    output = sampler.get_sampler_output()

    assert output.log_prob is not None
    assert not jnp.any(jnp.isnan(output.log_prob)), "log_prob contains NaN"
    assert "lambda_DY_J1614-2230" in output.samples
    assert "lambda_DY_J0740+6620" in output.samples
