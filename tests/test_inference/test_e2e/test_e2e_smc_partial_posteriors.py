"""End-to-end tests for the partial-posteriors (data-tempering) SMC sampler.

Tests the full pipeline: config -> sampler.sample() -> SamplerOutput, for a
sampler that tempers by *number of GW events included* rather than by the
usual lambda inverse-temperature (see
jesterTOV/inference/samplers/blackjax/smc/partial_posteriors.py).
"""

import copy

import pytest
import jax
import jax.numpy as jnp

from jesterTOV.inference.config.schema import InferenceConfig
from jesterTOV.inference.result import InferenceResult
from jesterTOV.inference.run_inference import (
    setup_prior,
    setup_transform,
    setup_likelihood,
    determine_keep_names,
)
from jesterTOV.inference.samplers import create_sampler

from .conftest import validate_sampler_output, NEP_PARAMS


def _run_partial_posteriors(config_dict: dict) -> tuple:
    """Build prior/transform/likelihood/sampler and run sample(). Returns
    (config, sampler) so callers can inspect both the parsed config and the
    post-sample() sampler state/metadata.

    Passes ``likelihood_configs=config.likelihoods`` so the always-on
    likelihood consistency check (exercised by the warm-start tests below)
    actually has something to compare against, matching how
    ``run_inference.py`` calls ``create_sampler``.
    """
    config = InferenceConfig(**config_dict)
    prior, _fixed_params = setup_prior(config)
    keep_names = determine_keep_names(config, prior)
    transform = setup_transform(config, prior=prior, keep_names=keep_names)
    likelihood = setup_likelihood(config, transform)
    sampler = create_sampler(
        config=config.sampler,
        prior=prior,
        likelihood=likelihood,
        likelihood_transforms=[transform],
        seed=config.seed,
        likelihood_configs=config.likelihoods,
    )
    sampler.sample(jax.random.PRNGKey(config.seed))
    return config, sampler


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.blackjax
class TestSMCPartialPosteriorsE2E:
    """End-to-end tests for the smc-partial-posteriors-rw sampler."""

    def test_partial_posteriors_gw_full_pipeline(
        self, smc_partial_posteriors_gw_config, e2e_temp_dir
    ):
        """Test full pipeline with two real GW event likelihoods.

        This exercises:
        - Config loading and validation
        - Prior/transform/likelihood setup (two GW events + EOS constraints)
        - Sampler creation via factory
        - The event-driven (not lambda-driven) SMC loop, including the
          per-event fractional mask sub-stepping
        - SamplerOutput generation
        """
        config = InferenceConfig(**smc_partial_posteriors_gw_config)

        prior, _fixed_params = setup_prior(config)
        keep_names = determine_keep_names(config, prior)
        transform = setup_transform(config, prior=prior, keep_names=keep_names)
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
        validate_sampler_output(output, expected_params=NEP_PARAMS, min_samples=50)

        assert "weights" in output.metadata, "SMC output missing weights"
        weights = output.metadata["weights"]
        assert jnp.isclose(jnp.sum(weights), 1.0, atol=0.01)

        # Partial-posteriors-specific: event_order recorded and matches config
        sampler_metadata = sampler.metadata  # type: ignore[attr-defined]
        assert sampler_metadata["event_order"] == ["GW170817", "GW190425"]
        assert sampler_metadata["n_events"] == 2
        assert len(sampler_metadata["ess_history"]) == 2
        assert len(sampler_metadata["acceptance_history"]) == 2
        assert len(sampler_metadata["n_substeps_history"]) == 2
        assert all(n >= 1 for n in sampler_metadata["n_substeps_history"])

        # n_substeps_history must survive the HDF5 round-trip (histories/
        # group), not just live in the in-memory sampler metadata.
        result = InferenceResult.from_sampler(
            sampler=sampler, config=config, runtime=0.0, fixed_params=_fixed_params
        )
        result_path = e2e_temp_dir / "result.h5"
        result.save(result_path)
        loaded = InferenceResult.load(result_path)
        assert loaded.histories is not None
        assert list(loaded.histories["n_substeps_history"]) == list(
            sampler_metadata["n_substeps_history"]
        )

    def test_partial_posteriors_cadence_groups_events(
        self, smc_partial_posteriors_gw_config, e2e_temp_dir
    ):
        """cadence=2 should ramp both configured events in jointly, as a
        single data-tempering step, instead of one at a time.
        """
        config_dict = copy.deepcopy(smc_partial_posteriors_gw_config)
        config_dict["sampler"]["cadence"] = 2
        _config, sampler = _run_partial_posteriors(config_dict)

        metadata = sampler.metadata  # type: ignore[attr-defined]
        assert metadata["event_order"] == ["GW170817", "GW190425"]
        assert metadata["cadence"] == 2
        assert metadata["event_groups"] == [["GW170817", "GW190425"]]
        # Both events ramped jointly -> a single tempering step, not two.
        assert len(metadata["ess_history"]) == 1
        assert len(metadata["acceptance_history"]) == 1

        output = sampler.get_sampler_output()
        validate_sampler_output(output, expected_params=NEP_PARAMS, min_samples=50)

        plot_outdir = e2e_temp_dir / "cadence_plots"
        sampler.plot_diagnostics(outdir=plot_outdir)  # type: ignore[attr-defined]
        assert (plot_outdir / "smc_diagnostics.png").exists()
        assert (
            plot_outdir / "substep_diagnostics" / "00_GW170817+GW190425.png"
        ).exists()

    def test_partial_posteriors_cadence_list_must_sum_to_new_events(
        self, smc_partial_posteriors_gw_config
    ):
        """A cadence list that doesn't sum to the number of new events must
        raise a clear error at sample time (not silently truncate/pad)."""
        config_dict = copy.deepcopy(smc_partial_posteriors_gw_config)
        config_dict["sampler"]["cadence"] = [1]  # only 1, but 2 events configured
        config = InferenceConfig(**config_dict)

        prior, _fixed_params = setup_prior(config)
        keep_names = determine_keep_names(config, prior)
        transform = setup_transform(config, prior=prior, keep_names=keep_names)
        likelihood = setup_likelihood(config, transform)
        sampler = create_sampler(
            config=config.sampler,
            prior=prior,
            likelihood=likelihood,
            likelihood_transforms=[transform],
            seed=config.seed,
        )

        with pytest.raises(ValueError, match="cadence"):
            sampler.sample(jax.random.PRNGKey(config.seed))

    def test_partial_posteriors_rejects_no_gw_events(
        self, smc_rw_prior_config, e2e_temp_dir
    ):
        """Sampler must fail fast if there's no GW event to temper over."""
        config_dict = dict(smc_rw_prior_config)
        # smc_rw_prior_config's sampler dict is flat (smc-rw shape:
        # n_particles/n_mcmc_steps/target_ess/random_walk_sigma at top
        # level); the partial-posteriors config nests those under "inner"
        # instead, so only output_dir/n_eos_samples carry over.
        config_dict["sampler"] = {
            "type": "smc-partial-posteriors-rw",
            "output_dir": config_dict["sampler"]["output_dir"],
            "n_eos_samples": config_dict["sampler"]["n_eos_samples"],
            "inner": {"n_mcmc_steps": 2},
        }
        config = InferenceConfig(**config_dict)

        prior, _fixed_params = setup_prior(config)
        keep_names = determine_keep_names(config, prior)
        transform = setup_transform(config, prior=prior, keep_names=keep_names)
        likelihood = setup_likelihood(config, transform)

        with pytest.raises(ValueError, match="at least one GW event"):
            create_sampler(
                config=config.sampler,
                prior=prior,
                likelihood=likelihood,
                likelihood_transforms=[transform],
                seed=config.seed,
            )

    def test_partial_posteriors_consistent_with_smc_rw(
        self, smc_partial_posteriors_gw_config, smc_rw_gw_config, e2e_temp_dir
    ):
        """Adding events one at a time should converge near the same target
        as tempering the joint likelihood by lambda, on the same combined
        likelihood. Loose tolerance: this is a consistency sanity check
        between two different SMC schedules over the same posterior, not an
        exact-match test (different sampler, different particle count,
        different random walk schedule).
        """
        pp_config = InferenceConfig(**smc_partial_posteriors_gw_config)
        rw_config = InferenceConfig(**smc_rw_gw_config)

        pp_prior, _ = setup_prior(pp_config)
        pp_keep_names = determine_keep_names(pp_config, pp_prior)
        pp_transform = setup_transform(
            pp_config, prior=pp_prior, keep_names=pp_keep_names
        )
        pp_likelihood = setup_likelihood(pp_config, pp_transform)
        pp_sampler = create_sampler(
            config=pp_config.sampler,
            prior=pp_prior,
            likelihood=pp_likelihood,
            likelihood_transforms=[pp_transform],
            seed=pp_config.seed,
        )
        pp_sampler.sample(jax.random.PRNGKey(pp_config.seed))
        pp_output = pp_sampler.get_sampler_output()

        rw_prior, _ = setup_prior(rw_config)
        rw_keep_names = determine_keep_names(rw_config, rw_prior)
        rw_transform = setup_transform(
            rw_config, prior=rw_prior, keep_names=rw_keep_names
        )
        rw_likelihood = setup_likelihood(rw_config, rw_transform)
        rw_sampler = create_sampler(
            config=rw_config.sampler,
            prior=rw_prior,
            likelihood=rw_likelihood,
            likelihood_transforms=[rw_transform],
            seed=rw_config.seed,
        )
        rw_sampler.sample(jax.random.PRNGKey(rw_config.seed))
        rw_output = rw_sampler.get_sampler_output()

        # Compare posterior mean of a couple of NEPs; loose tolerance given
        # small particle counts (lightweight e2e config) and two genuinely
        # different SMC schedules.
        for param in ["K_sat", "L_sym"]:
            pp_mean = float(jnp.mean(pp_output.samples[param]))
            rw_mean = float(jnp.mean(rw_output.samples[param]))
            param_range = {
                "K_sat": 300.0 - 150.0,
                "L_sym": 200.0 - 10.0,
            }[param]
            assert abs(pp_mean - rw_mean) < 0.5 * param_range, (
                f"{param}: partial-posteriors mean {pp_mean:.2f} vs "
                f"smc-rw mean {rw_mean:.2f} differ by more than half the "
                "prior range"
            )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.blackjax
class TestSMCPartialPosteriorsWarmStart:
    """End-to-end tests for Phase 2: warm_start_from.

    A first run assimilates only GW170817 and is saved to HDF5. A second
    run's config points `warm_start_from` at that HDF5 file and adds
    GW190425 -- it should resample its initial particles from the first
    run's posterior and skip re-assimilating GW170817 entirely.
    """

    def test_warm_start_skips_already_covered_event(
        self, smc_partial_posteriors_gw_config, e2e_temp_dir
    ):
        # Stage 1: assimilate only GW170817.
        stage1_dict = copy.deepcopy(smc_partial_posteriors_gw_config)
        stage1_dict["likelihoods"][1]["events"] = [{"name": "GW170817"}]
        stage1_config, stage1_sampler = _run_partial_posteriors(stage1_dict)

        stage1_result = InferenceResult.from_sampler(
            stage1_sampler, stage1_config, runtime=0.0
        )
        stage1_path = e2e_temp_dir / "stage1_result.h5"
        stage1_result.save(stage1_path)

        # Stage 2: warm-start from stage 1, add GW190425.
        stage2_dict = copy.deepcopy(smc_partial_posteriors_gw_config)
        stage2_dict["likelihoods"][1]["events"] = [
            {"name": "GW170817"},
            {"name": "GW190425"},
        ]
        stage2_dict["sampler"]["warm_start_from"] = str(stage1_path)
        _stage2_config, stage2_sampler = _run_partial_posteriors(stage2_dict)

        stage2_metadata = stage2_sampler.metadata  # type: ignore[attr-defined]
        assert stage2_metadata["event_order"] == ["GW170817", "GW190425"]
        assert stage2_metadata["n_events_replayed"] == 1
        assert stage2_metadata["warm_start_from"] == str(stage1_path)
        # Only the new event (GW190425) is stepped through, not GW170817.
        assert len(stage2_metadata["ess_history"]) == 1
        assert len(stage2_metadata["acceptance_history"]) == 1

        output = stage2_sampler.get_sampler_output()
        validate_sampler_output(output, expected_params=NEP_PARAMS, min_samples=50)

        # Regression test: plot_diagnostics() crashed on a warm-started run
        # because its x-axis used the full event_order (length 2) while
        # ess_history/acceptance_history/log_evidence_history only cover
        # the newly-assimilated events (length 1) -- matplotlib raised
        # "x and y must have same first dimension, but have shapes (2,)
        # and (1,)". Must not raise, and must produce both plot files.
        plot_outdir = e2e_temp_dir / "plots"
        stage2_sampler.plot_diagnostics(outdir=plot_outdir)  # type: ignore[attr-defined]
        assert (plot_outdir / "smc_diagnostics.png").exists()
        assert (plot_outdir / "substep_diagnostics" / "01_GW190425.png").exists()

    def test_warm_start_rejects_non_prefix_event_order(
        self, smc_partial_posteriors_gw_config, e2e_temp_dir
    ):
        """A previous run whose event_order isn't a strict prefix of the
        new run's event_order must fail fast, not silently misalign masks.
        """
        stage1_dict = copy.deepcopy(smc_partial_posteriors_gw_config)
        stage1_dict["likelihoods"][1]["events"] = [{"name": "GW190425"}]
        stage1_config, stage1_sampler = _run_partial_posteriors(stage1_dict)

        stage1_result = InferenceResult.from_sampler(
            stage1_sampler, stage1_config, runtime=0.0
        )
        stage1_path = e2e_temp_dir / "stage1_mismatched_result.h5"
        stage1_result.save(stage1_path)

        stage2_dict = copy.deepcopy(smc_partial_posteriors_gw_config)
        stage2_dict["likelihoods"][1]["events"] = [
            {"name": "GW170817"},
            {"name": "GW190425"},
        ]
        stage2_dict["sampler"]["warm_start_from"] = str(stage1_path)

        with pytest.raises(ValueError, match="strict prefix"):
            _run_partial_posteriors(stage2_dict)

    def test_warm_start_rejects_mismatched_always_on_likelihoods(
        self, smc_partial_posteriors_gw_config, e2e_temp_dir
    ):
        """If the always-on (non-GW) likelihoods differ between the
        warm-start source and this run, the cumulative logZ bookkeeping
        would be silently wrong -- must fail fast instead.
        """
        stage1_dict = copy.deepcopy(smc_partial_posteriors_gw_config)
        stage1_dict["likelihoods"][1]["events"] = [{"name": "GW170817"}]
        stage1_config, stage1_sampler = _run_partial_posteriors(stage1_dict)

        stage1_result = InferenceResult.from_sampler(
            stage1_sampler, stage1_config, runtime=0.0
        )
        stage1_path = e2e_temp_dir / "stage1_mismatched_always_on.h5"
        stage1_result.save(stage1_path)

        stage2_dict = copy.deepcopy(smc_partial_posteriors_gw_config)
        stage2_dict["likelihoods"][1]["events"] = [
            {"name": "GW170817"},
            {"name": "GW190425"},
        ]
        # Stage 1 had the always-on constraints_eos likelihood enabled;
        # disabling it here means the always-on likelihood sets no longer
        # match between the two runs.
        stage2_dict["likelihoods"][0]["enabled"] = False
        stage2_dict["sampler"]["warm_start_from"] = str(stage1_path)

        with pytest.raises(ValueError, match="always-on"):
            _run_partial_posteriors(stage2_dict)

    def test_warm_start_from_gw_free_run(
        self, smc_partial_posteriors_gw_config, e2e_temp_dir
    ):
        """warm_start_from also works when the source run had zero GW
        events (e.g. a radio- or ChiEFT-only run in a sequential-updating
        workflow): the empty event list is trivially a strict prefix, so
        every configured GW event is still assimilated from scratch, only
        the initial particles come from the source run's posterior instead
        of the prior.
        """
        # Stage 1: no GW events at all, only the always-on constraints_eos
        # likelihood (matching stage 2's always-on set below). Uses smc-rw
        # since partial-posteriors itself requires at least one GW event.
        stage1_dict = copy.deepcopy(smc_partial_posteriors_gw_config)
        stage1_dict["likelihoods"][1]["enabled"] = False
        stage1_dict["sampler"] = {
            "type": "smc-rw",
            "output_dir": stage1_dict["sampler"]["output_dir"],
            "n_eos_samples": stage1_dict["sampler"]["n_eos_samples"],
            "n_particles": 100,
            "n_mcmc_steps": 3,
            "target_ess": 0.9,
            "random_walk_sigma": 0.1,
        }
        stage1_config, stage1_sampler = _run_partial_posteriors(stage1_dict)
        assert stage1_sampler.metadata["kernel_type"] == "random_walk"  # type: ignore[attr-defined]

        stage1_result = InferenceResult.from_sampler(
            stage1_sampler, stage1_config, runtime=0.0
        )
        stage1_path = e2e_temp_dir / "stage1_gw_free_result.h5"
        stage1_result.save(stage1_path)

        # Stage 2: partial-posteriors run warm-starting from the GW-free
        # posterior, assimilating both configured events from scratch.
        stage2_dict = copy.deepcopy(smc_partial_posteriors_gw_config)
        stage2_dict["sampler"]["warm_start_from"] = str(stage1_path)
        _stage2_config, stage2_sampler = _run_partial_posteriors(stage2_dict)

        stage2_metadata = stage2_sampler.metadata  # type: ignore[attr-defined]
        assert stage2_metadata["event_order"] == ["GW170817", "GW190425"]
        assert stage2_metadata["n_events_replayed"] == 0
        assert stage2_metadata["warm_start_from"] == str(stage1_path)
        # Both events are stepped through -- nothing was already covered.
        assert len(stage2_metadata["ess_history"]) == 2
        assert len(stage2_metadata["acceptance_history"]) == 2

        output = stage2_sampler.get_sampler_output()
        validate_sampler_output(output, expected_params=NEP_PARAMS, min_samples=50)

    def test_warm_start_from_smc_rw_result(
        self, smc_rw_gw_config, smc_partial_posteriors_gw_config, e2e_temp_dir
    ):
        """warm_start_from doesn't require the source run to be a
        partial-posteriors run. A converged ``smc-rw`` posterior (tempering
        the full combined likelihood by lambda, not by event) works too,
        since the "already covered" events are now derived from the saved
        config's ``likelihoods`` section rather than sampler-specific
        ``event_order`` metadata that only the partial-posteriors sampler
        used to write.
        """
        # Stage 1: plain smc-rw run over GW170817 alone (lambda-tempered).
        stage1_dict = copy.deepcopy(smc_rw_gw_config)
        stage1_dict["likelihoods"][1]["events"] = [{"name": "GW170817"}]
        stage1_config = InferenceConfig(**stage1_dict)
        prior, _fixed_params = setup_prior(stage1_config)
        keep_names = determine_keep_names(stage1_config, prior)
        transform = setup_transform(stage1_config, prior=prior, keep_names=keep_names)
        likelihood = setup_likelihood(stage1_config, transform)
        stage1_sampler = create_sampler(
            config=stage1_config.sampler,
            prior=prior,
            likelihood=likelihood,
            likelihood_transforms=[transform],
            seed=stage1_config.seed,
            likelihood_configs=stage1_config.likelihoods,
        )
        stage1_sampler.sample(jax.random.PRNGKey(stage1_config.seed))
        assert stage1_sampler.metadata["kernel_type"] == "random_walk"  # type: ignore[attr-defined]

        stage1_result = InferenceResult.from_sampler(
            stage1_sampler, stage1_config, runtime=0.0
        )
        stage1_path = e2e_temp_dir / "stage1_smc_rw_result.h5"
        stage1_result.save(stage1_path)

        # Stage 2: partial-posteriors run warm-starting from the smc-rw
        # posterior, adding GW190425 via the adaptive ramp-in.
        stage2_dict = copy.deepcopy(smc_partial_posteriors_gw_config)
        stage2_dict["likelihoods"][1]["events"] = [
            {"name": "GW170817"},
            {"name": "GW190425"},
        ]
        stage2_dict["sampler"]["warm_start_from"] = str(stage1_path)
        _stage2_config, stage2_sampler = _run_partial_posteriors(stage2_dict)

        stage2_metadata = stage2_sampler.metadata  # type: ignore[attr-defined]
        assert stage2_metadata["event_order"] == ["GW170817", "GW190425"]
        assert stage2_metadata["n_events_replayed"] == 1
        # Only the new event (GW190425) is stepped through, not GW170817.
        assert len(stage2_metadata["ess_history"]) == 1

        output = stage2_sampler.get_sampler_output()
        validate_sampler_output(output, expected_params=NEP_PARAMS, min_samples=50)
