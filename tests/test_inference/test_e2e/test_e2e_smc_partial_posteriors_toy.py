"""E2E regression tests for the partial-posteriors SMC sampler using
locally-trained toy normalizing flows instead of the bundled GW170817/
GW190425 presets.

The bundled presets (used by ``smc_partial_posteriors_gw_config`` in
conftest.py) currently fail to load in this environment with an
``equinox.tree_deserialise_leaves`` shape mismatch between the checked-in
weights and the architecture ``create_flow`` builds today -- a pre-existing
issue unrelated to the per-batch active-event-set construction these tests
exercise (see ``jesterTOV/inference/samplers/blackjax/smc/partial_posteriors.py``:
``_build_ordered_event_vals_fn``, and
``jesterTOV/inference/likelihoods/gw.py``: ``StackedGWLikelihood.subset``).
These tests give equivalent coverage using freshly trained toy flows
(``_save_toy_flow``, dim=4, matching a GW event's (m1, m2, Lambda1, Lambda2)
likelihood), so that construction keeps a real, running multi-event,
multi-batch regression test independent of the preset issue.
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

from tests.test_inference.test_likelihoods import _save_toy_flow

from .conftest import (
    validate_sampler_output,
    NEP_PARAMS,
    LIGHTWEIGHT_EOS,
    LIGHTWEIGHT_TOV,
    SMC_PARTIAL_POSTERIORS_LIGHTWEIGHT,
    SMC_RW_LIGHTWEIGHT,
)


def _toy_gw_config(
    tmp_path,
    prior_file,
    outdir,
    sampler_config: dict,
    n_events: int = 3,
) -> dict:
    """A GW config identical in shape to conftest's ``build_gw_config``, but
    with ``n_events`` freshly-trained toy flows instead of the (currently
    broken) GW170817/GW190425 presets."""
    model_dirs = [
        _save_toy_flow(tmp_path / f"toy_event_{i}", seed=i) for i in range(n_events)
    ]
    events = [
        {"name": f"toy_event_{i}", "nf_model_dir": str(model_dirs[i])}
        for i in range(n_events)
    ]
    return {
        "seed": 42,
        "dry_run": False,
        "validate_only": False,
        "eos": {"type": "metamodel", "nb_CSE": 0, **LIGHTWEIGHT_EOS},
        "tov": {"type": "gr", **LIGHTWEIGHT_TOV},
        "prior": {"specification_file": str(prior_file)},
        "likelihoods": [
            {"type": "constraints_eos", "enabled": True},
            {
                "type": "gw",
                "enabled": True,
                "events": events,
                "N_masses_evaluation": 20,
                "N_masses_batch_size": 10,
            },
        ],
        "sampler": {**sampler_config, "output_dir": str(outdir), "n_eos_samples": 50},
        "postprocessing": {"enabled": False},
    }


def _run(config_dict: dict) -> tuple:
    config = InferenceConfig(**config_dict)
    prior, fixed_params = setup_prior(config)
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
    return config, sampler, fixed_params


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.blackjax
class TestSMCPartialPosteriorsToyFlowsE2E:
    """Exercises the per-batch active-event-set construction with a real,
    controllable multi-event likelihood: 3 toy events, so the active set
    genuinely grows across 3 (cadence=1), 2 (cadence=2), or a
    dynamically-decided number (cadence="auto") of data-tempering batches.
    """

    def test_three_events_cadence_one_grows_active_set_each_batch(
        self, tmp_path, minimal_prior_file, e2e_temp_dir
    ):
        sampler_config = {
            "type": "smc-partial-posteriors-rw",
            **SMC_PARTIAL_POSTERIORS_LIGHTWEIGHT,
        }
        config_dict = _toy_gw_config(
            tmp_path, minimal_prior_file, e2e_temp_dir, sampler_config, n_events=3
        )
        _config, sampler, _fixed = _run(config_dict)

        metadata = sampler.metadata  # type: ignore[attr-defined]
        assert metadata["event_order"] == [
            "toy_event_0",
            "toy_event_1",
            "toy_event_2",
        ]
        assert metadata["n_events"] == 3
        assert metadata["event_groups"] == [
            ["toy_event_0"],
            ["toy_event_1"],
            ["toy_event_2"],
        ]
        # One data-tempering step per event -- the active set grows from 1
        # event (batch_01) to 3 (batch_03).
        assert len(metadata["ess_history"]) == 3
        assert all(n >= 1 for n in metadata["n_substeps_history"])
        assert jnp.isfinite(jnp.array(metadata["logZ"]))

        output = sampler.get_sampler_output()
        validate_sampler_output(output, expected_params=NEP_PARAMS, min_samples=50)

    def test_three_events_cadence_two_groups_first_two_events(
        self, tmp_path, minimal_prior_file, e2e_temp_dir
    ):
        sampler_config = {
            "type": "smc-partial-posteriors-rw",
            **SMC_PARTIAL_POSTERIORS_LIGHTWEIGHT,
            "cadence": 2,
        }
        config_dict = _toy_gw_config(
            tmp_path, minimal_prior_file, e2e_temp_dir, sampler_config, n_events=3
        )
        _config, sampler, _fixed = _run(config_dict)

        metadata = sampler.metadata  # type: ignore[attr-defined]
        # cadence=2 -> groups of [2, 1]: batch_01 jumps the active set
        # straight from 0 to 2 events, batch_02 adds the 3rd.
        assert metadata["event_groups"] == [
            ["toy_event_0", "toy_event_1"],
            ["toy_event_2"],
        ]
        assert len(metadata["ess_history"]) == 2

        output = sampler.get_sampler_output()
        validate_sampler_output(output, expected_params=NEP_PARAMS, min_samples=50)

    def test_three_events_auto_cadence_high_threshold_matches_cadence_one(
        self, tmp_path, minimal_prior_file, e2e_temp_dir
    ):
        """A threshold of 1.0 triggers on every event (see the analogous
        preset-based test), so this should reproduce the cadence=1 batch
        structure while exercising the auto-cadence look-ahead predictor's
        own per-candidate active-set construction."""
        sampler_config = {
            "type": "smc-partial-posteriors-rw",
            **SMC_PARTIAL_POSTERIORS_LIGHTWEIGHT,
            "cadence": "auto",
            "auto_ess_threshold": 1.0,
        }
        config_dict = _toy_gw_config(
            tmp_path, minimal_prior_file, e2e_temp_dir, sampler_config, n_events=3
        )
        _config, sampler, _fixed = _run(config_dict)

        metadata = sampler.metadata  # type: ignore[attr-defined]
        assert metadata["event_groups"] == [
            ["toy_event_0"],
            ["toy_event_1"],
            ["toy_event_2"],
        ]
        assert len(metadata["auto_cadence_ess_history"]) == 3
        assert all(metadata["auto_cadence_triggered_history"])

        output = sampler.get_sampler_output()
        validate_sampler_output(output, expected_params=NEP_PARAMS, min_samples=50)

    def test_warm_start_only_replays_new_events_active_set(
        self, tmp_path, minimal_prior_file, e2e_temp_dir
    ):
        """Stage 1 assimilates 1 toy event; stage 2 warm-starts and adds 2
        more. Regression target for the active-set bookkeeping across a
        warm start: stage 2's batch_02 compute graph must cover exactly the
        already-assimilated event plus the 2 new ones (3 total), not stage
        1's event replayed from scratch and not the full configured set
        evaluated from batch 1."""
        sampler_config = {
            "type": "smc-partial-posteriors-rw",
            **SMC_PARTIAL_POSTERIORS_LIGHTWEIGHT,
        }
        full_config_dict = _toy_gw_config(
            tmp_path, minimal_prior_file, e2e_temp_dir, sampler_config, n_events=3
        )

        stage1_dict = copy.deepcopy(full_config_dict)
        stage1_dict["likelihoods"][1]["events"] = full_config_dict["likelihoods"][1][
            "events"
        ][:1]
        stage1_config, stage1_sampler, stage1_fixed = _run(stage1_dict)

        stage1_result = InferenceResult.from_sampler(
            stage1_sampler, stage1_config, runtime=0.0, fixed_params=stage1_fixed
        )
        stage1_path = e2e_temp_dir / "toy_stage1_result.h5"
        stage1_result.save(stage1_path)

        stage2_dict = copy.deepcopy(full_config_dict)
        stage2_dict["sampler"]["warm_start_from"] = str(stage1_path)
        _stage2_config, stage2_sampler, _stage2_fixed = _run(stage2_dict)

        stage2_metadata = stage2_sampler.metadata  # type: ignore[attr-defined]
        assert stage2_metadata["event_order"] == [
            "toy_event_0",
            "toy_event_1",
            "toy_event_2",
        ]
        assert stage2_metadata["n_events_replayed"] == 1
        # Only the 2 new events are stepped through.
        assert len(stage2_metadata["ess_history"]) == 2
        assert stage2_metadata["event_groups"] == [["toy_event_1"], ["toy_event_2"]]

        output = stage2_sampler.get_sampler_output()
        validate_sampler_output(output, expected_params=NEP_PARAMS, min_samples=50)

    def test_consistent_with_smc_rw_on_same_toy_likelihood(
        self, tmp_path, minimal_prior_file, e2e_temp_dir
    ):
        """Cross-check against lambda-tempered smc-rw on the identical toy
        combined likelihood -- same consistency argument as
        test_partial_posteriors_consistent_with_smc_rw, just not blocked by
        the broken presets."""
        pp_sampler_config = {
            "type": "smc-partial-posteriors-rw",
            **SMC_PARTIAL_POSTERIORS_LIGHTWEIGHT,
        }
        pp_config_dict = _toy_gw_config(
            tmp_path,
            minimal_prior_file,
            e2e_temp_dir / "pp",
            pp_sampler_config,
            n_events=2,
        )
        _pp_config, pp_sampler, _pp_fixed = _run(pp_config_dict)
        pp_output = pp_sampler.get_sampler_output()

        rw_sampler_config = {"type": "smc-rw", **SMC_RW_LIGHTWEIGHT}
        rw_config_dict = copy.deepcopy(pp_config_dict)
        rw_config_dict["sampler"] = {
            **rw_sampler_config,
            "output_dir": str(e2e_temp_dir / "rw"),
            "n_eos_samples": 50,
        }
        _rw_config, rw_sampler, _rw_fixed = _run(rw_config_dict)
        rw_output = rw_sampler.get_sampler_output()

        for param in ["K_sat", "L_sym"]:
            pp_mean = float(jnp.mean(pp_output.samples[param]))
            rw_mean = float(jnp.mean(rw_output.samples[param]))
            param_range = {"K_sat": 300.0 - 150.0, "L_sym": 200.0 - 10.0}[param]
            assert abs(pp_mean - rw_mean) < 0.5 * param_range, (
                f"{param}: partial-posteriors mean {pp_mean:.2f} vs "
                f"smc-rw mean {rw_mean:.2f} differ by more than half the "
                "prior range"
            )
