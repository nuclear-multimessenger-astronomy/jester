"""Tests for inference configuration system (parser and schema)."""

import pytest
import yaml
import jax
from pathlib import Path
from pydantic import ValidationError

from jesterTOV.inference.config import schema, parser


class TestEOSConfig:
    """Test EOS configuration validation."""

    def test_valid_metamodel_config(self):
        """Test valid metamodel configuration."""
        config = schema.MetamodelEOSConfig(
            type="metamodel",
            ndat_metamodel=100,
            nmax_nsat=2.0,
            nb_CSE=0,
            crust_name="DH",
        )
        assert config.type == "metamodel"
        assert config.nb_CSE == 0
        assert config.ndat_metamodel == 100

    def test_valid_metamodel_cse_config(self):
        """Test valid metamodel_cse configuration."""
        config = schema.MetamodelCSEEOSConfig(
            type="metamodel_cse",
            ndat_metamodel=100,
            nmax_nsat=25.0,
            nb_CSE=8,
            crust_name="DH",
        )
        assert config.type == "metamodel_cse"
        assert config.nb_CSE == 8
        assert config.nmax_nsat == 25.0

    def test_valid_spectral_config(self):
        """Test valid spectral configuration."""
        config = schema.SpectralEOSConfig(
            type="spectral",
            n_points_high=500,
            crust_name="SLy",
        )
        assert config.type == "spectral"
        assert config.n_points_high == 500
        assert config.crust_name == "SLy"

    def test_metamodel_with_nonzero_cse_fails(self):
        """Test that metamodel with nb_CSE != 0 fails validation."""
        with pytest.raises(ValidationError, match="nb_CSE must be 0"):
            schema.MetamodelEOSConfig(
                type="metamodel",
                nb_CSE=8,  # Should fail for type=metamodel
            )

    def test_metamodel_cse_with_zero_cse_fails(self):
        """Test that metamodel_cse with nb_CSE = 0 fails validation."""
        with pytest.raises(ValidationError, match="nb_CSE must be > 0"):
            schema.MetamodelCSEEOSConfig(
                type="metamodel_cse",
                nb_CSE=0,  # Should fail for type=metamodel_cse
            )

    def test_spectral_with_nonzero_cse_fails(self):
        """Test that spectral with nb_CSE != 0 fails validation."""
        with pytest.raises(ValidationError, match="nb_CSE must be 0"):
            schema.SpectralEOSConfig(
                type="spectral",
                nb_CSE=8,  # Should fail for type=spectral
            )

    def test_invalid_crust_name(self):
        """Test that invalid crust names fail validation."""
        with pytest.raises(ValidationError):
            schema.MetamodelEOSConfig(
                type="metamodel",
                crust_name="InvalidCrust",  # type: ignore[arg-type]  # intentionally wrong
            )

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = schema.MetamodelEOSConfig(type="metamodel", nb_CSE=0)
        assert config.ndat_metamodel == 100
        assert config.nmax_nsat == 25.0
        assert config.crust_name == "DH"

    def test_base_metamodel_eos_config_hierarchy(self):
        """Test that MetamodelEOSConfig and MetamodelCSEEOSConfig share BaseMetamodelEOSConfig."""
        mm = schema.MetamodelEOSConfig(type="metamodel", nb_CSE=0)
        cse = schema.MetamodelCSEEOSConfig(type="metamodel_cse", nb_CSE=8)
        spectral = schema.SpectralEOSConfig(type="spectral", crust_name="SLy")

        assert isinstance(mm, schema.BaseMetamodelEOSConfig)
        assert isinstance(cse, schema.BaseMetamodelEOSConfig)
        assert not isinstance(spectral, schema.BaseMetamodelEOSConfig)

    def test_metamodel_cse_ndat_cse_default(self):
        """Test that ndat_CSE defaults to 100 for MetamodelCSEEOSConfig."""
        config = schema.MetamodelCSEEOSConfig(type="metamodel_cse", nb_CSE=8)
        assert config.ndat_CSE == 100

    def test_metamodel_cse_ndat_cse_custom(self):
        """Test that ndat_CSE can be overridden for MetamodelCSEEOSConfig."""
        config = schema.MetamodelCSEEOSConfig(
            type="metamodel_cse", nb_CSE=8, ndat_CSE=50
        )
        assert config.ndat_CSE == 50

    def test_metamodel_cse_max_nbreak_nsat(self):
        """Test that max_nbreak_nsat can be set optionally on MetamodelCSEEOSConfig."""
        config_without = schema.MetamodelCSEEOSConfig(type="metamodel_cse", nb_CSE=8)
        assert config_without.max_nbreak_nsat is None

        config_with = schema.MetamodelCSEEOSConfig(
            type="metamodel_cse", nb_CSE=8, max_nbreak_nsat=2.0
        )
        assert config_with.max_nbreak_nsat == 2.0

    def test_eos_discriminated_union(self):
        """Test that EOSConfig discriminated union works correctly."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(schema.EOSConfig)

        # Test metamodel
        metamodel_dict = {"type": "metamodel", "nb_CSE": 0}
        config = adapter.validate_python(metamodel_dict)
        assert isinstance(config, schema.MetamodelEOSConfig)

        # Test metamodel_cse
        cse_dict = {"type": "metamodel_cse", "nb_CSE": 8}
        config = adapter.validate_python(cse_dict)
        assert isinstance(config, schema.MetamodelCSEEOSConfig)

        # Test spectral
        spectral_dict = {"type": "spectral", "crust_name": "SLy"}
        config = adapter.validate_python(spectral_dict)
        assert isinstance(config, schema.SpectralEOSConfig)


class TestTOVConfig:
    """Test TOV configuration validation."""

    def test_valid_tov_config(self):
        """Test valid TOV configuration."""
        config = schema.GRTOVConfig(
            type="gr",
            min_nsat_TOV=0.75,
            ndat_TOV=100,
            nb_masses=100,
        )
        assert config.type == "gr"
        assert config.min_nsat_TOV == 0.75
        assert config.ndat_TOV == 100

    def test_tov_default_values(self):
        """Test that TOV default values are set correctly."""
        config = schema.GRTOVConfig()
        assert config.type == "gr"
        assert config.min_nsat_TOV == 0.75
        assert config.ndat_TOV == 100
        assert config.nb_masses == 100

    def test_invalid_solver_type_fails(self):
        """Test that invalid TOV solver type fails validation."""
        from pydantic import TypeAdapter

        ta: TypeAdapter[schema.TOVConfig] = TypeAdapter(schema.TOVConfig)  # type: ignore[type-arg]
        with pytest.raises(ValidationError):
            ta.validate_python({"type": "invalid_solver"})


class TestPriorConfig:
    """Test PriorConfig validation."""

    def test_valid_prior_config(self):
        """Test valid prior configuration."""
        config = schema.PriorConfig(specification_file="test.prior")
        assert config.specification_file == "test.prior"

    def test_prior_without_extension_fails(self):
        """Test that prior files without .prior extension fail."""
        with pytest.raises(ValidationError, match="must have .prior extension"):
            schema.PriorConfig(specification_file="test.txt")

    def test_prior_with_wrong_extension_fails(self):
        """Test that prior files with wrong extension fail."""
        with pytest.raises(ValidationError, match="must have .prior extension"):
            schema.PriorConfig(specification_file="test.yaml")


class TestLikelihoodConfig:
    """Test LikelihoodConfig validation."""

    def test_zero_likelihood_config(self):
        """Test zero likelihood configuration (for prior-only sampling)."""
        config = schema.ZeroLikelihoodConfig(
            enabled=True,
        )
        assert config.type == "zero"
        assert config.enabled is True

    def test_gw_likelihood_config(self):
        """Test GW likelihood configuration."""
        config = schema.GWLikelihoodConfig(
            enabled=True,
            events=[{"name": "GW170817", "nf_model_dir": "/path/to/data"}],
            N_masses_evaluation=20,
        )
        assert config.type == "gw"
        assert len(config.events) == 1
        assert config.events[0].nf_model_dir == "/path/to/data"
        assert config.penalty_value == 0.0

    def test_gw_likelihood_missing_event_name_fails(self):
        """Test that GW likelihood without event name fails."""
        with pytest.raises(ValidationError):
            schema.GWLikelihoodConfig(
                events=[{"nf_model_dir": "/path/to/data"}],  # Missing 'name'
            )

    def test_gw_likelihood_empty_events_fails(self):
        """Test that GW likelihood with empty events list fails."""
        with pytest.raises(ValidationError):
            schema.GWLikelihoodConfig(
                events=[],  # Empty list
            )

    def test_nicer_likelihood_config(self):
        """Test NICER flow-based likelihood configuration."""
        config = schema.NICERLikelihoodConfig(
            enabled=True,
            pulsars=[
                {
                    "name": "J0030",
                    "amsterdam_model_dir": "/path/to/amsterdam_flow",
                    "maryland_model_dir": "/path/to/maryland_flow",
                }
            ],
            N_masses_evaluation=100,
        )
        assert config.type == "nicer"
        assert len(config.pulsars) == 1

    def test_nicer_kde_likelihood_config(self):
        """Test NICER KDE-based likelihood configuration."""
        config = schema.NICERKDELikelihoodConfig(
            enabled=True,
            pulsars=[
                {
                    "name": "J0030",
                    "amsterdam_samples_file": "/path/to/amsterdam.txt",
                    "maryland_samples_file": "/path/to/maryland.txt",
                }
            ],
            N_masses_evaluation=100,
        )
        assert config.type == "nicer_kde"
        assert len(config.pulsars) == 1

    def test_nicer_kde_likelihood_missing_files_fails(self):
        """Test that NICER KDE likelihood without sample files fails."""
        with pytest.raises(
            ValidationError, match="missing required 'amsterdam_samples_file' field"
        ):
            schema.NICERKDELikelihoodConfig(
                pulsars=[
                    {
                        "name": "J0030",
                        # Missing sample files
                    }
                ],
            )

    def test_radio_likelihood_config(self):
        """Test radio timing likelihood configuration."""
        config = schema.RadioLikelihoodConfig(
            enabled=True,
            pulsars=[{"name": "J0348+0432", "mass_mean": 2.01, "mass_std": 0.04}],
        )
        assert config.type == "radio"
        assert config.pulsars[0]["mass_mean"] == 2.01

    def test_radio_likelihood_missing_mass_fails(self):
        """Test that radio likelihood without mass parameters fails."""
        with pytest.raises(ValidationError, match="missing required fields"):
            schema.RadioLikelihoodConfig(
                pulsars=[{"name": "J0348+0432"}],  # Missing mass_mean and mass_std
            )

    def test_radio_likelihood_negative_std_fails(self):
        """Test that radio likelihood with negative mass_std fails."""
        with pytest.raises(ValidationError, match="must be a positive number"):
            schema.RadioLikelihoodConfig(
                pulsars=[{"name": "J0348+0432", "mass_mean": 2.01, "mass_std": -0.04}],
            )

    def test_chieft_likelihood_config(self):
        """Test ChiEFT likelihood configuration."""
        config = schema.ChiEFTLikelihoodConfig(
            enabled=True,
            low_filename="/path/to/low.dat",
            high_filename="/path/to/high.dat",
            nb_n=100,
        )
        assert config.type == "chieft"
        assert config.nb_n == 100

    def test_eos_constraints_config(self):
        """Test EOS constraints likelihood configuration."""
        config = schema.EOSConstraintsLikelihoodConfig(
            enabled=True,
            penalty_causality=-1e10,
            penalty_stability=-1e10,
        )
        assert config.type == "constraints_eos"
        assert config.penalty_causality == -1e10

    def test_tov_constraints_config(self):
        """Test TOV constraints likelihood configuration."""
        config = schema.TOVConstraintsLikelihoodConfig(
            enabled=True,
            penalty_tov=-1e10,
        )
        assert config.type == "constraints_tov"
        assert config.penalty_tov == -1e10

    def test_discriminated_union_from_dict(self):
        """Test that discriminated union works with dict input."""
        from pydantic import TypeAdapter

        # Create type adapter for LikelihoodConfig union
        adapter = TypeAdapter(schema.LikelihoodConfig)

        # Test GW likelihood with just a name (uses preset)
        gw_dict = {
            "type": "gw",
            "enabled": True,
            "events": [{"name": "GW170817"}],
        }
        gw_config = adapter.validate_python(gw_dict)
        assert isinstance(gw_config, schema.GWLikelihoodConfig)

        # Test NICER flow-based likelihood
        nicer_dict = {
            "type": "nicer",
            "enabled": True,
            "pulsars": [
                {
                    "name": "J0030",
                    "amsterdam_model_dir": "/path/to/amsterdam_flow",
                    "maryland_model_dir": "/path/to/maryland_flow",
                }
            ],
        }
        nicer_config = adapter.validate_python(nicer_dict)
        assert isinstance(nicer_config, schema.NICERLikelihoodConfig)

        # Test Zero likelihood
        zero_dict = {"type": "zero"}
        zero_config = adapter.validate_python(zero_dict)
        assert isinstance(zero_config, schema.ZeroLikelihoodConfig)

    def test_disabled_likelihood(self):
        """Test that likelihoods can be disabled."""
        config = schema.GWLikelihoodConfig(
            enabled=False,
            events=[{"name": "GW170817"}],
        )
        assert config.enabled is False


class TestSamplerConfig:
    """Test SamplerConfig validation."""

    def test_valid_sampler_config(self):
        """Test valid sampler configuration."""
        config = schema.FlowMCSamplerConfig(
            n_chains=20,
            n_loop_training=2,
            n_loop_production=2,
            n_local_steps=10,
            n_global_steps=10,
            n_epochs=20,
            learning_rate=0.001,
            output_dir="./outdir/",
        )
        assert config.n_chains == 20
        assert config.learning_rate == 0.001

    def test_negative_chains_fails(self):
        """Test that negative number of chains fails validation."""
        with pytest.raises(ValidationError):
            schema.FlowMCSamplerConfig(
                n_chains=-1,  # Should fail
                n_loop_training=2,
                n_loop_production=2,
            )

    def test_zero_chains_fails(self):
        """Test that zero chains fails validation."""
        with pytest.raises(ValidationError):
            schema.FlowMCSamplerConfig(
                n_chains=0,  # Should fail
                n_loop_training=2,
                n_loop_production=2,
            )

    def test_negative_learning_rate_fails(self):
        """Test that negative learning rate fails validation."""
        with pytest.raises(ValidationError):
            schema.FlowMCSamplerConfig(
                n_chains=4,
                learning_rate=-0.001,  # Should fail
            )


class TestInferenceConfig:
    """Test InferenceConfig (top-level configuration)."""

    def test_valid_full_config(self, sample_config_dict):
        """Test valid full configuration."""
        config = schema.InferenceConfig(**sample_config_dict)
        assert config.seed == 42
        assert config.eos.type == "metamodel"
        assert config.tov.type == "gr"
        assert len(config.likelihoods) == 1
        assert config.sampler.n_chains == 4

    def test_config_with_multiple_likelihoods(self, sample_config_dict):
        """Test configuration with multiple likelihoods."""
        config_dict = sample_config_dict.copy()
        config_dict["likelihoods"] = [
            {
                "type": "gw",
                "enabled": True,
                "events": [{"name": "GW170817", "nf_model_dir": "/path/to/data"}],
            },
            {
                "type": "nicer",
                "enabled": True,
                "pulsars": [
                    {
                        "name": "J0030",
                        "amsterdam_model_dir": "/path/to/amsterdam_flow",
                        "maryland_model_dir": "/path/to/maryland_flow",
                    }
                ],
            },
            {
                "type": "radio",
                "enabled": False,
                "pulsars": [
                    {"name": "J0348+0432", "mass_mean": 2.01, "mass_std": 0.04}
                ],  # Still need valid data even when disabled
            },
        ]
        config = schema.InferenceConfig(**config_dict)
        assert len(config.likelihoods) == 3
        assert isinstance(config.likelihoods[0], schema.GWLikelihoodConfig)
        assert isinstance(config.likelihoods[1], schema.NICERLikelihoodConfig)
        assert isinstance(config.likelihoods[2], schema.RadioLikelihoodConfig)
        assert config.likelihoods[2].enabled is False

    def test_config_with_cse(self, sample_config_dict):
        """Test configuration with CSE enabled."""
        config_dict = sample_config_dict.copy()
        config_dict["eos"]["type"] = "metamodel_cse"
        config_dict["eos"]["nb_CSE"] = 8
        config = schema.InferenceConfig(**config_dict)
        assert config.eos.type == "metamodel_cse"
        assert config.eos.nb_CSE == 8

    def test_missing_required_field_fails(self):
        """Test that missing required fields fail validation."""
        with pytest.raises(ValidationError):
            schema.InferenceConfig(
                # Missing eos, tov, prior, etc.
                sampler={"type": "flowmc", "n_chains": 4},
            )

    def test_debug_nans_default_false(self, sample_config_dict):
        """Test that debug_nans defaults to False."""
        config = schema.InferenceConfig(**sample_config_dict)
        assert config.debug_nans is False

    def test_debug_nans_can_be_enabled(self, sample_config_dict):
        """Test that debug_nans can be set to True."""
        config_dict = sample_config_dict.copy()
        config_dict["debug_nans"] = True
        config = schema.InferenceConfig(**config_dict)
        assert config.debug_nans is True


class TestConfigParser:
    """Test configuration parser functionality."""

    def test_load_config_from_file(self, sample_config_file):
        """Test loading configuration from YAML file."""
        config = parser.load_config(sample_config_file)
        assert isinstance(config, schema.InferenceConfig)
        assert config.seed == 42
        assert config.eos.type == "metamodel"

    def test_load_config_with_relative_paths(self, temp_dir, sample_config_dict):
        """Test that relative paths in config are resolved correctly."""
        # Create prior file manually in temp_dir
        prior_file = temp_dir / "relative.prior"
        prior_content = """K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
E_sym = UniformPrior(28.0, 45.0, parameter_names=["E_sym"])
"""
        prior_file.write_text(prior_content)

        # Create config with relative path
        config_dict = sample_config_dict.copy()
        config_dict["prior"]["specification_file"] = "relative.prior"

        config_file = temp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        # Load config - parser should resolve relative path
        config = parser.load_config(config_file)
        assert config.prior.specification_file.endswith("relative.prior")
        # Verify it's an absolute path
        assert Path(config.prior.specification_file).is_absolute()

    def test_load_nonexistent_file_fails(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            parser.load_config("nonexistent_config.yaml")

    def test_load_invalid_yaml_fails(self, temp_dir):
        """Test that invalid YAML raises error."""
        invalid_yaml = temp_dir / "invalid.yaml"
        invalid_yaml.write_text("this is: not: valid: yaml:")

        with pytest.raises(yaml.YAMLError):
            parser.load_config(invalid_yaml)

    def test_load_config_missing_required_fields_fails(self, temp_dir):
        """Test that YAML missing required fields fails validation.

        Note: parser.load_config() wraps ValidationError in ValueError
        for better error messages, so we expect ValueError here.
        """
        incomplete_config = temp_dir / "incomplete.yaml"
        incomplete_config.write_text(
            """
seed: 42
# Missing transform, prior, likelihoods, sampler
"""
        )

        with pytest.raises(ValueError, match="Configuration error in"):
            parser.load_config(incomplete_config)


class TestExtraFieldValidation:
    """Test that config models reject extra/unknown fields."""

    def test_eos_config_rejects_extra_fields(self):
        """Test that EOS config rejects unknown fields."""
        with pytest.raises(ValidationError, match="Unrecognized field"):
            schema.MetamodelEOSConfig(
                type="metamodel",
                nb_CSE=0,
                wrong_entry=500,  # Should be rejected
            )

    def test_tov_config_rejects_extra_fields(self):
        """Test that TOV config rejects unknown fields."""
        with pytest.raises(ValidationError, match="Unrecognized field"):
            schema.GRTOVConfig(
                type="gr",
                wrong_entry=500,  # type: ignore[call-arg]  # Should be rejected
            )

    def test_prior_config_rejects_extra_fields(self):
        """Test that PriorConfig rejects unknown fields."""
        with pytest.raises(ValidationError, match="Unrecognized field"):
            schema.PriorConfig(
                specification_file="test.prior",
                invalid_param="value",  # Should be rejected
            )

    def test_likelihood_config_rejects_extra_fields(self):
        """Test that LikelihoodConfig rejects unknown fields."""
        with pytest.raises(ValidationError, match="Unrecognized field"):
            schema.ZeroLikelihoodConfig(
                enabled=True,
                extra_field="should_fail",  # Should be rejected
            )

    def test_sampler_config_rejects_extra_fields(self):
        """Test that FlowMCSamplerConfig rejects unknown fields."""
        with pytest.raises(ValidationError, match="Unrecognized field"):
            schema.FlowMCSamplerConfig(
                type="flowmc",
                n_chains=4,
                n_loop_training=2,
                n_loop_production=2,
                invalid_option=True,  # Should be rejected
            )

    def test_postprocessing_config_rejects_extra_fields(self):
        """Test that PostprocessingConfig rejects unknown fields."""
        with pytest.raises(ValidationError, match="Unrecognized field"):
            schema.PostprocessingConfig(
                enabled=True,
                make_cornerplot=True,
                unknown_plot_type=True,  # Should be rejected
            )

    def test_inference_config_rejects_extra_fields(self, sample_config_dict):
        """Test that InferenceConfig rejects unknown fields with a helpful message."""
        config_dict = sample_config_dict.copy()
        config_dict["random_invalid_field"] = "should_fail"

        with pytest.raises(ValidationError, match="Unrecognized field"):
            schema.InferenceConfig(**config_dict)

    def test_nested_extra_fields_rejected(self, sample_config_dict):
        """Test that extra fields in nested config sections are rejected with a helpful message."""
        config_dict = sample_config_dict.copy()
        config_dict["eos"]["wrong_entry"] = 500  # Should be rejected

        with pytest.raises(ValidationError, match="Unrecognized field"):
            schema.InferenceConfig(**config_dict)


class TestGWEventConfig:
    """Tests for the GWEventConfig Pydantic model."""

    def test_gw_event_nf_model_dir_mode(self):
        """Valid config with only name and nf_model_dir."""
        event = schema.GWEventConfig(name="GW170817", nf_model_dir="./my_flow")
        assert event.name == "GW170817"
        assert event.nf_model_dir == "./my_flow"
        assert event.from_bilby_result is None
        assert event.flow_config is None
        assert event.retrain_flow is False

    def test_gw_event_name_only(self):
        """Valid config with just name (uses preset)."""
        event = schema.GWEventConfig(name="GW170817")
        assert event.name == "GW170817"
        assert event.nf_model_dir is None
        assert event.from_bilby_result is None

    def test_gw_event_bilby_mode_minimal(self):
        """Valid config with just from_bilby_result (no flow_config, no nf_model_dir)."""
        event = schema.GWEventConfig(
            name="GW170817",
            from_bilby_result="./GW170817_result.hdf5",
        )
        assert event.from_bilby_result == "./GW170817_result.hdf5"
        assert event.nf_model_dir is None
        assert event.flow_config is None
        assert event.retrain_flow is False

    def test_gw_event_bilby_mode_full(self):
        """Valid bilby config with from_bilby_result, flow_config, and retrain_flow."""
        event = schema.GWEventConfig(
            name="GW170817",
            from_bilby_result="./GW170817_result.hdf5",
            flow_config="./flow_config.yaml",
            retrain_flow=True,
        )
        assert event.from_bilby_result == "./GW170817_result.hdf5"
        assert event.flow_config == "./flow_config.yaml"
        assert event.retrain_flow is True

    def test_gw_event_flow_config_without_bilby_raises(self):
        """flow_config without from_bilby_result raises ValidationError."""
        with pytest.raises(ValidationError, match="'flow_config' is only valid"):
            schema.GWEventConfig(
                name="GW170817",
                flow_config="./flow_config.yaml",
            )

    def test_gw_event_retrain_flow_without_bilby_raises(self):
        """retrain_flow=True without from_bilby_result raises ValidationError."""
        with pytest.raises(ValidationError, match="'retrain_flow' is only valid"):
            schema.GWEventConfig(
                name="GW170817",
                retrain_flow=True,
            )

    def test_gw_event_extra_field_rejected(self):
        """Unknown field raises ValidationError with a helpful message."""
        with pytest.raises(ValidationError, match="Unrecognized field"):
            schema.GWEventConfig(  # type: ignore[call-arg]
                name="GW170817",
                unknown_field="bad",
            )

    def test_gw_event_both_modes_raises(self):
        """Specifying both nf_model_dir and from_bilby_result raises ValidationError."""
        with pytest.raises(ValidationError, match="Cannot specify both"):
            schema.GWEventConfig(
                name="GW170817",
                nf_model_dir="./my_flow",
                from_bilby_result="./result.hdf5",
            )

    def test_gw_event_npz_mode_minimal(self):
        """Valid config with just from_npz_file (no flow_config, no nf_model_dir)."""
        event = schema.GWEventConfig(
            name="GW170817",
            from_npz_file="./GW170817_posterior.npz",
        )
        assert event.from_npz_file == "./GW170817_posterior.npz"
        assert event.nf_model_dir is None
        assert event.from_bilby_result is None
        assert event.flow_config is None
        assert event.retrain_flow is False

    def test_gw_event_npz_mode_full(self):
        """Valid NPZ config with from_npz_file, flow_config, and retrain_flow."""
        event = schema.GWEventConfig(
            name="GW170817",
            from_npz_file="./GW170817_posterior.npz",
            flow_config="./flow_config.yaml",
            retrain_flow=True,
        )
        assert event.from_npz_file == "./GW170817_posterior.npz"
        assert event.flow_config == "./flow_config.yaml"
        assert event.retrain_flow is True

    def test_gw_event_nf_model_dir_and_npz_raises(self):
        """Specifying both nf_model_dir and from_npz_file raises ValidationError."""
        with pytest.raises(ValidationError, match="Cannot specify both"):
            schema.GWEventConfig(
                name="GW170817",
                nf_model_dir="./my_flow",
                from_npz_file="./posterior.npz",
            )

    def test_gw_event_bilby_and_npz_raises(self):
        """Specifying both from_bilby_result and from_npz_file raises ValidationError."""
        with pytest.raises(ValidationError, match="Cannot specify both"):
            schema.GWEventConfig(
                name="GW170817",
                from_bilby_result="./result.hdf5",
                from_npz_file="./posterior.npz",
            )

    def test_gw_event_flow_config_with_npz_valid(self):
        """flow_config with from_npz_file is valid."""
        event = schema.GWEventConfig(
            name="GW170817",
            from_npz_file="./posterior.npz",
            flow_config="./flow_config.yaml",
        )
        assert event.flow_config == "./flow_config.yaml"

    def test_gw_event_retrain_flow_with_npz_valid(self):
        """retrain_flow=True with from_npz_file is valid."""
        event = schema.GWEventConfig(
            name="GW170817",
            from_npz_file="./posterior.npz",
            retrain_flow=True,
        )
        assert event.retrain_flow is True

    def test_gw_likelihood_config_with_event_objects(self):
        """GWLikelihoodConfig accepts a list of GWEventConfig objects."""
        events = [
            schema.GWEventConfig(name="GW170817"),
            schema.GWEventConfig(name="GW190425", nf_model_dir="./my_flow"),
        ]
        config = schema.GWLikelihoodConfig(events=events)
        assert len(config.events) == 2
        assert config.events[0].name == "GW170817"
        assert config.events[1].nf_model_dir == "./my_flow"

    def test_gw_likelihood_config_from_dict_with_nf_model_dir(self):
        """GWLikelihoodConfig constructed from YAML-style dict with nf_model_dir."""
        config = schema.GWLikelihoodConfig(
            events=[{"name": "GW170817", "nf_model_dir": "./flows/GW170817"}],  # type: ignore[arg-type]
        )
        assert config.events[0].name == "GW170817"
        assert config.events[0].nf_model_dir == "./flows/GW170817"

    def test_gw_likelihood_old_model_dir_rejected(self):
        """Using the old 'model_dir' key (instead of 'nf_model_dir') raises ValidationError."""
        with pytest.raises(ValidationError, match="Unrecognized field"):
            schema.GWLikelihoodConfig(
                events=[{"name": "GW170817", "model_dir": "./my_flow"}],  # type: ignore[arg-type]
            )


@pytest.mark.integration
class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_roundtrip_config_to_yaml_and_back(
        self, temp_dir, sample_config_dict, sample_prior_file
    ):
        """Test that config can be saved and loaded without changes."""
        # Update prior path
        config_dict = sample_config_dict.copy()
        config_dict["prior"]["specification_file"] = str(sample_prior_file)

        # Create config
        config1 = schema.InferenceConfig(**config_dict)

        # Save to file
        config_file = temp_dir / "roundtrip.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_dict, f)

        # Load back
        config2 = parser.load_config(config_file)

        # Compare key fields
        assert config1.seed == config2.seed
        assert config1.eos.type == config2.eos.type
        assert config1.tov.type == config2.tov.type
        # Type narrowing: we know from sample_config_dict that this is FlowMC
        assert config1.sampler.type == "flowmc"  # type: ignore[attr-defined]
        assert config2.sampler.type == "flowmc"  # type: ignore[attr-defined]

    def test_example_configs_are_valid(self):
        """Test that all example config files are valid.

        NOTE: This test may fail if example configs have invalid paths or
        are missing required data files. If so, document the issue in CLAUDE.md
        and investigate - do not just skip the test!
        """
        from pathlib import Path

        # Find example configs
        repo_root = Path(__file__).parent.parent.parent
        example_dir = repo_root / "examples" / "inference"

        if not example_dir.exists():
            pytest.skip(f"Example directory not found: {example_dir}")

        config_files = list(example_dir.glob("**/config.yaml"))
        assert len(config_files) > 0, "No example config files found"

        # Track issues instead of failing immediately
        issues = []

        for config_file in config_files:
            try:
                config = parser.load_config(config_file)
                assert isinstance(config, schema.InferenceConfig)
            except Exception as e:
                issues.append(f"{config_file.name}: {str(e)}")

        # If there are issues, document them
        if issues:
            error_msg = "\n".join(
                [
                    "Issues found in example configs:",
                    *[f"  - {issue}" for issue in issues],
                    "\n⚠️  These issues should be investigated and documented in CLAUDE.md",
                    "    Do not just skip this test - fix the underlying issues!",
                ]
            )
            pytest.fail(error_msg)

    def test_debug_nans_config_integration(
        self, temp_dir, sample_config_dict, sample_prior_file
    ):
        """Test that debug_nans config properly controls JAX NaN debugging.

        This test verifies the full integration: YAML -> Config -> JAX setting.
        """
        # Save original JAX config state
        original_debug_nans = jax.config.jax_debug_nans

        try:
            # Test 1: debug_nans=False (default) - should not enable JAX NaN debugging
            config_dict = sample_config_dict.copy()
            config_dict["prior"]["specification_file"] = str(sample_prior_file)
            config_dict["debug_nans"] = False

            config_file = temp_dir / "config_debug_false.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_dict, f)

            config = parser.load_config(config_file)
            assert config.debug_nans is False

            # Test 2: debug_nans=True - should enable JAX NaN debugging
            config_dict["debug_nans"] = True
            config_file = temp_dir / "config_debug_true.yaml"
            with open(config_file, "w") as f:
                yaml.dump(config_dict, f)

            config = parser.load_config(config_file)
            assert config.debug_nans is True

            # Simulate what run_inference.py does
            if config.debug_nans:
                jax.config.update("jax_debug_nans", True)

            # Verify JAX config was updated
            assert jax.config.jax_debug_nans is True

        finally:
            # Restore original JAX config state
            jax.config.update("jax_debug_nans", original_debug_nans)
