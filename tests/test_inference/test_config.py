"""Tests for inference configuration system (parser and schema)."""

import pytest
import yaml
from pathlib import Path
from pydantic import ValidationError

from jesterTOV.inference.config import schema, parser


class TestTransformConfig:
    """Test TransformConfig validation."""

    def test_valid_metamodel_config(self):
        """Test valid metamodel configuration."""
        config = schema.TransformConfig(
            type="metamodel",
            ndat_metamodel=100,
            nmax_nsat=2.0,
            nb_CSE=0,
            min_nsat_TOV=0.75,
            ndat_TOV=100,
            nb_masses=100,
            crust_name="DH",
        )
        assert config.type == "metamodel"
        assert config.nb_CSE == 0
        assert config.ndat_metamodel == 100

    def test_valid_metamodel_cse_config(self):
        """Test valid metamodel_cse configuration."""
        config = schema.TransformConfig(
            type="metamodel_cse",
            ndat_metamodel=100,
            nmax_nsat=25.0,
            nb_CSE=8,
            min_nsat_TOV=0.75,
            ndat_TOV=100,
            nb_masses=100,
            crust_name="DH",
        )
        assert config.type == "metamodel_cse"
        assert config.nb_CSE == 8
        assert config.nmax_nsat == 25.0

    def test_metamodel_with_nonzero_cse_fails(self):
        """Test that metamodel with nb_CSE != 0 fails validation."""
        with pytest.raises(ValidationError, match="nb_CSE must be 0"):
            schema.TransformConfig(
                type="metamodel",
                nb_CSE=8,  # Should fail for type=metamodel
            )

    def test_invalid_crust_name(self):
        """Test that invalid crust names fail validation."""
        with pytest.raises(ValidationError):
            schema.TransformConfig(
                type="metamodel",
                crust_name="InvalidCrust",
            )

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = schema.TransformConfig(type="metamodel", nb_CSE=0)
        assert config.ndat_metamodel == 100
        assert config.ndat_TOV == 100
        assert config.nb_masses == 100
        assert config.crust_name == "DH"


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
        config = schema.LikelihoodConfig(
            type="zero",
            enabled=True,
            parameters={},
        )
        assert config.type == "zero"
        assert config.enabled is True
        assert config.parameters == {}

    def test_gw_likelihood_config(self):
        """Test GW likelihood configuration."""
        config = schema.LikelihoodConfig(
            type="gw",
            enabled=True,
            parameters={
                "events": [
                    {"name": "GW170817", "model_dir": "/path/to/data"}
                ],
                "penalty_value": -99999.0,
                "N_masses_evaluation": 20,
            },
        )
        assert config.type == "gw"
        assert len(config.parameters["events"]) == 1
        assert config.parameters["penalty_value"] == -99999.0

    def test_nicer_likelihood_config(self):
        """Test NICER likelihood configuration."""
        config = schema.LikelihoodConfig(
            type="nicer",
            enabled=True,
            parameters={
                "pulsars": [
                    {
                        "name": "J0030",
                        "amsterdam_samples_file": "/path/to/amsterdam.txt",
                        "maryland_samples_file": "/path/to/maryland.txt",
                    }
                ],
                "N_masses_evaluation": 100,
            },
        )
        assert config.type == "nicer"
        assert len(config.parameters["pulsars"]) == 1

    def test_radio_likelihood_config(self):
        """Test radio timing likelihood configuration."""
        config = schema.LikelihoodConfig(
            type="radio",
            enabled=True,
            parameters={
                "pulsars": [
                    {"name": "J0348+0432", "mass_mean": 2.01, "mass_std": 0.04}
                ],
                "nb_masses": 100,
            },
        )
        assert config.type == "radio"
        assert config.parameters["pulsars"][0]["mass_mean"] == 2.01

    def test_invalid_likelihood_type(self):
        """Test that invalid likelihood types fail validation."""
        with pytest.raises(ValidationError):
            schema.LikelihoodConfig(
                type="invalid_type",
                enabled=True,
            )

    def test_disabled_likelihood(self):
        """Test that likelihoods can be disabled."""
        config = schema.LikelihoodConfig(
            type="gw",
            enabled=False,
            parameters={},
        )
        assert config.enabled is False


class TestSamplerConfig:
    """Test SamplerConfig validation."""

    def test_valid_sampler_config(self):
        """Test valid sampler configuration."""
        config = schema.SamplerConfig(
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
            schema.SamplerConfig(
                n_chains=-1,  # Should fail
                n_loop_training=2,
                n_loop_production=2,
            )

    def test_zero_chains_fails(self):
        """Test that zero chains fails validation."""
        with pytest.raises(ValidationError):
            schema.SamplerConfig(
                n_chains=0,  # Should fail
                n_loop_training=2,
                n_loop_production=2,
            )

    def test_negative_learning_rate_fails(self):
        """Test that negative learning rate fails validation."""
        with pytest.raises(ValidationError):
            schema.SamplerConfig(
                n_chains=4,
                learning_rate=-0.001,  # Should fail
            )


class TestInferenceConfig:
    """Test InferenceConfig (top-level configuration)."""

    def test_valid_full_config(self, sample_config_dict):
        """Test valid full configuration."""
        config = schema.InferenceConfig(**sample_config_dict)
        assert config.seed == 42
        assert config.transform.type == "metamodel"
        assert len(config.likelihoods) == 1
        assert config.sampler.n_chains == 4

    def test_config_with_multiple_likelihoods(self, sample_config_dict):
        """Test configuration with multiple likelihoods."""
        config_dict = sample_config_dict.copy()
        config_dict["likelihoods"] = [
            {
                "type": "gw",
                "enabled": True,
                "parameters": {
                    "events": [{"name": "GW170817", "model_dir": "/path/to/data"}]
                },
            },
            {
                "type": "nicer",
                "enabled": True,
                "parameters": {
                    "pulsars": [
                        {
                            "name": "J0030",
                            "amsterdam_samples_file": "/path/to/amsterdam.txt",
                            "maryland_samples_file": "/path/to/maryland.txt",
                        }
                    ]
                },
            },
            {
                "type": "radio",
                "enabled": False,
                "parameters": {},  # Empty OK when disabled
            },
        ]
        config = schema.InferenceConfig(**config_dict)
        assert len(config.likelihoods) == 3
        assert config.likelihoods[0].type == "gw"
        assert config.likelihoods[1].type == "nicer"
        assert config.likelihoods[2].enabled is False

    def test_config_with_cse(self, sample_config_dict):
        """Test configuration with CSE enabled."""
        config_dict = sample_config_dict.copy()
        config_dict["transform"]["type"] = "metamodel_cse"
        config_dict["transform"]["nb_CSE"] = 8
        config = schema.InferenceConfig(**config_dict)
        assert config.transform.type == "metamodel_cse"
        assert config.transform.nb_CSE == 8

    def test_missing_required_field_fails(self):
        """Test that missing required fields fail validation."""
        with pytest.raises(ValidationError):
            schema.InferenceConfig(
                # Missing transform, prior, etc.
                sampler={"n_chains": 4},
            )


class TestConfigParser:
    """Test configuration parser functionality."""

    def test_load_config_from_file(self, sample_config_file):
        """Test loading configuration from YAML file."""
        config = parser.load_config(sample_config_file)
        assert isinstance(config, schema.InferenceConfig)
        assert config.seed == 42
        assert config.transform.type == "metamodel"

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
        incomplete_config.write_text("""
seed: 42
# Missing transform, prior, likelihoods, sampler
""")

        with pytest.raises(ValueError, match="Error validating configuration"):
            parser.load_config(incomplete_config)


class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_roundtrip_config_to_yaml_and_back(self, temp_dir, sample_config_dict, sample_prior_file):
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
        assert config1.transform.type == config2.transform.type
        assert config1.sampler.n_chains == config2.sampler.n_chains

    def test_example_configs_are_valid(self):
        """Test that all example config files are valid.

        NOTE: This test may fail if example configs have invalid paths or
        are missing required data files. If so, document the issue in CLAUDE.md
        and investigate - do not just skip the test!
        """
        import os
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
            error_msg = "\n".join([
                "Issues found in example configs:",
                *[f"  - {issue}" for issue in issues],
                "\n⚠️  These issues should be investigated and documented in CLAUDE.md",
                "    Do not just skip this test - fix the underlying issues!"
            ])
            pytest.fail(error_msg)
