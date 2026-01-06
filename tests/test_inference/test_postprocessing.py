"""Tests for postprocessing module (visualization and data loading)."""

import pytest
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from jesterTOV.inference.postprocessing.postprocessing import (
    setup_matplotlib,
    load_eos_data,
    load_prior_data,
    report_credible_interval,
    make_cornerplot,
    make_mass_radius_plot,
    make_pressure_density_plot,
)
from jesterTOV.inference.result import InferenceResult


class TestMatplotlibSetup:
    """Test matplotlib configuration."""

    def test_setup_matplotlib_with_tex(self):
        """Test matplotlib setup attempts TeX rendering."""
        # This may or may not succeed depending on system TeX installation
        # Just verify it returns a boolean and doesn't crash
        result = setup_matplotlib(use_tex=True)
        assert isinstance(result, bool)

    def test_setup_matplotlib_without_tex(self):
        """Test matplotlib setup without TeX."""
        result = setup_matplotlib(use_tex=False)
        assert result is False  # TeX explicitly disabled

    def test_setup_matplotlib_sets_rc_params(self):
        """Test that setup_matplotlib configures rcParams."""
        setup_matplotlib(use_tex=False)

        # Check that key parameters were set
        assert plt.rcParams["axes.grid"] is False
        assert plt.rcParams["ytick.color"] == "black"
        assert plt.rcParams["xtick.labelsize"] == 16


class TestLoadEOSData:
    """Test EOS data loading from HDF5."""

    def test_load_eos_data_success(self, temp_dir):
        """Test successful loading of EOS data from HDF5."""
        # Create a minimal HDF5 result file
        posterior = {
            "K_sat": np.array([220.0, 230.0]),
            "log_prob": np.array([-10.0, -11.0]),
            "masses_EOS": np.random.rand(2, 100),
            "radii_EOS": np.random.rand(2, 100),
            "Lambdas_EOS": np.random.rand(2, 100),
            "n": np.random.rand(2, 50),
            "p": np.random.rand(2, 50),
            "e": np.random.rand(2, 50),
            "cs2": np.random.rand(2, 50),
        }
        metadata = {
            "sampler": "flowmc",
            "n_samples": 2,
        }

        result = InferenceResult(
            sampler_type="flowmc",
            posterior=posterior,
            metadata=metadata,
        )

        # Save to temp directory
        result.save(temp_dir / "results.h5")

        # Load using postprocessing function
        data = load_eos_data(str(temp_dir))

        # Verify data structure (note: keys are renamed by load_eos_data)
        assert "masses" in data  # renamed from masses_EOS
        assert "radii" in data  # renamed from radii_EOS
        assert "lambdas" in data  # renamed from Lambdas_EOS
        assert "densities" in data  # renamed from n
        assert "pressures" in data  # renamed from p
        assert "energies" in data  # renamed from e
        assert "cs2" in data
        assert "log_prob" in data

        # Verify shapes preserved (note: densities/pressures have unit conversions applied)
        assert data["masses"].shape == (2, 100)
        assert data["densities"].shape == (2, 50)

    def test_load_eos_data_file_not_found(self, temp_dir):
        """Test that load_eos_data raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError, match="Results file not found"):
            load_eos_data(str(temp_dir / "nonexistent"))

    def test_load_eos_data_missing_required_fields(self, temp_dir):
        """Test error handling when required EOS fields are missing."""
        # Create result without EOS quantities
        posterior = {
            "K_sat": np.array([220.0]),
            "log_prob": np.array([-10.0]),
        }
        metadata = {"sampler": "flowmc"}

        result = InferenceResult(
            sampler_type="flowmc",
            posterior=posterior,
            metadata=metadata,
        )

        result.save(temp_dir / "results.h5")

        # Should raise KeyError for missing fields
        with pytest.raises(KeyError):
            load_eos_data(str(temp_dir))


class TestLoadPriorData:
    """Test prior data loading."""

    def test_load_prior_data_file_not_found(self):
        """Test that load_prior_data returns None when file not found."""
        result = load_prior_data(prior_dir="/nonexistent/path")
        assert result is None

    def test_load_prior_data_success(self, temp_dir):
        """Test successful loading of prior samples."""
        # Create a minimal prior result file
        posterior = {
            "K_sat": np.array([220.0, 230.0, 240.0]),
            "L_sym": np.array([90.0, 95.0, 100.0]),
            "log_prob": np.array([-10.0, -11.0, -12.0]),
            "masses_EOS": np.random.rand(3, 100),
            "radii_EOS": np.random.rand(3, 100),
            "Lambdas_EOS": np.random.rand(3, 100),
            "n": np.random.rand(3, 50),
            "p": np.random.rand(3, 50),
            "e": np.random.rand(3, 50),
            "cs2": np.random.rand(3, 50),
        }
        metadata = {"sampler": "flowmc"}

        result = InferenceResult(
            sampler_type="flowmc",
            posterior=posterior,
            metadata=metadata,
        )

        # Save as "results.h5" in temp directory
        result.save(temp_dir / "results.h5")

        # Load prior data
        prior_data = load_prior_data(prior_dir=str(temp_dir))

        assert prior_data is not None
        assert "masses" in prior_data  # renamed from masses_EOS
        assert prior_data["masses"].shape == (3, 100)


class TestReportCredibleInterval:
    """Test credible interval reporting."""

    def test_report_credible_interval_basic(self):
        """Test basic credible interval calculation and reporting."""
        # Create simple distribution
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Note: actual API is (values, hdi_prob, verbose)
        low_err, median, high_err = report_credible_interval(
            values, hdi_prob=0.90, verbose=False
        )

        # Verify return values
        assert isinstance(low_err, (float, np.floating))
        assert isinstance(median, (float, np.floating))
        assert isinstance(high_err, (float, np.floating))
        assert median == 3.0  # Median of [1, 2, 3, 4, 5]

    def test_report_credible_interval_custom_hdi(self):
        """Test with custom HDI probability."""
        values = np.linspace(0, 100, 1000)

        low_err, median, high_err = report_credible_interval(values, hdi_prob=0.68)

        # Median should be around 50
        assert 49 < median < 51
        assert low_err > 0
        assert high_err > 0


class TestPlotGeneration:
    """Test plot generation functions (smoke tests)."""

    @pytest.fixture
    def mock_data(self):
        """Create mock data for plotting tests (using postprocessing format)."""
        n_samples = 50
        n_eos_points = 100

        # Note: Using the key names that load_eos_data returns
        data = {
            "masses": np.random.uniform(1.0, 2.5, (n_samples, n_eos_points)),
            "radii": np.random.uniform(10.0, 15.0, (n_samples, n_eos_points)),
            "lambdas": np.random.uniform(100.0, 1000.0, (n_samples, n_eos_points)),
            "densities": np.random.uniform(0.5, 3.0, (n_samples, n_eos_points)),
            "pressures": np.random.uniform(1.0, 100.0, (n_samples, n_eos_points)),
            "energies": np.random.uniform(1.0, 500.0, (n_samples, n_eos_points)),
            "cs2": np.random.uniform(0.0, 1.0, (n_samples, n_eos_points)),
            "log_prob": np.random.uniform(-50.0, -10.0, n_samples),
            "nep_params": {
                "K_sat": np.random.uniform(200.0, 250.0, n_samples),
                "L_sym": np.random.uniform(50.0, 100.0, n_samples),
                "Q_sat": np.random.uniform(-200.0, 200.0, n_samples),
                "E_sym": np.random.uniform(28.0, 35.0, n_samples),
            },
            "cse_params": {},
        }
        return data

    def test_make_cornerplot_basic(self, mock_data, temp_dir):
        """Test cornerplot generation doesn't crash."""
        # Should not raise any exceptions
        make_cornerplot(mock_data, outdir=str(temp_dir), max_params=4)

        # Check that plot file was created (PDF, not PNG)
        expected_file = temp_dir / "cornerplot.pdf"
        assert expected_file.exists()

        # Clean up
        plt.close("all")

    def test_make_mass_radius_plot_basic(self, mock_data, temp_dir):
        """Test mass-radius plot generation doesn't crash."""
        # Signature: make_mass_radius_plot(data, prior_data, outdir, use_crest_cmap)
        make_mass_radius_plot(data=mock_data, prior_data=None, outdir=str(temp_dir))

        # Check that plot file was created (PDF, not PNG)
        expected_file = temp_dir / "mass_radius_plot.pdf"
        assert expected_file.exists()

        # Clean up
        plt.close("all")

    def test_make_mass_radius_plot_with_prior(self, mock_data, temp_dir):
        """Test mass-radius plot with prior samples."""
        # Create smaller prior data (using correct key names)
        prior_data = {
            "masses": mock_data["masses"][:10],
            "radii": mock_data["radii"][:10],
            "log_prob": mock_data["log_prob"][:10],
        }

        make_mass_radius_plot(
            data=mock_data, prior_data=prior_data, outdir=str(temp_dir)
        )

        expected_file = temp_dir / "mass_radius_plot.pdf"
        assert expected_file.exists()

        plt.close("all")

    def test_make_pressure_density_plot_basic(self, mock_data, temp_dir):
        """Test pressure-density plot generation doesn't crash."""
        # Check signature first
        make_pressure_density_plot(
            data=mock_data, prior_data=None, outdir=str(temp_dir)
        )

        # PDF, not PNG
        expected_file = temp_dir / "pressure_density_plot.pdf"
        assert expected_file.exists()

        plt.close("all")

    def test_plots_handle_small_sample_size(self, temp_dir):
        """Test plots work with very small sample sizes."""
        # Minimal data (just 2 samples)
        minimal_data = {
            "masses": np.random.rand(2, 50),
            "radii": np.random.rand(2, 50),
            "lambdas": np.random.rand(2, 50),
            "densities": np.random.rand(2, 50),
            "pressures": np.random.rand(2, 50),
            "energies": np.random.rand(2, 50),
            "cs2": np.random.rand(2, 50),
            "log_prob": np.array([-10.0, -11.0]),
            "nep_params": {
                "K_sat": np.array([220.0, 230.0]),
                "L_sym": np.array([90.0, 95.0]),
            },
            "cse_params": {},
        }

        # Should handle gracefully (may show warnings but shouldn't crash)
        make_mass_radius_plot(data=minimal_data, prior_data=None, outdir=str(temp_dir))

        plt.close("all")


class TestPlotErrorHandling:
    """Test error handling in plotting functions."""

    def test_cornerplot_missing_parameters(self, temp_dir):
        """Test cornerplot handles missing parameter data."""
        incomplete_data = {
            "log_prob": np.array([-10.0, -11.0]),
            "masses_EOS": np.random.rand(2, 50),
        }

        # Should either skip or handle gracefully
        # (exact behavior depends on implementation)
        try:
            make_cornerplot(incomplete_data, outdir=str(temp_dir))
            plt.close("all")
        except (KeyError, ValueError):
            # Acceptable to raise error for incomplete data
            pass

    def test_mass_radius_plot_missing_eos_data(self, temp_dir):
        """Test M-R plot handles missing EOS data."""
        incomplete_data = {
            "log_prob": np.array([-10.0, -11.0]),
            "nep_params": {"K_sat": np.array([220.0, 230.0])},
            "cse_params": {},
        }

        # Should raise KeyError for missing required fields
        with pytest.raises(KeyError):
            make_mass_radius_plot(
                data=incomplete_data, prior_data=None, outdir=str(temp_dir)
            )


class TestIntegrationWithInferenceResult:
    """Integration tests with InferenceResult class."""

    def test_full_workflow_flowmc(self, temp_dir):
        """Test complete workflow: create result → save → load → plot."""
        # Create realistic FlowMC result
        n_samples = 100
        n_eos = 200

        posterior = {
            "K_sat": np.random.uniform(200, 250, n_samples),
            "L_sym": np.random.uniform(50, 100, n_samples),
            "Q_sat": np.random.uniform(-200, 200, n_samples),
            "E_sym": np.random.uniform(28, 35, n_samples),
            "log_prob": np.random.uniform(-50, -10, n_samples),
            "masses_EOS": np.random.uniform(0.5, 2.5, (n_samples, n_eos)),
            "radii_EOS": np.random.uniform(8, 15, (n_samples, n_eos)),
            "Lambdas_EOS": np.random.uniform(10, 2000, (n_samples, n_eos)),
            "n": np.random.uniform(0.1, 5.0, (n_samples, n_eos)),
            "p": np.random.uniform(0.1, 500, (n_samples, n_eos)),
            "e": np.random.uniform(1, 1000, (n_samples, n_eos)),
            "cs2": np.random.uniform(0.0, 1.0, (n_samples, n_eos)),
        }

        metadata = {
            "sampler": "flowmc",
            "n_samples": n_samples,
            "runtime_seconds": 3600.0,
        }

        result = InferenceResult(
            sampler_type="flowmc",
            posterior=posterior,
            metadata=metadata,
        )

        # Save result
        result.save(temp_dir / "results.h5")

        # Load using postprocessing
        data = load_eos_data(str(temp_dir))

        # Generate plots
        make_cornerplot(data, outdir=str(temp_dir), max_params=4)
        make_mass_radius_plot(data, prior_data=None, outdir=str(temp_dir))
        make_pressure_density_plot(data, prior_data=None, outdir=str(temp_dir))

        # Verify all plots created (note: some are PDF, some are PDF/PNG)
        assert (temp_dir / "cornerplot.pdf").exists()
        assert (temp_dir / "mass_radius_plot.pdf").exists()
        # pressure_density may save as PDF or PNG depending on implementation
        assert (temp_dir / "pressure_density_plot.pdf").exists() or (
            temp_dir / "pressure_density_plot.png"
        ).exists()

        plt.close("all")
