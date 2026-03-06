r"""CUTER subprocess wrapper for the unified neutron star EOS.

This module provides :class:`UnifiedCrustEOS_CUTER`, which wraps the CUTER
Fortran/C backend to produce a thermodynamically consistent unified EOS.

**This class is NOT suitable for inference.** It:

- Calls the CUTER Fortran binary as a subprocess for each ``construct_eos`` call
- Is not JAX-traceable or JIT-compilable
- May take several seconds per call (Fortran compilation + execution)

**Purpose:** Cross-checking and validation of the JAX-native
:class:`~jesterTOV.eos.unified_crust.unified_crust.UnifiedCrustEOS_MetaModel`.

**Requirements:**
- CUTER Fortran backend must be compiled:
  ``cd cuter-v2/source/apps/eos_consistent-crust/ && make``
- Either set ``CUTER_DIR`` environment variable or pass ``cuter_dir`` argument.
- CUTER Python dependencies (numpy, scipy) must be available.

**Sign conventions:** Jester uses negative ``E_sat`` (binding energy < 0).
CUTER's JSON metadata uses positive ``E0_NMP``. This wrapper handles the conversion.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import numpy as np
from pathlib import Path


from jesterTOV.eos.base import Interpolate_EOS_model
from jesterTOV.tov.data_classes import EOSData
from jesterTOV.logging_config import get_logger

logger = get_logger("jester")


class UnifiedCrustEOS_CUTER(Interpolate_EOS_model):
    r"""Unified EOS via CUTER Fortran/C backend subprocess call.

    Calls the CUTER ``eos_cons-crust`` binary for each parameter set, reads the
    output table, and returns an :class:`~jesterTOV.tov.data_classes.EOSData`.

    Parameters
    ----------
    cuter_dir : str | None
        Path to the ``cuter-v2/`` directory. If ``None``, falls back to the
        ``CUTER_DIR`` environment variable. Raises ``RuntimeError`` if neither
        is set.
    bsk_model : int
        BSk model for the outer crust Pearson fit (22 or 24, default 24).
    ndat_out : int
        Number of density points in the returned EOS table (default 500).
    order : int
        Taylor expansion order for the metamodel (2, 3, or 4, default 4).

    Raises
    ------
    RuntimeError
        If ``cuter_dir`` is not set and ``CUTER_DIR`` env var is not defined.
    FileNotFoundError
        If the CUTER binary ``eos_cons-crust`` is not found. Run ``make`` in
        ``cuter-v2/source/apps/eos_consistent-crust/`` first.

    Notes
    -----
    **Sign convention:** The ``E_sat`` parameter in Jester is negative (e.g.,
    -16.05 MeV). CUTER's internal convention stores it as a positive ``Esat``
    value. The wrapper negates ``E_sat`` before writing the input file.

    Each ``construct_eos`` call:
    1. Writes ``eos_param.in`` and ``eos_input.table`` to a temp directory.
    2. Calls the CUTER binary (which also re-makes the binary if headers changed,
       but we skip re-compilation for speed).
    3. Reads the ``eos_reconstructed_mm_*.out`` output.
    4. Converts units and returns ``EOSData``.

    For BSk24 NEPs, typical execution time is ~5–30 seconds per call (dominated
    by the Fortran code). For 100 samples, use batched/parallel runs.
    """

    def __init__(
        self,
        cuter_dir: str | None = None,
        bsk_model: int = 24,
        ndat_out: int = 500,
        order: int = 4,
    ) -> None:
        """Initialize CUTER wrapper.

        Parameters
        ----------
        cuter_dir : str | None
            Path to cuter-v2/ directory (or use CUTER_DIR env var)
        bsk_model : int
            BSk model number (22 or 24)
        ndat_out : int
            Output EOS table size
        order : int
            Metamodel Taylor expansion order (2, 3, or 4)
        """
        if bsk_model not in (22, 24):
            raise ValueError(f"bsk_model must be 22 or 24, got {bsk_model}")
        if order not in (2, 3, 4):
            raise ValueError(f"order must be 2, 3, or 4, got {order}")

        cuter_dir = cuter_dir or os.environ.get("CUTER_DIR", "")
        if not cuter_dir or not Path(cuter_dir).exists():
            raise RuntimeError(
                "CUTER directory not found. Either:\n"
                "  1. Pass cuter_dir='/path/to/cuter-v2/'\n"
                "  2. Set CUTER_DIR environment variable\n"
                "See internal-jester-review/cuter/README.md for setup instructions."
            )

        self.cuter_dir = Path(cuter_dir)
        self.bsk_model = bsk_model
        self.ndat_out = ndat_out
        self.order = order

        # Locate the compiled binary
        self.binary = (
            self.cuter_dir
            / "source"
            / "apps"
            / "eos_consistent-crust"
            / "eos_cons-crust"
        )
        if not self.binary.exists():
            raise FileNotFoundError(
                f"CUTER binary not found: {self.binary}\n"
                "Compile with:\n"
                f"  cd {self.cuter_dir}/source/apps/eos_consistent-crust/\n"
                "  make clean && make"
            )

        # Input/output directories
        self._input_dir = self.cuter_dir / "source" / "input" / "eos_param"

        logger.info(
            f"UnifiedCrustEOS_CUTER initialized: binary={self.binary}, "
            f"BSk{bsk_model}, order={order}"
        )

    def get_required_parameters(self) -> list[str]:
        """Return the NEPs required by CUTER.

        Returns
        -------
        list[str]
            Same 9 NEPs as the metamodel.
        """
        return [
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

    def construct_eos(self, params: dict[str, float]) -> EOSData:
        """Call CUTER to build unified EOS from nuclear empirical parameters.

        Parameters
        ----------
        params : dict[str, float]
            Nuclear empirical parameters. Keys: E_sat, K_sat, Q_sat, Z_sat,
            E_sym, L_sym, K_sym, Q_sym, Z_sym. E_sat should be negative.

        Returns
        -------
        EOSData
            Unified EOS data in geometric units.

        Raises
        ------
        RuntimeError
            If CUTER subprocess fails or output cannot be parsed.
        """
        E_sat = params["E_sat"]  # negative in Jester
        K_sat = params["K_sat"]
        Q_sat = params["Q_sat"]
        Z_sat = params["Z_sat"]
        E_sym = params["E_sym"]
        L_sym = params["L_sym"]
        K_sym = params["K_sym"]
        Q_sym = params["Q_sym"]
        Z_sym = params["Z_sym"]

        # CUTER sign convention: nsat, Esat (positive), Ksat, Qsat, Zsat
        # Esat is positive in CUTER (binding energy magnitude)
        Esat_cuter = abs(E_sat)  # positive
        nsat = 0.16  # fm^-3 (fixed)
        meff = 1.0  # effective mass ratio
        delta = 0.0  # isospin split
        massn = 939.5654205  # neutron mass [MeV]
        massp = 938.2720882  # proton mass [MeV]

        # Write eos_param.in
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            param_file = tmpdir_path / "eos_param.in"
            table_file = tmpdir_path / "eos_input.table"

            # Write parameter file (CUTER format)
            param_content = (
                f"{self.order} \t ! order (in Taylor expansion)\n"
                f"{nsat:.4f}    {Esat_cuter:.4f}   {K_sat:.4f}    {Q_sat:.4f}    {Z_sat:.4f}"
                "\t\t !nsat, Esat, Ksat, Qsat, Zsat \n"
                f"{E_sym:.4f}    {L_sym:.4f}    {K_sym:.4f}     {Q_sym:.4f}    {Z_sym:.4f}"
                "\t\t ! Esym, Lsym, Ksym, Qsym, Zsym \n"
                f"{meff:.4f}    {delta:.4f} \t ! effmass and isosplit \n"
                "0 \t ! Boolean for TOV\n"
            )
            param_file.write_text(param_content)

            # Write empty eos_input.table (no external core EOS — pure metamodel)
            table_file.write_text("")

            # Copy param files to CUTER input directory
            cuter_input_dir = self.cuter_dir / "source" / "input" / "eos_param"
            cuter_input_dir.mkdir(parents=True, exist_ok=True)
            import shutil

            shutil.copy(param_file, cuter_input_dir / "eos_param.in")
            shutil.copy(table_file, cuter_input_dir / "eos_input.table")

            # Run CUTER binary
            result = subprocess.run(
                [str(self.binary)],
                cwd=str(self.cuter_dir / "source" / "apps" / "eos_consistent-crust"),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"CUTER binary failed (returncode={result.returncode}):\n"
                    f"STDERR: {result.stderr[-500:]}\n"
                    f"STDOUT: {result.stdout[-500:]}"
                )

            # Find output file (eos_reconstructed_mm_*.out)
            output_candidates = list(
                (self.cuter_dir / "source" / "apps" / "eos_consistent-crust").glob(
                    "eos_reconstructed_mm*.out"
                )
            )
            if not output_candidates:
                # Try filesOutput/
                output_candidates = list(
                    (self.cuter_dir / "filesOutput").glob("eos_reconstructed_mm*.out")
                )
            if not output_candidates:
                raise RuntimeError(
                    "CUTER output file 'eos_reconstructed_mm*.out' not found.\n"
                    f"CUTER stdout: {result.stdout[-500:]}"
                )

            output_file = output_candidates[0]
            data = np.loadtxt(str(output_file))

        if data.ndim != 2 or data.shape[1] < 3:
            raise RuntimeError(
                f"Unexpected CUTER output shape: {data.shape}. "
                "Expected at least 3 columns: nb, eps, P."
            )

        # CUTER output columns: nb [fm^-3], eps [g/cm^3], P [MeV/fm^3]
        # (columns may vary — check the first few rows)
        n_raw = data[:, 0]  # fm^-3
        # eps column 1 might be in g/cm^3 or MeV/fm^3 depending on CUTER version
        # Column 2 is pressure in MeV/fm^3 typically
        # We need to figure out units from magnitude:
        eps_raw = data[:, 1]
        p_raw = data[:, 2]

        # Detect eps units: if > 1e10, it's g/cm³; convert to MeV/fm³
        _g_cm3_to_MeV_fm3 = 1.0 / 1.78266184e12  # inverse of econvert
        if eps_raw[eps_raw > 0].min() > 1e10:
            eps_MeV_fm3 = eps_raw * _g_cm3_to_MeV_fm3
        else:
            eps_MeV_fm3 = eps_raw

        # Filter monotone and positive
        mask = (n_raw > 0) & (p_raw > 0) & (eps_MeV_fm3 > 0)
        n_raw, p_raw, eps_MeV_fm3 = n_raw[mask], p_raw[mask], eps_MeV_fm3[mask]

        # Ensure monotone n
        dn = np.diff(n_raw)
        if not np.all(dn > 0):
            mono = np.concatenate([[True], dn > 0])
            n_raw, p_raw, eps_MeV_fm3 = n_raw[mono], p_raw[mono], eps_MeV_fm3[mono]

        # Interpolate to ndat_out uniform points (log-spaced in density)
        from scipy.interpolate import interp1d as scipy_interp1d

        log_n = np.log(n_raw)
        log_n_out = np.linspace(log_n[0], log_n[-1], self.ndat_out)
        n_out = np.exp(log_n_out)
        p_out = np.exp(scipy_interp1d(log_n, np.log(p_raw), kind="linear")(log_n_out))
        e_out = np.exp(
            scipy_interp1d(log_n, np.log(eps_MeV_fm3), kind="linear")(log_n_out)
        )

        ns, ps, hs, es, dloge_dlogps = self.interpolate_eos(n_out, p_out, e_out)
        cs2 = ps / (es * dloge_dlogps)

        return EOSData(
            ns=ns,
            ps=ps,
            hs=hs,
            es=es,
            dloge_dlogps=dloge_dlogps,
            cs2=cs2,
            mu=None,
            extra_constraints=None,
        )
