r"""Neutron star crust models and data loading."""

import os
import jax.numpy as jnp
from jaxtyping import Array, Float

# Get the path to the crust directory (relative to this module's parent)
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
CRUST_DIR = f"{DEFAULT_DIR}/crust_files"


class Crust:
    r"""
    Neutron star crust equation of state data handler.

    This class provides validated loading, preprocessing, and access to crust EOS data
    with automatic zero-pressure filtering and density range masking. It eliminates
    code duplication across EOS models by centralizing all crust preprocessing logic.

    The crust data files contain tabulated values of number density :math:`n`,
    pressure :math:`p`, and energy density :math:`\varepsilon` for the low-density
    outer and inner crust regions of neutron stars.

    Parameters
    ----------
    name : str
        Name of the crust model to load (e.g., 'BPS', 'DH', 'SLy') or a filename
        with .npz extension for custom files.
    min_density : float, optional
        Minimum density cutoff [:math:`\mathrm{fm}^{-3}`]. Points with density
        below this value are excluded. Default is None (no minimum cutoff).
    max_density : float, optional
        Maximum density cutoff [:math:`\mathrm{fm}^{-3}`]. Points with density
        above this value are excluded. Default is None (no maximum cutoff).
    filter_zero_pressure : bool, optional
        If True, remove points with zero or negative pressure to avoid numerical
        issues in logarithmic calculations. Default is True.

    Raises
    ------
    ValueError
        If the specified crust model is not found in the crust directory.

    Attributes
    ----------
    n : Float[Array, "n_points"]
        Number densities [:math:`\mathrm{fm}^{-3}`] after filtering and masking
    p : Float[Array, "n_points"]
        Pressures [:math:`\mathrm{MeV} \, \mathrm{fm}^{-3}`] after filtering and masking
    e : Float[Array, "n_points"]
        Energy densities [:math:`\mathrm{MeV} \, \mathrm{fm}^{-3}`] after filtering and masking
    mu_lowest : Float
        Chemical potential at lowest density point: :math:`(\varepsilon_0 + p_0) / n_0`
    cs2 : Float[Array, "n_points"]
        Speed of sound squared :math:`c_s^2 = dp/d\varepsilon`
    max_density : Float
        Maximum density in filtered crust [:math:`\mathrm{fm}^{-3}`]
    min_density : Float
        Minimum density in filtered crust [:math:`\mathrm{fm}^{-3}`]

    Examples
    --------
    Load a crust model with default settings:

    >>> from jesterTOV.eos import Crust
    >>> crust = Crust("DH")
    >>> print(f"Loaded {len(crust)} crust points")
    >>> n, p, e = crust.get_data()

    Load with density range masking:

    >>> crust = Crust("DH", min_density=0.001, max_density=0.1)
    >>> print(f"Density range: {crust.min_density:.4f} - {crust.max_density:.4f} fm^-3")

    Check available crusts before loading:

    >>> available = Crust.list_available()
    >>> if Crust.validate("BPS"):
    ...     crust = Crust("BPS")

    Access derived quantities:

    >>> crust = Crust("DH")
    >>> print(f"Chemical potential: {crust.mu_lowest:.2f} MeV")
    >>> print(f"Sound speed squared: {crust.cs2[0]:.4f}")

    See Also
    --------
    SpectralDecomposition_EOS_model : Spectral EOS using crust stitching
    MetaModel_EOS_model : Meta-model EOS using crust stitching

    Notes
    -----
    Available built-in crust models can be found in the jesterTOV/crust/ directory.
    The class automatically handles:

    - Zero-pressure point filtering (avoids log(0) in calculations)
    - Density range masking (for crust-core matching)
    - Monotonicity validation
    - Caching of derived quantities for performance
    """

    def __init__(
        self,
        name: str,
        min_density: float | None = None,
        max_density: float | None = None,
        filter_zero_pressure: bool = True,
    ):
        """
        Initialize and load crust EOS data with optional preprocessing.

        The initialization process:
        1. Validates crust existence
        2. Loads raw data from .npz file
        3. Applies zero-pressure filtering (if enabled)
        4. Applies density range masking (if specified)
        5. Validates data quality (monotonicity, positivity)
        6. Stores filtered arrays for property access

        Parameters
        ----------
        name : str
            Crust model name or path to .npz file
        min_density : float, optional
            Minimum density cutoff [:math:`\\mathrm{fm}^{-3}`]
        max_density : float, optional
            Maximum density cutoff [:math:`\\mathrm{fm}^{-3}`]
        filter_zero_pressure : bool, optional
            Remove zero/negative pressure points (default: True)
        """
        self._name = name
        self._file_path = self._resolve_file_path(name)

        # Load raw data
        crust_data = jnp.load(self._file_path)
        n_raw = crust_data["n"]
        p_raw = crust_data["p"]
        e_raw = crust_data["e"]

        # Apply preprocessing pipeline
        n_filtered, p_filtered, e_filtered = self._preprocess(
            n_raw, p_raw, e_raw, min_density, max_density, filter_zero_pressure
        )

        # Store filtered data
        self._n = n_filtered
        self._p = p_filtered
        self._e = e_filtered

        # Cache for derived quantities (computed on first access)
        self._mu_lowest_cached = None
        self._cs2_cached = None

    @classmethod
    def list_available(cls) -> list[str]:
        """
        List all available crust model names.

        Returns
        -------
        list[str]
            List of available crust model names (without .npz extension)

        Examples
        --------
        >>> available = Crust.list_available()
        >>> print(f"Available crusts: {', '.join(available)}")
        Available crusts: BPS, DH, SLy
        """
        crust_files = [f for f in os.listdir(CRUST_DIR) if f.endswith(".npz")]
        return [f.split(".")[0] for f in crust_files]

    @classmethod
    def validate(cls, name: str) -> bool:
        """
        Check if a crust model exists without loading it.

        This is useful for validation before instantiation, allowing fail-fast
        behavior in configuration parsing or user input validation.

        Parameters
        ----------
        name : str
            Crust model name to validate

        Returns
        -------
        bool
            True if crust exists, False otherwise

        Examples
        --------
        >>> if Crust.validate("DH"):
        ...     crust = Crust("DH")
        >>> else:
        ...     print("Crust not found")
        """
        try:
            cls._resolve_file_path_static(name)
            return True
        except ValueError:
            return False

    @classmethod
    def get_crust_dir(cls) -> str:
        """
        Return the path to the crust data directory.

        Returns
        -------
        str
            Absolute path to the crust directory

        Examples
        --------
        >>> crust_dir = Crust.get_crust_dir()
        >>> print(f"Crust data stored in: {crust_dir}")
        """
        return CRUST_DIR

    @property
    def n(self) -> Float[Array, "n_points"]:
        r"""
        Number density [:math:`\mathrm{fm}^{-3}`] after filtering and masking.

        Returns
        -------
        Float[Array, "n_points"]
            Array of number densities
        """
        return self._n

    @property
    def p(self) -> Float[Array, "n_points"]:
        r"""
        Pressure [:math:`\mathrm{MeV} \, \mathrm{fm}^{-3}`] after filtering and masking.

        Returns
        -------
        Float[Array, "n_points"]
            Array of pressures
        """
        return self._p

    @property
    def e(self) -> Float[Array, "n_points"]:
        r"""
        Energy density [:math:`\mathrm{MeV} \, \mathrm{fm}^{-3}`] after filtering and masking.

        Returns
        -------
        Float[Array, "n_points"]
            Array of energy densities
        """
        return self._e

    @property
    def mu_lowest(self) -> Float:
        r"""
        Chemical potential at the lowest density point.

        Computed as :math:`\mu_0 = (\varepsilon_0 + p_0) / n_0` where subscript 0
        denotes the first (lowest density) point in the crust. This value is used
        as the starting point for integrating thermodynamic quantities in the
        meta-model EOS construction.

        Returns
        -------
        Float
            Chemical potential [:math:`\mathrm{MeV}`]

        Examples
        --------
        >>> crust = Crust("DH")
        >>> mu = crust.mu_lowest
        >>> print(f"Lowest chemical potential: {mu:.2f} MeV")
        """
        mu_lowest_val = (self._e[0] + self._p[0]) / self._n[0]
        return mu_lowest_val

    @property
    def cs2(self) -> Float[Array, "n_points"]:
        r"""
        Speed of sound squared :math:`c_s^2 = dp/d\varepsilon`.

        Computed using numerical differentiation via gradient.

        Returns
        -------
        Float[Array, "n_points"]
            Array of sound speed squared values (dimensionless)

        Examples
        --------
        >>> crust = Crust("DH")
        >>> cs2 = crust.cs2
        >>> print(f"Sound speed range: {cs2.min():.4f} - {cs2.max():.4f}")

        Notes
        -----
        The speed of sound should satisfy :math:`0 < c_s^2 \leq 1` (in units where
        :math:`c = 1`) for physical crust models.
        """
        # jnp.gradient with two 1D arrays returns a single Array
        # but type checker sees it as Array | list[Array]
        # Cast to satisfy type checker - we know it's an Array in this case
        cs2_vals = jnp.asarray(jnp.gradient(self._p, self._e))
        return cs2_vals

    @property
    def max_density(self) -> Float:
        r"""
        Maximum density in filtered crust [:math:`\mathrm{fm}^{-3}`].

        Returns
        -------
        Float
            Maximum number density value
        """
        return self._n[-1]

    @property
    def min_density(self) -> Float:
        r"""
        Minimum density in filtered crust [:math:`\mathrm{fm}^{-3}`].

        Returns
        -------
        Float
            Minimum number density value
        """
        return self._n[0]

    def get_data(
        self,
    ) -> tuple[
        Float[Array, "n_points"], Float[Array, "n_points"], Float[Array, "n_points"]
    ]:
        """
        Return (n, p, e) tuple for convenient unpacking.

        This method provides a convenient way to get all three arrays at once,
        useful for code patterns that expect tuple unpacking.

        Returns
        -------
        tuple[Float[Array, "n_points"], Float[Array, "n_points"], Float[Array, "n_points"]]
            Tuple of (number density, pressure, energy density) arrays

        Examples
        --------
        >>> crust = Crust("DH")
        >>> n, p, e = crust.get_data()
        >>> assert jnp.array_equal(n, crust.n)
        """
        return self._n, self._p, self._e

    def __len__(self) -> int:
        """
        Return the number of crust points after filtering.

        Returns
        -------
        int
            Number of points in filtered crust arrays

        Examples
        --------
        >>> crust = Crust("DH")
        >>> print(f"Crust has {len(crust)} points")
        """
        return len(self._n)

    def __repr__(self) -> str:
        """
        String representation of Crust instance.

        Returns
        -------
        str
            Human-readable string describing the crust

        Examples
        --------
        >>> crust = Crust("DH", max_density=0.1)
        >>> print(crust)
        Crust(name='DH', n_points=42, density_range=[0.0001, 0.1000] fm^-3)
        """
        return (
            f"Crust(name='{self._name}', n_points={len(self)}, "
            f"density_range=[{self.min_density:.4f}, {self.max_density:.4f}] fm^-3)"
        )

    # Private helper methods

    def _resolve_file_path(self, name: str) -> str:
        """
        Resolve crust name to file path.

        Parameters
        ----------
        name : str
            Crust model name or path

        Returns
        -------
        str
            Absolute path to .npz file

        Raises
        ------
        ValueError
            If crust not found
        """
        return self._resolve_file_path_static(name)

    @staticmethod
    def _resolve_file_path_static(name: str) -> str:
        """
        Static version of file path resolution for class methods.

        Parameters
        ----------
        name : str
            Crust model name or path

        Returns
        -------
        str
            Absolute path to .npz file

        Raises
        ------
        ValueError
            If crust not found
        """
        # Get available crust names
        available_crust_names = [
            f.split(".")[0] for f in os.listdir(CRUST_DIR) if f.endswith(".npz")
        ]

        # If name doesn't end with .npz, try to find it in crust directory
        if not name.endswith(".npz"):
            if name in available_crust_names:
                return os.path.join(CRUST_DIR, f"{name}.npz")
            else:
                raise ValueError(
                    f"Crust '{name}' not found in {CRUST_DIR}. "
                    f"Available crusts: {available_crust_names}"
                )
        else:
            # Name includes .npz extension - treat as file path
            if os.path.exists(name):
                return name
            else:
                raise ValueError(f"Crust file not found: {name}")

    def _preprocess(
        self,
        n: Float[Array, "n_raw"],
        p: Float[Array, "n_raw"],
        e: Float[Array, "n_raw"],
        min_density: float | None,
        max_density: float | None,
        filter_zero_pressure: bool,
    ) -> tuple[
        Float[Array, "n_filtered"],
        Float[Array, "n_filtered"],
        Float[Array, "n_filtered"],
    ]:
        """
        Apply preprocessing pipeline to raw crust data.

        The pipeline applies filters in this order:
        1. Zero-pressure filtering (if enabled)
        2. Density range masking (if specified)

        Parameters
        ----------
        n, p, e : Float[Array, "n_raw"]
            Raw crust data arrays
        min_density, max_density : float, optional
            Density range bounds
        filter_zero_pressure : bool
            Whether to filter zero/negative pressures

        Returns
        -------
        tuple[Float[Array, "n_filtered"], Float[Array, "n_filtered"], Float[Array, "n_filtered"]]
            Filtered (n, p, e) arrays
        """
        # Start with all points
        mask = jnp.ones(len(n), dtype=bool)

        # Apply zero-pressure filter
        if filter_zero_pressure:
            mask = mask & (p > 0)

        # Apply density range filters
        if min_density is not None:
            mask = mask & (n >= min_density)

        if max_density is not None:
            mask = mask & (n <= max_density)

        # Apply mask
        n_filtered = n[mask]
        p_filtered = p[mask]
        e_filtered = e[mask]

        # Check if we masked out all points, which might happen if we mess up units:
        # Specifically, we check if the sum is zero (which means all False)
        if jnp.sum(mask) == 0:
            raise ValueError(
                f"No crust points remain after filtering. Please check density units and range of the crust file: {self._file_path}"
            )

        # Validate filtered data
        if len(n_filtered) == 0:
            raise ValueError(
                "No crust points remain after filtering. Check density range and zero-pressure settings."
            )

        # Check monotonicity
        if not jnp.all(jnp.diff(n_filtered) > 0):
            raise ValueError(
                "Crust density is not monotonically increasing after filtering"
            )

        return n_filtered, p_filtered, e_filtered
