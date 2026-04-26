# Contributing to JESTER

Thank you for your interest in contributing to JESTER! This guide will help you get started with development.

JESTER is a scientific computing library for neutron star physics using JAX for hardware acceleration. We welcome contributions of new EOS models, TOV solvers, likelihood constraints, bug fixes, documentation improvements, and more.

## Quick Start

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/nuclear-multimessenger-astronomy/jester
cd jester

# Install with development dependencies
uv sync --extra dev

# Optional: Install CUDA support for GPU acceleration (helpful for inference)
uv sync --extra cuda12
```

### Running Tests

```bash
# Run all tests (excluding slow tests, while developing)
uv run pytest tests/ -m "not slow"

# Run all tests (if ready to merge the changes -- verifies if end-to-end inference still works)
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_eos/test_metamodel.py -v

# Run type checking
uv run pyright jesterTOV/

# Run pre-commit checks
uv run pre-commit run --all-files
```

### Building Documentation

```bash
# Build docs locally
uv run sphinx-build docs docs/_build/html

# Build in strict mode (same as CI) — ALWAYS run this before opening a PR
uv run sphinx-build -W --keep-going docs docs/_build/html

# Open in browser
open docs/_build/html/index.html  # macOS
```

One can also use `sphinx-autobuild` to automatically build the documentation on any edit, to quickly check and iterate:
```bash
uv run sphinx-autobuild docs docs/_build/html
```

> **Sphinx warnings are treated as errors in CI.** The `-W` flag used by the CI pipeline turns every Sphinx warning (undefined references, unexpected indentation, missing docstrings, etc.) into a hard build failure. A PR cannot be merged if the docs build fails. Always run the strict-mode build locally before pushing to catch issues early — it is much faster to fix a warning locally than to iterate on CI.

> **Note — `.. plot::` directive caching:** Sphinx caches the output of `.. plot::` directives
> and only regenerates them when the **Python script file** changes, not when any data files
> it reads change. If you update a data file (e.g. an `.npz` posterior file) that an embedded
> plot script reads, you must force regeneration manually:
>
> ```bash
> # Clear the cached plot output for a specific page, then rebuild
> rm -rf docs/_build/html/plot_directive/overview/likelihoods/
> uv run sphinx-build docs docs/_build/html
> ```
>
> Alternatively, touch the script file to trigger a rebuild:
> ```bash
> touch docs/overview/likelihoods/nicer_mr_plot.py
> uv run sphinx-build docs docs/_build/html
> ```

## Contributing New Features

We have detailed step-by-step guides for adding the most common new features:

### Adding a New EOS Model

Want to add a new equation of state parametrization? See the complete guide:

**📖 [Adding a New EOS Model](docs/developer_guide/adding_new_eos.md)**

Covers:
- Creating the EOS class
- Updating configuration schema and factory
- Parameter validation
- Comprehensive testing strategies
- JAX compatibility requirements
- Documentation and examples

### Adding a New TOV Solver

Implementing a modified gravity theory or new TOV equation variant? See:

**📖 [Adding a New TOV Solver](docs/developer_guide/adding_new_tov.md)**

Covers:
- Creating the solver class
- Diffrax ODE integration patterns
- Computing M-R-Λ family curves
- Type handling for vmap patterns
- Testing including GR limit checks
- Documentation with equations

### Adding a New Likelihood

Want to add a new observational constraint (GW, X-ray, radio, nuclear experiments)? See:

**📖 [Adding a New Likelihood](docs/developer_guide/adding_new_likelihood.md)**

Covers:
- Creating the likelihood class
- Data loading and management
- Integration with transform system
- Configuration and factory patterns
- Testing including full inference pipeline
- Statistical model documentation

## General Contribution Requirements

All contributions must meet these requirements:

### Code Quality

- **Type hints** - Try to include as many type annotations as possible in your code
  ```python
  def my_function(x: float, y: Array) -> Float[Array, "n"]:
      ...
  ```

- **JAX compatibility** - Ensure the code is compatible with `JAX` operations (`jit`, `vmap`, `grad`,...) in case they get used in inference 

- **Docstrings** - Use proper math formatting
  ```python
  def gamma_function(x: float) -> float:
      r"""Compute the gamma function.

      The gamma function is defined as :math:`\Gamma(x) = \int_0^\infty t^{x-1} e^{-t} dt`.

      Args:
          x: Input value where :math:`x > 0`

      Returns:
          Gamma function value
      """
  ```

### Testing

- **Unit tests** - Test individual components in isolation
- **Integration tests** - Test interactions with other components
- **Physical validation** - Verify results satisfy physics constraints
- **Comprehensive coverage** - Test edge cases, error handling

**Required test patterns:**
- Shape validation (array dimensions)
- Numerical stability (no NaN/Inf)
- Physical constraints (causality, monotonicity, positivity)
- Parameter sensitivity (results change appropriately)

### Documentation

- **Docstrings** - All public functions and classes
- **Math formatting** - Use reStructuredText `:math:` or `.. math::`
- **Citations** - Use `sphinxcontrib.bibtex` with `:cite:` directives; the bibliography style is set to `"unsrt"` in `conf.py`, which renders numeric labels `[1]`, `[2]`, etc. Keep it that way — do not switch to `"alpha"`.
- **User guide** - Add page to `docs/overview/` explaining the feature
- **API reference** - Auto-generated from docstrings (check it renders correctly)
- **Example config** - Provide working example in `examples/`

**After schema changes:**

The documentation page about settings for the config files is automatically generated to ensure it is up to date with the source code. For this, run:s
```bash
# Regenerate YAML reference
uv run python -m jesterTOV.inference.config.generate_yaml_reference
```

### Configuration Integration

For new EOS/TOV/Likelihood features:

1. Update `jesterTOV/inference/config/schema.py` with Pydantic model
2. Register in factory (`transforms/transform.py` or `likelihoods/factory.py`)
3. Regenerate YAML documentation
4. Add parameter validation if needed

## Pull Request Process

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes** following the guidelines above

3. **Run the test suite**:
   ```bash
   uv run pytest tests/ -m "not slow"
   uv run pyright jesterTOV/
   uv run pre-commit run --all-files
   ```

4. **Build documentation** (strict mode):
   ```bash
   uv run sphinx-build -W --keep-going docs docs/_build/html
   ```

5. **Commit with descriptive messages**:
   ```bash
   git add .
   git commit -m "Add spectral EOS model with 4-parameter expansion"
   ```

6. **Push and create PR**:
   ```bash
   git push origin feature/my-new-feature
   ```
   Then open a pull request on GitHub

### What Reviewers (and CI/CD pipeline) Look For

- All tests pass (including type checking)
- **Documentation builds without warnings** — Sphinx runs with `-W` in CI so any warning blocks the merge. Run `uv run sphinx-build -W --keep-going docs docs/_build/html` locally before pushing.
- Code follows project conventions (see `CLAUDE.md`)
- Comprehensive tests covering edge cases
- Clear commit messages
- No backwards compatibility breaks (pre-release, so less critical)

## Development Philosophy

> **Testing Philosophy**: When tests fail, investigate root causes rather than modifying tests to pass.

> **Documentation Style**: Write clear, concise documentation in full sentences as if by a human researcher. Avoid LLM-like verbosity.

> **JAX-first**: All physics calculations must be JAX-compatible for hardware acceleration and automatic differentiation.

## Getting Help

- **Documentation**: https://nuclear-multimessenger-astronomy.github.io/jester/
- **Architecture guide**: See `jesterTOV/inference/CLAUDE.md` for inference system details
- **Issues**: https://github.com/nuclear-multimessenger-astronomy/jester/issues
- **Discussions**: https://github.com/nuclear-multimessenger-astronomy/jester/discussions

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and professional in all interactions.

---

**Ready to contribute?** Pick a feature from the guides above and dive in! If you're unsure where to start, check the [good first issue](https://github.com/nuclear-multimessenger-astronomy/jester/labels/good%20first%20issue) label on GitHub.
