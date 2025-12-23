# JESTER Inference Documentation Index

**Complete guide to Bayesian EOS inference with JESTER**

## Documentation Overview

Choose the right guide for your needs:

### 🚀 Quick Start
**[Quick Start Guide](inference_quickstart.md)** - *5-10 minutes*
- Installation instructions
- Run your first inference
- Minimal configuration examples
- Load and plot results
- **Start here if**: You want to get up and running quickly

### 📘 Complete Reference
**[Complete Documentation](inference.md)** - *30-60 minutes*
- Full system architecture
- Configuration system in detail
- Prior specification format
- All likelihood types
- Transform mathematics
- Data management
- Advanced usage
- **Start here if**: You want comprehensive understanding

### 🏗️ Architecture Guide
**[Architecture Documentation](inference_architecture.md)** - *20-40 minutes*
- Module dependency graph
- Execution flow diagrams
- Component interfaces
- Data transformations
- Class hierarchy
- Design patterns
- **Start here if**: You want to understand implementation details or contribute code

### 📋 YAML Configuration Reference
**[YAML Reference](inference_yaml_reference.md)** - *Auto-generated reference*
- Complete list of all YAML options
- Auto-generated from Pydantic schemas
- Every field with type, default, and description
- Validation rules
- **Start here if**: You need the authoritative list of all configuration options

### 📁 Module README
**[Inference Module README](../jesterTOV/inference/README.md)** - *10-15 minutes*
- Module structure overview
- Basic workflow
- File connections
- Quick reference tables
- Development status
- **Start here if**: You're browsing the codebase

---

## Documentation by Task

### Getting Started

| Task | Documentation | Time |
|------|---------------|------|
| Install JESTER | [Quick Start](inference_quickstart.md#installation) | 2 min |
| Run first inference | [Quick Start](inference_quickstart.md#run-your-first-inference) | 5 min |
| Understand workflow | [Architecture](inference_architecture.md#execution-flow) | 10 min |

### Configuration

| Task | Documentation | Time |
|------|---------------|------|
| Create config file | [Quick Start](inference_quickstart.md#configuration-files-explained) | 5 min |
| See all YAML options | [YAML Reference](inference_yaml_reference.md) | 2 min |
| Understand YAML structure | [Complete Docs](inference.md#configuration-system) | 15 min |
| Validation rules | [Complete Docs](inference.md#configuration-validation) | 10 min |
| Path resolution | [Complete Docs](inference.md#path-resolution) | 5 min |

### Priors

| Task | Documentation | Time |
|------|---------------|------|
| Create prior file | [Quick Start](inference_quickstart.md#prior-file) | 5 min |
| Understand format | [Complete Docs](inference.md#prior-file-format) | 10 min |
| Inclusion rules | [Complete Docs](inference.md#prior-inclusion-rules) | 10 min |
| CSE grid parameters | [Complete Docs](inference.md#cse-grid-parameters) | 5 min |

### Transforms

| Task | Documentation | Time |
|------|---------------|------|
| Choose transform type | [Quick Start](inference_quickstart.md#use-metamodel--cse) | 5 min |
| MetaModel details | [Complete Docs](inference.md#1-metamodel-transform) | 10 min |
| MetaModel+CSE details | [Complete Docs](inference.md#2-metamodel-cse-transform) | 10 min |
| Transform workflow | [Architecture](inference_architecture.md#parameter--eos--observables) | 10 min |

### Likelihoods

| Task | Documentation | Time |
|------|---------------|------|
| Add data constraint | [Quick Start](inference_quickstart.md#add-real-observational-data) | 5 min |
| All likelihood types | [Complete Docs](inference.md#available-likelihood-types) | 20 min |
| Custom likelihood | [Complete Docs](inference.md#custom-likelihoods) | 30 min |
| Combined likelihood | [Architecture](inference_architecture.md#likelihood-system) | 10 min |

### Data Management

| Task | Documentation | Time |
|------|---------------|------|
| Configure data paths | [Complete Docs](inference.md#data-path-configuration) | 10 min |
| Understand caching | [Complete Docs](inference.md#caching-mechanism) | 5 min |
| DataLoader interface | [Architecture](inference_architecture.md#data-loading) | 10 min |

### Sampling

| Task | Documentation | Time |
|------|---------------|------|
| Configure sampler | [Quick Start](inference_quickstart.md#configuration-files-explained) | 5 min |
| Sampler parameters | [Complete Docs](inference.md#sampler-parameters) | 15 min |
| Sampling phases | [Complete Docs](inference.md#sampling-phases) | 10 min |
| Tune acceptance rates | [Complete Docs](inference.md#adjusting-step-sizes) | 10 min |

### Results Analysis

| Task | Documentation | Time |
|------|---------------|------|
| Load results | [Quick Start](inference_quickstart.md#analyze-results) | 5 min |
| Output file structure | [Complete Docs](inference.md#output-files) | 5 min |
| Corner plots | [Quick Start](inference_quickstart.md#make-corner-plot) | 5 min |
| M-R diagrams | [Quick Start](inference_quickstart.md#plot-m-r-diagram) | 5 min |

### Advanced Topics

| Task | Documentation | Time |
|------|---------------|------|
| Add custom likelihood | [Complete Docs](inference.md#custom-likelihoods) | 30 min |
| Add custom prior | [Complete Docs](inference.md#custom-priors) | 20 min |
| Parallel runs | [Complete Docs](inference.md#parallel-runs) | 10 min |
| Understand architecture | [Architecture](inference_architecture.md) | 40 min |

### Troubleshooting

| Task | Documentation | Time |
|------|---------------|------|
| Common issues | [Quick Start](inference_quickstart.md#common-issues) | 10 min |
| Performance tips | [Quick Start](inference_quickstart.md#performance-tips) | 10 min |

---

## Documentation by Role

### 👤 End User (Running Inference)

**Recommended path**:
1. Start: [Quick Start Guide](inference_quickstart.md)
2. Deep dive: [Complete Documentation](inference.md)
3. Reference: [Module README](../jesterTOV/inference/README.md)

**Focus areas**:
- Configuration system
- Prior specification
- Likelihood types
- Results analysis

### 👨‍💻 Developer (Contributing Code)

**Recommended path**:
1. Start: [Architecture Documentation](inference_architecture.md)
2. Reference: [Complete Documentation](inference.md)
3. Code: [Module README](../jesterTOV/inference/README.md)

**Focus areas**:
- Module structure
- Class hierarchy
- Component interfaces
- Design patterns

### 👨‍🔬 Researcher (Method Development)

**Recommended path**:
1. Start: [Complete Documentation](inference.md)
2. Details: [Architecture Documentation](inference_architecture.md)
3. Examples: Check `examples/inference/`

**Focus areas**:
- Transform mathematics
- Likelihood evaluation
- Sampling methodology
- Custom components

---

## Examples

All documentation references working examples in `jester/examples/inference/`:

| Example | Description | Config | Prior | Use Case |
|---------|-------------|--------|-------|----------|
| `prior/` | Prior-only sampling | [config.yaml](../examples/inference/prior/config.yaml) | [prior.prior](../examples/inference/prior/prior.prior) | Testing, prior predictive |
| `nicer_only/` | NICER data only | [config.yaml](../examples/inference/nicer_only/config.yaml) | [prior.prior](../examples/inference/nicer_only/prior.prior) | X-ray timing constraint |
| `gw170817_only/` | GW170817 only | [config.yaml](../examples/inference/gw170817_only/config.yaml) | [prior.prior](../examples/inference/gw170817_only/prior.prior) | GW constraint |
| `full_inference/` | All constraints | [config.yaml](../examples/inference/full_inference/config.yaml) | [prior.prior](../examples/inference/full_inference/prior.prior) | Multi-messenger |
| `MM_CSE/` | MetaModel+CSE | [config.yaml](../examples/inference/MM_CSE/config.yaml) | [prior.prior](../examples/inference/MM_CSE/prior.prior) | High-density EOS |

---

## Quick Reference

### Command Line

```bash
# Run inference
run_jester_inference config.yaml

# Validate only
run_jester_inference config.yaml --validate-only

# Dry run
run_jester_inference config.yaml --dry-run

# Override output dir
run_jester_inference config.yaml --output-dir ./results/
```

### Python API

```python
# Load configuration
from jesterTOV.inference.config.parser import load_config
config = load_config("config.yaml")

# Parse prior
from jesterTOV.inference.priors.parser import parse_prior_file
prior = parse_prior_file("prior.prior", nb_CSE=8)

# Create transform
from jesterTOV.inference.transforms.factory import create_transform
transform = create_transform(config.transform)

# Create likelihood
from jesterTOV.inference.likelihoods.factory import create_combined_likelihood
from jesterTOV.inference.data.loader import DataLoader
data_loader = DataLoader(data_paths=config.data_paths)
likelihood = create_combined_likelihood(config.likelihoods, data_loader)

# Setup sampler
from jesterTOV.inference.samplers.flowmc import setup_flowmc_sampler
sampler = setup_flowmc_sampler(config.sampler, prior, likelihood, transform)

# Run sampling
import jax
sampler.sample(jax.random.PRNGKey(43))
samples = sampler.get_samples(training=False)
```

### Configuration Snippets

**MetaModel transform**:
```yaml
transform:
  type: "metamodel"
  ndat_metamodel: 100
  nmax_nsat: 25.0
```

**MetaModel+CSE transform**:
```yaml
transform:
  type: "metamodel_cse"
  nb_CSE: 8
```

**NICER likelihood**:
```yaml
likelihoods:
  - type: "nicer"
    enabled: true
    parameters:
      targets: ["J0030", "J0740"]
```

**GW likelihood**:
```yaml
likelihoods:
  - type: "gw"
    enabled: true
    parameters:
      event_name: "GW170817"
      sample_masses: true
```

### Prior Snippets

**Standard NEP priors**:
```python
K_sat = UniformPrior(150.0, 300.0, parameter_names=["K_sat"])
L_sym = UniformPrior(10.0, 200.0, parameter_names=["L_sym"])
```

**CSE breaking density** (only if nb_CSE > 0):
```python
nbreak = UniformPrior(0.16, 0.32, parameter_names=["nbreak"])
```

---

## Documentation Maintenance

### Version Information

- **Documentation Version**: 1.0
- **Last Updated**: December 2024
- **JESTER Version**: Compatible with v0.1.0+
- **Status**: Production-ready

### Contributing to Documentation

If you find errors or want to improve the documentation:

1. **Minor fixes**: Edit the relevant `.md` file
2. **Major changes**: Discuss in GitHub issues first
3. **New sections**: Follow existing structure and style

### Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| `inference_index.md` | Navigation hub | All |
| `inference_quickstart.md` | Quick start guide | End users |
| `inference.md` | Complete reference | All |
| `inference_architecture.md` | Technical details | Developers |
| `../jesterTOV/inference/README.md` | Module overview | Developers |

---

## Getting Help

### Documentation Not Enough?

1. **Check examples**: `jester/examples/inference/` for working configs
2. **Read source code**: Modules have detailed docstrings
3. **GitHub Issues**: Report bugs or request features
4. **Community**: Ask questions in discussions

### Reporting Documentation Issues

Please report:
- Broken links
- Unclear explanations
- Missing information
- Outdated content
- Typos

**Where to report**: GitHub Issues with label `documentation`

---

## Related Resources

### JESTER Core Documentation

- [Main README](../README.md) - Project overview
- [API Documentation](../docs/) - Auto-generated API docs
- [Examples](../examples/) - Code examples

### External Resources

- **flowMC**: https://github.com/ThibeauWouters/flowMC
- **Jim**: https://github.com/ThibeauWouters/jim
- **JAX**: https://jax.readthedocs.io/

---

**Happy Inferencing! 🚀**

For questions, issues, or contributions, visit: https://github.com/nuclear-multimessenger-astronomy/jester
