# Inference Documentation Maintenance Guide

**For developers: How to keep documentation in sync with code**

## Overview

The JESTER inference documentation uses a **hybrid approach**:
- **Auto-generated**: YAML configuration reference (from Pydantic schemas)
- **Manual**: Narrative documentation, guides, examples

This ensures accuracy while maintaining readability.

---

## Auto-Generated Documentation

### YAML Configuration Reference

**File**: `docs/inference_yaml_reference.md`
**Source**: `jesterTOV/inference/config/schema.py` (Pydantic models)
**Generator**: `jesterTOV/inference/config/generate_yaml_reference.py`

#### When to Regenerate

Regenerate whenever you modify:
- `config/schema.py` - Any changes to Pydantic models
- Field names, types, or defaults
- Validation rules
- Documentation strings in `Field(...)`

#### How to Regenerate

```bash
# From repository root
uv run python -m jesterTOV.inference.config.generate_yaml_reference

# This creates/updates:
#   docs/inference_yaml_reference.md
```

#### Automated Reminder

The `config/schema.py` file has a docstring reminder:
```python
"""
IMPORTANT: When you modify these schemas, regenerate the YAML reference:
    uv run python -m jesterTOV.inference.config.generate_yaml_reference
"""
```

#### Pre-Commit Hook (Recommended)

Add to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: regenerate-yaml-reference
      name: Regenerate YAML reference
      entry: uv run python -m jesterTOV.inference.config.generate_yaml_reference
      language: system
      files: ^jesterTOV/inference/config/schema\.py$
      pass_filenames: false
```

This automatically regenerates the reference when `schema.py` changes.

---

## Manual Documentation

### Files to Update Manually

| File | Update When | Contents |
|------|-------------|----------|
| `inference.md` | Major features added | Complete reference, explanations |
| `inference_quickstart.md` | User workflow changes | Quick start guide, examples |
| `inference_architecture.md` | Architecture changes | Module structure, data flow |
| `inference_index.md` | New docs added | Navigation, table of contents |
| `jesterTOV/inference/README.md` | Module structure changes | Module overview |

### Update Checklist

When adding a **new likelihood type**:
- [ ] Add to `likelihoods/my_likelihood.py`
- [ ] Update `likelihoods/factory.py`
- [ ] Update `config/schema.py` (add to `LikelihoodConfig.type` Literal)
- [ ] Regenerate YAML reference (auto)
- [ ] Document in `inference.md` → "Likelihoods" section (manual)
- [ ] Add example to `inference_quickstart.md` (manual)
- [ ] Update `inference_yaml_reference.md` "Likelihood-Specific Parameters" (manual - generator doesn't know about `parameters` dict contents)

When adding a **new prior type**:
- [ ] Add to `priors/simple_priors.py`
- [ ] Update `priors/parser.py` namespace
- [ ] Document in `inference.md` → "Prior Specification" section (manual)
- [ ] Add example to `inference_quickstart.md` (manual)

When adding a **new transform type**:
- [ ] Add to `transforms/my_transform.py`
- [ ] Update `transforms/factory.py`
- [ ] Update `config/schema.py` (add to `TransformConfig.type` Literal)
- [ ] Regenerate YAML reference (auto)
- [ ] Document in `inference.md` → "Transforms" section (manual)
- [ ] Add example configuration (manual)

When modifying **configuration fields**:
- [ ] Modify `config/schema.py`
- [ ] Regenerate YAML reference (auto)
- [ ] Update examples in `inference.md` if field is commonly used (manual)
- [ ] Update `inference_quickstart.md` if it affects quick start (manual)

---

## Documentation Structure

### Source of Truth Hierarchy

1. **Code** (ultimate truth)
   - `config/schema.py` - Configuration validation
   - Module docstrings - API documentation
   - Type hints - Function signatures

2. **Auto-generated docs** (always in sync with code)
   - `docs/inference_yaml_reference.md` - All YAML options

3. **Manual docs** (requires human updates)
   - `docs/inference.md` - Complete reference
   - `docs/inference_quickstart.md` - Quick start
   - `docs/inference_architecture.md` - Architecture
   - `docs/inference_index.md` - Navigation

### Avoiding Duplication

**Don't duplicate information that can be auto-generated**:
- ❌ List all YAML fields manually in `inference.md`
- ✅ Link to `inference_yaml_reference.md` for complete list
- ✅ Show key examples and explain concepts

**Example**:

```markdown
<!-- Good: Explain concept, link to reference -->
## Configuration System

JESTER uses YAML files for configuration. For a complete list of all
available options, see the [YAML Reference](inference_yaml_reference.md).

Key configuration sections:
- `transform`: How to convert parameters to observables
- `prior`: Prior distributions
- `likelihoods`: Observational constraints
...

<!-- Bad: Duplicate auto-generated content -->
## Configuration System

All available fields:
- seed: int, default 43
- transform: TransformConfig (required)
  - type: "metamodel" | "metamodel_cse" (required)
  - ndat_metamodel: int, default 100
  ... (this will get out of sync!)
```

---

## Testing Documentation

### Manual Testing Checklist

Before committing documentation changes:

- [ ] **Links work**: Check all internal links resolve
- [ ] **Code examples run**: Test YAML configs and Python snippets
- [ ] **Formatting renders**: Check markdown renders correctly
- [ ] **Examples are current**: Verify examples match latest code

### Link Checking

```bash
# Check for broken links (requires markdown-link-check)
npm install -g markdown-link-check
markdown-link-check docs/inference*.md
```

### Code Example Testing

Extract and test code examples:

```bash
# Test YAML examples
cat docs/inference_quickstart.md | \
  sed -n '/```yaml/,/```/p' | \
  sed '/```/d' > /tmp/test_config.yaml

uv run python -m jesterTOV.inference.run_inference \
  --config /tmp/test_config.yaml \
  --validate-only
```

---

## Version Control

### Documentation Commits

Follow these conventions:

```bash
# When regenerating auto-docs
git commit -m "docs: regenerate YAML reference after schema changes"

# When updating manual docs
git commit -m "docs: add likelihood type XYZ to inference guide"

# When fixing docs issues
git commit -m "docs: fix broken link in inference quickstart"
```

### Pull Request Checklist

When your PR changes inference code:

- [ ] Updated/regenerated auto-generated docs (if schema changed)
- [ ] Updated manual docs (if user-facing features changed)
- [ ] Added/updated examples (if workflow changed)
- [ ] Tested documentation links
- [ ] Checked code examples still work

---

## Documentation Review

### Self-Review Checklist

Before requesting review:

1. **Accuracy**: Does the doc match the code?
2. **Completeness**: Are all new features documented?
3. **Clarity**: Can a new user understand it?
4. **Examples**: Are there working examples?
5. **Links**: Do all references work?

### Reviewer Checklist

When reviewing documentation PRs:

1. **Verify auto-generation**: If `schema.py` changed, was reference regenerated?
2. **Check examples**: Do code examples run?
3. **Test links**: Click through major navigation paths
4. **Assess clarity**: Is the explanation understandable?
5. **Look for duplication**: Is info duplicated that could be auto-generated?

---

## Common Pitfalls

### ❌ Don't Do This

**Pitfall 1: Forgetting to regenerate**
```python
# You modify config/schema.py
class SamplerConfig(BaseModel):
    n_chains: int = 20
    new_field: int = 100  # Added

# ❌ Commit without regenerating reference
# → docs/inference_yaml_reference.md is now out of sync!
```

**Solution**: Always run generator after modifying schemas.

**Pitfall 2: Duplicating auto-generated content**
```markdown
<!-- ❌ Don't copy-paste from inference_yaml_reference.md -->
All sampler fields:
- n_chains: int, default 20
- n_loop_training: int, default 3
...

<!-- ✅ Instead, link and explain -->
See [YAML Reference](inference_yaml_reference.md#sampler-configuration)
for all sampler fields. Key parameters to tune:
- `n_chains`: More chains = better convergence but slower
- `learning_rate`: Controls NF training speed
```

**Pitfall 3: Hardcoding examples that will break**
```markdown
<!-- ❌ Example that will break if defaults change -->
Run with default settings (20 chains, 3 training loops):
```yaml
sampler:
  n_chains: 20
  n_loop_training: 3
```

<!-- ✅ Show minimal example, reference defaults -->
Run with default settings (see [defaults](inference_yaml_reference.md)):
```yaml
sampler:
  output_dir: "./outdir/"  # Only override what you need
```
```

---

## Future Improvements

### Potential Automation

1. **API documentation from docstrings**
   - Use Sphinx autodoc for module reference
   - Generate from `jesterTOV.inference` package docstrings

2. **Example validation in CI**
   - Extract code examples from markdown
   - Run `--validate-only` on all YAML examples
   - Fail CI if examples are broken

3. **Link checking in CI**
   - Automated broken link detection
   - Run on every PR that touches docs

4. **Documentation coverage**
   - Track what % of public APIs are documented
   - Alert when new public functions lack docstrings

### Suggested Pre-Commit Hook (Complete)

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      # Regenerate YAML reference when schema changes
      - id: regenerate-yaml-reference
        name: Regenerate YAML reference
        entry: uv run python -m jesterTOV.inference.config.generate_yaml_reference
        language: system
        files: ^jesterTOV/inference/config/schema\.py$
        pass_filenames: false

      # Check documentation links
      - id: markdown-link-check
        name: Check markdown links
        entry: markdown-link-check
        language: node
        files: \.md$
        additional_dependencies: ['markdown-link-check']

      # Validate YAML examples in documentation
      - id: validate-yaml-examples
        name: Validate YAML examples
        entry: scripts/validate_yaml_examples.sh
        language: system
        files: ^docs/inference.*\.md$
        pass_filenames: false
```

---

## Summary

### Quick Reference

| Task | Command |
|------|---------|
| Regenerate YAML reference | `uv run python -m jesterTOV.inference.config.generate_yaml_reference` |
| Check links | `markdown-link-check docs/inference*.md` |
| Validate config example | `run_jester_inference config.yaml --validate-only` |

### Golden Rule

> **If the code changes, the documentation must change.**
> **If the schema changes, regenerate the auto-docs.**

### Documentation Workflow

```
Code Change
    ↓
Modify schema.py? → Yes → Regenerate YAML reference
    ↓                           ↓
    ↓                      (auto-updated)
    ↓
User-facing change? → Yes → Update manual docs
    ↓                           ↓
    ↓                      (inference.md, quickstart.md, etc.)
    ↓
Commit
    ↓
PR Review
    ↓
Merge
```

---

**Maintainer Note**: Keep this guide updated as the documentation system evolves!

**Last Updated**: December 2024
