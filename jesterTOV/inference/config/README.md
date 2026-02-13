# Configuration Documentation Generation

This directory contains the auto-generation system for YAML configuration reference documentation.

## Architecture

The documentation generation system uses **Jinja2 templates** to separate data extraction from formatting:

```
config/
├── generate_yaml_reference.py   # Extracts field data from Pydantic schemas
├── templates/
│   └── yaml_reference.md.j2     # MyST markdown template
└── schema.py                     # Pydantic configuration models (source of truth)
```

## How It Works

1. **Data Extraction** (`generate_yaml_reference.py`):
   - Introspects Pydantic models in `schema.py`
   - Extracts field names, types, defaults, descriptions
   - Organizes data into structured dictionaries

2. **Template Rendering** (`templates/yaml_reference.md.j2`):
   - Jinja2 template defines document structure
   - Uses MyST markdown with dropdown directives
   - Loops over extracted data to generate sections

3. **Output**:
   - Generates `docs/inference/yaml_reference.md`
   - Auto-included in Sphinx documentation build

## Usage

Regenerate documentation after modifying `schema.py`:

```bash
uv run python -m jesterTOV.inference.config.generate_yaml_reference
```

## Benefits of Template Approach

### Separation of Concerns
- **Python code** handles data extraction (logic)
- **Jinja2 template** handles formatting (presentation)
- Changes to format don't require touching Python code

### Easy Maintenance
- Update template to change layout/styling
- Add new sections by extending template
- Template clearly shows document structure

### Format Flexibility
- Easy to add dropdown sections, admonitions, tabs
- MyST markdown syntax fully supported
- Future format changes are template-only edits

### No Manual Sync Issues
- Documentation auto-generated from source of truth
- Field changes in `schema.py` automatically reflected
- No risk of docs drifting out of sync

## Comparison to Previous Approach

**Old approach** (manual f-strings):
```python
doc = f"### {field_name}\n\n"
doc += f"**Type**: `{field_type}`\n\n"
# ... 400+ lines of string formatting
```

**New approach** (template):
```jinja2
### {{ field.name }}

**Type**: `{{ field.type }}`
```

## Template Syntax Notes

- Use `{% for item in items %}...{% endfor %}` for loops
- Use `{{ variable }}` for value interpolation
- Use `{% if condition %}...{% endif %}` for conditionals
- Avoid naming fields `items` (conflicts with Jinja2 dict.items())

## Adding New Sections

1. Add extraction function to `generate_yaml_reference.py`:
   ```python
   def extract_new_section() -> list[dict[str, Any]]:
       return [{"name": "...", "description": "..."}]
   ```

2. Add to data dict in `generate_documentation()`:
   ```python
   data = {
       "new_section": extract_new_section(),
       # ...
   }
   ```

3. Add to template `yaml_reference.md.j2`:
   ```jinja2
   ## New Section

   {% for item in new_section %}
   - {{ item.name }}: {{ item.description }}
   {% endfor %}
   ```

4. Regenerate: `uv run python -m jesterTOV.inference.config.generate_yaml_reference`

## Important Limitation: Likelihood Parameters

⚠️ **Manual Sync Required**: `LikelihoodConfig` uses a generic `parameters: dict` field documented in docstrings rather than typed Pydantic fields. This means:

1. **Parameter definitions are manually maintained** in `extract_likelihoods()` function
2. **Must be kept in sync** with:
   - `schema.py` LikelihoodConfig docstring (lines 125-197)
   - `schema.py` `validate_likelihood_parameters()` method
3. **Verification needed** when modifying likelihood schema

**When you modify likelihood parameters in schema.py:**
1. Update the docstring in `LikelihoodConfig`
2. Update validation logic in `validate_likelihood_parameters()`
3. Update `extract_likelihoods()` in this script
4. Regenerate documentation

**Future Improvement**: Refactor `schema.py` to use discriminated unions with typed Pydantic models for each likelihood type. This would enable true automatic introspection.

## CI/CD Integration

The documentation build in CI/CD will fail if:
- Template syntax is invalid
- Generated markdown has Sphinx warnings
- Schema changes without regenerating docs (manual check)

**Recommendation**: Add pre-commit hook to auto-regenerate docs when `schema.py` changes.
