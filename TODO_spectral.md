docs/inference_yaml_reference.md (3)
115-120: Fix inconsistency between documented options and validation rules.

Line 115 documents crust_name as accepting "DH" | "BPS" | "DH_fixed" | "SLy", but line 120 in the validation rules only mentions "DH", "BPS", or "DH_fixed". The validation rule needs to be updated to include "SLy".

388-407: Add validation rules for spectral transform type.

The validation rules section documents requirements for "metamodel" and "metamodel_cse" transforms but omits constraints for the "spectral" type. The schema enforces:

Spectral crust requirement: crust_name must be "SLy" (LALSuite compatibility)
Spectral CSE constraint: nb_CSE must be 0 (inherited from metamodel behavior)
Spectral-specific field: n_points_high defines high-density sampling points
Additionally, the likelihood section references "constraints_gamma" as spectral-only, which should be clarified in the validation rules section.