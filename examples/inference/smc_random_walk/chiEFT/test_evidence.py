#!/usr/bin/env python3
"""
Test script to verify SMC Bayesian evidence calculation.

This script:
1. Loads the results.h5 file from outdir
2. Verifies evidence is stored in metadata
3. Checks evidence value is valid (not NaN/inf)
4. Displays evidence and other SMC diagnostics
"""

import sys
import os
import numpy as np

# Add jester to path
sys.path.insert(0, '/Users/Woute029/Documents/Code/projects/jester_review/jester')

from jesterTOV.inference.result import InferenceResult

def test_evidence_calculation():
    """Test that SMC evidence is correctly computed and stored."""

    print("="*70)
    print("SMC Bayesian Evidence Test")
    print("="*70)

    # Load results from outdir
    results_path = "outdir/results.h5"
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found!")
        print(f"Current directory: {os.getcwd()}")
        return False

    print(f"\n✓ Loading results from {results_path}...")
    result = InferenceResult.load(results_path)

    # Check sampler type
    print(f"\n✓ Sampler type: {result.sampler_type}")
    if result.sampler_type != "blackjax_smc":
        print(f"WARNING: Expected blackjax_smc, got {result.sampler_type}")
        return False

    # Check metadata contains evidence
    print(f"\n✓ Checking metadata for evidence fields...")
    has_logZ = 'logZ' in result.metadata
    has_logZ_err = 'logZ_err' in result.metadata

    print(f"  - 'logZ' present: {has_logZ}")
    print(f"  - 'logZ_err' present: {has_logZ_err}")

    if not has_logZ:
        print("ERROR: logZ not found in metadata!")
        print(f"Available metadata keys: {list(result.metadata.keys())}")
        return False

    # Extract evidence
    logZ = result.metadata['logZ']
    logZ_err = result.metadata.get('logZ_err', 0.0)

    print(f"\n{'='*70}")
    print("BAYESIAN EVIDENCE")
    print(f"{'='*70}")
    print(f"  log(Z) = {logZ:.6f}")
    if logZ_err > 0:
        print(f"  Error  = ±{logZ_err:.6f}")
    else:
        print(f"  Error  = {logZ_err:.6f} (placeholder, error estimation not yet implemented)")
    print(f"  Z      = {np.exp(logZ):.6e}")

    # Validate evidence value
    print(f"\n✓ Validating evidence value...")
    is_valid = True

    if np.isnan(logZ):
        print("  ERROR: log(Z) is NaN!")
        is_valid = False
    elif np.isinf(logZ):
        print("  ERROR: log(Z) is infinite!")
        is_valid = False
    else:
        print(f"  ✓ log(Z) is finite and valid")

    # Display SMC-specific metadata
    print(f"\n{'='*70}")
    print("SMC METADATA")
    print(f"{'='*70}")

    smc_keys = ['n_particles', 'n_mcmc_steps', 'kernel_type',
                'annealing_steps', 'final_ess', 'mean_ess']

    for key in smc_keys:
        if key in result.metadata:
            value = result.metadata[key]
            if isinstance(value, float):
                print(f"  {key:20s}: {value:.4f}")
            else:
                print(f"  {key:20s}: {value}")

    # Check histories
    print(f"\n{'='*70}")
    print("SMC HISTORIES")
    print(f"{'='*70}")

    if result.histories is not None:
        for key in ['lmbda_history', 'ess_history', 'acceptance_history']:
            if key in result.histories:
                history = result.histories[key]
                print(f"  {key:25s}: shape={history.shape}, "
                      f"min={history.min():.4f}, max={history.max():.4f}, "
                      f"mean={history.mean():.4f}")
    else:
        print("  No histories stored")

    # Display posterior summary
    print(f"\n{'='*70}")
    print("POSTERIOR SUMMARY")
    print(f"{'='*70}")
    print(f"  Number of samples: {len(result.posterior.get('log_prob', []))}")

    param_keys = [k for k in result.posterior.keys() if k in ['L_sym', 'K_sat', 'Q_sat', 'K_sym']]
    print(f"  Sample parameters: {param_keys}")

    if 'log_prob' in result.posterior:
        log_prob = result.posterior['log_prob']
        print(f"  log_prob range: [{log_prob.min():.2f}, {log_prob.max():.2f}]")
        print(f"  log_prob mean: {log_prob.mean():.2f}")

    # Final verdict
    print(f"\n{'='*70}")
    if is_valid:
        print("✓ SUCCESS: Evidence calculation appears correct!")
    else:
        print("✗ FAILED: Issues detected with evidence calculation")
    print(f"{'='*70}\n")

    return is_valid


if __name__ == "__main__":
    success = test_evidence_calculation()
    sys.exit(0 if success else 1)
