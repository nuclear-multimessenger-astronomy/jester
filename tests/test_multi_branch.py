"""Tests for multi-branch stable stellar segment detection and interpolation."""

import jax.numpy as jnp
import pytest

from jesterTOV.utils import (
    N_MAX_BRANCHES,
    detect_stable_segments,
    interp_family_multi_branch,
)


class TestDetectStableSegments:
    def test_single_branch_monotone(self):
        """Strictly monotone M(pc) → exactly 1 valid segment."""
        pc = jnp.linspace(1e-4, 1e-2, 50)
        m = jnp.linspace(0.5, 2.0, 50)
        starts, ends, masks = detect_stable_segments(pc, m)
        assert int(jnp.sum(masks)) == 1
        assert bool(masks[0])

    def test_double_branch(self):
        """Fold-back in M(pc) → exactly 2 valid segments."""
        m = jnp.array(
            [0.3, 0.5, 0.7, 0.9, 1.1, 1.0, 0.95, 0.96, 1.05, 1.2],
            dtype=jnp.float64,
        )
        pc = jnp.linspace(1e-4, 1e-2, len(m))
        starts, ends, masks = detect_stable_segments(pc, m)
        assert int(jnp.sum(masks)) == 2

    def test_padded_entries_have_false_mask(self):
        """Unused segment entries must have masks=False."""
        pc = jnp.linspace(1e-4, 1e-2, 20)
        m = jnp.linspace(0.5, 2.0, 20)
        _, _, masks = detect_stable_segments(pc, m)
        # Only first entry is valid for a single-branch EOS
        assert bool(masks[0])
        for b in range(1, N_MAX_BRANCHES):
            assert not bool(masks[b])

    def test_single_branch_segment_covers_all_points(self):
        """All n points are in branch 0 for a pure monotone sequence."""
        n = 30
        pc = jnp.linspace(1e-4, 1e-2, n)
        m = jnp.linspace(0.5, 2.0, n)
        starts, ends, masks = detect_stable_segments(pc, m)
        assert int(starts[0]) == 0
        assert int(ends[0]) == n - 1

    def test_double_branch_segment_indices(self):
        """Segment indices correctly identify the two stable branches."""
        m = jnp.array(
            [0.3, 0.5, 0.7, 0.9, 1.1, 1.0, 0.95, 0.96, 1.05, 1.2],
            dtype=jnp.float64,
        )
        pc = jnp.linspace(1e-4, 1e-2, len(m))
        starts, ends, masks = detect_stable_segments(pc, m)
        # Branch 0: indices 0..4 (m rises 0.3→1.1)
        assert int(starts[0]) == 0
        assert int(ends[0]) == 4
        # Branch 1: indices 6..9 (m rises 0.95→1.2)
        assert int(starts[1]) == 6
        assert int(ends[1]) == 9


class TestInterpFamilyMultiBranch:
    def _make_single_branch(self, ndat=100):
        """Build masses/values/branch_ids for a pure single-branch EOS."""
        masses = jnp.linspace(0.5, 2.0, ndat)
        values = jnp.linspace(14.0, 10.0, ndat)  # radii decrease with mass
        branch_ids = jnp.zeros(ndat, dtype=jnp.int32)
        return masses, values, branch_ids

    def test_single_branch_in_range(self):
        """Query inside the single branch returns in_range[0]=True."""
        masses, values, bids = self._make_single_branch()
        _, in_range = interp_family_multi_branch(1.0, masses, values, bids)
        assert bool(in_range[0])
        assert int(jnp.sum(in_range)) == 1

    def test_single_branch_value_matches_interp(self):
        """Single branch: result equals jnp.interp at the same query."""
        masses, values, bids = self._make_single_branch()
        result, _ = interp_family_multi_branch(1.2, masses, values, bids)
        expected = jnp.interp(1.2, masses, values)
        assert jnp.allclose(result[0], expected, atol=1e-6)

    def test_absent_branch_not_in_range(self):
        """All-sentinel branch_ids: no branch is in_range."""
        ndat = 50
        masses = jnp.linspace(0.5, 2.0, ndat)
        values = jnp.ones(ndat)
        bids = jnp.full(ndat, N_MAX_BRANCHES, dtype=jnp.int32)
        _, in_range = interp_family_multi_branch(1.0, masses, values, bids)
        assert int(jnp.sum(in_range)) == 0

    def test_out_of_range_not_in_range(self):
        """Query below/above branch mass range: in_range[0]=False."""
        masses, values, bids = self._make_single_branch()
        _, in_range_low = interp_family_multi_branch(0.1, masses, values, bids)
        _, in_range_high = interp_family_multi_branch(3.0, masses, values, bids)
        assert not bool(in_range_low[0])
        assert not bool(in_range_high[0])

    def test_two_overlapping_branches(self):
        """Two branches with overlapping mass ranges: both in_range at overlap mass."""
        ndat = 20
        # Branch 0: masses 0.3→1.2, Branch 1: masses 0.9→1.5
        m0 = jnp.linspace(0.3, 1.2, ndat)
        v0 = jnp.linspace(14.0, 12.0, ndat)
        m1 = jnp.linspace(0.9, 1.5, ndat)
        v1 = jnp.linspace(11.0, 10.0, ndat)

        masses = jnp.concatenate([m0, m1])
        values = jnp.concatenate([v0, v1])
        bids = jnp.concatenate(
            [
                jnp.zeros(ndat, dtype=jnp.int32),
                jnp.ones(ndat, dtype=jnp.int32),
            ]
        )
        _, in_range = interp_family_multi_branch(1.0, masses, values, bids)
        assert bool(in_range[0])
        assert bool(in_range[1])
        assert int(jnp.sum(in_range)) == 2

    def test_returns_n_max_branches_elements(self):
        """Output arrays always have shape (N_MAX_BRANCHES,)."""
        masses, values, bids = self._make_single_branch()
        result, in_range = interp_family_multi_branch(1.0, masses, values, bids)
        assert result.shape == (N_MAX_BRANCHES,)
        assert in_range.shape == (N_MAX_BRANCHES,)


class TestBranchIdsAssignment:
    """Tests for branch_ids produced by _create_family_data (via construct_family)."""

    @pytest.fixture
    def single_branch_family(self):
        """A simple GR TOV family for a well-behaved metamodel EOS."""
        import jax

        jax.config.update("jax_enable_x64", True)
        from jesterTOV.eos.metamodel import MetaModel_EOS_model
        from jesterTOV.tov.gr import GRTOVSolver

        # nmax_nsat=2.0 keeps the EOS sub-luminal (cs2 <= 1) throughout
        eos = MetaModel_EOS_model(crust_name="DH", nmax_nsat=2.0)
        # Use known-valid NEP parameters (causal EOS)
        params = {
            "E_sat": -16.0,
            "K_sat": 220.0,
            "Q_sat": 0.0,
            "Z_sat": 0.0,
            "E_sym": 32.0,
            "L_sym": 90.0,
            "K_sym": 0.0,
            "Q_sym": 0.0,
            "Z_sym": 0.0,
        }
        eos_data = eos.construct_eos(params)
        solver = GRTOVSolver()
        return solver.construct_family(eos_data, ndat=50, min_nsat=0.75, tov_params={})

    def test_branch_ids_only_stable_and_sentinel(self, single_branch_family):
        """All branch_ids are either 0 (stable) or N_MAX_BRANCHES (sentinel)."""
        bids = single_branch_family.branch_ids
        valid = (bids == 0) | (bids == N_MAX_BRANCHES)
        assert bool(jnp.all(valid))

    def test_single_branch_has_stable_points(self, single_branch_family):
        """At least some points are on branch 0 for a normal EOS."""
        bids = single_branch_family.branch_ids
        assert int(jnp.sum(bids == 0)) > 0

    def test_masses_monotone_within_branch(self, single_branch_family):
        """Masses within branch 0 are strictly increasing (pc-sorted guarantee)."""
        family = single_branch_family
        mask = family.branch_ids == 0
        stable_masses = family.masses[mask]
        assert bool(jnp.all(jnp.diff(stable_masses) > 0))

    def test_pcs_sorted(self, single_branch_family):
        """log10pcs must be monotone increasing (pc-sorted storage)."""
        pcs = single_branch_family.log10pcs
        assert bool(jnp.all(jnp.diff(pcs) > 0))
