"""JAX-native Bilby adaptive DE kernel for unit hypercube sampling.

Copied and modified from:
https://github.com/mrosep/blackjax_ns_gw/blob/main/src/custom_kernels/acceptance_walk.py
"""

from typing import Callable, NamedTuple, Dict
from functools import partial

import jax
import jax.numpy as jnp
from jax import flatten_util

from blackjax.base import SamplingAlgorithm
from blackjax.ns.base import (
    NSState,
    NSInfo,
    StateWithLogLikelihood,
    init_state_strategy,
    delete_fn as default_delete_fn,
)
from blackjax.ns.adaptive import (
    AdaptiveNSState,
    build_kernel as build_adaptive_kernel,
    init as adaptive_init,
)
from blackjax.types import Array, ArrayTree, ArrayLikeTree


class PartitionedState(NamedTuple):
    """Simplified state for DE kernel without loglikelihood_birth tracking."""

    position: ArrayLikeTree
    logprior: Array
    loglikelihood: Array


class DEInfo(NamedTuple):
    """Diagnostic information for a single DE MCMC step."""

    is_accepted: jax.Array  # Scalar boolean array
    evals: jax.Array  # Scalar int array
    likelihood_evals: jax.Array  # Count only in-bounds proposals (like bilby)


class DEWalkInfo(NamedTuple):
    """Diagnostic information for a full DE MCMC walk."""

    n_accept: jax.Array  # Scalar int array
    walks_completed: jax.Array  # Actual number of walks completed (was n_steps)
    n_likelihood_evals: jax.Array  # Total likelihood evaluations (in-bounds only)
    total_proposals: (
        jax.Array
    )  # Total DE proposals made (including failed prior checks)


class DEKernelParams(NamedTuple):
    """Static pytree for DE kernel parameters."""

    live_points: ArrayLikeTree | ArrayTree  # Can be pytree of arrays
    loglikelihoods: jax.Array  # Log-likelihoods of all live points
    mix: float
    scale: float | jax.Array  # Can be array for JAX operations
    num_walks: jax.Array
    walks_float: jax.Array
    n_accept_total: jax.Array
    n_likelihood_evals_total: (
        jax.Array
    )  # Total likelihood evaluations (bilby-style counting)


def de_rwalk_one_step_unit_cube(
    rng_key: jax.Array,
    state: StateWithLogLikelihood,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    loglikelihood_0: float,
    params: DEKernelParams,
    stepper_fn: Callable,
    num_survivors: int,
    max_proposals: int = 1000,
):
    """Single DE step in unit hypercube space with prior bounds checking first."""

    # While loop to find valid prior point
    def body_fun(carry):
        is_valid, key, pos, logp, count = carry
        key_a, key_b, key_mix, key_gamma, new_key = jax.random.split(key, 5)

        # DE proposal
        _, top_indices = jax.lax.top_k(params.loglikelihoods, num_survivors)
        pos_a = jax.random.randint(key_a, (), 0, num_survivors)
        pos_b_raw = jax.random.randint(key_b, (), 0, num_survivors - 1)
        pos_b = jnp.where(pos_b_raw >= pos_a, pos_b_raw + 1, pos_b_raw)

        point_a = jax.tree_util.tree_map(
            lambda x: x[top_indices[pos_a]], params.live_points
        )
        point_b = jax.tree_util.tree_map(
            lambda x: x[top_indices[pos_b]], params.live_points
        )
        delta = jax.tree_util.tree_map(lambda a, b: a - b, point_a, point_b)

        is_small_step = jax.random.uniform(key_mix) < params.mix
        gamma = jnp.where(
            is_small_step, params.scale * jax.random.gamma(key_gamma, 4.0) * 0.25, 1.0
        )

        new_pos = stepper_fn(state.position, delta, gamma)
        new_logp = logprior_fn(new_pos)
        new_is_valid = jnp.isfinite(new_logp)

        return (new_is_valid, new_key, new_pos, new_logp, count + 1)

    def cond_fun(carry):
        is_valid, _, _, _, count = carry
        return jnp.logical_and(jnp.logical_not(is_valid), count < max_proposals)

    # Run while loop
    init = (False, rng_key, state.position, state.logdensity, jnp.array(0))
    is_valid, _, pos_prop, logp_prop, n_proposals = jax.lax.while_loop(
        cond_fun, body_fun, init
    )

    # Check prior constraint one more time (cheap)
    # logp_final = logprior_fn(pos_prop)
    # is_in_bounds = jnp.isfinite(logp_final)

    # Always evaluate likelihood for final point
    logl_prop = loglikelihood_fn(pos_prop)
    is_accepted = jnp.logical_and(is_valid, logl_prop > loglikelihood_0)
    # is_above_threshold = logl_prop > loglikelihood_0

    # # Accept only if both constraints satisfied
    # is_accepted = jnp.logical_and(is_in_bounds, is_above_threshold)

    # Update state
    final_pos = jax.tree_util.tree_map(
        lambda p, c: jnp.where(is_accepted, p, c), pos_prop, state.position
    )
    final_logp = jnp.where(is_accepted, logp_prop, state.logdensity)
    final_logl = jnp.where(is_accepted, logl_prop, state.loglikelihood)

    # Ensure final_logl is an Array (pyright inference issue workaround)
    final_logl_array = jnp.asarray(final_logl)
    # Create state with same structure as input (StateWithLogLikelihood)
    new_state = StateWithLogLikelihood(
        position=final_pos,
        logdensity=final_logp,
        loglikelihood=final_logl_array,
        loglikelihood_birth=state.loglikelihood_birth,  # Keep birth threshold
    )
    # Ensure is_valid is a JAX array before calling astype
    likelihood_evals = jnp.asarray(is_valid, dtype=jnp.bool_).astype(jnp.int32)
    info = DEInfo(
        is_accepted=is_accepted, evals=n_proposals, likelihood_evals=likelihood_evals
    )

    return new_state, info


def de_rwalk_dynamic_unit_cube(
    rng_key: jax.Array,
    state: StateWithLogLikelihood,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    loglikelihood_0: float,
    params: DEKernelParams,
    stepper_fn: Callable,
    num_survivors: int,
    max_proposals: int = 1000,
    max_mcmc: int = 5000,
):
    """MCMC walk in unit cube, capped by total proposal budget."""
    # Use partial to create a new function with num_survivors "baked in"
    one_step_with_static_k = partial(
        de_rwalk_one_step_unit_cube,
        num_survivors=num_survivors,
        max_proposals=max_proposals,
    )

    def single_step_fn(rng_key, state, loglikelihood_0):
        return one_step_with_static_k(
            rng_key=rng_key,
            state=state,
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            loglikelihood_0=loglikelihood_0,
            params=params,
            stepper_fn=stepper_fn,
        )

    def cond_fun(carry):
        """Continue looping if within walk limits AND under proposal budget."""
        _, _, _, _, total_proposals, walks_completed = carry
        within_walks_limit = walks_completed < params.num_walks
        under_budget = total_proposals < max_mcmc
        return jnp.logical_and(within_walks_limit, under_budget)

    def body_fun(carry):
        """Perform one MCMC step and update accumulators."""
        (
            key,
            current_state,
            n_accept,
            n_likelihood_evals,
            total_proposals,
            walks_completed,
        ) = carry

        step_key, next_key = jax.random.split(key)
        new_state, info = single_step_fn(step_key, current_state, loglikelihood_0)

        # Update the carry state for the next iteration
        return (
            next_key,
            new_state,
            n_accept + info.is_accepted,
            n_likelihood_evals + info.likelihood_evals,
            total_proposals
            + info.evals,  # Add ALL proposals made (including failed prior checks)
            walks_completed + 1,
        )

    # Initialize and run the loop
    init_val = (
        rng_key,
        state,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32),
    )

    (
        _final_key,
        final_state,
        final_n_accept,
        final_n_likelihood_evals,
        final_total_proposals,
        final_walks_completed,
    ) = jax.lax.while_loop(cond_fun, body_fun, init_val)

    info = DEWalkInfo(
        n_accept=final_n_accept,
        walks_completed=final_walks_completed,
        n_likelihood_evals=final_n_likelihood_evals,
        total_proposals=final_total_proposals,
    )
    return final_state, info


def update_bilby_walks_fn(
    ns_state: AdaptiveNSState,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    n_target: int,
    max_mcmc: int,
    n_delete: int,
) -> Dict[str, ArrayTree]:
    """Bilby batch-level adaptation for unit cube sampling.

    Returns a dict with keys 'logprior_fn', 'loglikelihood_fn', 'params'
    for compatibility with blackjax build_adaptive_kernel which unpacks
    inner_kernel_params as **kwargs.
    """
    # inner_kernel_params is a dict with 'params' key containing DEKernelParams
    inner_params_dict: Dict[str, ArrayTree] = ns_state.inner_kernel_params  # type: ignore[assignment]
    prev_params: DEKernelParams = inner_params_dict["params"]  # type: ignore[assignment]

    # Check sentinel value instead of None (JAX-compatible)
    is_uninitialized = prev_params.n_accept_total < 0

    # ==================== FIXED SECTION START ====================

    # --- 1. Define default values with explicit dtypes ---
    # These are the values to use on the first run (initialization).
    default_walks_float = jnp.array(100.0, dtype=jnp.float32)
    default_n_accept_total = jnp.array(0, dtype=jnp.int32)
    default_current_walks = jnp.array(100, dtype=jnp.int32)
    default_n_likelihood_evals_total = jnp.array(0, dtype=jnp.int32)

    # --- 2. Get values from previous state and explicitly cast to ensure type match ---
    # These are the values from the previous step. We cast them to the same
    # dtypes as the defaults to prevent any mismatch.
    param_walks_float = prev_params.walks_float.astype(jnp.float32)
    param_n_accept_total = prev_params.n_accept_total.astype(jnp.int32)
    param_current_walks = prev_params.num_walks.astype(jnp.int32)
    param_n_likelihood_evals_total = prev_params.n_likelihood_evals_total.astype(
        jnp.int32
    )

    # --- 3. Use jnp.where for branchless, type-safe selection ---
    # This replaces lax.cond and is robust to type differences since we
    # have already ensured the types of both branches are identical.
    walks_float = jnp.where(is_uninitialized, default_walks_float, param_walks_float)
    n_accept_total = jnp.where(
        is_uninitialized, default_n_accept_total, param_n_accept_total
    )
    current_walks = jnp.where(
        is_uninitialized, default_current_walks, param_current_walks
    )
    n_likelihood_evals_total = jnp.where(
        is_uninitialized,
        default_n_likelihood_evals_total,
        param_n_likelihood_evals_total,
    )

    # ===================== FIXED SECTION END =====================

    leaves = jax.tree_util.tree_leaves(ns_state.particles)
    nlive = leaves[0].shape[0]
    og_delay = nlive // 10 - 1
    delay = jnp.maximum(og_delay // n_delete, 1)

    # Keep bilby's walk length tuning formula (uses total walks, not likelihood evals)
    avg_accept_per_particle = n_accept_total / n_delete
    accept_prob = jnp.maximum(0.5, avg_accept_per_particle) / jnp.maximum(
        1.0, current_walks
    )

    new_walks_float = (walks_float * delay + n_target / accept_prob) / (delay + 1)
    new_walks_float = jnp.where(n_accept_total == 0, walks_float, new_walks_float)

    num_walks_int = jnp.minimum(jnp.ceil(new_walks_float).astype(jnp.int32), max_mcmc)

    # Calculate ndim from position dict (not full state)
    example_particle = jax.tree_util.tree_map(lambda x: x[0], ns_state.particles.position)
    flat_particle, _ = flatten_util.ravel_pytree(example_particle)
    n_dim = flat_particle.shape[0]

    new_de_params = DEKernelParams(
        live_points=ns_state.particles.position,  # Just the position dict, not full state
        loglikelihoods=ns_state.particles.loglikelihood,
        mix=0.5,
        scale=2.38 / jnp.sqrt(2 * n_dim),
        num_walks=jnp.array(num_walks_int, dtype=jnp.int32),
        walks_float=jnp.array(new_walks_float, dtype=jnp.float32),
        n_accept_total=jnp.array(0, dtype=jnp.int32),
        n_likelihood_evals_total=jnp.array(0, dtype=jnp.int32),
    )

    # Return dict with only array-like data (no callables!)
    # logprior_fn and loglikelihood_fn are baked into the kernel via partial
    return {"params": new_de_params}


def bilby_adaptive_de_sampler_unit_cube(
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    nlive: int,
    n_target: int = 60,
    max_mcmc: int = 5000,
    num_delete: int = 1,
    stepper_fn: Callable | None = None,
    max_proposals: int = 1000,
) -> SamplingAlgorithm:
    """Bilby adaptive DE sampler for unit hypercube."""
    if stepper_fn is None:
        raise ValueError("stepper_fn must be provided for unit cube sampling")

    # Calculate num_survivors statically as a Python integer
    num_survivors = nlive - num_delete

    delete_fn = partial(default_delete_fn, num_delete=num_delete)

    def update_fn(
        rng_key, ns_state: NSState, ns_info: NSInfo, params: Dict[str, ArrayTree]
    ) -> Dict[str, ArrayTree]:
        """Update function compatible with blackjax adaptive kernel interface.

        Note: At runtime ns_state is AdaptiveNSState (duck typed as NSState).
        Returns dict with keys 'logprior_fn', 'loglikelihood_fn', 'params'.
        """
        # Type cast: blackjax passes AdaptiveNSState through NSState-typed interface
        adaptive_state: AdaptiveNSState = ns_state  # type: ignore[assignment]
        return update_bilby_walks_fn(
            ns_state=adaptive_state,
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            n_target=n_target,
            max_mcmc=max_mcmc,
            n_delete=num_delete,
        )

    # Bake logprior_fn and loglikelihood_fn into the kernel via partial
    # These are callables that cannot be stored in JAX-traced pytrees
    kernel_with_stepper = partial(
        de_rwalk_dynamic_unit_cube,
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        stepper_fn=stepper_fn,
        num_survivors=num_survivors,
        max_proposals=max_proposals,
        max_mcmc=max_mcmc,
    )

    # Wrapper to match blackjax API: inner_kernel(rng_keys, states, loglikelihood_0, **kwargs)
    # blackjax does partial(inner_kernel, **inner_kernel_params) then calls with 3 positional args
    # Since vmap applies to ALL args (including kwargs!), we need to capture params in closure
    # after partial, so it's not subject to vmap
    def make_vmapped_kernel(params):
        """Create a vmapped kernel with params captured in closure."""

        def inner_kernel_closure(rng_key, state, loglikelihood_0):
            # Pass all remaining args by keyword to avoid positional conflicts
            # partial() already bound: logprior_fn, loglikelihood_fn, stepper_fn,
            # num_survivors, max_proposals, max_mcmc
            return kernel_with_stepper(
                rng_key=rng_key,
                state=state,
                loglikelihood_0=loglikelihood_0,
                params=params,
            )

        return jax.vmap(inner_kernel_closure, in_axes=(0, 0, None))

    # The inner_kernel passed to build_adaptive_kernel should accept params as kwarg
    # blackjax will do: partial(inner_kernel, params=de_params)
    # Then call: partial_kernel(rng_keys, states, loglikelihood_0)
    # But we need to avoid vmapping params, so we use a different approach:
    # Return a function that extracts params from kwargs and creates the vmapped kernel on-the-fly
    def inner_kernel_wrapper(rng_keys, states, loglikelihood_0, *, params):
        vmapped_kernel = make_vmapped_kernel(params)
        return vmapped_kernel(rng_keys, states, loglikelihood_0)

    base_kernel_step = build_adaptive_kernel(
        delete_fn,
        inner_kernel_wrapper,
        update_fn,  # type: ignore[arg-type]
    )

    def init_fn(particles, rng_key=None):
        # Create init_state_fn that captures logprior_fn and loglikelihood_fn
        def _init_state_fn(positions):
            return init_state_strategy(
                positions, jax.vmap(logprior_fn), jax.vmap(loglikelihood_fn)
            )

        # Calculate proper scale from particle dimensionality
        example_particle = jax.tree_util.tree_map(lambda x: x[0], particles)
        flat_particle, _ = flatten_util.ravel_pytree(example_particle)
        n_dim = flat_particle.shape[0]
        scale = 2.38 / jnp.sqrt(2 * n_dim)

        # Create function to initialize inner kernel params
        # This is called by adaptive_init to set up initial inner kernel kwargs
        # build_adaptive_kernel unpacks these as: partial(inner_kernel, **inner_kernel_params)
        # Note: logprior_fn and loglikelihood_fn are baked into kernel_with_stepper
        # via partial, since callables can't be stored in JAX-traced pytrees
        def _init_inner_kernel_params_fn(key, base_state, info, params):
            de_params = DEKernelParams(
                live_points=particles,
                loglikelihoods=base_state.particles.loglikelihood,
                mix=0.5,
                scale=scale,
                num_walks=jnp.array(100, dtype=jnp.int32),
                walks_float=jnp.array(100.0, dtype=jnp.float32),
                n_accept_total=jnp.array(-1, dtype=jnp.int32),  # Sentinel flag
                n_likelihood_evals_total=jnp.array(-1, dtype=jnp.int32),  # Sentinel flag
            )
            # Return dict with only array-like data (no callables!)
            return {"params": de_params}

        # Use adaptive_init which returns AdaptiveNSState with inner_kernel_params
        state = adaptive_init(
            positions=particles,
            init_state_fn=_init_state_fn,
            update_inner_kernel_params_fn=_init_inner_kernel_params_fn,
            rng_key=rng_key,
        )

        return state

    def step_fn(rng_key, state: AdaptiveNSState):
        new_state, info = base_kernel_step(rng_key, state)

        # NSInfo has update_info (not inner_kernel_info) - this contains DEWalkInfo
        inner_info = info.update_info
        batch_n_accept = jnp.sum(inner_info.n_accept)
        batch_n_likelihood_evals = jnp.sum(inner_info.n_likelihood_evals)

        # inner_kernel_params is a dict with only 'params' key containing DEKernelParams
        # (callables are baked into the kernel via partial, not stored in state)
        inner_params_dict: Dict[str, ArrayTree] = new_state.inner_kernel_params  # type: ignore[assignment]
        de_params: DEKernelParams = inner_params_dict["params"]  # type: ignore[assignment]

        updated_de_params = de_params._replace(
            n_accept_total=batch_n_accept,
            n_likelihood_evals_total=batch_n_likelihood_evals,
        )

        # Create updated dict with the new DEKernelParams
        updated_inner_params = {"params": updated_de_params}

        final_state = new_state._replace(inner_kernel_params=updated_inner_params)
        return final_state, info

    # BlackJAX API compatibility: Our functions work correctly but type signatures
    # don't match generic SamplingAlgorithm interface exactly. This is safe.
    return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]
