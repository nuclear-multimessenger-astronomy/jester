"""JAX-native Bilby adaptive DE kernel for unit hypercube sampling.

Copied and modified from:
https://github.com/mrosep/blackjax_ns_gw/blob/main/src/custom_kernels/acceptance_walk.py
"""

from typing import Callable, NamedTuple, Dict, Any
from functools import partial

import jax
import jax.numpy as jnp
from jax import flatten_util

from blackjax.base import SamplingAlgorithm
from blackjax.ns.base import PartitionedState, NSState, NSInfo, init as base_init, delete_fn as default_delete_fn
from blackjax.ns.adaptive import build_kernel as build_adaptive_kernel
from blackjax.types import ArrayTree, ArrayLikeTree


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
    total_proposals: jax.Array  # Total DE proposals made (including failed prior checks)


class DEKernelParams(NamedTuple):
    """Static pytree for DE kernel parameters."""
    live_points: ArrayLikeTree | ArrayTree  # Can be pytree of arrays
    loglikelihoods: jax.Array  # Log-likelihoods of all live points
    mix: float
    scale: float | jax.Array  # Can be array for JAX operations
    num_walks: jax.Array
    walks_float: jax.Array
    n_accept_total: jax.Array
    n_likelihood_evals_total: jax.Array  # Total likelihood evaluations (bilby-style counting)


def de_rwalk_one_step_unit_cube(
    rng_key: jax.Array,
    state: PartitionedState,
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
        
        point_a = jax.tree_util.tree_map(lambda x: x[top_indices[pos_a]], params.live_points)
        point_b = jax.tree_util.tree_map(lambda x: x[top_indices[pos_b]], params.live_points)
        delta = jax.tree_util.tree_map(lambda a, b: a - b, point_a, point_b)
        
        is_small_step = jax.random.uniform(key_mix) < params.mix
        gamma = jnp.where(is_small_step, 
                         params.scale * jax.random.gamma(key_gamma, 4.0) * 0.25, 
                         1.0)
        
        new_pos = stepper_fn(state.position, delta, gamma)
        new_logp = logprior_fn(new_pos)
        new_is_valid = jnp.isfinite(new_logp)
        
        return (new_is_valid, new_key, new_pos, new_logp, count + 1)
    
    def cond_fun(carry):
        is_valid, _, _, _, count = carry
        return jnp.logical_and(jnp.logical_not(is_valid), count < max_proposals)
    
    # Run while loop
    init = (False, rng_key, state.position, state.logprior, jnp.array(0))
    is_valid, _, pos_prop, logp_prop, n_proposals = jax.lax.while_loop(cond_fun, body_fun, init)
    
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
    final_logp = jnp.where(is_accepted, logp_prop, state.logprior)
    final_logl = jnp.where(is_accepted, logl_prop, state.loglikelihood)

    # Ensure final_logl is an Array (pyright inference issue workaround)
    final_logl_array = jnp.asarray(final_logl)
    new_state = PartitionedState(final_pos, final_logp, final_logl_array)
    # Ensure is_valid is a JAX array before calling astype
    likelihood_evals = jnp.asarray(is_valid, dtype=jnp.bool_).astype(jnp.int32)
    info = DEInfo(is_accepted=is_accepted, evals=n_proposals, likelihood_evals=likelihood_evals)
    
    return new_state, info


def de_rwalk_dynamic_unit_cube(
    rng_key: jax.Array,
    state: PartitionedState,
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
        key, current_state, n_accept, n_likelihood_evals, total_proposals, walks_completed = carry
        
        step_key, next_key = jax.random.split(key)
        new_state, info = single_step_fn(step_key, current_state, loglikelihood_0)
        
        # Update the carry state for the next iteration
        return (
            next_key, 
            new_state, 
            n_accept + info.is_accepted, 
            n_likelihood_evals + info.likelihood_evals,
            total_proposals + info.evals,  # Add ALL proposals made (including failed prior checks)
            walks_completed + 1
        )

    # Initialize and run the loop
    init_val = (
        rng_key, 
        state, 
        jnp.array(0, dtype=jnp.int32), 
        jnp.array(0, dtype=jnp.int32),
        jnp.array(0, dtype=jnp.int32), 
        jnp.array(0, dtype=jnp.int32)
    )
    
    _final_key, final_state, final_n_accept, final_n_likelihood_evals, final_total_proposals, final_walks_completed = jax.lax.while_loop(
        cond_fun, body_fun, init_val
    )

    info = DEWalkInfo(
        n_accept=final_n_accept, 
        walks_completed=final_walks_completed, 
        n_likelihood_evals=final_n_likelihood_evals,
        total_proposals=final_total_proposals
    )
    return final_state, info


def update_bilby_walks_fn(
    ns_state: NSState,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    n_target: int,
    max_mcmc: int,
    n_delete: int,
) -> DEKernelParams:
    """Bilby batch-level adaptation for unit cube sampling."""
    # Type annotation to help pyright understand the type
    prev_params: DEKernelParams = ns_state.inner_kernel_params  # type: ignore[assignment]
    
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
    param_n_likelihood_evals_total = prev_params.n_likelihood_evals_total.astype(jnp.int32)

    # --- 3. Use jnp.where for branchless, type-safe selection ---
    # This replaces lax.cond and is robust to type differences since we
    # have already ensured the types of both branches are identical.
    walks_float = jnp.where(is_uninitialized, default_walks_float, param_walks_float)
    n_accept_total = jnp.where(is_uninitialized, default_n_accept_total, param_n_accept_total)
    current_walks = jnp.where(is_uninitialized, default_current_walks, param_current_walks)
    n_likelihood_evals_total = jnp.where(is_uninitialized, default_n_likelihood_evals_total, param_n_likelihood_evals_total)
    
    # ===================== FIXED SECTION END =====================

    leaves = jax.tree_util.tree_leaves(ns_state.particles)
    nlive = leaves[0].shape[0]
    og_delay = nlive // 10 - 1
    delay = jnp.maximum(og_delay // n_delete, 1)
    
    # Keep bilby's walk length tuning formula (uses total walks, not likelihood evals)
    avg_accept_per_particle = n_accept_total / n_delete
    accept_prob = jnp.maximum(0.5, avg_accept_per_particle) / jnp.maximum(1.0, current_walks)
    
    new_walks_float = (walks_float * delay + n_target / accept_prob) / (delay + 1)
    new_walks_float = jnp.where(n_accept_total == 0, walks_float, new_walks_float)

    num_walks_int = jnp.minimum(jnp.ceil(new_walks_float).astype(jnp.int32), max_mcmc)

    example_particle = jax.tree_util.tree_map(lambda x: x[0], ns_state.particles)
    flat_particle, _ = flatten_util.ravel_pytree(example_particle)
    n_dim = flat_particle.shape[0]
    
    return DEKernelParams(
        live_points=ns_state.particles,
        loglikelihoods=ns_state.loglikelihood,
        mix=0.5,
        scale=2.38 / jnp.sqrt(2 * n_dim),
        num_walks=jnp.array(num_walks_int, dtype=jnp.int32),
        walks_float=jnp.array(new_walks_float, dtype=jnp.float32),
        n_accept_total=jnp.array(0, dtype=jnp.int32),
        n_likelihood_evals_total=jnp.array(0, dtype=jnp.int32),
    )


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

    def update_fn(ns_state: NSState, ns_info: NSInfo, params: Dict[str, ArrayTree]) -> DEKernelParams:
        """Update function compatible with blackjax adaptive kernel interface."""
        return update_bilby_walks_fn(
            ns_state=ns_state,
            logprior_fn=logprior_fn,
            loglikelihood_fn=loglikelihood_fn,
            n_target=n_target,
            max_mcmc=max_mcmc,
            n_delete=num_delete,
        )

    kernel_with_stepper = partial(
        de_rwalk_dynamic_unit_cube, 
        stepper_fn=stepper_fn,
        num_survivors=num_survivors,
        max_proposals=max_proposals,
        max_mcmc=max_mcmc,
        )

    # BlackJAX API compatibility: DEKernelParams (NamedTuple) works as pytree but type checker
    # expects Dict[str, ArrayTree]. This is safe since NamedTuple is a valid pytree.
    base_kernel_step = build_adaptive_kernel(
        logprior_fn,
        loglikelihood_fn,
        delete_fn,
        jax.vmap(kernel_with_stepper, in_axes=(0, 0, None, None, None, None)),
        update_fn,  # type: ignore[arg-type]
    )

    def init_fn(particles):
        state = base_init(
            particles=particles,
            logprior_fn=jax.vmap(logprior_fn),
            loglikelihood_fn=jax.vmap(loglikelihood_fn),
        )
        
        # Calculate proper scale from particle dimensionality
        example_particle = jax.tree_util.tree_map(lambda x: x[0], particles)
        flat_particle, _ = flatten_util.ravel_pytree(example_particle)
        n_dim = flat_particle.shape[0]
        scale = 2.38 / jnp.sqrt(2 * n_dim)
        
        # Create initial DEKernelParams with sentinel value
        initial_de_params = DEKernelParams(
            live_points=particles,
            loglikelihoods=state.loglikelihood,
            mix=0.5,
            scale=scale,
            num_walks=jnp.array(100, dtype=jnp.int32),
            walks_float=jnp.array(100.0, dtype=jnp.float32),
            n_accept_total=jnp.array(-1, dtype=jnp.int32),  # Sentinel flag
            n_likelihood_evals_total=jnp.array(-1, dtype=jnp.int32),  # Sentinel flag
        )
        
        # Set our sentinel state manually
        return state._replace(inner_kernel_params=initial_de_params)
    
    def step_fn(rng_key, state: NSState):
        new_state, info = base_kernel_step(rng_key, state)

        inner_info = info.inner_kernel_info
        batch_n_accept = jnp.sum(inner_info.n_accept)
        batch_n_likelihood_evals = jnp.sum(inner_info.n_likelihood_evals)

        updated_params = new_state.inner_kernel_params._replace(
            n_accept_total=batch_n_accept,
            n_likelihood_evals_total=batch_n_likelihood_evals
        )
        
        final_state = new_state._replace(inner_kernel_params=updated_params)
        return final_state, info

    # BlackJAX API compatibility: Our functions work correctly but type signatures
    # don't match generic SamplingAlgorithm interface exactly. This is safe.
    return SamplingAlgorithm(init_fn, step_fn)  # type: ignore[arg-type]
