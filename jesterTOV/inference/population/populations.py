import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def UniformPopulation(key: jax.random.PRNGKey, params: dict[str, Float], size: int) -> Array:

    subkeys = jax.random.split(key, num=3)

    m_min = params.get("m_min", 1.0)
    m_max = params.get("m_max", 2.5)

    mass_1_source = jax.random.uniform(subkeys[1], shape=(size,), minval=m_min, maxval=m_max)
    mass_2_source  = jax.random.uniform(subkeys[2], shape=(size,), minval=m_min, maxval=m_max)

    mass_1_source, mass_2_source = jnp.maximum(mass_1_source, mass_2_source), jnp.minimum(mass_1_source, mass_2_source)

    return jnp.stack([mass_1_source, mass_2_source])


def MassRatioPowerLaw(key: jax.random.PRNGKey, params: dict[str, Float], size: int) -> Array:

    subkeys = jax.random.split(key, num=3)

    m_min = params.get("m_min", 1.1)
    m_max = params.get("m_max", 2.0)
    alpha = params.get("alpha", 2.0)

    m1_arr = jnp.linspace(m_min, m_max, 200)
    cdf1_arr = jnp.where(alpha!=1, 
                         m1_arr**2 / 2 - m_min**2 / 2 - m_min**(alpha+1) * m1_arr**(1-alpha) / (1-alpha) + m_min**2 / (1-alpha), 
                         m1_arr**2 / 2 - m_min**2 / 2 - m_min**(alpha+1) * jnp.log(m1_arr / m_min))
    cdf1_arr /= cdf1_arr[-1]

    u1 = jax.random.uniform(subkeys[1], (size,))
    mass_1_source = jnp.interp(u1, cdf1_arr, m1_arr)

    u2 = jax.random.uniform(subkeys[2], (size,))
    mass_2_source = (u2*(mass_1_source**3 - m_min**3) + m_min**3)**(1/3)

    return jnp.stack([mass_1_source, mass_2_source])


def RecycledBinary(key: jax.random.PRNGKey, params: dict[str, Float], size: int) -> Array:

    subkeys = jax.random.split(key, num=4)

    alpha = params.get("alpha", 0.68)
    mu_1 = params.get("mu_1", 1.34)
    mu_2 = params.get("mu_2", 1.43)
    sigma_1 = params.get("sigma_1", 0.02)
    sigma_2 = params.get("sigma_2", 0.15)

    m_min = params.get("m_min", 1.16)
    m_max = params.get("m_max", 1.42)


    ms_samples = jax.random.uniform(subkeys[1], shape=(size,), minval=m_min, maxval=m_max)

    comp = jax.random.bernoulli(subkeys[2], shape=(size,), p=alpha)
    z = jax.random.normal(subkeys[3], shape=(size,))
    mr_samples = jnp.where(comp, mu_1 + sigma_1 * z, mu_2 + sigma_2* z)
    
    mass_1_source, mass_2_source = jnp.maximum(mr_samples, ms_samples), jnp.minimum(mr_samples, ms_samples)

    return jnp.stack([mass_1_source, mass_2_source])