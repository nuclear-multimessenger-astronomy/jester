import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def UniformPopulation(key: jax.random.PRNGKey, params: dict[str, Float], size: int) -> Array:
    """
    BNS population model where both masses are simply drawn uniformly between an interval.

    Parameters
    ----------
    key: PRNGKey
        Random sampling key.
    params : dict[str, Float]
        Input parameter dictionary containing
        m_min: Lowest NS mass
        m_max: Highest NS Mass
    size: int
        Number of samples to generate.
    Returns
    -------
    Array[size, 2]
        mass_1, mass_2 in source frame
    """

    subkeys = jax.random.split(key, num=3)

    m_min = params["m_min"]
    m_max = params["m_max"]

    mass_1_source = jax.random.uniform(subkeys[1], shape=(size,), minval=m_min, maxval=m_max)
    mass_2_source  = jax.random.uniform(subkeys[2], shape=(size,), minval=m_min, maxval=m_max)

    mass_1_source, mass_2_source = jnp.maximum(mass_1_source, mass_2_source), jnp.minimum(mass_1_source, mass_2_source)

    return jnp.stack([mass_1_source, mass_2_source])


def MassRatioPowerLaw(key: jax.random.PRNGKey, params: dict[str, Float], size: int) -> Array:
    """
    BNS population model where the masses are in a limited interval and the mass ratio follows a power law.
    Based on Landry et al. https://arxiv.org/abs/2107.04559 who analyze GW170817 and GW190425.

    Parameters
    ----------
    key: PRNGKey
        Random sampling key.
    params : dict[str, Float]
        Input parameter dictionary containing 
            m_min: Lowest NS mass (GW170817+GW190425: 1.1)
            m_max: Highest NS mass (GW170817+GW190424: 2.0)
            alpha: Power law index for mass ratio (GW170817+GW190425: 2.0)
    size: int
        Number of samples to generate.
    Returns
    -------
    Array[size, 2]
        mass_1, mass_2 in source frame

    """
    subkeys = jax.random.split(key, num=3)

    m_min = params["m_min"]
    m_max = params["m_max"]
    alpha = params["alpha"]

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
    """
    BNS population model where the recycled pulsar is drawn from a double Gaussian
    and the slow pulsar from a uniform distribution. 
    The recycled and slow mass are then sorted to give back m1, m2. 
    Based on Farrow et al. https://arxiv.org/abs/1902.03300.
    We list their best fit parameters here from the galactic DNS distribution.

    Parameters
    ----------
    key: PRNGKey
        Random sampling key.
    params : dict[str, Float]
        Input parameter dictionary containing 
            mu_1: First gaussian peak (galactic DNS: 1.34)
            sigma_1: First gaussian width (galactic DNS: 1.47)
            mu_2: Second gaussian peak (galactic DNS: 0.02)
            sigma_2: Second gaussian width (galactic DNS: 0.15)
            alpha: Relative proportions of the Gaussians (galactic DNS: 0.68)
            m_min: Minimum mass of the slow mass (galactic DNS: 1.16)
            m_max: Maximum mass of the slow mass (galactic DNS: 1.42)
    size: int
        Number of samples to generate.

    Returns
    -------
    Array[size, 2]
        mass_1, mass_2 in source frame

    """

    subkeys = jax.random.split(key, num=4)

    alpha = params["alpha"]
    mu_1 = params["mu_1"]
    mu_2 = params["mu_2"]
    sigma_1 = params["sigma_1"]
    sigma_2 = params["sigma_2"]

    m_min = params["m_min"]
    m_max = params["m_max"]


    ms_samples = jax.random.uniform(subkeys[1], shape=(size,), minval=m_min, maxval=m_max)

    comp = jax.random.bernoulli(subkeys[2], shape=(size,), p=alpha)
    z = jax.random.normal(subkeys[3], shape=(size,))
    mr_samples = jnp.where(comp, mu_1 + sigma_1 * z, mu_2 + sigma_2* z)
    
    mass_1_source, mass_2_source = jnp.maximum(mr_samples, ms_samples), jnp.minimum(mr_samples, ms_samples)

    return jnp.stack([mass_1_source, mass_2_source])