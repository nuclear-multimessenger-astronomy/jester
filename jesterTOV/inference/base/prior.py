"""Prior base classes for JESTER inference system.

This module contains prior classes that were originally from Jim (jimgw v0.2.0).
They are copied here to remove the dependency on jimgw.

Note: These classes follow the Jim/jimgw architecture and are designed to work
with flowMC's Distribution base class.
"""

from dataclasses import field
from typing import Any

import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from flowMC.nfmodel.base import Distribution
from jaxtyping import Array, Float, PRNGKeyArray, jaxtyped

from .transform import BijectiveTransform, LogitTransform, ScaleTransform, OffsetTransform


class Prior(Distribution):
    """
    A thin wrapper built on top of flowMC distributions to do book keeping.

    Note: This class follows the Jim/jimgw architecture. Should not be used directly
    since it does not implement any of the real methods.

    The rationale behind this is to have a class that can be used to keep track of
    the names of the parameters and the transforms that are applied to them.
    """

    parameter_names: list[str]
    composite: bool = False

    @property
    def n_dim(self) -> int:
        return len(self.parameter_names)

    def __init__(self, parameter_names: list[str]) -> None:
        """
        Parameters
        ----------
        parameter_names : list[str]
            A list of names for the parameters of the prior.
        """
        self.parameter_names = parameter_names

    def add_name(self, x: Float[Array, " n_dim"]) -> dict[str, Float]:
        """
        Turn an array into a dictionary.

        Parameters
        ----------
        x : Array
            An array of parameters. Shape (n_dim,).

        Returns
        -------
        dict[str, Float]
            Dictionary mapping parameter names to values.
        """
        return dict(zip(self.parameter_names, x))

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from the prior distribution.

        Parameters
        ----------
        rng_key : PRNGKeyArray
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : dict[str, Float[Array, " n_samples"]]
            Samples from the distribution. The keys are the names of the parameters.
        """
        raise NotImplementedError

    def log_prob(self, z: dict[str, Float | Array]) -> Float:
        """
        Evaluate the log probability of the prior.

        Parameters
        ----------
        z : dict[str, Array]
            Dictionary of parameter names to values.

        Returns
        -------
        log_prob : Float
            The log probability.
        """
        raise NotImplementedError


@jaxtyped(typechecker=typechecker)
class LogisticDistribution(Prior):
    """
    Logistic distribution prior.

    Note: This class follows the Jim/jimgw architecture.
    """

    def __repr__(self) -> str:
        return f"LogisticDistribution(parameter_names={self.parameter_names})"

    def __init__(self, parameter_names: list[str], **kwargs: Any) -> None:
        super().__init__(parameter_names)
        self.composite = False
        assert self.n_dim == 1, "LogisticDistribution needs to be 1D distributions"

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from a logistic distribution.

        Parameters
        ----------
        rng_key : PRNGKeyArray
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : dict
            Samples from the distribution. The keys are the names of the parameters.
        """
        samples = jax.random.uniform(rng_key, (n_samples,), minval=0.0, maxval=1.0)
        samples = jnp.log(samples / (1 - samples))
        return self.add_name(samples[None])

    def log_prob(self, z: dict[str, Float]) -> Float:
        """
        Evaluate the log probability.

        Parameters
        ----------
        z : dict[str, Float]
            Dictionary of parameter names to values.

        Returns
        -------
        log_prob : Float
            The log probability.
        """
        variable = z[self.parameter_names[0]]
        return -variable - 2 * jnp.log(1 + jnp.exp(-variable))


class SequentialTransformPrior(Prior):
    """
    Transform a prior distribution by applying a sequence of transforms.

    Note: This class follows the Jim/jimgw architecture.

    The space before the transform is named as x,
    and the space after the transform is named as z.
    """

    base_prior: Prior
    transforms: list[BijectiveTransform]

    def __repr__(self) -> str:
        return f"Sequential(priors={self.base_prior}, parameter_names={self.parameter_names})"

    def __init__(
        self,
        base_prior: Prior,
        transforms: list[BijectiveTransform],
    ) -> None:
        self.base_prior = base_prior
        self.transforms = transforms
        self.parameter_names = base_prior.parameter_names
        for transform in transforms:
            self.parameter_names = transform.propagate_name(self.parameter_names)
        self.composite = True

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from the transformed prior.

        Parameters
        ----------
        rng_key : PRNGKeyArray
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : dict
            Samples from the transformed distribution.
        """
        output = self.base_prior.sample(rng_key, n_samples)
        return jax.vmap(self.transform)(output)

    def log_prob(self, z: dict[str, Float]) -> Float:
        """
        Evaluate the probability of the transformed variable z.

        This is what flowMC should sample from.

        Parameters
        ----------
        z : dict[str, Float]
            Dictionary of parameter names to values in transformed space.

        Returns
        -------
        log_prob : Float
            The log probability including Jacobian correction.
        """
        output = 0
        for transform in reversed(self.transforms):
            z, log_jacobian = transform.inverse(z)
            output += log_jacobian
        output += self.base_prior.log_prob(z)
        return output

    def transform(self, x: dict[str, Float]) -> dict[str, Float]:
        """
        Apply forward transforms to x.

        Parameters
        ----------
        x : dict[str, Float]
            Dictionary of parameter names to values.

        Returns
        -------
        z : dict[str, Float]
            Transformed dictionary.
        """
        for transform in self.transforms:
            x = transform.forward(x)
        return x


class CombinePrior(Prior):
    """
    A prior class constructed by joining multiple priors together to form a multivariate prior.

    Note: This class follows the Jim/jimgw architecture.

    This assumes the priors composing the Combine class are independent.
    """

    base_prior: list[Prior] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"Combine(priors={self.base_prior}, parameter_names={self.parameter_names})"
        )

    def __init__(
        self,
        priors: list[Prior],
    ) -> None:
        """
        Parameters
        ----------
        priors : list[Prior]
            List of independent prior distributions to combine.
        """
        parameter_names = []
        for prior in priors:
            parameter_names += prior.parameter_names
        self.base_prior = priors
        self.parameter_names = parameter_names
        self.composite = True

    def sample(
        self, rng_key: PRNGKeyArray, n_samples: int
    ) -> dict[str, Float[Array, " n_samples"]]:
        """
        Sample from the combined prior by sampling from each component.

        Parameters
        ----------
        rng_key : PRNGKeyArray
            A random key to use for sampling.
        n_samples : int
            The number of samples to draw.

        Returns
        -------
        samples : dict
            Combined samples from all priors.
        """
        output = {}
        for prior in self.base_prior:
            rng_key, subkey = jax.random.split(rng_key)
            output.update(prior.sample(subkey, n_samples))
        return output

    def log_prob(self, z: dict[str, Float]) -> Float:
        """
        Evaluate the log probability by summing over independent priors.

        Parameters
        ----------
        z : dict[str, Float]
            Dictionary of parameter names to values.

        Returns
        -------
        log_prob : Float
            The combined log probability.
        """
        output = 0.0
        for prior in self.base_prior:
            output += prior.log_prob(z)
        return output


@jaxtyped(typechecker=typechecker)
class UniformPrior(SequentialTransformPrior):
    """
    Uniform prior distribution over [xmin, xmax].

    Note: This class follows the Jim/jimgw architecture. It is implemented
    as a composition of a logistic base distribution with transforms.
    """

    xmin: float
    xmax: float

    def __repr__(self) -> str:
        return f"UniformPrior(xmin={self.xmin}, xmax={self.xmax}, parameter_names={self.parameter_names})"

    def __init__(
        self,
        xmin: float,
        xmax: float,
        parameter_names: list[str],
    ) -> None:
        """
        Parameters
        ----------
        xmin : float
            Lower bound of the uniform distribution.
        xmax : float
            Upper bound of the uniform distribution.
        parameter_names : list[str]
            Names of the parameters (must be 1D).
        """
        self.parameter_names = parameter_names
        assert self.n_dim == 1, "UniformPrior needs to be 1D distributions"
        self.xmax = xmax
        self.xmin = xmin
        super().__init__(
            LogisticDistribution([f"{self.parameter_names[0]}_base"]),
            [
                LogitTransform(
                    (
                        [f"{self.parameter_names[0]}_base"],
                        [f"({self.parameter_names[0]}-({xmin}))/{(xmax-xmin)}"],
                    )
                ),
                ScaleTransform(
                    (
                        [f"({self.parameter_names[0]}-({xmin}))/{(xmax-xmin)}"],
                        [f"{self.parameter_names[0]}-({xmin})"],
                    ),
                    xmax - xmin,
                ),
                OffsetTransform(
                    ([f"{self.parameter_names[0]}-({xmin})"], self.parameter_names),
                    xmin,
                ),
            ],
        )
