"""Prior specification and parsing for jesterTOV inference system."""

from .parser import parse_prior_file, ParsedPrior

__all__ = [
    "parse_prior_file",
    "ParsedPrior",
]
