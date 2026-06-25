from importlib.metadata import PackageNotFoundError, version

from jax import config

config.update("jax_enable_x64", True)

try:
    __version__: str = version("jesterTOV")
except PackageNotFoundError:
    __version__ = "unknown"


def get_version_info() -> dict[str, str]:
    """Return package version and git commit hash.

    Returns
    -------
    dict[str, str]
        ``{"version": ..., "git_hash": ...}``
    """
    import subprocess
    from pathlib import Path

    git_hash = "unknown"
    try:
        repo_root = Path(__file__).parent.parent
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            git_hash = result.stdout.strip()
    except Exception:
        pass

    return {"version": __version__, "git_hash": git_hash}
