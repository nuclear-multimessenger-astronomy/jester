"""Generate a tiny synthetic gwfast Fisher-forecast dataset for this example.

Produces a (gwfast_result.h5, injection_catalog.h5) pair with the schema
GWFisherLikelihood expects (see docs/overview/likelihoods/gw_fisher.rst), for a
handful of toy BNS sources. This is NOT real gwfast output -- it exists purely so
this example is fast and self-contained, without depending on a real (much larger)
gwfast forecast + injection catalog pair.

Run this once before `run_jester_inference config.yaml`:

    python generate_synthetic_gwfast_data.py
"""

import h5py
import jax.numpy as jnp
import numpy as np

from jesterTOV import utils

# Toy "true" BNS sources: source-frame component masses (Msun) and network SNR.
# Roughly matches the observed ranges in a real gwfast SFHo forecast (q in
# [0.49, 1.0], Mc_src in [0.87, 1.79] Msun, SNR in [8, 440]).
SOURCES = [
    dict(m1=1.35, m2=1.30, snr=25.0),
    dict(m1=1.55, m2=1.20, snr=40.0),
    dict(m1=1.70, m2=1.45, snr=18.0),
    dict(m1=1.45, m2=1.42, snr=60.0),
    dict(m1=1.90, m2=1.10, snr=15.0),
]

# Toy tidal-deformability relation Lambda(M) ~ Lambda_1.4 * (M / 1.4 Msun)^-6, a
# common rough approximation -- NOT a real EOS, purely to get a self-consistent
# synthetic dataset (heavier stars have smaller Lambda).
LAMBDA_1P4 = 400.0


def lambda_of_mass(mass: np.ndarray) -> np.ndarray:
    return LAMBDA_1P4 * (mass / 1.4) ** (-6.0)


def main() -> None:
    m1 = np.array([s["m1"] for s in SOURCES])
    m2 = np.array([s["m2"] for s in SOURCES])
    snr = np.array([s["snr"] for s in SOURCES])
    lambda1 = lambda_of_mass(m1)
    lambda2 = lambda_of_mass(m2)
    eta = m1 * m2 / (m1 + m2) ** 2

    lambda_tilde = np.asarray(
        utils.lambda_tilde_from_lambda1_lambda2(
            jnp.asarray(lambda1), jnp.asarray(lambda2), jnp.asarray(eta)
        )
    )

    # Toy Fisher-like fractional errors, roughly SNR-scaled (louder sources are
    # better measured) -- not derived from an actual Fisher matrix.
    err_m1 = 0.03 * m1 * (20.0 / snr)
    err_m2 = 0.03 * m2 * (20.0 / snr)
    err_lambda_tilde = 0.15 * lambda_tilde * (20.0 / snr)

    z = np.zeros_like(m1)
    Mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2  # detector frame; == Mc_src since z=0

    n = len(SOURCES)
    with h5py.File("gwfast_result.h5", "w") as f:
        f.create_dataset("err_LambdaTilde", data=err_lambda_tilde)
        f.create_dataset("err_m1_src", data=err_m1)
        f.create_dataset("err_m2_src", data=err_m2)
        f.create_dataset("idx_det_in_cat", data=np.arange(n))
        f.create_dataset("snrs", data=snr)

    with h5py.File("injection_catalog.h5", "w") as f:
        f.create_dataset("m1_src", data=m1)
        f.create_dataset("m2_src", data=m2)
        f.create_dataset("Mc", data=Mc)
        f.create_dataset("z", data=z)
        f.create_dataset("Lambda1", data=lambda1)
        f.create_dataset("Lambda2", data=lambda2)
        f.create_dataset("eta", data=eta)

    q = m2 / m1
    print(f"Wrote {n} synthetic sources to gwfast_result.h5 and injection_catalog.h5")
    print(f"  q range: [{q.min():.3f}, {q.max():.3f}]")
    print(f"  Mc_src range: [{Mc.min():.3f}, {Mc.max():.3f}] Msun")
    print(f"  SNR range: [{snr.min():.1f}, {snr.max():.1f}]")


if __name__ == "__main__":
    main()
