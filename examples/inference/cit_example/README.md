# CIT example — HTCondor GPU job

Combined radio (J0740+6620) + GW170817 inference run using SMC-RW,
designed for the LIGO CIT cluster (HTCondor, GPU).

## Setup: build the Apptainer container (once)

The job runs inside an Apptainer container so it works on both CIT-local machines
and remote OSG/IGWN grid nodes (which do not mount the CIT home directory).
`jester.def` lives in `containers/` at the repo root.

**Option A — build directly on a CIT login node** (requires internet access):

```bash
# Verify internet access (needed to fetch the blackjax git dependency)
curl -I https://github.com

# Build (~10–20 minutes)
cd /path/to/jester
apptainer build /home/thibeau.wouters/jester_analyses/jester.sif containers/jester.def
```

**Option B — build with Docker locally, transfer to CIT** (if apptainer not installed locally):

```bash
# On your laptop
docker build -t jester -f containers/Dockerfile .
apptainer build jester.sif docker-daemon://jester:latest
scp jester.sif thibeau.wouters@ldas-pcdev1.ligo.caltech.edu:/home/thibeau.wouters/jester_analyses/
```

**Test interactively before submitting:**

```bash
apptainer exec --nv /home/thibeau.wouters/jester_analyses/jester.sif \
    run_jester_inference config.yaml --dry-run
```

(`--nv` passes through the host GPU; omit on nodes without a GPU.)

## Submit and monitor

```bash
condor_submit submit.sub
condor_q
condor_q -analyze <job_id>     # diagnose idle jobs
condor_tail -f <job_id>        # stream stdout once running
```

## Accounting groups

| Use | Tag |
|---|---|
| Testing | `ligo.dev.o4.cbc.extremematter.bilby` |
| Production | `ligo.prod.o4.cbc.extremematter.bilby` |

Change `accounting_group` in `submit.sub` accordingly.

## Files

| File | Purpose |
|---|---|
| `jester.def` | Apptainer definition file — builds `jester.sif` |
| `submit.sub` | HTCondor submit file (container-based, works on full grid) |
| `run_jester.sh` | Legacy venv wrapper — only works on CIT-local machines |
| `config.yaml` | Inference configuration |
| `prior.prior` | MetaModel+CSE prior specification |
| `outdir/` | Job logs and results land here |

## Notes

- `MY.SingularityImage` in `submit.sub` tells HTCondor which container to use.
  The container is portable — no home directory mounting needed on remote nodes.
- `requirements = (HAS_SINGULARITY=?=True) && ...` ensures jobs only land on
  container-capable nodes. Do not restrict to `ligo.caltech.edu` machines — that
  defeats the distributed grid and reduces GPU availability.
- `getenv = true` is disabled on IGWN clusters. The container is self-contained
  so this is irrelevant.
- `initialdir` in `submit.sub` still sets the working directory so relative paths
  in `config.yaml` (e.g. `outdir/`) resolve correctly.
- If you update JESTER and need a fresh container, rebuild the `.sif` image with
  `apptainer build` and re-submit.
