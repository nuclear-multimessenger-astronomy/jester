# CIT example — HTCondor GPU job

Combined radio (J0740+6620) + GW170817 inference run using SMC-RW,
designed for the LIGO CIT cluster (HTCondor, GPU).

## Setup (once, on a login node)

A uv venv is installed here: `/home/thibeau.wouters/jester_analyses/jester/.venv/`

## Submit and monitor

```bash
condor_submit submit.sub
condor_q
condor_q -analyze <job_id>
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
| `submit.sub` | HTCondor submit file (GPU, absolute-path approach) |
| `run_jester.sh` | Wrapper script alternative — activates venv explicitly |
| `config.yaml` | Inference configuration |
| `prior.prior` | MetaModel+CSE prior specification |
| `outdir/` | Job logs and results land here |

## Notes

- `transfer_executable = false` in `submit.sub` is required — the venv binary
  uses relative RPATH links that break if HTCondor copies it to the worker node.
- `getenv = true` is disabled on IGWN clusters. The absolute-path approach in
  `submit.sub` needs no `getenv`. If you switch to `run_jester.sh`, that also
  needs no `getenv` (the wrapper sources the venv itself).
- Data paths in `config.yaml` are relative to this directory (`initialdir` in
  `submit.sub` ensures the job runs from here).
