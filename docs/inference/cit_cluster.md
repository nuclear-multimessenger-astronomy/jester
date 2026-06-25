(running-on-cit)=
# Running on the CIT cluster

This page explains how to run ``jester`` inferences on the LIGO CIT cluster using HTCondor and GPU-capable Apptainer containers.
The example below uses a combined radio (J0740+6620) + GW170817 inference run with the SMC-RW sampler, but the same setup applies to any ``jester`` configuration.
The full example files are in `examples/inference/cit_example/` at the repository root.

## Setup: build the Apptainer container

The job runs inside an Apptainer container so it works on both CIT-local machines and remote OSG/IGWN grid nodes, which do not mount the CIT home directory.
The container is stored on the [Open Science Data Federation (OSDF)](https://osg-htc.org/services/osdf.html) CIT staging area so HTCondor can fetch it on any execution site — no home directory mounting or file transfers needed.
A pre-built container for the latest version of ``jester`` is available at:

```
osdf:///igwn/cit/staging/thibeau.wouters/containers/jester.sif
```

### Building and uploading a new container to OSDF

The above container is intended to have the latest version of `jester`. 
However, if you need to rebuild the container (e.g. after a major ``jester`` update), here are some instructions to get started. 

First, build the `.sif` on a CIT login node and upload it to OSDF staging with the `osdf` CLI tool.
The container definition file `jester.def` lives in `containers/` at the repository root.

```bash
# Build the container on a CIT login node (takes a few minutes)
cd /path/to/jester
apptainer build /home/<username>/jester_analyses/containers/jester.sif containers/jester.def

# Upload to OSDF CIT staging
osdf object put /home/<username>/jester_analyses/containers/jester.sif \
    /igwn/cit/staging/<username>/containers/jester.sif

# Verify the upload
osdf object ls /igwn/cit/staging/<username>/containers/
```

The `osdf` command is available on CIT login nodes. The authentication uses SciTokens: refer to the [LIGO computing docs](https://computing.docs.ligo.org/guide/auth/scitokens/?h=scit) for more information.

## Submission script

An example submission script is given in ``examples/inference/cit_example``.
The key lines that enable OSDF and SciTokens authentication are:

```ini
universe        = container
container_image = osdf:///igwn/cit/staging/<username>/containers/jester.sif

use_oauth_services = scitokens
```

`universe = container` tells HTCondor to run the job inside the Apptainer container fetched from OSDF.
`use_oauth_services = scitokens` is required so that remote glidein nodes (e.g. OSG sites) can authenticate against OSDF to fetch the `.sif` image.

Don't forget to choose the appropriate accounting group (as well as update usernames and paths) for CIT by setting `accounting_group` in `submit.sub`.
For EOS studies, the following can be considered:

| Use | Tag |
|---|---|
| Testing | `ligo.dev.o4.cbc.extremematter.bilby` |
| Production | `ligo.prod.o4.cbc.extremematter.bilby` |

Then, you can submit the job and monitor it with the standard HTCondor commands:

```bash
condor_submit submit.sub
condor_q
condor_q -analyze <job_id>     # diagnose idle jobs (nodes available,...)
condor_tail -f <job_id>        # stream stdout once running
```

## Notes

`getenv = true` recently got disabled on IGWN clusters; the container is self-contained and therefore bypasses this issue.

`initialdir` in `submit.sub` sets the working directory so relative paths in `config.yaml` (e.g. `outdir/`) resolve correctly.

If you update JESTER and need a fresh container, rebuild the `.sif` image with `apptainer build` and re-submit.