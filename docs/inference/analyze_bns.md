# Analyzing a binary neutron star signal

This page in the inference guide helps you set up the `jester` inference to analyze the equation of state from gravitational wave (GW) data from a binary neutron star (BNS) merger.

## Get GW posteriors

`jester` takes posterior samples of the GW source parameters as input to then constrain the EOS. 
Therefore, the first step is to run your favourite GW sampler in order to obtain these posterior samples. 
In particular, we require the source-frame component masses and the tidal deformabilities, which are named `mass_1_source`, `mass_2_source`, `lambda_1`, `lambda_2`. 

There are various ways to then store the posterior samples:
- As `bilby` result HDF5 file: `bilby` creates the result file at the end of the inference run. We have some internal preprocessing functions to load in the HDF5 file for the `jester` inference.
- As `npz` file: In order to handle the output from any sampler, we also support `npz` files, which are a lightweight way to store your data. You can create the `npz` files yourself with a small utilities script by calling the [`np.savez` function](https://numpy.org/devdocs/reference/generated/numpy.savez.html), and storing the data arrays as `mass_1_source`, `mass_2_source`, `lambda_1`, `lambda_2`. 

## GW posterior postprocessing: training a normalizing flow

In order to use the posterior samples in the EOS inference, we need to approximate the 4D marginal posterior on the source-frame masses and tidal deformabilities. 
To achieve this, `jester` makes use of density estimators.
In practice, we train a normalizing flow (NF) on the samples to make their density tractable for inference. 

There are various ways to train the normalizing flow. 
Currently, we have two ways to train the flow automatically when setting up inference.
The `config.yaml` below shows how to start from the bilby result HDF5 file.
```bash
- type: gw
  enabled: true
  events:
  - name: GW170817
    from_bilby_result: path/to/bilby/result.h5
```
The following `config.yaml` shows how to, instead, start from an `npz` file.
```bash
- type: gw
  enabled: true
  events:
  - name: GW170817
    from_npz_file: path/to/npz/file.npz
```

You can choose to use the default hyperparameters and settings for training the flow, which are chosen to be robust enough and should give good results for use in `jester`.
However, in case you suspect the settings are suboptimal and want to pass your own flow hyperparameters, you can do so by adding a flag to the config file as follows, which points to a `config.yaml` file. 
```bash
- type: gw
  enabled: true
  events:
  - name: GW170817
    from_npz_file: path/to/npz/file.npz
    flow_config: path/to/flow_config.yaml
```
<!-- TODO: make a nice, dedicated docs page for this -->
This `flow_config.yaml` file should contain the settings necessary for `FlowTrainingConfig` (see `jester/jesterTOV/inference/flows/config.py`). 

After the flow has been trained, the flow weights and some metadata is saved inside the specified output directory in a subdirectory called `gw_flow_cache/`. 
**NOTE**: If you rerun the `jester` inference and no changes have been made to the flow settings, `jester` will load in the flow and *not* retrain the flow. If you wish to retrain the flow (e.g., if you changed the dataset), then you have to remove the directory containing the flow first. 

<!-- TODO: this also needs the NF docs page -->
Once a trained flow exists, you can easily re-use it in subsequent `jester` inferences by pointing to the NF model directory containing the weights and metadata to load it into `jester`.
```bash
- type: gw
  enabled: true
  events:
  - name: GW170817
    nf_model_dir: path/to/model_dir/
```

## Running `jester`

Once the flow training has been set up in the `config.yaml` file, `jester` is good to go to analyze the BNS data!
You can learn more about running the `jester` inference in the following pages:

- {doc}`quickstart` — a step-by-step guide to running your first inference
- {doc}`yaml_reference` — full reference for all configuration options
- {doc}`workflow` — how the inference components connect and data flows through the system