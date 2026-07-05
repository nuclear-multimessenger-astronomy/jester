# Analyzing a binary neutron star signal

This page in the inference guide helps you set up the `jester` inference to analyze the equation of state from gravitational wave (GW) data from a binary neutron star (BNS) merger.

## 1. Get GW posteriors

`jester` takes posterior samples of the GW source parameters as input to then constrain the EOS. 
Therefore, the first step is to run your favourite GW sampler in order to obtain these posterior samples. 
In particular, we require the source-frame component masses and the tidal deformabilities, which are named `mass_1_source`, `mass_2_source`, `lambda_1`, `lambda_2`. 

There are various ways to then store the posterior samples to feed to ``jester``:

- As `bilby` result HDF5 file: `bilby` creates the result file at the end of the inference run. We have some internal preprocessing functions to load in the HDF5 file for the `jester` inference.
- As `npz` file: In order to handle the output from any sampler, we also support `npz` files, which are a lightweight way to store your data. You can create the `npz` files yourself with a small utilities script by calling the [`np.savez` function](https://numpy.org/devdocs/reference/generated/numpy.savez.html), and storing the data arrays as `mass_1_source`, `mass_2_source`, `lambda_1`, `lambda_2`. 

### Converting a bilby result to `npz` without installing bilby

If you have a bilby result HDF5 file but do not have (or do not want to install) bilby, `jester` ships a standalone CLI tool (which only depends on ``h5py``, which is already installed in ``jester``) that reads the posterior group directly from the HDF5 file and saves the four required arrays as a `npz` file:

```bash
jester_extract_gw_posterior_bilby result.h5
```

By default the output is written next to the input file with a `_gw_jester_posterior.npz` suffix.
You can also specify a custom output path:

```bash
jester_extract_gw_posterior_bilby result.h5 --output /path/to/posterior.npz
```

The tool extracts `mass_1_source`, `mass_2_source`, `lambda_1`, and `lambda_2` from the `posterior` group of the bilby HDF5 file.
No bilby installation is needed, only `h5py` (already a `jester` dependency).
The resulting `npz` file can then be used directly in the `jester` configuration as shown in the next section.

In case this tool fails (e.g., because the result file was saved with a different h5py version or convention), please save manually save the four required arrays as a `npz` file using the `np.savez` function, and make sure to name the arrays `mass_1_source`, `mass_2_source`, `lambda_1`, `lambda_2` for them to be correctly loaded into `jester`.

## 2. GW posterior postprocessing: training a normalizing flow

In order to use the posterior samples in the EOS inference, we need to approximate the 4D marginal posterior on the source-frame masses and tidal deformabilities. 
To achieve this, `jester` makes use of normalizing flows: flexible density estimators parametrized by neural networks that can learn to approximate complex distributions.
The flow is trained on the samples to make their density tractable for inference. 

Starting from the dataset with the posterior samples, there are various ways to train the normalizing flow, which is a necessary step before running the `jester` inference.

### Pointing to the dataset for automatic flow training

``jester`` allows you to set up the flow training directly from the dataset with the posterior samples, which is a convenient way to get started with the inference without having to worry too much about the flow training settings.

To do this, a first option is to specify the training dataset as a `bilby` result HDF5 file, `jester` will then train a flow with default settings and automatically proceed to the inference stage afterwards.

```bash
- type: gw
  enabled: true
  events:
  - name: my_bns
    from_bilby_result: path/to/bilby/result.h5
```

Alternatively, point ``jester`` to the `npz` file containing the posterior samples, as follows:

```bash
- type: gw
  enabled: true
  events:
  - name: my_bns
    from_npz_file: path/to/npz/file.npz
```

For both of these options, should you wish to change the training settings, this can be done by creating a YAML file with your desired flow training hyperparameters, which you can point to in the main ``jester`` config file:

```bash
- type: gw
  enabled: true
  events:
  - name: my_bns
    from_npz_file: path/to/npz/file.npz
    flow_config: path/to/flow_config.yaml
```

For more information about flow training, refer to the {doc}`training_flows` page. 
The API for the flows can be found in {doc}`../api/jesterTOV.inference.flows`.

This `flow_config.yaml` file should contain the settings necessary for `FlowTrainingConfig` (see `jester/jesterTOV/inference/flows/config.py`). 
In case you want more information on how to train your own normalizing flows and gain more control on how this is done, check out {doc}`training_flows`.

After the flow has been trained, the flow weights and some metadata is saved inside the specified output directory in a subdirectory called `gw_flow_cache/`. 

```{note}
If you rerun the `jester` inference and no changes have been made to the flow settings, `jester` will load in the flow and *not* retrain the flow. If you wish to retrain the flow (e.g., if you changed the dataset), then you have to remove the directory containing the flow first. 
```

### Pointing to an already trained flow

In case you wish to have more control on the flow training, you can train the flow separately with your desired settings and then point to the trained flow directory in the `jester` config file.

Afterwards, point to an already trained flow directory from ``jester`` to load the flow directly without retraining. For example, ``jester`` ships flows trained for a handful of datasets of posterior samples on the GW170817 and GW190425 events, which you can directly load in your inference by pointing to the corresponding flow directory. These flows are stored in ``jester/jesterTOV/inference/flows/models/gw_maf``. 

Once a trained flow exists, you can easily re-use it in subsequent `jester` inferences by pointing to the model directory containing the weights and metadata to load it into `jester`.

```bash
- type: gw
  enabled: true
  events:
  - name: my_bns
    nf_model_dir: path/to/model_dir/
```

## 3. Running `jester`

Once the flow training has been set up in the `config.yaml` file, `jester` is good to go to analyze the BNS data!
You can learn more about running the `jester` inference in the following pages:

- {doc}`quickstart` — a step-by-step guide to running your first inference
- {doc}`yaml_reference` — full reference for all configuration options
- {doc}`workflow` — how the inference components connect and data flows through the system
- {doc}`training_flows` — how to train your own normalizing flows