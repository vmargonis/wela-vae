# wela-vae

This repository is a minimilized and cleaned-up version of the code for reproducing
the experiments in "WeLa-VAE: Learning alternative disentangled representations using
weak labels" [[arxiv preprint](https://arxiv.org/abs/2008.09879)].


## Notes
1. The synthetic dataset that this code generates is much smaller than the one used in
the paper. However, the results can still be reproduced on this tiny version using
smaller MLPs.
2. Results are **very** sensitive to random seeds: Identical python & package versions
have yielded different latent representations on different OS. Some "good"
hyperparameters and seeds are mentioned at the bottom of `lib/train_config.yaml`.
3. Although WeLa-VAE is presented as an extension of
[[TCVAE](https://arxiv.org/pdf/1802.04942.pdf)], it can also be combined with other
VAE variants suited for disentaglement, like
[[beta-VAE](https://openreview.net/pdf?id=Sy2fzU9gl)] and
[[DIP-VAE](https://openreview.net/pdf?id=H1kG7GZAW)] which are also
implemented here.


## Installation & Setup

Recommended python version for reproducibility: 3.9.13

1. Clone the repository
2. `cd` into the repository or activate virtual environment, e.g. `workon wela-vae`
3. Install requirements: `pip install -r requirements/base.txt`
4. Create a python file `local_settings.py` inside `lib`,
and set the following **absolute** paths:
```
    BLOBS_PATH = "desired/directory/to/store/blobs/data"
    RESULTS_PATH = "desired/directory/to/store/results"
```

## Usage

### "Blobs" dataset generation

```
    python -m lib.blob_generator
```

### Training

Open `lib/train_config.yaml` and set the desired parameters for the model as well as
for the training process. To train a model, run
```
    python -m lib.main --type [tcvae|betavae|dipvae] --seed 42
```

Command Parameters:
- `-t/--type` - required: VAE type
- `-s/--seed` - required: Layer weight random initialization seed
- `-w/--wela`: Train WeLa variant
- `-l/--loss`: Produce a loss history figure
- `-no--verbose`: Silent keras.fit() verbosity

For more info
```
    python -m lib.main --help
```

### Experiment

To train a model for multiple weight initialization seeds, a simple `bash` script can be
used:

```
    #!/bin/bash

    for wseed in {1..50}
    do
       echo "---------- Training with weight_seed = $wseed ---------------"
       python -m lib.main --t tcvae --s $wseed --loss --no-verbose
    done
```


### Results

Once the training is finished, the "cartesian" and "polar" MSEs are printed. A low
cartesian MSE indicates a disentagled, axis-alinged cartesian latent space, while a low
polar MSE is evidence that the model has learned a disentagled polar representation
instead. Also, a couple of figures are generated:

1. A qualitative evaluation figure in `RESULTS_PATH/results/eval/<model type>/`
2. If `--loss`: training loss history in `RESULTS_PATH/results/history/<model type>/`

The name of the figures will display the important parameters of the model. See the
preprint for more info on the qualitative and quantitative evaluation.
