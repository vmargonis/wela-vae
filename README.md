# WeLa-VAE: Learning alternative disentangled representations using weak labels

This is a minimilised version of the implementation of the respective
[manuscript](https://arxiv.org/abs/2008.09879).

The code was originally intended for research work. However, it is also useful for
1. Providing simple `keras` implementations for famous VAE variants
    - BetaVAE (vanilla VAE if `beta=1`), TCVAE and DIP-VAE.
    - The implementation considers only dense layered encoder/decoder, but it can
be easily extended to any other architecture (e.g. LSTM, Convolution).
2. Showcasing the effect of these variants on learning disentangled representations
through a simple, small toy-dataset. Training takes at most ~5min on CPU


## Installation / Setup

Developed in Python 3.9. Python 3.7-3.10 should also work.

1. Clone the repository
2. `cd` into the repository, e.g. `cd Desktop/repos/wela-vae` or activate virtual
environment, e.g. `workon wela-vae`
3. Install requirements: `pip install -r requirements/base.txt`
4. Create a python file `local_settings.py` inside `lib`,
and set the following **absolute** paths:
```
    BLOBS_PATH = "desired/directory/to/store/blobs/data"
    RESULTS_PATH = "desired/directory/to/store/results"
```

## Usage

### Blobs dataset generation

```
    python -m lib.blob_generator
```

### Training

Open `lib/train_config.yaml` and set the desired parameters for the model as well as
for the training process. To train, for example, a TCVAE, run
```
    python -m lib.train --type tcvae
```

Other choices are `betavae` and `dipvae`. To train the **WeLa** variants, make sure
that `label_dim` and `gamma` parameters are set in `train_config.yaml` and just use
the `--wela` option in the command:
```
    python -m lib.train --type tcvae --wela
```

### Results

Once the training has stopped (or interrupted by the user), two figures are generated:
1. The qualitative evaluation figure in `RESULTS_PATH/results/eval/<model type>/`
2. The training loss history in `RESULTS_PATH/results/history/<model type>/`

The name of the figures contain the type of the model, number of latent dimensions,
number of label dimensions (for **WeLa**), weight initialization seed,
as well as the variant's loss parameters (e.g., beta for TCVAE).
