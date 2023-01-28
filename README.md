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

Developed in Python 3.9. Python 3.7-3.10 should also work. The project can also run in
docker containers. See instruction below.

- Clone the repository
- Create a dedicated virtual enviroment, e.g. with `venv`
- `cd` into the repository, e.g. `cd .../projects/wela-vae`
- Create and activate virtual environment
```shell
  python -m venv venv && source venv/bin/activate
```
- Install requirements: `pip install -r requirements/base.txt`

## Usage

### Blobs dataset generation
The following command will create the dataset into the directory `.../wela-vae/blobs`
```shell
  python -m lib.blob_generator
```

### Training

Open `lib/train_config.yaml` and set the desired parameters for the model as well as
for the training process. To train, for example, a TCVAE, run
```shell
  python -m lib.train --type tcvae
```

Other choices are `betavae` and `dipvae`. To train the **WeLa** variants, make sure
that `label_dim` and `gamma` parameters are set in `train_config.yaml` and just use
the `--wela` option in the command:
```shell
  python -m lib.train --type tcvae --wela
```

### Results

Once the training has stopped (or interrupted by the user), two figures are generated:
1. The qualitative evaluation figure in `.../wela-vae/results/eval/<model type>/`
2. The training loss history in `.../wela-vae/results/history/<model type>/`

The name of the figures contain the type of the model, number of latent dimensions,
number of label dimensions (for **WeLa**), weight initialization seed,
as well as the variant's loss parameters (e.g., beta for TCVAE).

## Run in docker container

- `cd` into the repository, e.g. `cd .../projects/wela-vae`
- Create docker image with tag `welavae:main`
```shell
  docker build . -t welavae:main
```
- Create blobs dataset using a volume mount so that the data are stored locally
```shell
  docker run --rm \
    -v $(pwd)/blobs:/app/blobs \
    welavae:main python -m lib.blob_generator
```
- Train the NN. Use volume mounts to read the blob dataset and store results locally.
There is an extra volume mount for the configuration `.yaml` file. Using this mount
allows any change in the configuration file to be seen by the container, eliminating
the need to re-build the image.
```shell
  docker run --rm \
    -v $(pwd)/blobs:/app/blobs \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/lib/train_config.yaml:/app/lib/train_config.yaml \
    welavae:main python -m lib.train --type tcvae
```
