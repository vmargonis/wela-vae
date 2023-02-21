import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras import backend as K
from evaluation.visualize import quality
from vae.vae_models import TCVAE

import os
import random
from os.path import exists
from os import mkdir

# SET RANDOM SEEDS
SEED = 12
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

for res_dir in ["results/models", "results/repres", "results/history"]:
    if not exists(res_dir):
        mkdir(res_dir)

# LOAD BLOBS DATA SETS
dataset_zip = np.load("blobs/data/blobs64.npz")
blobs = dataset_zip["arr_0"]
print("blobs shape:", blobs.shape)

# Flatten blobs: training set
X_train = blobs.reshape((len(blobs), np.prod(blobs.shape[1:])))
print("blobs_flat shape:", X_train.shape)

# Parameters
num_samples, initial_dim = X_train.shape
latent_dim = 5
num_epochs = 20
batch_size = 64

vae_config = {
    "weight_seed": SEED,
    "initial_dim": initial_dim,
    "latent_dim": latent_dim,
    "is_binary_input": False,
    "output_dist": "bernoulli",
    "encoder": {
        "units": [512, 64],
        "activation": ["relu", "relu"],
        "output_activation": "linear"
    },
    "decoder": {
        "units": [512, 64],
        "activation": ["relu", "relu"],
        "output_activation": "sigmoid",
    },
    "optimizer": {
        "type": "Adam",
        "learning_rate": 1e-04
    },
}

dip_vae_type = "ii"
# model = 'dipvae_{}'.format(dip_vae_type)
model = "tcvae"  # betavae
fig_dir = f"results/vae/{model}"
if not exists(fig_dir):
    mkdir(fig_dir)

# lambda_od = 1200
# lambda_d = lambda_od
# multipliers = [5]
# lods = [200]


betas = [40]
weight_init_seeds = [4]  # list(range(1, 26))
for weight_seed in weight_init_seeds:
    for beta in betas:

        # vae_config['random_seed'] = SEED
        #
        # lambda_d = multiplier * lambda_od
        # params = {'dip_vae_type': dip_vae_type,
        #           'lambda_od': lambda_od,
        #           'lambda_d': lambda_d}
        # vae = DIPVae(vae_config, params)

        vae_config["weight_seed"] = weight_seed
        params = {"beta": beta}
        vae = TCVAE(vae_config, params)

        # TRAINING
        try:
            history = vae.vae.fit(
                X_train,
                None,
                epochs=num_epochs,
                batch_size=batch_size,
                shuffle=True,
            )

            sns.set_style("white")
            plt.figure(figsize=(10, 7))
            plt.plot(history.history["loss"][1:])
            plt.title(vae.str_repr)
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.savefig(f"results/history/hist_{vae.str_repr}")

        except KeyboardInterrupt:
            print("Training interrupted.")

        finally:
            means, log_vars = vae.encoder.predict(X_train)
            print("means shape:", means.shape)
            print("log_vars shape:", log_vars.shape)

            quality(
                dataset=X_train,
                mean_vec=means,
                log_var_vec=log_vars,
                decoder=vae.decoder,
                out_dir=fig_dir,
                save_str=vae.str_repr,
                is_wela=False,
            )

            print("Saved qualitative evaluation figure.")

            # SAVE REPRESENTATIONS AND LOG VARS
            means_str = f"results/repres/means_{vae.str_repr}.npz"
            log_vars_str = f"results/repres/log_vars_{vae.str_repr}.npz"
            np.savez_compressed(means_str, means)
            np.savez_compressed(log_vars_str, log_vars)
            print("Saved representations and log_vars.")

            # SAVE MODELS
            enc_str = f"results/models/blob_enc_{vae.str_repr}.h5"
            dec_str = f"results/models/blob_dec_{vae.str_repr}.h5"
            vae.encoder.save(enc_str)
            vae.decoder.save(dec_str)
            print("Saved models to disk")
