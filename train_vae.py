import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from keras import backend as K
from evaluation.visualize import quality
from vae.vae_models import TCVAE
from os.path import exists
from os import mkdir

for res_dir in ["results/models", "results/repres", "results/history"]:
    if not exists(res_dir):
        mkdir(res_dir)


# LOAD BLOBS DATA SETS
dataset_zip = np.load('blobs/data/blobs64.npz')
blobs = dataset_zip['arr_0']
print('blobs shape:', blobs.shape)

# Flatten blobs
blobs_flat = blobs.reshape((len(blobs), np.prod(blobs.shape[1:])))
print('blobs_flat shape:', blobs_flat.shape)

# random permutation of dsprites data set
SEED = 12
np.random.seed(SEED)

# random permutation of blobs data set
permuted_idx = np.random.permutation(len(blobs))

# training set
x_train = blobs_flat[permuted_idx]
print('training shape:', x_train.shape)

FIG_SIZES = {
    2: (15, 5),
    5: (15, 8.75),
    10: (15, 15)
}

# Parameters
num_samples, initial_dim = x_train.shape
latent_dim = 5
num_epochs = 100
batch_size = 64
fig_size = FIG_SIZES[latent_dim]

vae_config = {
    'random_seed': SEED,
    'initial_dim': initial_dim,
    'latent_dim': latent_dim,
    'IsBinaryInput': False,
    'output_dist': 'bernoulli',
    'encoder': {
        'units': [512, 64],
        'activation': ['relu', 'relu'],
        'output_activation': 'linear'
    },
    'decoder': {
        'units': [512, 64],
        'activation': ['relu', 'relu'],
        'output_activation': 'sigmoid',
    },
    'optimizer': {
        'type': 'Adam',
        'learning_rate': 1e-04
    },
}

dip_vae_type = 'ii'
# model = 'dipvae_{}'.format(dip_vae_type)
model = 'tcvae'  # betavae
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
        # print("Progress: lambda_od={}, multiplier={}, seed={}".format(
        #     lambda_od,
        #     multiplier, SEED))
        # vae_config['random_seed'] = SEED
        #
        # lambda_d = multiplier * lambda_od
        # params = {'dip_vae_type': dip_vae_type,
        #           'lambda_od': lambda_od,
        #           'lambda_d': lambda_d}
        # vae = DIPVae(vae_config, params)
        #
        # st = '{}_L{}_lod{}_ld{}_seed{}'.format(model, latent_dim,
        #                                        lambda_od, lambda_d, SEED)

        vae_config['random_seed'] = weight_seed
        params = {"beta": beta}
        vae = TCVAE(vae_config, params)
        st = f"{model}_L{latent_dim}_b{beta}_seed{weight_seed}"

        # TRAINING
        try:
            history = vae.vae.fit(
                x_train,
                None,
                epochs=num_epochs,
                batch_size=batch_size,
                shuffle=False,
            )

            sns.set_style("white")
            plt.figure(figsize=(10, 7))
            plt.plot(history.history['loss'][1:])
            plt.title(st)
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.savefig(f"results/history/hist_{st}")

        except KeyboardInterrupt:
            print("Training interrupted.")

        finally:
            means, log_vars = vae.encoder.predict(blobs_flat)
            print('means shape:', means.shape)
            print('log_vars shape:', log_vars.shape)

            quality(
                dataset=blobs_flat,
                mean_vec=means,
                log_var_vec=log_vars,
                decode_model=vae.decoder,
                n_cols=10,
                image_res=64,
                a=-3,
                b=3,
                seed=12,
                out_dir=fig_dir,
                save_str=st,
                fig_size=fig_size
            )

            print("Saved qualitative evaluation figure.")

            # SAVE REPRESENTATIONS AND LOG VARS
            means_str = f'results/repres/means_{st}.npz'
            log_vars_str = f'results/repres/log_vars_{st}.npz'
            np.savez_compressed(means_str, means)
            np.savez_compressed(log_vars_str, log_vars)
            print("Saved representations and log_vars.")

            # SAVE MODELS
            enc_str = f'results/models/blob_enc_{st}.h5'
            dec_str = f'results/models/blob_dec_{st}.h5'
            vae.encoder.save(enc_str)
            vae.decoder.save(dec_str)
            print("Saved models to disk")
