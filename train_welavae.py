from os import mkdir
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras import backend as K

from lib.evaluation.qualitative import quality_welavae
from lib.models.welavae import WeLaTCVae

# Session settings to avoid pre allocating all the GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

out_dir = "blobs/results"
if not exists(out_dir):
    mkdir(out_dir)

out_dir = "blobs/models"
if not exists(out_dir):
    mkdir(out_dir)

out_dir = "blobs/repres"
if not exists(out_dir):
    mkdir(out_dir)

dataset_zip = np.load("blobs/data/blobs64.npz")
blobs = dataset_zip["arr_0"]
print("blobs shape:", blobs.shape)

# Flatten blobs
blobs_flat = blobs.reshape((len(blobs), np.prod(blobs.shape[1:])))
print("blobs_flat shape:", blobs_flat.shape)

# label resolution
res = 7

# load angle evidence
label_zip = np.load("blobs/data/blobs64_anglelabels_res{}.npz".format(res))
angle_labels = label_zip["arr_0"]
print("angle labels shape:", angle_labels.shape)

# load distance evidence
label_zip = np.load("blobs/data/blobs64_distlabels_res{}.npz".format(res))
dist_labels = label_zip["arr_0"]
print("distance labels shape:", dist_labels.shape)

# random permutation of samples for training
SEED = 12
np.random.seed(SEED)
permuted_idx = np.random.permutation(len(blobs))

# training set shuffle
x_train = blobs_flat[permuted_idx]
angle_shuffled = angle_labels[permuted_idx]
dist_shuffled = dist_labels[permuted_idx]
print("training shape:", x_train.shape)

# Parameters
num_samples, initial_dim = x_train.shape
label_dim = angle_labels.shape[1]
latent_dim = 2  # L
num_epochs = 150
batch_size = 256
fig_size = (15, 5)  # (15, 15) for L=10, (15, 5) for L=2, (15, 8.75) for L=5

config = {
    "random_seed": SEED,
    "initial_dim": initial_dim,
    "evidence_dim": label_dim,
    "latent_dim": latent_dim,
    "IsBinaryInput": False,
    "output_dist": "bernoulli",
    "encoder": {
        "units": [1200, 1200],
        "activation": ["tanh", "tanh"],
        "output_activation": "linear",
    },
    "decoder": {
        "units": [1200, 1200],
        "activation": ["tanh", "tanh"],
        "output_activation": "sigmoid",
    },
    "optimizer": {"type": "Adam", "learning_rate": 1e-04},
}


# dip_vae_type = 'i'
# model = 'dipvae_{}'.format(dip_vae_type)
model = "tcvae"  # betavae
fig_dir = "blobs/results/double_labels/{}/corner_labels/res{}".format(model, res)
if not exists(fig_dir):
    mkdir(fig_dir)

# lambda_od = 10
# lambda_d = 500
# lods = [200]
# multipliers = [5]


betas = [40]
gammas = [600]

# res = 2 -> gamma = 2000
# res = 3 -> gamma = 1500
# res = 4 -> gamma = 1000
# res = 5 -> gamma = 800
# res = 6 -> gamma = 750
# res = 7 -> gamma = 600
# res = 8 -> gamma = 500

# good beta for tcvae: 40
# good gamma for tcvae: 1200
# good seeds for tcvae: 3,5,12,34,54
# seeds = [3, 5, 12, 34, 54, 2, 6, 7, 8, 9, 11]
# seeds = [52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64]
seeds_0 = list(range(1, 26))
seeds_1 = list(range(26, 51))
# seeds_all = list(range(1,51))
for SEED in seeds_0:
    for gamma in gammas:
        for beta in betas:
            config["random_seed"] = SEED
            # print('Progress:', beta, gamma, seed)

            # params = {'dip_vae_type': dip_vae_type,
            #           'lambda_od': lambda_od,
            #           'lambda_d': lambda_d,
            #           'gamma': gamma}
            # welavae = EviDIPVae(config, params)
            # st = '{}_L{}_lod{}_ld{}_g{}_seed{}'.format(model, latent_dim,
            #                                            lambda_od, lambda_d,
            #                                            gamma,
            #                                            SEED)

            params = {"beta": beta, "gamma": gamma}
            # welavae = EviBetaVae(config, params)
            welavae = WeLaTCVae(config, params)
            st = "{}_L{}_b{}_g{}_seed{}_res{}".format(
                model, latent_dim, beta, gamma, SEED, res
            )

            # TRAINING
            try:
                history = welavae.welavae.fit(
                    [x_train, angle_shuffled, dist_shuffled],
                    None,
                    epochs=num_epochs,
                    batch_size=batch_size,
                    shuffle=False,
                )

                sns.set_style("white")
                plt.figure(figsize=(10, 7))
                plt.plot(history.history["loss"][1:])
                plt.title(st)
                plt.ylabel("Loss")
                plt.xlabel("Epoch")
                plt.savefig("blobs/results/hist/hist_{}".format(st))

            except KeyboardInterrupt:
                print("Training interrupted.")

            finally:
                means, log_vars = welavae.encoder.predict(
                    [blobs_flat, angle_labels, dist_labels]
                )
                print("means shape:", means.shape)
                print("log_vars shape:", log_vars.shape)

                quality_welavae(
                    dataset=blobs_flat,
                    mean_vec=means,
                    log_var_vec=log_vars,
                    decode_model=welavae.decoder,
                    n_cols=10,
                    image_res=64,
                    a=-3,
                    b=3,
                    seed=10,
                    out_dir=fig_dir,
                    save_str=st,
                    fig_size=fig_size,
                )
                print("Saved qualitative evaluation figure.")

                # SAVE REPRESENTATIONS AND LOG VARS
                means_str = "blobs/repres/means_{}.npz".format(st)
                log_vars_str = "blobs/repres/log_vars_{}.npz".format(st)
                np.savez_compressed(means_str, means)
                np.savez_compressed(log_vars_str, log_vars)
                print("Saved representations and log_vars.")

                # SAVE MODELS
                enc_str = "blobs/models/blob_enc_{}.h5".format(st)
                dec_str = "blobs/models/blob_dec_{}.h5".format(st)
                welavae.encoder.save(enc_str)
                welavae.decoder.save(dec_str)
                print("Saved models to disk")
