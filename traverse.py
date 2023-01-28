import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score


def hot_one(labels):
    """
    :param labels: NxL array, one-hot-encoded
    :return: 1D array with N values, the index of max element for each line.
    """
    return np.argmax(labels, axis=1)


# LOAD BLOBS DATA SETS
dataset_zip = np.load('blobs/data/blobs64.npz')
blobs = dataset_zip['arr_0']
print('blobs shape:', blobs.shape)

# Flatten blobs
blobs_flat = blobs.reshape((len(blobs), np.prod(blobs.shape[1:])))
print('blobs_flat shape:', blobs_flat.shape)
num_samples, initial_dim = blobs_flat.shape

# CHOOSE MODEL FOR TRAVERSALS

# label resolution
res = 4
model = 'tcvae'  # or dipvae_ii, betavae, tcvae
beta = 40  # for betaVAE and TCVA
gamma = 1000
latent_dim = 2
seeds = [14]

fig_dir = "blobs/results/unsupervised/{}".format(model)
# fig_dir = "blobs/results/double_labels/{}/corner_labels/".format(model)
fig_size = (15, 8.75)  # (15, 15) for L=10, (15, 5) for L=2, (15, 8.75) for L=5

# load angle evidence
label_zip = np.load('blobs/data/blobs64_anglelabels_res{}.npz'.format(res))
angle_labels = label_zip['arr_0']
print('angle labels shape:', angle_labels.shape)

# load distance evidence
label_zip = np.load('blobs/data/blobs64_distlabels_res{}.npz'.format(res))
dist_labels = label_zip['arr_0']
print('distance labels shape:', dist_labels.shape)


for SEED in seeds:

    # UNSUPERVISED
    # st = '{}_L{}_b{}_seed{}'.format(model, latent_dim, beta, SEED)

    # LABELS
    st = '{}_L{}_b{}_g{}_seed{}_res{}'.format(model, latent_dim, beta,
                                              gamma, SEED, res)

    # LOAD MEANS AND LOG_VARS
    means_str = 'blobs/repres/means_{}.npz'.format(st)
    log_vars_str = 'blobs/repres/log_vars_{}.npz'.format(st)
    repre_zip = np.load(means_str)
    logvars_zip = np.load(log_vars_str)
    means = repre_zip['arr_0']  # shape: (N, latent_dim)
    log_vars = logvars_zip['arr_0']

    # Load decoder:
    dec_str = 'blobs/models/blob_dec_{}.h5'.format(st)
    decoder = load_model(dec_str)

    # quality(dataset=blobs_flat, mean_vec=means,
    #         log_var_vec=log_vars,
    #         decode_model=decoder, n_cols=10, image_res=64,
    #         a=-3, b=3,
    #         seed=12, out_dir=fig_dir, save_str=st,
    #         fig_size=fig_size)

    # quality_welavae(dataset=blobs_flat, mean_vec=means,
    #                 log_var_vec=log_vars,
    #                 decode_model=decoder, n_cols=10,
    #                 image_res=64,
    #                 a=-3, b=3,
    #                 seed=10, out_dir=fig_dir, save_str=st,
    #                 fig_size=fig_size)
    print("Saved qualitative evaluation figure.")

    [_, y_pred_angle, y_pred_dist] = decoder.predict(means)

    print("res:", res)
    angle_acc = accuracy_score(hot_one(angle_labels), hot_one(y_pred_angle))
    print("accuracy on angle = {}".format(angle_acc))

    dist_acc = accuracy_score(hot_one(dist_labels), hot_one(y_pred_dist))
    print("accuracy on distance = {}".format(dist_acc))
