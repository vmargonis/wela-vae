import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def traverse_wela(mean_vec, decode_model, late_dim, init_dim, num, a, b):
    interval = np.linspace(a, b, num)
    traversals = np.zeros((late_dim, num, init_dim))

    for i in range(late_dim):
        aux = mean_vec[i, :].copy()
        aux = np.expand_dims(aux, axis=1)
        for j in range(num):
            aux[i, 0] = interval[j]
            # [traversals[i, j, :], _] = decode_model.predict(aux.T)
            [traversals[i, j, :], _, _] = decode_model.predict(aux.T)

    return traversals


def traverse(mean_vec, decode_model, late_dim, init_dim, num, a, b):
    interval = np.linspace(a, b, num)
    traversals = np.zeros((late_dim, num, init_dim))

    for i in range(late_dim):
        aux = mean_vec[i, :].copy()
        aux = np.expand_dims(aux, axis=1)
        for j in range(num):
            aux[i, 0] = interval[j]
            traversals[i, j, :] = decode_model.predict(aux.T)

    return traversals


def make_heatmap(mean_vec, late_dim, img_res, posit_samples):
    r_prime = mean_vec.reshape((img_res, img_res, posit_samples, late_dim))

    heatmap = np.zeros((late_dim, img_res, img_res))
    for lat in range(late_dim):
        for i in range(img_res):
            for j in range(img_res):
                aux = r_prime[i, j, :, lat].copy()
                # aux = aux.flatten()
                heatmap[lat, i, j] = np.mean(aux)

    return heatmap


def quality(dataset, mean_vec, log_var_vec, decode_model,
            n_cols, image_res, a, b, seed, out_dir, save_str,
            fig_size):

    np.random.seed(seed)

    num_samples, init_dim = dataset.shape
    posit_samples = num_samples // init_dim
    late_dim = mean_vec.shape[1]

    # construct position heat map with all data:
    pos_map = make_heatmap(mean_vec, late_dim, image_res, posit_samples)

    # TWO FIRST ROWS OF FIGURE (ORIGINALS AND RECONSTRUCTIONS)
    idx = np.random.choice(num_samples, n_cols)
    test_images = dataset[idx]
    test_means = mean_vec[idx]
    test_recons = decode_model.predict(test_means)

    sns.set()
    sns.set_style("white")

    plt.figure(figsize=fig_size)
    for col in range(n_cols):
        # display original
        ax = plt.subplot(late_dim+2, n_cols+2, col+2)
        plt.imshow(test_images[col].reshape(image_res, image_res), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == n_cols - 1:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel('orig.', fontsize=22, rotation=0, ha='left',
                          va='center')

        # display reconstruction
        ax = plt.subplot(late_dim+2, n_cols+2, col+2 + (n_cols+2))
        plt.imshow(test_recons[col].reshape(image_res, image_res), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == n_cols - 1:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel('recon.', fontsize=22, rotation=0, ha='left',
                          va='center')

    # CONSTRUCT THE REST OF THE FIGURE
    mean_variances = np.mean(np.exp(log_var_vec), axis=0)
    sorted_idx = np.argsort(mean_variances)

    # PICK late_dim random images:
    idx_2 = np.array(late_dim * [15])
    # idx_2 = np.random.choice(num_samples, late_dim)
    originals = dataset[idx_2]
    representations = mean_vec[idx_2]

    # take traversals:
    traversed = traverse(representations, decode_model, late_dim,
                         init_dim, n_cols, a, b)

    for i in range(late_dim):

        ax = plt.subplot(late_dim + 2, n_cols+2, (i+2) * (n_cols+2) + 1)
        plt.imshow(originals[sorted_idx[i]].reshape(image_res, image_res),
                   cmap='gray')
        ax.grid(False)
        # ax.set_ylabel(r'${}$'.format(sorted_idx[i]), fontsize=22, rotation=0,
        #               ha='right', va='center')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(late_dim + 2, n_cols+2,
                         (i+2) * (n_cols+2) + 1 + n_cols + 1)
        # noinspection PyUnboundLocalVariable
        plt.imshow(pos_map[sorted_idx[i], :, :], vmin=a, vmax=b, cmap='jet')
        ax.yaxis.set_label_position("right")
        # ax.set_ylabel(r'${:.2f}$'.format(mean_variances[sorted_idx[i]]),
        #               fontsize=22, rotation=0, ha='left', va='center')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        for col in range(n_cols):
            ax = plt.subplot(late_dim+2, n_cols+2, (i+2) * (n_cols+2) + col+2)
            plt.imshow(traversed[sorted_idx[i], col, :].reshape(image_res,
                                                                image_res),
                       cmap='gray')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
    ax.autoscale(enable=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(out_dir+'/quality_{}.pdf'.format(save_str),
                bbox_inches='tight')

    return None


def quality_welavae(dataset, mean_vec, log_var_vec, decode_model,
                    n_cols, image_res, a, b, seed, out_dir, save_str,
                    fig_size):

    np.random.seed(seed)

    num_samples, init_dim = dataset.shape
    posit_samples = num_samples // init_dim
    late_dim = mean_vec.shape[1]

    # construct position heat map with all data:
    pos_map = make_heatmap(mean_vec, late_dim, image_res, posit_samples)

    # TWO FIRST ROWS OF FIGURE (ORIGINALS AND RECONSTRUCTIONS)
    idx = np.random.choice(num_samples, n_cols)
    test_images = dataset[idx]
    test_means = mean_vec[idx]
    # [test_recons, _] = decode_model.predict(test_means)
    [test_recons, _, _] = decode_model.predict(test_means)

    sns.set()
    sns.set_style("white")

    plt.figure(figsize=fig_size)
    for col in range(n_cols):
        # display original
        ax = plt.subplot(late_dim+2, n_cols+2, col+2)
        plt.imshow(test_images[col].reshape(image_res, image_res), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == n_cols - 1:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel('orig.', fontsize=22, rotation=0, ha='left',
                          va='center')

        # display reconstruction
        ax = plt.subplot(late_dim+2, n_cols+2, col+2 + (n_cols+2))
        plt.imshow(test_recons[col].reshape(image_res, image_res), cmap='gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == n_cols - 1:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel('recon.', fontsize=22, rotation=0, ha='left',
                          va='center')

    # CONSTRUCT THE REST OF THE FIGURE
    mean_variances = np.mean(np.exp(log_var_vec), axis=0)
    sorted_idx = np.argsort(mean_variances)

    # PICK late_dim random images:
    idx_2 = np.array(late_dim * [52001])
    # idx_2 = np.random.choice(num_samples, late_dim)
    originals = dataset[idx_2]
    representations = mean_vec[idx_2]

    # take traversals:
    traversed = traverse_wela(representations, decode_model, late_dim,
                              init_dim, n_cols, a, b)

    for i in range(late_dim):

        ax = plt.subplot(late_dim + 2, n_cols+2, (i+2) * (n_cols+2) + 1)
        plt.imshow(originals[sorted_idx[i]].reshape(image_res, image_res),
                   cmap='gray')
        ax.grid(False)
        ax.set_ylabel(r'${}$'.format(sorted_idx[i]), fontsize=22, rotation=0,
                      ha='right', va='center')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(late_dim + 2, n_cols+2,
                         (i+2) * (n_cols+2) + 1 + n_cols + 1)
        # noinspection PyUnboundLocalVariable
        plt.imshow(pos_map[sorted_idx[i], :, :], vmin=a, vmax=b, cmap='jet')
        ax.yaxis.set_label_position("right")
        ax.set_ylabel(r'${:.2f}$'.format(mean_variances[sorted_idx[i]]),
                      fontsize=22, rotation=0, ha='left', va='center')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        for col in range(n_cols):
            ax = plt.subplot(late_dim+2, n_cols+2, (i+2) * (n_cols+2) + col+2)
            plt.imshow(traversed[sorted_idx[i], col, :].reshape(image_res,
                                                                image_res),
                       cmap='gray')
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
    ax.autoscale(enable=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(out_dir+'/quality_{}'.format(save_str),
                bbox_inches='tight')

    return None
