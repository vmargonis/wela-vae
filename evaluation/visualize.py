import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("white")

# Figure variables
STD_STEP = 3  # Traversals: How many stds away from mean
TRAVERSE_STEP = 10  # How many steps in the traversal
FIG_SIZE_MAP = {  # latent dim to figure size
    2: (15, 5),
    5: (15, 8.75),
    10: (15, 15)
}


def _traverse(
        mean_vec: np.array,
        decoder: keras.Model,
        latent_dim: int,
        initial_dim: int,
        is_wela: bool,
) -> np.array:

    interval = np.linspace(-STD_STEP, STD_STEP, TRAVERSE_STEP)
    traversals = np.zeros((latent_dim, TRAVERSE_STEP, initial_dim))

    for i in range(latent_dim):
        aux = mean_vec[i, :].copy()
        aux = np.expand_dims(aux, axis=1)
        for j in range(TRAVERSE_STEP):
            aux[i, 0] = interval[j]

            if is_wela:
                [traversals[i, j, :], _, _] = decoder.predict(aux.T)
            else:
                traversals[i, j, :] = decoder.predict(aux.T)

    return traversals


def _make_heatmap(
        mean_vec: np.array,
        latent_dim: int,
        image_resolution: int,
        samples_per_position: int,
) -> np.array:

    r_prime = mean_vec.reshape((
        image_resolution,
        image_resolution,
        samples_per_position,
        latent_dim
    ))

    heatmap = np.zeros((latent_dim, image_resolution, image_resolution))
    for lat in range(latent_dim):
        for i in range(image_resolution):
            for j in range(image_resolution):
                aux = r_prime[i, j, :, lat].copy()
                # aux = aux.flatten()
                heatmap[lat, i, j] = np.mean(aux)

    return heatmap


def quality(
        dataset,
        mean_vec,
        log_var_vec,
        decoder,
        out_dir,
        save_str,
        is_wela,
) -> None:

    n_samples, initial_dim = dataset.shape
    samples_per_position = n_samples // initial_dim
    image_resolution = np.sqrt(initial_dim).astype(int)
    latent_dim = mean_vec.shape[1]

    if latent_dim not in [2, 5, 10]:
        raise ValueError(f"No fig size set for latent_dim={latent_dim}")
    else:
        fig_size = FIG_SIZE_MAP[latent_dim]

    # construct position heat map with all data:
    pos_map = _make_heatmap(
        mean_vec,
        latent_dim,
        image_resolution,
        samples_per_position
    )

    # TWO FIRST ROWS OF FIGURE (ORIGINALS AND RECONSTRUCTIONS)
    np.random.seed(42)
    idx = np.random.choice(n_samples, TRAVERSE_STEP)
    test_images = dataset[idx]
    test_means = mean_vec[idx]
    if is_wela:
        [test_recons, _, _] = decoder.predict(test_means)
    else:
        test_recons = decoder.predict(test_means)

    plt.figure(figsize=fig_size)
    for col in range(TRAVERSE_STEP):

        # display original
        ax = plt.subplot(latent_dim+2, TRAVERSE_STEP+2, col+2)
        plt.imshow(
            test_images[col].reshape(image_resolution, image_resolution),
            cmap='gray'
        )
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == TRAVERSE_STEP - 1:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(
                'orig.',
                fontsize=22,
                rotation=0,
                ha='left',
                va='center'
            )

        # display reconstruction
        ax = plt.subplot(latent_dim+2, TRAVERSE_STEP+2, col+2+TRAVERSE_STEP+2)
        plt.imshow(
            test_recons[col].reshape(image_resolution, image_resolution),
            cmap='gray'
        )
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == TRAVERSE_STEP - 1:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(
                'recon.',
                fontsize=22,
                rotation=0,
                ha='left',
                va='center'
            )

    # CONSTRUCT THE REST OF THE FIGURE
    mean_variances = np.mean(np.exp(log_var_vec), axis=0)
    sorted_idx = np.argsort(mean_variances)

    # PICK late_dim random images:
    idx_2 = np.array(latent_dim * [2048])
    # idx_2 = np.random.choice(num_samples, late_dim)
    originals = dataset[idx_2]
    representations = mean_vec[idx_2]

    # take traversals:
    traversed = _traverse(
        representations,
        decoder,
        latent_dim,
        initial_dim,
        is_wela,
    )

    for i in range(latent_dim):

        ax = plt.subplot(
            latent_dim + 2,
            TRAVERSE_STEP+2,
            (i+2) * (TRAVERSE_STEP+2) + 1
        )

        plt.imshow(
            originals[sorted_idx[i]].reshape(
                image_resolution,
                image_resolution),
            cmap='gray'
        )
        ax.grid(False)
        # ax.set_ylabel(r'${}$'.format(sorted_idx[i]), fontsize=22, rotation=0,
        #               ha='right', va='center')
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(
            latent_dim + 2,
            TRAVERSE_STEP+2,
            (i+2) * (TRAVERSE_STEP+2) + 1 + TRAVERSE_STEP + 1
        )

        plt.imshow(
            pos_map[sorted_idx[i], :, :],
            vmin=-STD_STEP,
            vmax=STD_STEP,
            cmap='jet'
        )
        ax.yaxis.set_label_position("right")
        # ax.set_ylabel(r'${:.2f}$'.format(mean_variances[sorted_idx[i]]),
        #               fontsize=22, rotation=0, ha='left', va='center')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        for col in range(TRAVERSE_STEP):
            ax = plt.subplot(
                latent_dim+2,
                TRAVERSE_STEP+2,
                (i+2) * (TRAVERSE_STEP+2) + col+2
            )

            plt.imshow(
                traversed[sorted_idx[i], col, :].reshape(
                    image_resolution,
                    image_resolution),
                cmap='gray'
            )
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

    ax.autoscale(enable=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(out_dir+f'/quality_{save_str}.pdf', bbox_inches='tight')

    return None
