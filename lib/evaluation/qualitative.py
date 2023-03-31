import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_style("white")

# Figure variables
STD_STEP = 3  # Traversals: maximum stds away from mean
TRAVERSE_STEP = 10  # How many steps in the traversal


def _traverse(
    mean_vec: np.array,
    decoder: keras.Model,
    latent_dim: int,
    initial_dim: int,
    is_wela: bool,
) -> np.array:
    """Traverses each latent channel in [-3, +3] while keeping the others fixed.
    At each step of the traversal, the reconstruction is saved to be displayed later.
    """
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
    """Makes the positional heatmaps of the qualitative figure."""

    r_prime = mean_vec.reshape(
        (image_resolution, image_resolution, samples_per_position, latent_dim)
    )

    heatmap = np.zeros((latent_dim, image_resolution, image_resolution))
    for lat in range(latent_dim):
        for i in range(image_resolution):
            for j in range(image_resolution):
                aux = r_prime[i, j, :, lat].copy()
                heatmap[lat, i, j] = np.mean(aux)

    return heatmap


def make_qualitative_evaluation_figure(
    dataset: np.array,
    mean_vec: np.array,
    log_var_vec: np.array,
    decoder: keras.Model,
    is_wela: bool,
    figure_name: str,
    output_directory: str,
) -> None:
    """Constructs the qualitative evaluation figure."""

    n_samples, initial_dim = dataset.shape
    samples_per_position = n_samples // initial_dim
    image_resolution = np.sqrt(initial_dim).astype(int)
    latent_dim = mean_vec.shape[1]

    # construct position heat map with all data:
    pos_map = _make_heatmap(
        mean_vec, latent_dim, image_resolution, samples_per_position
    )

    # TWO FIRST ROWS OF FIGURE (ORIGINALS AND RECONSTRUCTIONS)
    np.random.seed(42)
    recon_ids = np.random.choice(n_samples, TRAVERSE_STEP)
    test_images = dataset[recon_ids]
    test_means = mean_vec[recon_ids]
    if is_wela:
        [test_recons, _, _] = decoder.predict(test_means)
    else:
        test_recons = decoder.predict(test_means)

    plt.figure(figsize=(15, 5+(latent_dim-2)*1.25))
    for col in range(TRAVERSE_STEP):
        # display original
        ax = plt.subplot(latent_dim + 2, TRAVERSE_STEP + 2, col + 2)
        plt.imshow(
            test_images[col].reshape(image_resolution, image_resolution), cmap="gray"
        )
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == TRAVERSE_STEP - 1:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel("orig.", fontsize=22, rotation=0, ha="left", va="center")

        # display reconstruction
        ax = plt.subplot(latent_dim + 2, TRAVERSE_STEP + 2, col + 2 + TRAVERSE_STEP + 2)
        plt.imshow(
            test_recons[col].reshape(image_resolution, image_resolution), cmap="gray"
        )
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == TRAVERSE_STEP - 1:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel("recon.", fontsize=22, rotation=0, ha="left", va="center")

    # CONSTRUCT THE REST OF THE FIGURE
    mean_variances = np.mean(np.exp(log_var_vec), axis=0)
    sorted_idx = np.argsort(mean_variances)

    # PICK late_dim images for traversals:
    traverse_ids = np.array(latent_dim * [2080])
    # traverse_ids = np.random.choice(num_samples, late_dim)
    originals = dataset[traverse_ids]
    representations = mean_vec[traverse_ids]

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
            latent_dim + 2, TRAVERSE_STEP + 2, (i + 2) * (TRAVERSE_STEP + 2) + 1
        )

        plt.imshow(
            originals[sorted_idx[i]].reshape(image_resolution, image_resolution),
            cmap="gray",
        )
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(
            latent_dim + 2,
            TRAVERSE_STEP + 2,
            (i + 2) * (TRAVERSE_STEP + 2) + 1 + TRAVERSE_STEP + 1,
        )

        plt.imshow(
            pos_map[sorted_idx[i], :, :], vmin=-STD_STEP, vmax=STD_STEP, cmap="jet"
        )
        ax.yaxis.set_label_position("right")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        for col in range(TRAVERSE_STEP):
            ax = plt.subplot(
                latent_dim + 2,
                TRAVERSE_STEP + 2,
                (i + 2) * (TRAVERSE_STEP + 2) + col + 2,
            )

            plt.imshow(
                traversed[sorted_idx[i], col, :].reshape(
                    image_resolution, image_resolution
                ),
                cmap="gray",
            )
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

    ax.autoscale(enable=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(output_directory + f"/quality_{figure_name}.pdf", bbox_inches="tight")

    return None
