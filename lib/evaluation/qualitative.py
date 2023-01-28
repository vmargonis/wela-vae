import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model


def _traverse(
    mean_vec: np.ndarray,
    decoder: Model,
    latent_dim: int,
    initial_dim: int,
    is_wela: bool,
    std_bound: float = 3.0,
) -> np.ndarray:
    """
    Traverses each latent channel in [-std_bound, +std_bound] while keeping the others
    fixed. Each channel is a normal distribution N(0, 1) so [-3, +3] suffices for most
    cases. At each step of the traversal, the reconstruction is stored for display.
    """
    interval = np.linspace(-std_bound, std_bound, 10)
    traversals = np.zeros((latent_dim, 10, initial_dim))

    for i in range(latent_dim):
        aux = mean_vec[i, :].copy()
        aux = np.expand_dims(aux, axis=1)
        for j in range(10):
            aux[i, 0] = interval[j]

            if is_wela:
                [traversals[i, j, :], _, _] = decoder.predict(aux.T, verbose=0)
            else:
                traversals[i, j, :] = decoder.predict(aux.T, verbose=0)

    return traversals


def _make_heatmap(
    mean_vec: np.ndarray,
    latent_dim: int,
    image_resolution: int,
    samples_per_position: int,
) -> np.ndarray:
    """
    Makes the positional heatmaps of the qualitative figure.
    """
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
    dataset: np.ndarray,
    mean_vec: np.ndarray,
    log_var_vec: np.ndarray,
    decoder: Model,
    is_wela: bool,
    figure_name: str,
    output_directory: str,
    std_bound: float = 3.0,
) -> None:
    """
    Constructs the qualitative evaluation figure.
    """
    n_samples, initial_dim = dataset.shape
    samples_per_position = n_samples // initial_dim
    image_resolution = np.sqrt(initial_dim).astype(int)
    latent_dim = mean_vec.shape[1]

    # construct position heat map with all data:
    pos_map = _make_heatmap(
        mean_vec, latent_dim, image_resolution, samples_per_position
    )

    # two first rows of figure: originals and reconstructions
    np.random.seed(42)
    recon_ids = np.random.choice(n_samples, 10)
    test_images = dataset[recon_ids]
    test_means = mean_vec[recon_ids]
    if is_wela:
        [test_recons, _, _] = decoder.predict(test_means, verbose=0)
    else:
        test_recons = decoder.predict(test_means, verbose=0)

    plt.figure(figsize=(15, 5 + (latent_dim - 2) * 1.25))
    for col in range(10):
        # display original
        ax = plt.subplot(latent_dim + 2, 12, col + 2)
        plt.imshow(
            test_images[col].reshape(image_resolution, image_resolution), cmap="gray"
        )
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 9:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel("orig.", fontsize=22, rotation=0, ha="left", va="center")

        # display reconstruction
        ax = plt.subplot(latent_dim + 2, 12, col + 14)
        plt.imshow(
            test_recons[col].reshape(image_resolution, image_resolution), cmap="gray"
        )
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 9:
            ax.yaxis.set_label_position("right")
            ax.set_ylabel("recon.", fontsize=22, rotation=0, ha="left", va="center")

    # construct rest of figure
    mean_variances = np.mean(np.exp(log_var_vec), axis=0)
    sorted_idx = np.argsort(mean_variances)

    # pick late_dim images for traversals:
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
        # plot original image left
        ax = plt.subplot(latent_dim + 2, 12, (i + 2) * 12 + 1)
        plt.imshow(
            originals[sorted_idx[i]].reshape(image_resolution, image_resolution),
            cmap="gray",
        )
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # plot heatmap right
        ax = plt.subplot(latent_dim + 2, 12, (i + 2) * 12 + 12)
        plt.imshow(
            pos_map[sorted_idx[i], :, :], vmin=-std_bound, vmax=std_bound, cmap="jet"
        )
        ax.yaxis.set_label_position("right")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # plot traversals in-between
        for col in range(10):
            ax = plt.subplot(latent_dim + 2, 12, (i + 2) * 12 + col + 2)
            plt.imshow(
                traversed[sorted_idx[i], col, :].reshape(
                    image_resolution, image_resolution
                ),
                cmap="gray",
            )
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

    ax.autoscale(enable=True)  # noqa
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(output_directory + f"/qa_{figure_name}.png", bbox_inches="tight")

    return
