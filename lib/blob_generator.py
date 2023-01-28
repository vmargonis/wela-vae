import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from lib.config import Config

IMAGE_SIZE = 64  # images are 64x64 pixels
N_SAMPLES = IMAGE_SIZE**2

DISK = cv.circle(
    img=np.zeros((41, 41)),
    center=(20, 20),
    radius=10,
    color=1,
    thickness=-1,
)

# blur disk with gaussian filter
BLOB = cv.GaussianBlur(
    src=DISK,
    ksize=(21, 21),
    sigmaX=0,
)


def make_directories() -> None:
    """
    Makes subdirectories of `blobs_path` to store data and examples.
    """
    for sub_dir in ["data", "images"]:
        par_dir = f"{Config.blobs_path}/{sub_dir}"
        os.mkdir(par_dir) if not os.path.exists(par_dir) else None
    return


def embed_image_in_canvas(
    image: np.ndarray,
    canvas_size: int,
    pos_x: int,
    pos_y: int,
) -> np.ndarray:
    """
    Places a square image inside a black canvas of specified size.

    (pos_x, pos_y) references the center of the source image in the canvas.
    Source image must be square and of odd dimensions.
    """
    img_size = image.shape[0]
    padding = (img_size - 1) // 2

    canvas = np.zeros((canvas_size + 2 * padding, canvas_size + 2 * padding))
    canvas[pos_x : pos_x + img_size, pos_y : pos_y + img_size] = image

    out = canvas[padding : padding + canvas_size, padding : padding + canvas_size]
    return out


def make_example_image() -> None:
    """
    Plot a blob example.
    """
    example = embed_image_in_canvas(BLOB, 64, 15, 15)

    plt.figure(figsize=(5, 5))
    plt.imshow(example, cmap="gray")
    plt.title("Blob example")
    plt.savefig(f"{Config.blobs_path}/images/sample64.png", bbox_inches="tight")
    return


def generate_blobs() -> None:
    """
    Generates blob images.
    """
    blobset64 = np.zeros((N_SAMPLES, IMAGE_SIZE, IMAGE_SIZE))
    ground_truth_factors = np.zeros((N_SAMPLES, 2))

    num = 0
    for x in range(IMAGE_SIZE):
        for y in range(IMAGE_SIZE):
            blobset64[num, :, :] = embed_image_in_canvas(BLOB, IMAGE_SIZE, x, y)
            ground_truth_factors[num, :] = np.array([x, y])
            num += 1

    np.savez_compressed(f"{Config.blobs_path}/data/blobs64", blobset64)
    np.savez_compressed(
        f"{Config.blobs_path}/data/blobs64_ground_truth",
        ground_truth_factors,
    )

    print(f"Dataset generated, size={blobset64.shape}")
    return


def generate_polar_labels() -> None:
    """
    Generates polar labels for each image.
    """
    for label_resolution in range(2, 11):
        angle_labels = np.zeros((N_SAMPLES, label_resolution))
        distance_labels = np.zeros((N_SAMPLES, label_resolution))

        angle_step = np.pi / (2 * label_resolution)
        distance_step = np.sqrt(IMAGE_SIZE**2 + IMAGE_SIZE**2) / label_resolution

        num = 0
        for x in range(IMAGE_SIZE):
            for y in range(IMAGE_SIZE):
                phi = np.arctan2(y, x)  # angle from upper-left corner
                dis = np.sqrt(x**2 + y**2)  # distance from upper-left corner

                for k in range(label_resolution):
                    if k * angle_step < phi <= (k + 1) * angle_step:
                        angle_labels[num, k] = 1

                    if k * distance_step < dis <= (k + 1) * distance_step:
                        distance_labels[num, k] = 1
                num += 1

        # small correction: assign to label 0 when k=0 and k * angle_step = phi
        problematic_ids = angle_labels.sum(axis=1) == 0
        angle_labels[problematic_ids, 0] = 1

        np.savez_compressed(
            f"{Config.blobs_path}/data/blobs64_anglelabels_res{label_resolution}",
            angle_labels,
        )
        np.savez_compressed(
            f"{Config.blobs_path}/data/blobs64_distlabels_res{label_resolution}",
            distance_labels,
        )

        # plot angle/distance labels + export images
        angle_map = angle_labels.argmax(axis=1).reshape(IMAGE_SIZE, IMAGE_SIZE)
        plt.figure(figsize=(5, 5))
        plt.imshow(angle_map, cmap="gray")
        plt.title(f"Angle labels, res={label_resolution}")
        plt.savefig(
            f"{Config.blobs_path}/images/anglelabels_res{label_resolution}.png",
            bbox_inches="tight",
        )

        dis_map = distance_labels.argmax(axis=1).reshape(IMAGE_SIZE, IMAGE_SIZE)
        plt.figure(figsize=(5, 5))
        plt.imshow(dis_map, cmap="gray")
        plt.title(f"Distance labels, res={label_resolution}")
        plt.savefig(
            f"{Config.blobs_path}/images/distlabels_res{label_resolution}.png",
            bbox_inches="tight",
        )

    print("Labels generated.")
    return


if __name__ == "__main__":
    make_directories()
    make_example_image()
    generate_blobs()
    generate_polar_labels()
