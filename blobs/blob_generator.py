import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

sns.set()
sns.set_style("white")

DATA_PATH = "data"
if not os.path.exists(DATA_PATH):
    os.mkdir(DATA_PATH)

IMAGE_PATH = "images"
if not os.path.exists(IMAGE_PATH):
    os.mkdir(IMAGE_PATH)

IMAGE_SIZE = 64  # all images are 64x64 pixels
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


def embed_image_in_canvas(
    image: np.array,
    canvas_size: int,
    pos_x: int,
    pos_y: int,
) -> np.array:
    """Places a square image inside a black canvas of specified size.
    (pos_x, pos_y) references the center of the source image in the canvas.
    Source image must be of odd dimensions.
    """

    assert image.shape[0] == image.shape[1]
    assert image.shape[0] % 2 == 1

    img_size = image.shape[0]
    padding = (img_size - 1) // 2

    canvas = np.zeros((canvas_size + 2 * padding, canvas_size + 2 * padding))
    canvas[pos_x : pos_x + img_size, pos_y : pos_y + img_size] = image

    out = canvas[padding : padding + canvas_size, padding : padding + canvas_size]
    return out


def make_example_image() -> None:
    """Plot a blob example."""
    example = embed_image_in_canvas(BLOB, 64, 15, 15)
    plt.figure(figsize=(5, 5))
    plt.imshow(example, cmap="gray")
    plt.title("Blob example")
    plt.savefig(f"{IMAGE_PATH}/sample64.png", bbox_inches="tight")
    return None


def generate_blobs() -> None:
    blobset64 = np.zeros((N_SAMPLES, IMAGE_SIZE, IMAGE_SIZE))
    ground_truth_factors = np.zeros((N_SAMPLES, 2))

    num = 0
    for x in tqdm(range(IMAGE_SIZE)):
        for y in range(IMAGE_SIZE):
            blobset64[num, :, :] = embed_image_in_canvas(BLOB, IMAGE_SIZE, x, y)
            ground_truth_factors[num, :] = np.array([x, y])
            num += 1

    np.savez_compressed(f"{DATA_PATH}/blobs64", blobset64)
    np.savez_compressed(f"{DATA_PATH}/blobs64_ground_truth", ground_truth_factors)

    print(f"Dataset generated, size={blobset64.shape}")
    return None


# Generate weak labels of angle and distance
print("\n")
print("Generating Angle/Distance labels...")

for label_resolution in tqdm(range(2, 11)):
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

    assert (
        angle_labels.sum(axis=1).min() == angle_labels.sum(axis=1).max()
    )  # min=max=1; labels are one-hot
    assert angle_labels.sum(axis=1).min() == angle_labels.sum(axis=1).max()

    np.savez_compressed(
        f"{DATA_PATH}/blobs64_anglelabels_res{label_resolution}", angle_labels
    )
    np.savez_compressed(
        f"{DATA_PATH}/blobs64_distlabels_res{label_resolution}", distance_labels
    )

    # plot angle/distance labels + export images
    angle_map = angle_labels.argmax(axis=1).reshape(IMAGE_SIZE, IMAGE_SIZE)
    plt.figure(figsize=(5, 5))
    plt.imshow(angle_map, cmap="gray")
    plt.title(f"Angle labels, res={label_resolution}")
    plt.savefig(
        f"{IMAGE_PATH}/anglelabels_res{label_resolution}.png", bbox_inches="tight"
    )

    dis_map = distance_labels.argmax(axis=1).reshape(IMAGE_SIZE, IMAGE_SIZE)
    plt.figure(figsize=(5, 5))
    plt.imshow(dis_map, cmap="gray")
    plt.title(f"Distance labels, res={label_resolution}")
    plt.savefig(
        f"{IMAGE_PATH}/distlabels_res{label_resolution}.png", bbox_inches="tight"
    )

print("Labels generated.")
