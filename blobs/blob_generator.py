import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
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
SAMPLES_PER_POSITION = 25  # images per (x, y) position

# A white disk
DISK = cv.circle(
    img=np.zeros((41, 41)),
    center=(20, 20),
    radius=10,
    color=1,
    thickness=-1
)

# blur disk with gaussian filter; to be placed on a certain position on the canvas
BLOB = cv.GaussianBlur(
    src=DISK,
    ksize=(21, 21),
    sigmaX=0
)


# function that places a square matrix inside a black canvas
def embed(image, out_size, pos_x, pos_y):
    # image must be square with odd dims
    img_size = image.shape[0]

    stride = (img_size - 1) // 2
    new_canvas = np.zeros((out_size + 2*stride, out_size + 2*stride))

    new_canvas[pos_x:pos_x + img_size, pos_y:pos_y + img_size] = image

    out = new_canvas[stride: stride + out_size, stride:stride + out_size]
    return out


# SAVE A BLOB EXAMPLE 64x64 AS IMAGE
example = embed(BLOB, 64, 15, 15)
plt.figure(figsize=(5, 5))
plt.imshow(example, cmap='gray')
plt.title("Blob example")
plt.savefig(f'{IMAGE_PATH}/sample64.png', bbox_inches='tight')

print("Generating Blob dataset...")

num_samples = IMAGE_SIZE ** 2 * SAMPLES_PER_POSITION
blobset64 = np.zeros((num_samples, IMAGE_SIZE, IMAGE_SIZE))
ground_truth_factors = np.zeros((num_samples, 2))

num = 0
for x in tqdm(range(IMAGE_SIZE)):
    for y in range(IMAGE_SIZE):
        for s in range(SAMPLES_PER_POSITION):

            blobset64[num, :, :] = embed(BLOB, IMAGE_SIZE, x, y)
            ground_truth_factors[num, :] = np.array([x, y])  # ground truth factors: (pos_x, pos_y)
            num += 1

print(blobset64.shape)
np.savez_compressed(f'{DATA_PATH}/blobs64', blobset64)
np.savez_compressed(f'{DATA_PATH}/blobs64_ground_truth', ground_truth_factors)

print("Dataset and ground truth labels generated.")

# Generate weak labels of angle and distance
print("\n")
print("Generating Angle/Distance labels...")

for label_resolution in tqdm(range(2, 11)):

    angle_labels = np.zeros((num_samples, label_resolution))
    distance_labels = np.zeros((num_samples, label_resolution))

    angle_step = np.pi / (2*label_resolution)
    distance_step = np.sqrt(IMAGE_SIZE**2 + IMAGE_SIZE**2) / label_resolution

    num = 0
    for x in range(IMAGE_SIZE):
        for y in range(IMAGE_SIZE):
            for s in range(SAMPLES_PER_POSITION):

                # create angle labels
                phi = np.arctan2(y, x)  # angle from upper-left corner
                dis = np.sqrt(x**2 + y**2)  # distance from upper-left corner

                for k in range(label_resolution):

                    # angle label (one-hot)
                    if k * angle_step < phi <= (k + 1) * angle_step:
                        angle_labels[num, k] = 1

                    # distance label (one-hot)
                    if k * distance_step < dis <= (k + 1) * distance_step:
                        distance_labels[num, k] = 1

                num += 1

    # small correction: assign to label 0 when k=0 and k * angle_step = phi
    problematic_ids = angle_labels.sum(axis=1) == 0
    angle_labels[problematic_ids, 0] = 1

    assert angle_labels.sum(axis=1).min() == angle_labels.sum(axis=1).max()  # min=max=1; labels are one-hot
    assert angle_labels.sum(axis=1).min() == angle_labels.sum(axis=1).max()

    np.savez_compressed(f'{DATA_PATH}/blobs64_anglelabels_res{label_resolution}', angle_labels)
    np.savez_compressed(f'{DATA_PATH}/blobs64_distlabels_res{label_resolution}', distance_labels)

    # plot angle/distance labels + export images
    ids = [i * SAMPLES_PER_POSITION for i in range(IMAGE_SIZE ** 2)]

    angle_map = angle_labels[ids, :].argmax(axis=1).reshape(IMAGE_SIZE, IMAGE_SIZE)
    plt.figure(figsize=(5, 5))
    plt.imshow(angle_map, cmap='gray')
    plt.title(f"Angle labels, res={label_resolution}")
    plt.savefig(f'{IMAGE_PATH}/anglelabels_res{label_resolution}.png', bbox_inches='tight')

    dis_map = distance_labels[ids, :].argmax(axis=1).reshape(IMAGE_SIZE, IMAGE_SIZE)
    plt.figure(figsize=(5, 5))
    plt.imshow(dis_map, cmap='gray')
    plt.title(f"Distance labels, res={label_resolution}")
    plt.savefig(f'{IMAGE_PATH}/distancelabels_res{label_resolution}.png', bbox_inches='tight')

print("Labels generated.")
