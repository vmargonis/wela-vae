import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import exists
from os import mkdir


def polar_to_cartesian(theta, distance):
    pos_x = np.ceil(distance * np.cos(theta)).astype(int)
    pos_y = np.ceil(distance * np.sin(theta)).astype(int)

    return pos_x, pos_y


# function that places a square matrix inside a black canvas
def embed(image, out_size, pos_x, pos_y):
    # image must be square with odd dims
    img_size = image.shape[0]

    stride = (img_size - 1) // 2
    new_canvas = np.zeros((out_size + 2*stride, out_size + 2*stride))

    new_canvas[pos_x:pos_x + img_size, pos_y:pos_y + img_size] = image

    out = new_canvas[stride: stride + out_size, stride:stride + out_size]
    return out


def portion(region, shape):
    """

    :param region: [0,1] matrix, indicates a region inside a square image
    :param shape: [0,1] matrix, a shape, matrices must have same shape
    :return: the portion of the shape that intersects the region
    """

    denom = np.sum(shape)
    intersection = region * shape
    numer = np.sum(intersection)
    return numer / denom


def portion_labelling(regions, blobset):
    num_regions = regions.shape[0]
    num_examples = blobset.shape[0]
    portions = np.zeros((num_examples, num_regions))
    for i in range(num_examples):
        for j in range(num_regions):
            portions[i, j] = portion(regions[j, :, :], blobset[i, :, :])

    return portions


sns.set()
sns.set_style("white")
out_dir = "data"
if not exists(out_dir):
    mkdir(out_dir)

# radial_regions = np.zeros((8, 64, 64))  # 8 regions to create labels
# for x in range(64):
#     for y in range(64):
#         if x < 32:
#             if y >= 32:
#                 if x + y >= 64:
#                     radial_regions[0, x, y] = 1
#                 else:
#                     radial_regions[1, x, y] = 1
#             else:
#                 if y > x:
#                     radial_regions[2, x, y] = 1
#                 else:
#                     radial_regions[3, x, y] = 1
#         else:
#             if y < 32:
#                 if x + y < 64:
#                     radial_regions[4, x, y] = 1
#                 else:
#                     radial_regions[5, x, y] = 1
#             else:
#                 if x >= y:
#                     radial_regions[6, x, y] = 1
#                 else:
#                     radial_regions[7, x, y] = 1

# EXAMPLE 64x64
disk = cv.circle(np.zeros((41, 41)), center=(20, 20), radius=10, color=1, 
                 thickness=-1)
# blur with gaussian filter
disk = cv.GaussianBlur(disk, (21, 21), 0)
test = embed(disk, 64, 15, 15)
plt.figure(figsize=(5, 5))
plt.imshow(test, cmap='gray')
plt.savefig('data/sample64.pdf', bbox_inches='tight')

# DATASET 64x64
size = 64
position_samples = 25
num_samples = size ** 2 * position_samples

max_distance = np.sqrt(size**2 + size**2)
label_resolution = 10
angle_labels = np.zeros((num_samples, label_resolution))
dis_labels = np.zeros((num_samples, label_resolution))

angle_step = np.pi / (2*label_resolution)
distance_step = max_distance / label_resolution

# blobset64 = np.zeros((num_samples, size, size))
#
# num = 0
# for x in range(size):
#     for y in range(size):
#         for s in range(position_samples):
#             # draw disk
#             disk = cv.circle(np.zeros((41, 41)), center=(20, 20), radius=10,
#                              color=1, thickness=-1)
#
#             # blur with gaussian filter
#             blob = cv.GaussianBlur(disk, (21, 21), 0)
#
#             blobset64[num, :, :] = embed(blob, size, x, y)
#             num += 1
#
# print(blobset64.shape)
# np.savez_compressed('data/blobs64', blobset64)

ground_truth_factors = np.zeros((num_samples, 2))

num = 0
for x in range(size):
    for y in range(size):
        for s in range(position_samples):

            ground_truth_factors[num, :] = np.array([x, y])
            num += 1

print(ground_truth_factors.shape)
np.savez_compressed('data/blobs64_ground_truth', ground_truth_factors)

# # DENSITY 64
# plt.figure(figsize=(5, 5))
# plt.imshow(blobset64.mean(axis=0), interpolation='nearest', cmap='gray')
# plt.savefig('data/density64.pdf', bbox_inches='tight')

num = 0
for x in range(size):
    for y in range(size):
        for s in range(position_samples):
            # create angle labels
            phi = np.arctan2(y, x)  # angle from upper-left corner
            for k in range(label_resolution):
                if k * angle_step < phi <= (k + 1) * angle_step:
                    angle_labels[num, k] = 1

            # create distance labels
            dis = np.sqrt(x**2 + y**2)  # distance from upper-left corner
            for k in range(label_resolution):
                if k * distance_step < dis <= (k + 1) * distance_step:
                    dis_labels[num, k] = 1

            num += 1


print(angle_labels.shape)
print(dis_labels.shape)

# portion_labels = portion_labelling(radial_regions, blobset64)
res_str = '_res{}'.format(label_resolution)
np.savez_compressed('data/blobs64_anglelabels'+res_str, angle_labels)
np.savez_compressed('data/blobs64_distlabels'+res_str, dis_labels)
# np.savez_compressed('blobs_data/blobs64portionlabels', portion_labels)

# plot angle labels
A = np.zeros((64, 64))
for i in range(64):
    for j in range(64):
        phi = np.arctan2(j, i)
        for k in range(label_resolution):
            if k * angle_step < phi <= (k+1) * angle_step:
                A[i,j] = k
plt.figure(figsize=(5, 5))
plt.imshow(A, cmap='gray')
plt.savefig('data/anglelabels'+res_str+'.pdf', bbox_inches='tight')

# plot distance labels
A = np.zeros((64, 64))
for i in range(64):
    for j in range(64):
        dis = np.sqrt(i**2 + j**2)
        for k in range(label_resolution):
            if k * distance_step < dis <= (k+1) * distance_step:
                A[i,j] = k
plt.figure(figsize=(5, 5))
plt.imshow(A, cmap='gray')
plt.savefig('data/distancelabels'+res_str+'.pdf', bbox_inches='tight')


# CREATE DATA SET WITH BIASED DISTANCE AND VARYING ANGLE
# size = 64
# num_thetas = 50
# offset = 0
# thetas = np.linspace(0 + offset, 2*np.pi, num_thetas)
# distances = np.arange(19, 20)  # distance 0 to 30
#
# position_samples = 1500
#
# num_samples = thetas.shape[0] * distances.shape[0] * position_samples
# blobs64radius = np.zeros((num_samples, size, size))
# i = 0
# for theta in thetas:
#     for dis in distances:
#         for s in range(position_samples):
#             # draw disk
#             disk = cv.circle(np.zeros((41, 41)), center=(20, 20), radius=10,
#                              color=1, thickness=-1)
#
#             # blur with gaussian filter
#             blob = cv.GaussianBlur(disk, (21, 21), 0)
#
#             x, y = polar_to_cartesian(theta, dis)
#
#             blobs64radius[i, :, :] = embed(blob, size, x + 31, y + 31)
#
#             i += 1
#
# print(blobs64radius.shape)
# np.savez_compressed('blobs_data/blobs64rad', blobs64radius)
#
# # DENSITY 64 RADIUS
# _, ax = plt.subplots()
# ax.imshow(blobs64radius.mean(axis=0), interpolation='nearest', cmap='gray')
# ax.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
# plt.savefig('blobs_data/density64rad')

# # CREATE DATA SET WITH BIASED DISTANCE AND VARYING ANGLE
# size = 64
# num_thetas = 50
# offset = 0.5
# thetas = np.linspace(0 + offset, 2*np.pi, num_thetas)
# distances = np.arange(19, 20)  # distance 0 to 30
#
# position_samples = 1500
#
# num_samples = thetas.shape[0] * distances.shape[0] * position_samples
# blobset64radius = np.zeros((num_samples, size, size))
# i = 0
# for theta in thetas:
#     for dis in distances:
#         for s in range(position_samples):
#             # draw disk
#             disk = cv.circle(np.zeros((41, 41)), center=(20, 20), radius=10,
#                              color=1, thickness=-1)
#
#             # blur with gaussian filter
#             blob = cv.GaussianBlur(disk, (21, 21), 0)
#
#             x, y = polar_to_cartesian(theta, dis)
#
#             blobset64radius[i, :, :] = embed(blob, size, x + 31, y + 31)
#
#             i += 1
#
# print(blobset64radius.shape)
# np.savez_compressed('blobs_data/blobs64radoff', blobset64radius)
#
# # DENSITY 64 RADIUS
# _, ax = plt.subplots()
# ax.imshow(blobset64radius.mean(axis=0), interpolation='nearest', cmap='gray')
# ax.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
# plt.savefig('blobs_data/density64radoff')
