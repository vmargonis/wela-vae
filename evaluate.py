import numpy as np
from evaluation.cartesian_metric import compute_metric_cartesian
from evaluation.polar_metric import compute_metric_polar

# LOAD GROUND TRUTH FACTORS
dataset_zip = np.load('blobs/data/blobs64_ground_truth.npz')
ground_truth = dataset_zip['arr_0']
print('ground_truth shape:', ground_truth.shape)

np.set_printoptions(precision=2)
# Parameters to retrieve representations:
model = 'tcvae'  # or dipvae_ii, betavae, tcvae
beta = 40  # for betaVAE and TCVAE
latent_dim = 5
res = 3
gamma = 1500
# res = 2 -> gamma = 2000
# res = 3 -> gamma = 1500
# res = 4 -> gamma = 1000
# res = 5 -> gamma = 800
# res = 6 -> gamma = 750
# res = 7 -> gamma = 600
# res = 8 -> gamma = 500

seeds = list(range(1, 51))
scores = []
for SEED in seeds:

    # string for DIP-VAE
    # st = '{}_L{}_lod{}_ld{}_seed{}'.format(model, latent_dim, lambda_od,
    #                                        lambda_d, SEED)
    # string for beta-VAE and TCVAE
    st = '{}_L{}_b{}_seed{}'.format(model, latent_dim, beta, SEED)

    # LABELS
    # st = '{}_L{}_lod{}_ld{}_g{}_seed{}'.format(model, latent_dim,
    #                                            lambda_od, lambda_d,
    #                                            gamma,
    #                                            SEED)
    # st = '{}_L{}_b{}_g{}_seed{}_res{}'.format(model, latent_dim, beta,
    #                                           gamma, SEED, res)

    # LOAD REPRESENTATIONS
    means_str = 'blobs/repres/means_{}.npz'.format(st)
    repre_zip = np.load(means_str)
    representations = repre_zip['arr_0']  # shape: (N, latent_dim)

    # for i in range(latent_dim):
    #     print(np.min(representations[:, i]), np.max(representations[:, i]))

    # COMPUTE SCORE
    # score = compute_metric_cartesian(ground_truth, representations)
    score = compute_metric_polar(ground_truth, representations)
    scores.append(score)


# REPORT SCORES
scores = np.array(scores)
sorted_id = np.argsort(scores)

top_five_scores = scores[sorted_id[:5]]
top_five_score_seeds = np.array(seeds)[sorted_id[:5]]
mean_top_five = np.mean(top_five_scores)

mean_all = np.mean(scores)

worst_score = scores[sorted_id[-1]]
worst_score_seed = np.array(seeds)[sorted_id[-1]]

print(scores)
print("Top five scores: {}".format(top_five_scores))
print("Top five score seeds: {}".format(top_five_score_seeds))
print("Mean top 5: {:.2f}". format(mean_top_five))
print("Mean overall: {:.2f}". format(mean_all))
print("Worst Score: {:.2f}, corresponding seed: {}".format(worst_score,
                                                           worst_score_seed))
