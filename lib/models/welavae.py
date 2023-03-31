from keras.models import Model
from keras.initializers import glorot_uniform
from lib.models.losses import *
from legacy.welavae_layers import stacked_encoder, stacked_decoder, reparameterize
from lib.models.utils import add_optimizer


class WeLaBetaVae:
    # Constructor / Build topology
    # Args: config: dictionary of betavae configuration
    def __init__(self, config, params):
        try:
            # glorot weight initializer
            w_init = glorot_uniform(seed=config['random_seed'])

            # Encoder
            self._in, self._label_1, self._label_2, self.mean, self.log_var, \
                self.encoder = stacked_encoder(w_init, config)
            self.reparam = reparameterize(config)
            self.decoder = stacked_decoder(w_init, config)

            # WeLa-VAE
            self.z = self.reparam(self.encoder(
                [self._in, self._label_1, self._label_2]))
            self._out, self._label_out_1, self._label_out_2 = self.decoder(
                self.z)

            self.welavae = Model([self._in, self._label_1, self._label_2],
                                 [self._out, self._label_out_1,
                                  self._label_out_2])

            # ADD LOSS
            loss = self.wela_betavae_loss(config, params)
            self.welavae.add_loss(loss)

            # ADD OPTIMIZER
            opt = add_optimizer(config)
            self.welavae.compile(optimizer=opt)

        except KeyError:
            raise KeyError

    def wela_betavae_loss(self, config, params):
        # KL DIVERGENCE
        kl = kl_divergence(self.mean, self.log_var)

        # RECONSTRUCTION LOSS
        if config['output_dist'] == 'bernoulli':
            recon_loss = bernoulli_loss(self._in, self._out,
                                        config['IsBinaryInput'])
        elif config['output_dist'] == 'gaussian':
            recon_loss = gaussian_loss(self._in, self._out)
        else:
            raise NotImplementedError("Loss not supported.")

        # LABEL RECONSTRUCTION - CROSS-ENTROPY
        label_1_loss = K.categorical_crossentropy(self._label_1,
                                                  self._label_out_1)
        label_2_loss = K.categorical_crossentropy(self._label_2,
                                                  self._label_out_2)
        label_loss = K.mean(label_1_loss + label_2_loss, axis=-1)

        # ELBO
        beta_elbo = params['beta']*kl + recon_loss  # shape:()
        # TOTAL LOSS, shape:()
        batch_loss = beta_elbo + params['gamma']*label_loss

        return batch_loss


class WeLaTCVae:
    # Constructor / Build topology
    # Args: config: dictionary of betavae configuration
    def __init__(self, config, params):
        try:
            # glorot weight initializer
            w_init = glorot_uniform(seed=config['random_seed'])

            # Encoder
            self._in, self._label_1, self._label_2, self.mean, self.log_var, \
                self.encoder = stacked_encoder(w_init, config)
            self.reparam = reparameterize(config)
            self.decoder = stacked_decoder(w_init, config)

            # WeLa-VAE
            self.z = self.reparam(self.encoder(
                [self._in, self._label_1, self._label_2]))
            self._out, self._label_out_1, self._label_out_2 = self.decoder(
                self.z)

            self.welavae = Model([self._in, self._label_1, self._label_2],
                                 [self._out, self._label_out_1,
                                  self._label_out_2])

            # ADD LOSS
            loss = self.wela_tcvae_loss(config, params)
            self.welavae.add_loss(loss)

            # ADD OPTIMIZER
            opt = add_optimizer(config)
            self.welavae.compile(optimizer=opt)

        except KeyError:
            raise KeyError

    def wela_tcvae_loss(self, config, params):
        # KL DIVERGENCE
        kl = kl_divergence(self.mean, self.log_var)

        # RECONSTRUCTION LOSS
        if config['output_dist'] == 'bernoulli':
            recon_loss = bernoulli_loss(self._in, self._out,
                                        config['IsBinaryInput'])
        elif config['output_dist'] == 'gaussian':
            recon_loss = gaussian_loss(self._in, self._out)
        else:
            raise NotImplementedError("Loss not supported.")

        # ELBO
        elbo = kl + recon_loss  # shape:()

        # TOTAL CORRELATION
        tc = total_correlation(self.z, self.mean, self.log_var)

        # LABEL RECONSTRUCTION - CROSS-ENTROPY
        label_1_loss = K.categorical_crossentropy(self._label_1,
                                                  self._label_out_1)
        label_2_loss = K.categorical_crossentropy(self._label_2,
                                                  self._label_out_2)
        label_loss = K.mean(label_1_loss + label_2_loss, axis=-1)

        # TOTAL LOSS, shape:()
        batch_loss = elbo + (params['beta']-1)*tc + params['gamma']*label_loss

        return batch_loss


class WeLaDIPVae:
    # Constructor / Build topology
    # Args: config: dictionary of betavae configuration
    def __init__(self, config, params):
        try:
            # glorot weight initializer
            w_init = glorot_uniform(seed=config['random_seed'])

            # Encoder
            self._in, self._label_1, self._label_2, self.mean, self.log_var, \
                self.encoder = stacked_encoder(w_init, config)
            self.reparam = reparameterize(config)
            self.decoder = stacked_decoder(w_init, config)

            # WeLa-VAE
            self.z = self.reparam(self.encoder(
                [self._in, self._label_1, self._label_2]))
            self._out, self._label_out_1, self._label_out_2 = self.decoder(
                self.z)

            self.welavae = Model([self._in, self._label_1, self._label_2],
                                 [self._out, self._label_out_1,
                                  self._label_out_2])

            # ADD LOSS
            loss = self.wela_dipvae_loss(config, params)
            self.welavae.add_loss(loss)

            # ADD OPTIMIZER
            opt = add_optimizer(config)
            self.welavae.compile(optimizer=opt)

        except KeyError:
            raise KeyError

    def wela_dipvae_loss(self, config, params):
        # KL DIVERGENCE
        kl = kl_divergence(self.mean, self.log_var)

        # RECONSTRUCTION LOSS
        if config['output_dist'] == 'bernoulli':
            recon_loss = bernoulli_loss(self._in, self._out,
                                        config['IsBinaryInput'])
        elif config['output_dist'] == 'gaussian':
            recon_loss = gaussian_loss(self._in, self._out)
        else:
            raise NotImplementedError("Loss not supported.")

        # ELBO
        elbo = kl + recon_loss  # shape:()

        # DIP VAE REGULARIZATION
        if params['dip_vae_type'] == 'i':
            cov_matrix = compute_covariance_mean(self.mean)
        elif params['dip_vae_type'] == 'ii':
            cov_matrix = compute_covariance_z(self.mean, self.log_var)
        else:
            raise NotImplementedError("dip lib type not supported.")

        regularizer = dip_vae_regularizer(cov_matrix,
                                          params['lambda_od'],
                                          params['lambda_d'])

        # LABEL RECONSTRUCTION - CROSS-ENTROPY
        label_1_loss = K.categorical_crossentropy(self._label_1,
                                                  self._label_out_1)
        label_2_loss = K.categorical_crossentropy(self._label_2,
                                                  self._label_out_2)
        label_loss = K.mean(label_1_loss + label_2_loss, axis=-1)

        # TOTAL LOSS, shape:()
        batch_loss = elbo + regularizer + params['gamma'] * label_loss

        return batch_loss

############################################################
# EXAMPLE
############################################################

# vae_config = {'random_seed': SEED,
#               'initial_dim': initial_dim,
#               'evidence_dim': evidence_dim,
#               'latent_dim': latent_dim,
#               'IsBinaryInput': True,
#               'output_dist': 'bernoulli',
#               'encoder': {'units': [1200, 1200],
#                           'activation': ['tanh', 'tanh'],
#                           'output_activation': 'linear'},
#               'decoder': {'units': [1200, 1200],
#                           'activation': ['tanh', 'tanh'],
#                           'output_activation': 'sigmoid'},
#               'optimizer': {'type': 'Adam',
#                             'learning_rate': 0.001}
#                }
#
# params = {'beta': beta,
#           'gamma': gamma}
# bv = BetaVae(vae_config, params)
#
# params = {'dip_vae_type': 'i',
#           'lambda_od': 100,
#           'lambda_d': 200,
#           'gamma': gamma}
# dv = DIPVae(vae_config, params)
