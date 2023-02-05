from keras.models import Model
from keras.initializers.initializers_v2 import GlorotUniform
from vae.vae_layers import reparameterize, stacked_encoder, stacked_decoder
from vae.losses import *
from vae.utils import add_optimizer


class BetaVae:
    # Constructor / Build topology
    # Args: config: dictionary of betavae configuration
    def __init__(self, config, params):
        try:
            # glorot weight initializer
            w_init = GlorotUniform(seed=config['random_seed'])

            self._in, self.mean, self.log_var, self.encoder = stacked_encoder(
                w_init,
                config
            )
            self.reparam = reparameterize(config)
            self.decoder = stacked_decoder(w_init, config)
            self._out = self.decoder(self.reparam(self.encoder(self._in)))
            self.vae = Model(self._in, self._out)

            # ADD LOSS
            loss = self.beta_vae_loss(config, params)
            self.vae.add_loss(loss)

            # ADD OPTIMIZER
            opt = add_optimizer(config)
            self.vae.compile(optimizer=opt)

        except KeyError:
            raise KeyError

    def beta_vae_loss(self, config, params):
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

        batch_loss = params['beta'] * kl + recon_loss  # shape:()

        return batch_loss


class TCVae:
    # Constructor / Build topology
    # Args: config: dictionary of betavae configuration
    def __init__(self, config, params):
        try:
            # glorot weight initializer
            w_init = GlorotUniform(seed=config['random_seed'])

            self._in, self.mean, self.log_var, self.encoder = stacked_encoder(
                w_init,
                config,
            )

            self.reparam = reparameterize(config)
            self.decoder = stacked_decoder(w_init, config)

            self.z = self.reparam(self.encoder(self._in))
            self._out = self.decoder(self.z)
            self.vae = Model(self._in, self._out)

            # ADD LOSS
            loss = self.tcvae_loss(config, params)
            self.vae.add_loss(loss)

            # ADD OPTIMIZER
            opt = add_optimizer(config)
            self.vae.compile(optimizer=opt)

        except KeyError:
            raise KeyError

    def tcvae_loss(self, config, params):
        # KL DIVERGENCE
        kl = kl_divergence(self.mean, self.log_var)

        # RECONSTRUCTION LOSS
        if config['output_dist'] == 'bernoulli':
            recon_loss = bernoulli_loss(
                self._in,
                self._out,
                config['IsBinaryInput'],
            )
        elif config['output_dist'] == 'gaussian':
            recon_loss = gaussian_loss(self._in, self._out)
        else:
            raise NotImplementedError("Loss not supported.")

        # ELBO
        elbo = kl + recon_loss  # shape:()
        # TOTAL CORRELATION
        tc = total_correlation(self.z, self.mean, self.log_var)
        batch_loss = elbo + (params['beta']-1) * tc  # shape:()

        return batch_loss


class DIPVae:

    def __init__(self, config, params):
        try:
            # glorot weight initializer
            w_init = GlorotUniform(seed=config['random_seed'])

            self._in, self.mean, self.log_var, self.encoder = stacked_encoder(
                w_init,
                config,
            )
            self.reparam = reparameterize(config)
            self.decoder = stacked_decoder(w_init, config)
            self._out = self.decoder(self.reparam(self.encoder(self._in)))
            self.vae = Model(self._in, self._out)

            # ADD LOSS
            loss = self.dip_vae_loss(config, params)
            self.vae.add_loss(loss)

            # ADD OPTIMIZER
            opt = add_optimizer(config)
            self.vae.compile(optimizer=opt)

        except KeyError:
            raise KeyError

    def dip_vae_loss(self, config, params):
        # KL DIVERGENCE
        kl = kl_divergence(self.mean, self.log_var)

        # RECONSTRUCTION LOSS
        if config['output_dist'] == 'bernoulli':
            recon_loss = bernoulli_loss(
                self._in,
                self._out,
                config['IsBinaryInput']
            )
        elif config['output_dist'] == 'gaussian':
            recon_loss = gaussian_loss(self._in, self._out)
        else:
            raise NotImplementedError("Loss not supported.")

        # ELBO
        elbo = kl + recon_loss

        # DIP VAE REGULARIZATION
        if params['dip_vae_type'] == 'i':
            cov_matrix = compute_covariance_mean(self.mean)
        elif params['dip_vae_type'] == 'ii':
            cov_matrix = compute_covariance_z(self.mean, self.log_var)
        else:
            raise NotImplementedError("dip vae type not supported.")

        regularizer = dip_vae_regularizer(
            cov_matrix,
            params['lambda_od'],
            params['lambda_d']
        )

        batch_loss = elbo + regularizer  # shape:()
        return batch_loss

############################################################
# EXAMPLES
############################################################

# vae_config = {'random_seed': SEED,
#               'initial_dim': initial_dim,
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
# params = {'beta': beta}
# vae = BetaVae(vae_config, params)
#
# params = {'dip_vae_type': 'i',
#           'lambda_od': 100,
#           'lambda_d': 200}
# vae = DIPVae(vae_config, params)
