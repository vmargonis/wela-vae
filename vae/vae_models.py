from keras.models import Model
from keras.initializers.initializers_v2 import GlorotUniform
from vae.vae_layers import reparameterize, stacked_encoder, stacked_decoder
from vae.losses import *
from vae.utils import add_optimizer, get_reconstruction_loss


class BetaVAE:
    def __init__(self, config, params):
        try:

            # string representation
            self.str_repr = (
                f"betavae"
                f"_L{config['latent_dim']}"
                f"_beta{params['beta']}"
                f"_wseed{config['weight_seed']}"
            )

            # glorot weight initializer
            w_init = GlorotUniform(seed=config['weight_seed'])

            self._in, self.mean, self.log_var, self.encoder = stacked_encoder(
                w_init,
                config
            )
            self.reparam = reparameterize(config)
            self.decoder = stacked_decoder(w_init, config)
            self._out = self.decoder(self.reparam(self.encoder(self._in)))
            self.vae = Model(self._in, self._out)

            # Add Loss / Optimizer
            self.vae.add_loss(self.beta_vae_loss(config, params))
            self.vae.compile(optimizer=add_optimizer(config))

        except KeyError:
            raise KeyError

    def beta_vae_loss(self, config, params):
        """BETAVae Batch Loss:
        beta * KL divergence + reconstruction loss
        """

        kl = kl_divergence(self.mean, self.log_var)

        reconstruction_loss = get_reconstruction_loss(
            self._in,
            self._out,
            config
        )

        return params["beta"] * kl + reconstruction_loss


class TCVAE:

    def __init__(self, config, params):
        try:

            # string representation
            self.str_repr = (
                f"tcvae"
                f"_L{config['latent_dim']}"
                f"_beta{params['beta']}"
                f"_wseed{config['weight_seed']}"
            )

            # glorot weight initializer
            w_init = GlorotUniform(seed=config['weight_seed'])

            self._in, self.mean, self.log_var, self.encoder = stacked_encoder(
                w_init,
                config,
            )

            self.reparam = reparameterize(config)
            self.decoder = stacked_decoder(w_init, config)

            self.z = self.reparam(self.encoder(self._in))
            self._out = self.decoder(self.z)
            self.vae = Model(self._in, self._out)

            # Add Loss / Optimizer
            self.vae.add_loss(self.tcvae_loss(config, params))
            self.vae.compile(optimizer=add_optimizer(config))

        except KeyError:
            raise KeyError

    def tcvae_loss(self, config, params):
        """TCVAE Batch Loss:
        KL divergence + reconstruction loss + beta * total correlation
        """

        kl = kl_divergence(self.mean, self.log_var)

        reconstruction_loss = get_reconstruction_loss(
            self._in,
            self._out,
            config
        )

        tc = total_correlation(self.z, self.mean, self.log_var)

        return kl + reconstruction_loss + (params["beta"]-1) * tc


class DIPVAE:

    def __init__(self, config, params):
        try:

            # string representation
            self.str_repr = (
                f"dipvae"
                f"_type{params['dip_vae_type']}"
                f"_L{config['latent_dim']}"
                f"_loffdiag{params['lambda_off_diag']}"
                f"_ldiag{params['lambda_diag']}"
                f"_wseed{config['weight_seed']}"
            )

            # glorot weight initializer
            w_init = GlorotUniform(seed=config['weight_seed'])

            self._in, self.mean, self.log_var, self.encoder = stacked_encoder(
                w_init,
                config,
            )
            self.reparam = reparameterize(config)
            self.decoder = stacked_decoder(w_init, config)
            self._out = self.decoder(self.reparam(self.encoder(self._in)))
            self.vae = Model(self._in, self._out)

            # Add Loss / Optimizer
            self.vae.add_loss(self.dip_vae_loss(config, params))
            self.vae.compile(optimizer=add_optimizer(config))

        except KeyError:
            raise KeyError

    def dip_vae_loss(self, config, params):
        """DIPVAE Batch Loss:
        KL divergence + reconstruction loss + DIP regularizer
        """
        kl = kl_divergence(self.mean, self.log_var)

        reconstruction_loss = get_reconstruction_loss(
            self._in,
            self._out,
            config
        )

        # DIP VAE REGULARIZATION
        if params["dip_vae_type"] == "i":
            cov_matrix = compute_covariance_mean(self.mean)

        elif params["dip_vae_type"] == "ii":
            cov_matrix = compute_covariance_z(self.mean, self.log_var)

        else:
            raise NotImplementedError("dip vae type not supported.")

        regularizer = dip_vae_regularizer(
            cov_matrix,
            params["lambda_off_diag"],
            params["lambda_diag"],
        )

        return kl + reconstruction_loss + regularizer
