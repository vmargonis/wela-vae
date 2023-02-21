from keras.models import Model
from vae.vae_layers import reparameterize, stacked_encoder, stacked_decoder
from vae.losses import (
    kl_divergence,
    total_correlation,
    compute_covariance_z,
    compute_covariance_mean,
    dip_vae_regularizer,
)
from vae.utils import add_optimizer, get_reconstruction_loss


class BaseVAE:
    """Base VAE architecture"""
    def __init__(self, config):

        self._in, self.mean, self.log_var, self.encoder = stacked_encoder(
            config
        )
        self.reparam = reparameterize(config)
        self.decoder = stacked_decoder(config)
        self._z = self.reparam(self.encoder(self._in))
        self._out = self.decoder(self.reparam(self.encoder(self._in)))
        self.vae = Model(self._in, self._out)


class BetaVAE(BaseVAE):
    def __init__(self, config, params):

        super().__init__(config)

        # string representation
        self.str_repr = (
            f"betavae"
            f"_L{config['latent_dim']}"
            f"_beta{params['beta']}"
            f"_wseed{config['weight_seed']}"
        )

        # Add Loss
        self.vae.add_loss(self.beta_vae_loss(config, params))
        self.vae.compile(optimizer=add_optimizer(config))

    def beta_vae_loss(self, config, params):
        """BETAVae Batch Loss:
        beta * KL divergence + reconstruction loss
        """

        kl = kl_divergence(self.mean, self.log_var)

        reconstruction_loss = get_reconstruction_loss(
            self._in,
            self._out,
            config,
        )

        return params["beta"] * kl + reconstruction_loss


class TCVAE(BaseVAE):

    def __init__(self, config, params):

        super().__init__(config)

        # string representation
        self.str_repr = (
            f"tcvae"
            f"_L{config['latent_dim']}"
            f"_beta{params['beta']}"
            f"_wseed{config['weight_seed']}"
        )

        # Add Loss / Optimizer
        self.vae.add_loss(self.tcvae_loss(config, params))
        self.vae.compile(optimizer=add_optimizer(config))

    def tcvae_loss(self, config, params):
        """TCVAE Batch Loss:
        KL divergence + reconstruction loss + beta * total correlation
        """

        kl = kl_divergence(self.mean, self.log_var)

        reconstruction_loss = get_reconstruction_loss(
            self._in,
            self._out,
            config,
        )

        tc = total_correlation(self._z, self.mean, self.log_var)

        return kl + reconstruction_loss + (params["beta"]-1) * tc


class DIPVAE(BaseVAE):

    def __init__(self, config, params):

        super().__init__(config)

        # string representation
        self.str_repr = (
            f"dipvae"
            f"_type{params['dip_vae_type']}"
            f"_L{config['latent_dim']}"
            f"_loffdiag{params['lambda_off_diag']}"
            f"_ldiag{params['lambda_diag']}"
            f"_wseed{config['weight_seed']}"
        )

        # Add Loss / Optimizer
        self.vae.add_loss(self.dip_vae_loss(config, params))
        self.vae.compile(optimizer=add_optimizer(config))

    def dip_vae_loss(self, config, params):
        """DIPVAE Batch Loss:
        KL divergence + reconstruction loss + DIP regularizer
        """
        kl = kl_divergence(self.mean, self.log_var)

        reconstruction_loss = get_reconstruction_loss(
            self._in,
            self._out,
            config,
        )

        # DIP VAE REGULARIZATION
        if params["dip_vae_type"] == "i":
            cov_matrix = compute_covariance_mean(self.mean)
        elif params["dip_vae_type"] == "ii":
            cov_matrix = compute_covariance_z(self.mean, self.log_var)
        else:
            raise ValueError(
                f"Unknown DIPVAE type {params['dip_vae_type']}."
                "Choose either 'i' or 'ii'."
            )

        regularizer = dip_vae_regularizer(
            cov_matrix,
            params["lambda_off_diag"],
            params["lambda_diag"],
        )

        return kl + reconstruction_loss + regularizer
