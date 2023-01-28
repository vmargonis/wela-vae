from tensorflow.keras.models import Model

from lib.models.layers import dense_decoder, dense_encoder, reparameterize
from lib.models.losses import kl_divergence, total_correlation
from lib.models.utils import get_dip_vae_regularizer, get_reconstruction_loss


class BaseVAE:
    """
    Base VAE architecture.
    """

    def __init__(self, config):
        self._in, self.mean, self.log_var, self.encoder = dense_encoder(config)
        self._reparam = reparameterize(config)
        self.decoder = dense_decoder(config)
        self._z = self._reparam(self.encoder(self._in))
        self._out = self.decoder(self._z)
        self.vae = Model(self._in, self._out)


class BetaVAE(BaseVAE):
    """
    Beta VAE.
    """

    def __init__(self, config):
        super().__init__(config)

        self.str_repr = (
            f"betavae"
            f"_L{config['latent_dim']}"
            f"_beta{config['beta']}"
            f"_wseed{config['weight_seed']}"
        )

        self.vae.add_loss(self.beta_vae_loss(config))

    def beta_vae_loss(self, config):
        """
        BETAVae Batch Loss: beta * KL divergence + reconstruction loss.
        """
        kl = kl_divergence(self.mean, self.log_var)
        reconstruction_loss = get_reconstruction_loss(self._in, self._out, config)
        return config["beta"] * kl + reconstruction_loss


class TCVAE(BaseVAE):
    """
    Total Correlation VAE.
    """

    def __init__(self, config):
        super().__init__(config)

        self.str_repr = (
            f"tcvae"
            f"_L{config['latent_dim']}"
            f"_beta{config['beta']}"
            f"_wseed{config['weight_seed']}"
        )

        self.vae.add_loss(self.tcvae_loss(config))

    def tcvae_loss(self, config):
        """
        TCVAE Batch Loss:
        KL divergence + reconstruction loss + beta * total correlation
        """
        kl = kl_divergence(self.mean, self.log_var)
        reconstruction_loss = get_reconstruction_loss(self._in, self._out, config)
        tc = total_correlation(self._z, self.mean, self.log_var)

        return kl + reconstruction_loss + config["beta"] * tc


class DIPVAE(BaseVAE):
    """
    Disentangled Inferred Prior VAE.
    """

    def __init__(self, config):
        super().__init__(config)

        self.str_repr = (
            f"dipvae"
            f"_type{config['dip_vae_type']}"
            f"_L{config['latent_dim']}"
            f"_loffdiag{config['lambda_off_diag']}"
            f"_ldiag{config['lambda_diag']}"
            f"_wseed{config['weight_seed']}"
        )

        self.vae.add_loss(self.dip_vae_loss(config))

    def dip_vae_loss(self, config):
        """
        DIPVAE Batch Loss:
        KL divergence + reconstruction loss + DIP regularizer
        """
        kl = kl_divergence(self.mean, self.log_var)
        reconstruction_loss = get_reconstruction_loss(self._in, self._out, config)
        dip_regularizer = get_dip_vae_regularizer(self.mean, self.log_var, config)

        return kl + reconstruction_loss + dip_regularizer
