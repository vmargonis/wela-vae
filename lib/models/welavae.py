from tensorflow.keras.models import Model

from lib.models.layers import (
    dense_decoder_with_labels,
    dense_encoder_with_labels,
    reparameterize,
)
from lib.models.losses import kl_divergence, total_correlation
from lib.models.utils import (
    get_dip_vae_regularizer,
    get_label_loss,
    get_reconstruction_loss,
)


class BaseWeLaVAE:
    """
    Base WeLaVAE architecture.
    """

    def __init__(self, config):
        (
            self._in,
            self._label_1_in,
            self._label_2_in,
            self.mean,
            self.log_var,
            self.encoder,
        ) = dense_encoder_with_labels(config)
        self._reparam = reparameterize(config)
        self.decoder = dense_decoder_with_labels(config)

        self._z = self._reparam(
            self.encoder([self._in, self._label_1_in, self._label_2_in])
        )
        self._out, self._label_1_out, self._label_2_out = self.decoder(self._z)
        self.vae = Model(
            [self._in, self._label_1_in, self._label_2_in],
            [self._out, self._label_1_out, self._label_2_out],
        )


class WeLaBetaVAE(BaseWeLaVAE):
    """
    Weak Label variant of Beta VAE.
    """

    def __init__(self, config):
        super().__init__(config)

        self.str_repr = (
            f"wela_betavae"
            f"_L{config['latent_dim']}"
            f"_labeldim{config['label_dim']}"
            f"_beta{config['beta']}"
            f"_gamma{config['gamma']}"
            f"_wseed{config['weight_seed']}"
        )

        self.vae.add_loss(self.wela_beta_vae_loss(config))

    def wela_beta_vae_loss(self, config):
        """
        WeLa BetaVAE Batch Loss:
        beta * KL divergence + reconstruction loss * gamma + label_loss
        """
        kl = kl_divergence(self.mean, self.log_var)
        reconstruction_loss = get_reconstruction_loss(self._in, self._out, config)

        label_loss = get_label_loss(
            self._label_1_in,
            self._label_1_out,
            self._label_2_in,
            self._label_2_out,
        )

        return config["beta"] * kl + reconstruction_loss + config["gamma"] * label_loss


class WeLaTCVAE(BaseWeLaVAE):
    """
    Weak Label variant of Total Correlation VAE.
    """

    def __init__(self, config):
        super().__init__(config)

        self.str_repr = (
            f"wela_tcvae"
            f"_L{config['latent_dim']}"
            f"_labeldim{config['label_dim']}"
            f"_beta{config['beta']}"
            f"_gamma{config['gamma']}"
            f"_wseed{config['weight_seed']}"
        )

        self.vae.add_loss(self.wela_tcvae_loss(config))

    def wela_tcvae_loss(self, config):
        """
        TCVAE Batch Loss:
        KL divergence + reconstruction loss + beta * total correlation
        + gamma * label loss.
        """
        kl = kl_divergence(self.mean, self.log_var)
        reconstruction_loss = get_reconstruction_loss(self._in, self._out, config)
        tc = total_correlation(self._z, self.mean, self.log_var)

        label_loss = get_label_loss(
            self._label_1_in,
            self._label_1_out,
            self._label_2_in,
            self._label_2_out,
        )

        return (
            kl
            + reconstruction_loss
            + config["beta"] * tc
            + config["gamma"] * label_loss
        )


class WeLaDIPVAE(BaseWeLaVAE):
    """
    Weak Label variant of Disentangled Inferred Prior VAE.
    """

    def __init__(self, config):
        super().__init__(config)

        self.str_repr = (
            f"wela_dipvae"
            f"_type{config['dip_vae_type']}"
            f"_L{config['latent_dim']}"
            f"_labeldim{config['label_dim']}"
            f"_loffdiag{config['lambda_off_diag']}"
            f"_ldiag{config['lambda_diag']}"
            f"_gamma{config['gamma']}"
            f"_wseed{config['weight_seed']}"
        )

        self.vae.add_loss(self.wela_dip_vae_loss(config))

    def wela_dip_vae_loss(self, config):
        """
        DIPVAE Batch Loss:
        KL divergence + reconstruction loss + DIP regularizer
        """
        kl = kl_divergence(self.mean, self.log_var)
        reconstruction_loss = get_reconstruction_loss(self._in, self._out, config)
        dip_regularizer = get_dip_vae_regularizer(self.mean, self.log_var, config)

        label_loss = get_label_loss(
            self._label_1_in,
            self._label_1_out,
            self._label_2_in,
            self._label_2_out,
        )

        return kl + reconstruction_loss + dip_regularizer + config["gamma"] * label_loss
