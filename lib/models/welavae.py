from keras.models import Model
from lib.models.layers import reparameterize, dense_encoder_with_labels, dense_decoder
from lib.models.losses import kl_divergence, total_correlation
from lib.models.utils import (
    add_optimizer,
    get_reconstruction_loss,
    get_label_loss,
    get_dip_vae_regularizer,
)


class BaseWeLaVAE:
    """Base WeLaVAE architecture."""

    def __init__(self, config):
        (
            self._in,
            self._label_1_in,
            self._label_2_in,
            self.mean,
            self.log_var,
            self.encoder,
        ) = dense_encoder_with_labels(config)
        self.reparam = reparameterize(config)
        self.decoder = dense_decoder(config)

        self._z = self.reparam(
            self.encoder([self._in, self._label_1_in, self._label_2_in])
        )
        self._out, self._label_1_out, self._label_2_out = self.decoder(self._z)
        self.welavae = Model(
            [self._in, self._label_1_in, self._label_2_in],
            [self._out, self._label_1_out, self._label_2_out],
        )


class WeLaBetaVAE(BaseWeLaVAE):
    """Weak Label variant of Beta VAE."""

    def __init__(self, config, params):
        super().__init__(config)

        # string representation
        self.str_repr = (
            f"wela_betavae"
            f"_L{config['latent_dim']}"
            f"_labeldim{config['label_dim']}"
            f"_beta{params['beta']}"
            f"_wseed{config['weight_seed']}"
        )

        # Add Loss
        self.welavae.add_loss(self.wela_beta_vae_loss(config, params))
        self.welavae.compile(optimizer=add_optimizer(config))

    def wela_beta_vae_loss(self, config, params):
        """WeLa BetaVAE Batch Loss:
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

        return params["beta"] * kl + reconstruction_loss + params["gamma"] * label_loss


class WeLaTCVAE(BaseWeLaVAE):
    """Weak Label variant of Total Correlation VAE."""

    def __init__(self, config, params):
        super().__init__(config)

        # string representation
        self.str_repr = (
            f"wela_tcvae"
            f"_L{config['latent_dim']}"
            f"_labeldim{config['label_dim']}"
            f"_beta{params['beta']}"
            f"_wseed{config['weight_seed']}"
        )

        # Add Loss / Optimizer
        self.welavae.add_loss(self.wela_tcvae_loss(config, params))
        self.welavae.compile(optimizer=add_optimizer(config))

    def wela_tcvae_loss(self, config, params):
        """TCVAE Batch Loss:
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
            + (params["beta"] - 1) * tc
            + params["gamma"] * label_loss
        )


class WeLaDIPVAE(BaseWeLaVAE):
    """Weak Label variant of Disentangled Inferred Prior VAE."""

    def __init__(self, config, params):
        super().__init__(config)

        # string representation
        self.str_repr = (
            f"wela_dipvae"
            f"_type{params['dip_vae_type']}"
            f"_L{config['latent_dim']}"
            f"_labeldim{config['label_dim']}"
            f"_loffdiag{params['lambda_off_diag']}"
            f"_ldiag{params['lambda_diag']}"
            f"_wseed{config['weight_seed']}"
        )

        # Add Loss / Optimizer
        self.welavae.add_loss(self.wela_dip_vae_loss(config, params))
        self.welavae.compile(optimizer=add_optimizer(config))

    def wela_dip_vae_loss(self, config, params):
        """DIPVAE Batch Loss:
        KL divergence + reconstruction loss + DIP regularizer
        """
        kl = kl_divergence(self.mean, self.log_var)
        reconstruction_loss = get_reconstruction_loss(self._in, self._out, config)
        dip_regularizer = get_dip_vae_regularizer(self.mean, self.log_var, params)

        label_loss = get_label_loss(
            self._label_1_in,
            self._label_1_out,
            self._label_2_in,
            self._label_2_out,
        )

        return kl + reconstruction_loss + dip_regularizer + params["gamma"] * label_loss
