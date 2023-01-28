from keras import backend as K
from keras.models import Model
from keras.optimizers import Adagrad, RMSprop, Adam, SGD
from keras.layers import Dense, Input, Lambda, concatenate
from keras.initializers import glorot_uniform
# from vae.losses import *


def kl_loss(mu, log_var):
    kl = K.square(mu) + K.exp(log_var) - log_var - 1
    dkl = 0.5 * K.sum(kl, axis=-1)  # shape: (?,)
    return dkl


def multi_bernoulli(target, out):
    out_clipped = K.clip(out, 1e-7, 1 - 1e-7)
    a = target * K.log(out_clipped)
    b = (1 - target) * K.log(1 - out_clipped)
    xent = - K.sum(a + b, axis=-1)  # shape:(?,)
    return xent


def xent_lower_bound(target):
    dist = K.clip(target, 1e-7, 1 - 1e-7)
    xent_lb = -K.sum(dist * K.log(dist), axis=1)  # shape:(?,)
    return xent_lb


# Loss for bernoulli decoder
def bernoulli_loss(mu, log_var, evi, evi_out, beta, gamma, is_binary_input):

    # D_KL term
    dkl = kl_loss(mu, log_var)
    beta_dkl = beta * dkl

    # EVIDENCE RECONSTRUCTION - CROSS-ENTROPY
    evi_loss = gamma * K.categorical_crossentropy(evi, evi_out)  # shape:(?,)

    def custom_loss(vae_in, vae_out):

        # if input is not binary, lower bound in cross entropy is not zero
        if is_binary_input is True:
            xent_lb = 0
        else:
            xent_lb = xent_lower_bound(vae_in)  # shape:(?,)

        # cross-entropy
        xent = multi_bernoulli(vae_in, vae_out) - xent_lb  # shape:(?,)

        sample_loss = beta_dkl + xent + evi_loss   # shape:(?,)
        batch_loss = K.mean(sample_loss, axis=-1)  # shape:()

        return batch_loss

    return custom_loss


def gauss_loss(mu, log_var, evi, evi_out, beta, gamma):
    # D_KL term
    dkl = kl_loss(mu, log_var)
    beta_dkl = beta * dkl

    # EVIDENCE RECONSTRUCTION - CROSS-ENTROPY
    evi_loss = gamma * K.categorical_crossentropy(evi, evi_out)  # shape:(?,)

    def custom_loss(vae_in, vae_out):

        l2_loss = K.sum(K.square(vae_in - vae_out), axis=-1)  # shape: (?,)

        sample_loss = beta_dkl + l2_loss + evi_loss  # shape:(?,)
        batch_loss = K.mean(sample_loss, axis=-1)  # shape:()

        return batch_loss

    return custom_loss


class EviVae2:
    # Constructor / Build topology
    # Args: config: dictionary of betavae configuration
    def __init__(self, config, evidence):
        try:
            # glorot weight initializer
            w_init = glorot_uniform(seed=config['random_seed'])

            self._evi = evidence  # K.variable(evidence[:128, :])

            # Encoder
            self._in, self._mu, self._log_var, self.encoder \
                = self.build_encoder(w_init, config)

            # Reparameterizer
            self.reparam = self.reparameterize(config)

            # Decoder
            self.decoder = self.build_decoder(w_init, config)

            # EviVAE
            self._out, self._evi_out = self.decoder(
                self.reparam(self.encoder(self._in)))

            self.vae = Model(self._in, [self._out, self._evi_out])

            # Add optimizer
            lr = config['optimizer']['learning_rate']  # learning rate

            if config['optimizer']['type'] == 'Adagrad':
                opt = Adagrad(lr=lr)
            elif config['optimizer']['type'] == 'Adam':
                opt = Adam(lr=lr)
            elif config['optimizer']['type'] == 'RMSprop':
                opt = RMSprop(lr=lr)
            elif config['optimizer']['type'] == 'SGD':
                opt = SGD(lr=lr)
            else:
                raise NotImplementedError("Optimizer not supported.")

            if config['model_loss'] == 'bernoulli':
                # Compile with optimizer and bernoulli loss
                self.vae.compile(loss=bernoulli_loss(self._mu,
                                                     self._log_var,
                                                     self._evi,
                                                     self._evi_out,
                                                     config['beta'],
                                                     config['gamma'],
                                                     config['IsBinaryInput']),
                                 optimizer=opt)

            elif config['model_loss'] == 'gaussian':
                # Compile with optimizer and gaussian loss
                self.vae.compile(loss=gauss_loss(self._mu,
                                                 self._log_var,
                                                 self._evi,
                                                 self._evi_out,
                                                 config['beta'],
                                                 config['gamma']),
                                 optimizer=opt)
            else:
                raise NotImplementedError("Loss not supported.")

            # self.loss_func(config)

        except KeyError:
            raise KeyError

    # Build encoder
    @staticmethod
    def build_encoder(w_init, config):

        # input placeholder
        input_ = Input(shape=(config['initial_dim'],))
        # evidence = Input(shape=(config['evidence_dim'],))

        # hidden layer(s)
        num_layers = len(config['encoder']['units'])

        # First hidden is connected to input
        hidden = Dense(units=config['encoder']['units'][0],
                       activation=config['encoder']['activation'][0],
                       kernel_initializer=w_init)(input_)

        # rest of layers connected to each other
        for i in range(1, num_layers):
            hidden = Dense(units=config['encoder']['units'][i],
                           activation=config['encoder']['activation'][i],
                           kernel_initializer=w_init)(hidden)

        # Mu Layer (Latent representation)
        mu = Dense(units=config['latent_dim'],
                   activation=config['encoder']['output_activation'],
                   kernel_initializer=w_init,
                   name='mu_enc')(hidden)

        # Log var layer
        log_var = Dense(units=config['latent_dim'],
                        activation=config['encoder']['output_activation'],
                        kernel_initializer=w_init,
                        name='log_sigma_squared_enc')(hidden)

        # encoder definition
        encoder = Model(inputs=input_,
                        outputs=[mu, log_var],
                        name='encoder')

        return input_, mu, log_var, encoder

    # Reparameterization layer
    @staticmethod
    def reparameterize(config):
        mu_in = Input(shape=(config['latent_dim'],))
        log_var_in = Input(shape=(config['latent_dim'],))

        z = Lambda(reparam_trick,
                   output_shape=(config['latent_dim'],),
                   name='Z')([mu_in, log_var_in])

        reparam = Model(inputs=[mu_in, log_var_in],
                        outputs=z,
                        name='reparameterize')

        return reparam

    # Build decoder
    @staticmethod
    def build_decoder(w_init, config):
        # input placeholder
        z_in = Input(shape=(config['latent_dim'],))

        # hidden layer(s)
        num_layers = len(config['decoder']['units'])

        # First hidden is connected to input
        hidden = Dense(units=config['decoder']['units'][0],
                       activation=config['decoder']['activation'][0],
                       kernel_initializer=w_init)(z_in)

        # rest of layers connected to each other
        for i in range(1, num_layers):
            hidden = Dense(units=config['decoder']['units'][i],
                           activation=config['decoder']['activation'][i],
                           kernel_initializer=w_init)(hidden)

        # output
        x_out = Dense(units=config['initial_dim'],
                      activation=config['decoder']['output_activation'],
                      kernel_initializer=w_init,
                      name='x_tilde')(hidden)

        evi_out = Dense(units=config['evidence_dim'],
                        activation='softmax',
                        kernel_initializer=w_init,
                        name='evi_tilde')(hidden)

        # decoder definition
        decoder = Model(inputs=z_in, outputs=[x_out, evi_out], name='decoder')

        return decoder

############################################################
# EXAMPLE
############################################################


# vae_config = {'random_seed': SEED,
#               'initial_dim': initial_dim,
#               'evidence_dim': evidence_dim,
#               'latent_dim': latent_dim,
#               'beta': beta,
#               'gamma': gamma,
#               'IsBinaryInput': True,
#               'model_loss': 'bernoulli',
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
# ev = EviVae(vae_config)
