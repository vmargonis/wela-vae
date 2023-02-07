from keras.models import Model
from keras.layers import Dense, Input, Lambda
from vae.utils import reparameterization_trick


def reparameterize(config):

    means = Input(shape=(config['latent_dim'],))
    log_var = Input(shape=(config['latent_dim'],))

    z = Lambda(
        reparameterization_trick,
        output_shape=(config['latent_dim'],),
        name='Z'
    )([means, log_var])

    reparam = Model(
        inputs=[means, log_var],
        outputs=z,
        name='reparameterize'
    )

    return reparam


def stacked_encoder(w_init, config):

    # input placeholder
    input_ = Input(shape=(config['initial_dim'],))

    # hidden layer(s)
    num_layers = len(config['encoder']['units'])

    # First hidden is connected to input
    hidden = Dense(
        units=config['encoder']['units'][0],
        activation=config['encoder']['activation'][0],
        kernel_initializer=w_init
    )(input_)

    # rest of layers connected to each other
    for i in range(1, num_layers):
        hidden = Dense(
            units=config['encoder']['units'][i],
            activation=config['encoder']['activation'][i],
            kernel_initializer=w_init
        )(hidden)

    # Means Layer (Latent representation)
    mu = Dense(
        units=config['latent_dim'],
        activation=config['encoder']['output_activation'],
        kernel_initializer=w_init,
        name='mu_enc'
    )(hidden)

    # Log vars layer
    log_var = Dense(
        units=config['latent_dim'],
        activation=config['encoder']['output_activation'],
        kernel_initializer=w_init,
        name='log_sigma_squared_enc'
    )(hidden)

    # encoder definition
    encoder = Model(
        inputs=input_,
        outputs=[mu, log_var],
        name='encoder'
    )

    return input_, mu, log_var, encoder


def stacked_decoder(w_init, config):
    # input placeholder
    z_in = Input(shape=(config['latent_dim'],))

    # hidden layer(s)
    num_layers = len(config['decoder']['units'])

    # First hidden is connected to input
    hidden = Dense(
        units=config['decoder']['units'][0],
        activation=config['decoder']['activation'][0],
        kernel_initializer=w_init
    )(z_in)

    # rest of layers connected to each other
    for i in range(1, num_layers):
        hidden = Dense(units=config['decoder']['units'][i],
                       activation=config['decoder']['activation'][i],
                       kernel_initializer=w_init)(hidden)

    # output
    y_out = Dense(
        units=config['initial_dim'],
        activation=config['decoder']['output_activation'],
        kernel_initializer=w_init,
        name='y_out'
    )(hidden)

    # decoder definition
    decoder = Model(
        inputs=z_in,
        outputs=y_out,
        name='decoder'
    )

    return decoder
