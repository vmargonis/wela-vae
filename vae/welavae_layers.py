from keras.models import Model
from keras.layers import Dense, Input, Lambda, concatenate
from vae.utils import reparam_trick


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


def stacked_encoder(w_init, config):

    # input placeholder
    input_ = Input(shape=(config['initial_dim'],))
    label_1 = Input(shape=(config['label_dim'],))
    label_2 = Input(shape=(config['label_dim'],))

    # concatenate
    combined_in = concatenate([input_, label_1, label_2])

    # hidden layer(s)
    num_layers = len(config['encoder']['units'])

    # First hidden is connected to input
    hidden = Dense(units=config['encoder']['units'][0],
                   activation=config['encoder']['activation'][0],
                   kernel_initializer=w_init)(combined_in)

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
    encoder = Model(inputs=[input_, label_1, label_2],
                    outputs=[mu, log_var],
                    name='encoder')

    return input_, label_1, label_2, mu, log_var, encoder


def stacked_decoder(w_init, config):
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

    label_out_1 = Dense(units=config['evidence_dim'],
                        activation='softmax',
                        kernel_initializer=w_init,
                        name='evi_tilde')(hidden)

    label_out_2 = Dense(units=config['evidence_dim'],
                        activation='softmax',
                        kernel_initializer=w_init,
                        name='evi_tilde2')(hidden)

    # decoder definition
    decoder = Model(inputs=z_in, outputs=[x_out, label_out_1, label_out_2],
                    name='decoder')

    return decoder
