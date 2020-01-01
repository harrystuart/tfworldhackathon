# Import modules
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda, Dense, LSTM, Activation, Input, Bidirectional, Dropout
from tensorflow.keras.layers import Reshape, Conv2DTranspose, TimeDistributed, Conv1D, LeakyReLU, Layer, ReLU
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

def Generator(d, num_samples, c=16):

    input_layer = Input(shape=(100,))

    # output shape = (None, 16, 16d)
    dense_layer0 = Dense(16*c*d)(input_layer)
    reshape_layer0 = Reshape((c, c*d))(dense_layer0)
    relu_layer0 = ReLU()(reshape_layer0)

    # Upsampling
    # output shape = (None, 64, 8d)
    c //= 2
    expanded_layer0 = Lambda(lambda x: K.expand_dims(x, axis=1))(relu_layer0)
    conv1d_t_layer0 = Conv2DTranspose(c*d, (1, 25), strides=(1, 4), padding='same')(expanded_layer0)
    slice_layer0 = Lambda(lambda x: x[:, 0])(conv1d_t_layer0)
    relu_layer1 = ReLU()(slice_layer0)

    # output shape = (None, 256, 4d)
    c //= 2
    expanded_layer1 = Lambda(lambda x: K.expand_dims(x, axis=1))(relu_layer1)
    conv1d_t_layer1 = Conv2DTranspose(c*d, (1, 25), strides=(1, 4), padding='same')(expanded_layer1)
    slice_layer1 = Lambda(lambda x: x[:, 0])(conv1d_t_layer1)
    relu_layer2 = ReLU()(slice_layer1)

    # output shape = (None, 1024, 2d)
    c //= 2
    expanded_layer2 = Lambda(lambda x: K.expand_dims(x, axis=1))(relu_layer2)
    conv1d_t_layer2 = Conv2DTranspose(c*d, (1, 25), strides=(1, 4), padding='same')(expanded_layer2)
    slice_layer2 = Lambda(lambda x: x[:, 0])(conv1d_t_layer2)
    relu_layer3 = ReLU()(slice_layer2)

    # output shape = (None, 4096, d)
    c //= 2
    expanded_layer3 = Lambda(lambda x: K.expand_dims(x, axis=1))(relu_layer3)
    conv1d_t_layer3 = Conv2DTranspose(c*d, (1, 25), strides=(1, 4), padding='same')(expanded_layer3)
    slice_layer3 = Lambda(lambda x: x[:, 0])(conv1d_t_layer3)
    relu_layer4 = ReLU()(slice_layer3)

    # output shape = (None, 16384, 1)
    expanded_layer4 = Lambda(lambda x: K.expand_dims(x, axis=1))(relu_layer4)
    conv1d_t_layer4 = Conv2DTranspose(1, (1, 25), strides=(1, 4), padding='same')(expanded_layer4)
    slice_layer4 = Lambda(lambda x: x[:, 0])(conv1d_t_layer4)

    #### The number of transposed convolution operations  should be modified
    #### in accordance with num_samples. This current architecture expects
    #### num_samples == 16384

    # Squeeze values between (-1, 1)
    tanh_layer0 = Activation('tanh')(slice_layer4)

    model = Model(inputs=input_layer, outputs=tanh_layer0)

    return model

# Makes critic invariant to upsampling artifacts of generator to avoid the critic learning to
# easily identify generated audio from said artifacts
def _apply_phaseshuffle(x, rad=2, pad_type='reflect'):
    b, x_len, nch = x.get_shape().as_list()

    phase = tf.random.uniform([], minval=-rad, maxval=rad + 1, dtype=tf.int32)
    pad_l = tf.maximum(phase, 0)
    pad_r = tf.maximum(-phase, 0)
    phase_start = pad_r
    x = tf.pad(x, [[0, 0], [pad_l, pad_r], [0, 0]], mode=pad_type)

    x = x[:, phase_start:phase_start+x_len]
    x.set_shape([b, x_len, nch])

    return x

def Critic(d, num_samples, c=1):

    input_layer = Input(shape=(num_samples, 1))

    # Downsampling
    # output shape = (None, 4096, d)
    conv1d_layer0 = Conv1D(c*d, 25, strides=4, padding='same')(input_layer)
    LReLU_layer0 = LeakyReLU(alpha=0.2)(conv1d_layer0)
    phaseshuffle_layer0 = Lambda(lambda x: _apply_phaseshuffle(x))(LReLU_layer0)

    # output shape = (None, 1024, 2d)
    c *= 2
    conv1d_layer1 = Conv1D(c*d, 25, strides=4, padding='same')(phaseshuffle_layer0)
    LReLU_layer1 = LeakyReLU(alpha=0.2)(conv1d_layer1)
    phaseshuffle_layer1 = Lambda(lambda x: _apply_phaseshuffle(x))(LReLU_layer1)

    # output shape = (None, 256, 4d)
    c *= 2
    conv1d_layer2 = Conv1D(c*d, 25, strides=4, padding='same')(phaseshuffle_layer1)
    LReLU_layer2 = LeakyReLU(alpha=0.2)(conv1d_layer2)
    phaseshuffle_layer2 = Lambda(lambda x: _apply_phaseshuffle(x))(LReLU_layer2)

    # output shape = (None, 64, 8d)
    c *= 2
    conv1d_layer3 = Conv1D(c*d, 25, strides=4, padding='same')(phaseshuffle_layer2)
    LReLU_layer3 = LeakyReLU(alpha=0.2)(conv1d_layer3)
    phaseshuffle_layer3 = Lambda(lambda x: _apply_phaseshuffle(x))(LReLU_layer3)

    # output shape = (None, 16, 16d)
    c *= 2
    conv1d_layer4 = Conv1D(c*d, 25, strides=4, padding='same')(phaseshuffle_layer3)
    LReLU_layer4 = LeakyReLU(alpha=0.2)(conv1d_layer4)

    #### The number of convolution operations  should be modified
    #### in accordance with num_samples. This current architecture expects
    #### num_samples == 16384

    # output shape = (None, 256d)
    reshape_layer0 = Reshape((16*c*d,))(LReLU_layer4)#

    # Output a critic score
    dense_layer1 = Dense(1)(reshape_layer0)

    model = Model(inputs=input_layer, outputs=dense_layer1)

    return model