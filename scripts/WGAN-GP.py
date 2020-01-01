# Import modules

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, RMSprop
import numpy as np
import librosa
import random
import os
import sys
import time

import GANModels

# Setup GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

# Define hyperparameters
MODEL_DIMS = 64
NUM_SAMPLES = 16384
D_UPDATES_PER_G_UPDATE = 5
GRADIENT_PENALTY_WEIGHT = 10.0
NOISE_LEN = 100
EPOCHS = 128
EPOCHS_PER_SAMPLE = 2
BATCH_SIZE = 16
Fs = 16000

DATA_DIR = r"D:\ML_Datasets\mancini_piano\piano\train"

# Define class that contains GAN infrastructure
class GAN:
    def __init__(self, model_dims=MODEL_DIMS, num_samples=NUM_SAMPLES, 
                 gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT,
                 noise_len=NOISE_LEN, batch_size=BATCH_SIZE, sr=Fs):
        self.model_dims = model_dims
        self.num_samples = num_samples
        self.noise_dims = (noise_len,)
        self.batch_size = batch_size
        
        self.G = GANModels.Generator(self.model_dims, num_samples)
        print(self.G.summary())

        self.D = GANModels.Critic(self.model_dims, num_samples)
        print(self.D.summary())
        
        self.G_optimizer = Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        self.D_optimizer = Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
        
        self.gradient_penalty_weight = gradient_penalty_weight
        
        self.sr = sr

    # Loss function for critic
    def _d_loss_fn(self, r_logit, f_logit):
        r_loss = - tf.reduce_mean(r_logit)
        f_loss = tf.reduce_mean(f_logit)
        return r_loss, f_loss
    
    # Loss function for generator
    def _g_loss_fn(self, f_logit):
        f_loss = - tf.reduce_mean(f_logit)
        return f_loss

    # Calculates gradient penalty
    def _gradient_penalty(self, real, fake):
        def _interpolate(a, b):
            shape = [tf.shape(a)[0]] + [1] * (a.shape.ndims - 1)
            alpha = tf.random.uniform(shape=shape, minval=0., maxval=1.)
            inter = a + alpha * (b - a)
            inter.set_shape(a.shape)
            return inter
            
        x = _interpolate(real, fake)
        with tf.GradientTape() as t:
            t.watch(x)
            pred = self.D(x, training=True)
            
        grad = t.gradient(pred, x)
        norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp
        
    # Trains generator by keeping critic constant
    @tf.function
    def train_G(self):
        with tf.GradientTape() as t:
            z = tf.random.normal(shape=(self.batch_size,) + self.noise_dims)
            x_fake = self.G(z, training=True)
            x_fake_d_logit = self.D(x_fake, training=True)
            G_loss = self._g_loss_fn(x_fake_d_logit)

        G_grad = t.gradient(G_loss, self.G.trainable_variables)
        self.G_optimizer.apply_gradients(zip(G_grad, self.G.trainable_variables))

        return {'g_loss': G_loss}

    # Trains critic by keeping generator constant
    @tf.function
    def train_D(self, x_real):
        with tf.GradientTape() as t:
            z = tf.random.normal(shape=(x_real.shape[0],) + self.noise_dims)
            x_fake = self.G(z, training=True)

            x_real_d_logit = self.D(x_real, training=True)
            x_fake_d_logit = self.D(x_fake, training=True)

            x_real_d_loss, x_fake_d_loss = self._d_loss_fn(x_real_d_logit, x_fake_d_logit)
            gp = self._gradient_penalty(x_real, x_fake)

            D_loss = (x_real_d_loss + x_fake_d_loss) + gp * self.gradient_penalty_weight

        D_grad = t.gradient(D_loss, self.D.trainable_variables)
        self.D_optimizer.apply_gradients(zip(D_grad, self.D.trainable_variables))

        return {'d_loss': x_real_d_loss + x_fake_d_loss, 'gp': gp}
        
    # Creates music samples and saves current generator model
    def sample(self, epoch, num_samples=10):
        self.G.save(f"models/{epoch}.h5")
        z = tf.random.normal(shape=(num_samples,) + self.noise_dims)
        result = self.G(z, training=False)
        for i in range(num_samples):
            audio = result[i, :, :]
            audio = np.reshape(audio, (self.num_samples,))
            librosa.output.write_wav(f"output/piano/{epoch}-{i}.wav", audio, sr=self.sr)

    
###############################################################################################

# Instantiate model
gan = GAN()

# Create training data
X_train = []
for file in os.listdir(DATA_DIR): ### Modify for your data directory
    with open(DATA_DIR + fr"\{file}", "rb") as f:
        samples, _ = librosa.load(f, Fs)
        # Pad short audio files to NUM_SAMPLES duration
        if len(samples) < NUM_SAMPLES:
            audio = np.array([np.array([sample]) for sample in samples])
            padding = np.zeros(shape=(NUM_SAMPLES - len(samples), 1), dtype='float32')
            X_train.append(np.append(audio, padding, axis=0))
        # Create slices of length NUM_SAMPLES from long audio
        else:
            p = len(samples) // (NUM_SAMPLES)
            for i in range(p - 1):
                sample = np.expand_dims(samples[i*NUM_SAMPLES:(i+1)*NUM_SAMPLES], axis=1)
                X_train.append(sample)

print(f"X_train shape = {(len(X_train),) + X_train[0].shape}")

# Save some random training data slices and create baseline generated data for comparison
for i in range(10):
    librosa.output.write_wav(f"output/piano/real-{i}.wav", 
                             X_train[random.randint(0, len(X_train) - 1)], sr=Fs)

gan.sample("fake")

train_summary_writer = tf.summary.create_file_writer("logs/train")
    
# Train GAN
with train_summary_writer.as_default():
    steps_per_epoch = len(X_train) // BATCH_SIZE

    for e in range(EPOCHS):
        for i in range(steps_per_epoch):
            D_loss_sum = 0
        
            # Update dcritic a set number of times for each update of the generator
            for n in range(D_UPDATES_PER_G_UPDATE):
                gan.D.reset_states()
                D_loss_dict = gan.train_D(np.array(random.sample(X_train, BATCH_SIZE)))
                D_loss_sum += D_loss_dict['d_loss']
            
            # Calculate average loss of critic for current step
            D_loss = D_loss_sum / D_UPDATES_PER_G_UPDATE
            
            G_loss_dict = gan.train_G()
            G_loss = G_loss_dict['g_loss']
        
            # Write logs
            tf.summary.scalar('d_loss', D_loss, step=(e*steps_per_epoch)+i)
            tf.summary.scalar('g_loss', G_loss, step=(e*steps_per_epoch)+i)
        
            print(f"step {(e*steps_per_epoch)+i}: d_loss = {D_loss} g_loss = {G_loss}")
        
        # Periodically sample generator
        if e % EPOCHS_PER_SAMPLE == 0:
            gan.sample(e)