from keras.models import Model
import keras
from tensorflow.keras import layers, regularizers
import tensorflow as tf
from keras import Sequential
from keras.models import Model
from tensorflow_addons.layers import SpectralNormalization


class Inner_layer(layers.Layer):
    def __init__(self):
        super().__init__()
        self.dense = SpectralNormalization(layers.Dense(7*50, activation='relu'))
        self.conv1d_1 = SpectralNormalization(layers.Conv1D(45, kernel_size=2))
        self.rehape = layers.Reshape((7, 50), input_shape=(7*50,))
        # self.bn = layers.BatchNormalization()
        self.leakyrelu = layers.LeakyReLU(alpha=0.5)
        self.conv1d_2 = SpectralNormalization(layers.Conv1D(37, kernel_size=2, activation='relu'))
        self.lstm = layers.LSTM(37, return_sequences=True, activation='relu')
    def call(self, input):
        x = self.dense(input)
        x = self.rehape(x)
        x = self.conv1d_1(x)
        x = self.leakyrelu(x)
        # x = self.bn(x)
        x = self.conv1d_2(x)
        x = self.leakyrelu(x)
        # x = self.bn(x)
        return self.lstm(x)


def build_generator():
    input1 = layers.Input(shape=(100,))
    input2 = layers.Input(shape=(100,))

    output1 = Inner_layer()(input1)
    output2 = Inner_layer()(input2)

    output = tf.concat([output1, output2], 2)

    model = Model(inputs=[input1, input2], outputs=output)
    return model

def build_discriminator():
    model = Sequential([
        layers.Input(shape=(5, 74)),
        SpectralNormalization(layers.Conv1D(128, kernel_size=2)),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.05),
        layers.LSTM(32, return_sequences=True, activation='relu'),
        layers.Flatten(),
        SpectralNormalization(layers.Dense(64, activation='relu')),
        SpectralNormalization(layers.Dense(32, activation='relu')),
        layers.Dense(1, )
    ])
    return model

def g_and_d(g, d):
    input1 = layers.Input(shape=(100,))
    input2 = layers.Input(shape=(100,))
    x = g([input1, input2])
    y = d(x)
    d.trainable = False
    return Model([input1, input2], y)
    # model = Sequential()
    # model.add(g)
    # d.trainable = False
    # model.add(d)
    # return model



