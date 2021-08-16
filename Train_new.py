import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import Model
from tensorflow.keras.callbacks import TensorBoard
import Parameters
from tensorflow import keras
import split
tf.compat.v1.disable_eager_execution()




allData = np.load('V5_data/V5_data_4d.npy', allow_pickle=True)
x_train = allData[0][0]
y_train = allData[0][1]
x_test = allData[0][2]
y_test = allData[0][3]
# split
x_train_former, x_train_latter = split.split_data(x_train)
x_test_former, x_test_latter = split.split_data(x_test)


# print(x_train_former[5, :, :])
# print(x_train_latter[5, :, :])
#
# import sys
# sys.exit()



g = Model.build_generator()
d = Model.build_discriminator()
g_and_d = Model.g_and_d(g, d)
g_optim = keras.optimizers.Adam(learning_rate=Parameters.lr)
d_optim = keras.optimizers.Adam(learning_rate=Parameters.lr, beta_1=0, beta_2=0.9)
g.compile(optimizer='Adam', loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
d.compile(optimizer=d_optim, loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
d.trainable = False
g_and_d.compile(optimizer=g_optim, loss=keras.losses.BinaryCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
g.summary()
d.summary()

d_acc = []
d_loss = []
g_acc = []
g_loss = []
val_d_acc = []
val_d_loss = []
val_g_acc = []
val_g_loss = []

batch_size = Parameters.batch
batch_num = int(x_train.shape[0]/batch_size)
half_batch = int(batch_size/2)

def get_noise(size):
    return np.random.uniform(-1, 1, size=(size, 100))

def get_real_data(former, latter, index):
    real_data_former = former[index]
    real_data_latter = latter[index]
    real_data_concat = np.concatenate((real_data_former, real_data_latter), axis=2)
    return real_data_concat

# def label_smoothing(inputs, epsilon=0.1):
#     # K = inputs.get_shape().as_list()[-1]    # number of channels
#
#     for index, label in enumerate(inputs):
#
#         if label == 0:
#             inputs[index] =
#     return ((1-epsilon) * inputs) + (epsilon / K)

for epoch in range(Parameters.epoch):
    print('epoch {}/{}'.format(epoch, Parameters.epoch))
    print('=========================')
    # train D
    test_count = 0
    noise1 = get_noise(half_batch)
    noise2 = get_noise(half_batch)
    index_train = np.random.randint(0, x_train_former.shape[0], half_batch)
    index_test = np.random.randint(0, x_test_former.shape[0], half_batch)
    index_test_val = np.random.randint(0, x_test_former.shape[0], batch_size)
    train_real_data = get_real_data(x_train_former, x_train_latter, index_train)
    test_real_data = get_real_data(x_test_former, x_test_latter, index_test)
    fake_data = g.predict([noise1, noise2])
    X_fedIn_d = np.concatenate((train_real_data, fake_data), axis=0)
    test_fedIn_d = np.concatenate((test_real_data, fake_data), axis=0)
    Y_fedIn_d = np.array([1]*half_batch + [0]*half_batch)
    d_metrics = d.train_on_batch(X_fedIn_d, Y_fedIn_d)
    val_d_metrics = d.test_on_batch(test_fedIn_d, Y_fedIn_d)
    # train G
    noise3 = get_noise(batch_size)
    noise4 = get_noise(batch_size)
    valid_y = np.array([1]*batch_size)
    g_metrics = g_and_d.train_on_batch([noise3, noise4], valid_y)
    val_g_metrics = g.test_on_batch([noise3, noise4], get_real_data(x_test_former, x_test_latter, index_test_val))
    d_loss.append(d_metrics[0])
    d_acc.append(d_metrics[1])
    val_d_loss.append(val_d_metrics[0])
    val_d_acc.append(val_d_metrics[1])
    g_loss.append(g_metrics[0])
    g_acc.append(g_metrics[1])
    val_g_loss.append(val_g_metrics[0])
    val_g_acc.append(val_g_metrics[1])
    print('D_loss:{}'.format(d_metrics[0]))
    print('D_acc:{}'.format(d_metrics[1]))
    print('G_loss:{}'.format(g_metrics[0]))
    print('G_acc:{}'.format(g_metrics[1]), '\n')
    g.save_weights(filepath='Weights/g_weights')
    d.save_weights(filepath='Weights/d_weights')









plt.subplot(1, 2, 1)
plt.plot(g_acc, label='Generator Accuracy')
plt.plot(d_acc, label='Discriminator Accuracy')
plt.plot(val_g_acc, label='val_g_acc')
plt.plot(val_d_acc, label='val_d_acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(g_loss, label='Generator Loss')
plt.plot(d_loss, label='Discriminator Loss')
plt.plot(val_g_loss, label='val_g_loss')
plt.plot(val_d_loss, label='val_d_loss')
plt.title('Loss')
plt.legend()
plt.show()