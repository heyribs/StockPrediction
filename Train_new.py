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



def get_noise():
    return np.random.uniform(-1, 1, size=(batch_size, 100))

allData = np.load('V5_data/V5_data_4d.npy', allow_pickle=True)
x_train = allData[0][0]
y_train = allData[0][1]
x_test = allData[0][2]
y_test = allData[0][3]
# split
x_train_former, x_train_latter = split.split_data(x_train)
x_test_former, x_test_latter = split.split_data(x_test)



print(x_train_latter.shape)
print(x_test_former.shape)




g = Model.build_generator()
d = Model.build_discriminator()
g_and_d = Model.g_and_d(g, d)
g_optim = keras.optimizers.Adam(learning_rate=Parameters.lr, beta_1=0, beta_2=0.9)
d_optim = keras.optimizers.Adam(learning_rate=Parameters.lr, beta_1=0, beta_2=0.9)
g.compile(optimizer='Adam', loss='mse', metrics=['accuracy'])
g_and_d.compile(optimizer=g_optim, loss='binary_crossentropy', metrics=['accuracy'])
d.trainable = True
d.compile(optimizer=d_optim, loss='binary_crossentropy', metrics=['accuracy'])
g.summary()
d.summary()

batch_size = Parameters.batch
batch_num = int(x_train.shape[0]/batch_size)

d_acc = []
d_loss = []
g_acc = []
g_loss = []

val_d_acc = []
val_d_loss = []
val_g_acc = []
val_g_loss = []



for epoch in range(Parameters.epoch):
    print('epoch {}/{}'.format(epoch, Parameters.epoch))
    print('=========================')
    test_count = 0
    for index in range(batch_num):
        noise1 = get_noise()
        noise2 = get_noise()
        real_data_former = x_train_former[index*batch_size:(index+1)*batch_size, :, :]
        real_data_latter = x_train_latter[index*batch_size:(index+1)*batch_size, :, :]
        fake_data1 = g.predict([noise1, noise2])
        fake_data2 = g.predict([noise2, noise1])
        real_data = np.concatenate((real_data_former, real_data_latter), axis=2)
        X_fedIn_d = np.concatenate((real_data, fake_data1))
        Y_fedIn_d = [1] * batch_size + [0] * batch_size
        d_metrics = d.train_on_batch(X_fedIn_d, np.array(Y_fedIn_d))
        # print(len(d_metrics))
        d_loss.append(d_metrics[0])
        d_acc.append(d_metrics[1])
        print("bach %d d_loss %f" % (index, d_metrics[0]))
        print("bach %d d_acc %f" % (index, d_metrics[1]), '\n')
        d.trainable = False
        noise3 = get_noise()
        noise4 = get_noise()
        g_metrics = g_and_d.train_on_batch([noise3, noise4], np.array([1] * batch_size))
        print("bach %d g_loss %f" % (index, g_metrics[0]))
        print("bach %d g_acc %f" % (index, g_metrics[1]), '\n')
        g_loss.append(g_metrics[0])
        g_acc.append(g_metrics[1])
        if index % 3 == 0:
            test_data_former = x_test_former[test_count*batch_size:(test_count+1)*batch_size, :, :]
            test_data_latter = x_test_latter[test_count*batch_size:(test_count+1)*batch_size, :, :]
            test_count += 1
            test_data = np.concatenate((test_data_former, test_data_latter), axis=2)
            valX_fedIn_d = np.concatenate((test_data, fake_data2))
            val_d_metrics = d.test_on_batch(valX_fedIn_d, np.array(Y_fedIn_d))
            val_d_loss.append(val_d_metrics[0])
            val_d_acc.append(val_d_metrics[1])
            val_g_metrics = g_and_d.test_on_batch([noise2, noise1], np.array([1] * batch_size))
            val_g_loss.append(val_g_metrics[0])
            val_g_acc.append(val_g_metrics[1])
        d.trainable = True
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