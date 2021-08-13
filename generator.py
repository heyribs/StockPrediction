import Parameters
import Model
import numpy as np
import split

allData = np.load('V5_data/V5_data_4d.npy', allow_pickle=True)
x_train = allData[0][0]
y_train = allData[0][1]
x_test = allData[0][2]
y_test = allData[0][3]

x_train_former, x_train_latter = split.split_data(x_train)


g = Model.build_generator()
g.compile(optimizer='Adam', loss='MSE')
g.load_weights(filepath='Weights/g_weights')

noise1 = np.random.uniform(-1, 1, size=(Parameters.batch, 100))
noise2 = np.random.uniform(-1, 1, size=(Parameters.batch, 100))

generated_metrix = g.predict([noise1, noise2], verbose=1)

print('former:')
print(generated_metrix[5, :, :37], '\n')
print('latter:')
print(generated_metrix[5, :, 37:])

print('====================================')
print(x_train_former[5, :, :])