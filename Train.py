import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
tf.compat.v1.disable_eager_execution()

allData = np.load('V5_data/V5_data.npy', allow_pickle=True)
# train_1 = np.load('pre_dataV4.npy', allow_pickle=True)

x_train = allData[0][0]
y_train = allData[0][1]
x_test = allData[0][2]
y_test = allData[0][3]
# x_predict = train_1[0]
print(x_train[0:3, :, :])
print(x_test.shape)
# print(y_train.shape)
# print(x_train[:3, :, :])


import sys
sys.exit()

'''from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X, y)
sorted(Counter(y_resampled).items())'''

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv1D(128, kernel_size=2, strides=1, activation='relu'))
# model.add(tf.keras.layers.MaxPool1D(pool_size=2,strides=1,padding='valid'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Conv1D(256, kernel_size=2, strides=1, activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.LSTM(32, return_sequences=True, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.GRU(64, return_sequences=True, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.GRU(64, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.25))
'''    
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Dense(64, activation='relu'))
'''
model.add(tf.keras.layers.Dense(32, activation='relu', kernel_regularizer='l1'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy']
              )

history = model.fit(x_train, y_train, batch_size=1024, epochs=100, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[TensorBoard(log_dir='./tmp/log')])


y = model.predict(x_predict)
print('预测结果（y>1为风险警示）')
y = list(y)
file = open('his.txt', 'w')
his = []
for i in y:
    i = list(i)
    a = str(i[0])
    his.append(round(float(a)))

print(len(his))
writer = pd.ExcelWriter('history_compileV4.xlsx')
(pd.DataFrame(his)).to_excel(writer, sheet_name='result')
writer.save()
writer.close()


model.summary()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
