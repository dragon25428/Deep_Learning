import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import EarlyStopping

def unpickle(file):
    with open(file, 'rb') as fo:
        dicts = pickle.load(fo, encoding='bytes')
    return dicts

# load data
data_batch_1 = unpickle(file='cifar-10-batches-py/data_batch_1')
data_batch_2 = unpickle(file='cifar-10-batches-py/data_batch_2')
data_batch_3 = unpickle(file='cifar-10-batches-py/data_batch_3')
data_batch_4 = unpickle(file='cifar-10-batches-py/data_batch_4')
data_batch_5 = unpickle(file='cifar-10-batches-py/data_batch_5')
test_batch = unpickle(file='cifar-10-batches-py/test_batch')
all_image = np.vstack([data_batch_1[b'data'], data_batch_2[b'data'], data_batch_3[b'data'], data_batch_4[b'data'], data_batch_5[b'data']])
labels = np.hstack([data_batch_1[b'labels'], data_batch_2[b'labels'], data_batch_3[b'labels'], data_batch_4[b'labels'], data_batch_5[b'labels']])

# convert data array into the correct shape
X = np.zeros(shape=(50000, 32, 32, 3), dtype='uint8')
x_test = np.zeros(shape=(10000, 32, 32, 3), dtype='uint8')
for i in range(50000):
    X[i] = all_image[i].reshape(-1, 32, 32).transpose(1, 2, 0)
for i in range(10000):
    x_test[i] = test_batch[b'data'][i].reshape(-1, 32, 32).transpose(1, 2, 0)
Y = keras.utils.to_categorical(labels)
y_test = keras.utils.to_categorical(test_batch[b'labels'])

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.3)

# create Sequential model
model = Sequential()
model.add(layer=Conv2D(filters=32, kernel_size=(2, 2), activation='relu', input_shape=x_train.shape[1:]))
model.add(layer=Conv2D(filters=32, kernel_size=(2, 2), activation='relu'))
model.add(layer=MaxPool2D(pool_size=(2, 2)))
model.add(layer=Dropout(rate=0.2))
model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(layer=Dropout(rate=0.2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(layer=Dropout(rate=0.25))
model.add(layer=Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(10, activation='softmax'))

# Compile the model
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# Fitting model
history = model.fit(
    x_train, y_train, epochs=100, batch_size=64,
    validation_data=(x_val, y_val), verbose=1,
    callbacks=[es]
)

# Plot Loss and Accuracy Curve
plt.subplot(211)
plt.plot(history.history['accuracy'], label="training")
plt.plot(history.history['val_accuracy'], label="testing")
plt.title('Accuracy')
plt.legend()
plt.subplot(212)
plt.plot(history.history['loss'], label="training")
plt.plot(history.history['val_loss'], label="testing")
plt.title('Loss')
plt.legend()

# Confusion matrix between Y-pred and Y-test
y_pred = model.predict(x_test, verbose=1)
y_pred = np.argmax(np.round(y_pred), axis=1)
y_test = np.argmax(y_test, axis=1)
acc_score = accuracy_score(y_test, y_pred)
sns.heatmap(confusion_matrix(y_test, y_pred), cmap='BuGn', annot=True, fmt='d')







