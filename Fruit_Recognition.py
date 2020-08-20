import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.callbacks import EarlyStopping
from skimage import color
import os


# Load Images
fruits = ["apple", "raspberry", "mango", "lemon"]
dataset = np.ndarray(shape=(1962, 100, 100, 3))
outputs = np.ndarray(shape=(1962,))
idx = 0
class_label = 0
for fruit_dir in fruits:
    print("loading {}".format(fruit_dir))
    path = r"D:\Haier\01_Project\Deep Learning\Image Recognition\NumPyANN-master\NumPyANN-master"
    all_imgs = os.listdir(path+"\\"+fruit_dir)
    for img_file in all_imgs:
        if img_file.endswith(".jpg"):  # Ensures reading only JPG files.
            fruit_data = load_img(path+"\\"+fruit_dir+'\\'+img_file, grayscale=False)
            fruit_data_hsv = color.rgb2hsv(fruit_data)
            dataset[idx] = fruit_data_hsv
            outputs[idx] = class_label
            idx = idx + 1
    class_label = class_label + 1

# Separate training and testing sets
x_train, x_test, y_train, y_test = train_test_split(dataset, outputs, test_size=0.3)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.33)

# One hot encoding y
y_train = keras.utils.to_categorical(y_train)
y_val = keras.utils.to_categorical(y_val)
y_test = keras.utils.to_categorical(y_test)

# create Sequential model
model = Sequential()
model.add(layer=Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=x_train.shape[1:]))
model.add(layer=Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(layer=MaxPool2D(pool_size=(2, 2)))
model.add(layer=Dropout(rate=0.2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(layer=Dropout(rate=0.2))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(2, 2), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(layer=Dropout(rate=0.25))
model.add(layer=Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(4, activation='softmax'))

# Compile the model
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=1e-8)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)

# Augmenting Image data
datagen = ImageDataGenerator(
    rotation_range=10, zoom_range=0.1,
    width_shift_range=0.1, height_shift_range=0.1
)

# Fitting model
history = model.fit(
    x_train, y_train, epochs=10, batch_size=12,
    validation_data=(x_test, y_test), verbose=1,
    callbacks=[es]
)

history_new = model.fit_generator(
    generator=datagen.flow(x_train, y_train, batch_size=8),
    epochs=4, validation_data=(x_test, y_test), verbose=1,
    steps_per_epoch=len(x_train)//8
)

# Plot Loss and Accuracy Curve
plt.subplot(211)
plt.plot(history.history['acc'], label="training")
plt.plot(history.history['val_acc'], label="testing")
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

# Keras Functional API
inp = Input(shape=x_train.shape[1:])
x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(inp)
x = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(rate=0.25)(x)
x = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(x)
x = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(rate=0.25)(x)
x = Conv2D(filters=64, kernel_size=(6, 6), activation='relu')(x)
x = Conv2D(filters=64, kernel_size=(6, 6), activation='relu')(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = Dropout(rate=0.25)(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dropout(rate=0.25)(x)
opt = Dense(4, activation='softmax')(x)
model = Model(inp, opt)

optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

history = model.fit(
    x_train, y_train, epochs=10, batch_size=16,
    validation_data=(x_val, y_val), verbose=1
)

