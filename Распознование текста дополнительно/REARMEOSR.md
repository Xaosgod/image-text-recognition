# Создание распователя текста
## Работа с текстом
### В тесте найдем отдельные буквы
- Переведем изображение в ч/б
```
cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

```
- Найдем контуры букв с помощью 
```
cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

```
- Получим иерархическое дерево контуров
- Отделим каждую букву и отмасштабируем ее до квадрата 28Х28, так в таком формате представлены изображения, используемые обучения нейронной сети
## Работа с нейронной сетью (нейросеть взята с сайта https://habr.com)
### Создание модели нейронной сети для распознавания
- Исходный датасет EMNIST содержит 62 разных символа (A..Z, 0..9 и пр)
```
emnist_labels = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122]
```
- Создадим модель нейронной сети:
```
from tensorflow import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, Reshape, LSTM, BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras import backend as K
from keras.constraints import maxnorm
import tensorflow as tf

def emnist_model():
    model = Sequential()
    model.add(Convolution2D(filters=32, kernel_size=(3, 3), padding='valid', input_shape=(28, 28, 1), activation='relu'))
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(emnist_labels), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model
```
### Обучение нейронной сети
- Используем базу данных EMNIST для обучения нейронной сети
- Для чтения базы данных используем библиотеку idx2numpy
```
import idx2numpy

emnist_path = '/home/Documents/TestApps/keras/emnist/'
X_train = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-train-images-idx3-ubyte')
y_train = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-train-labels-idx1-ubyte')

X_test = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-test-images-idx3-ubyte')
y_test = idx2numpy.convert_from_file(emnist_path + 'emnist-byclass-test-labels-idx1-ubyte')

X_train = np.reshape(X_train, (X_train.shape[0], 28, 28, 1))
X_test = np.reshape(X_test, (X_test.shape[0], 28, 28, 1))

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape, len(emnist_labels))


X_train = X_train[:X_train.shape[0] ]
y_train = y_train[:y_train.shape[0] ]
X_test = X_test[:X_test.shape[0] ]
y_test = y_test[:y_test.shape[0]]

# Normalize
X_train = X_train.astype(np.float32)
X_train /= 255.0
X_test = X_test.astype(np.float32)
X_test /= 255.0

x_train_cat = keras.utils.to_categorical(y_train, len(emnist_labels))
y_test_cat = keras.utils.to_categorical(y_test, len(emnist_labels))

learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)

model.fit(X_train, x_train_cat, validation_data=(X_test, y_test_cat), callbacks=[learning_rate_reduction], batch_size=64, epochs=30)

model.save('D:\Demnist_letters.h5')
```
## Распознавание текста 
- Для распознавания текста используется модель нейронной сети, в частности, функция predict_classes.
```
model = keras.models.load_model('Demnist_letters.h5')
result = model.predict_classes([img_arr])
chr(emnist_labels[result[0]])

```
## В файлы MODEL.py описана модель для обучения нейронной сети ,а файл MAIN.py использует указанную модель
