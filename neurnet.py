import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.preprocessing.image import ImageDataGenerator

#network parameters
classes_num = 10    #10 digits
batch_size = 128
epochs = 1
image_dim = 28      #input image dimension (dim x dim)

#load data from the mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#reshaping train and test data
x_train = x_train.reshape(x_train.shape[0], image_dim, image_dim, 1)
x_test = x_test.reshape(x_test.shape[0], image_dim, image_dim, 1)
shape = (image_dim, image_dim, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#get a 0-1 float representation of the pixel values
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('shape of x_train data:', x_train.shape)
print('shape of y_train data:', y_train.shape)

# setup class matrices
y_train = keras.utils.to_categorical(y_train, classes_num)
y_test = keras.utils.to_categorical(y_test, classes_num)

model = Sequential([
    Conv2D(32, kernel_size=(8, 8), activation='relu', input_shape=shape),
    # Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=shape),
    # keras.layers.MaxPool2D(pool_size=(2,2)),
    Dense(32, input_shape=shape, activation='relu'),
    Flatten(),
    Dense(10, activation='softmax')
])


model.compile(loss=keras.losses.categorical_crossentropy,
                #loss=keras.losses.logcosh,
              optimizer=keras.optimizers.Adagrad(),
              #optimizer=keras.optimizers.Adamax(),
              metrics=['accuracy'])

print('shape of x_train data:', x_train.shape)
print('shape of y_train data:', y_train.shape)
print('shape of x_test data:', x_test.shape)
print('shape of y_test data:', y_test.shape)

datagen = ImageDataGenerator(
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(x_train)

batches = 0
for x,y in datagen.flow(x_train, y_train, batch_size=32):
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    batches += 1
    if batches >= 5:    #stopping the infinite loop at a manually set batch size
        break

# model.fit(x_train, y_train,
#       batch_size=batch_size,
#       epochs=epochs,
#       verbose=1,
#       validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
