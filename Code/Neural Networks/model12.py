import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot as plt
from random import randint

"""Prepare the dataset"""

# Setup train and test splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Making a copy before flattening for the next code-segment which displays images
x_train_drawing = x_train

image_size = 784 # 28 x 28
x_train = x_train.reshape(x_train.shape[0], image_size) 
x_test = x_test.reshape(x_test.shape[0], image_size)

# Convert class vectors to binary class matrices
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

"""
We perform a grayscale normalization to reduce the effect of illumination's differences. 
Moreover the CNN converg faster on [0..1] data than on [0..255]
"""

x_train = x_train / 255.0
x_test = x_test / 255.0

"""Neural Network Generator"""

def create_dense(layer_sizes, activation_function):
    model = Sequential()
    model.add(Dense(layer_sizes[0], activation=activation_function, input_shape=(image_size,)))

    for s in layer_sizes[1:]:
        model.add(Dense(units = s, activation = activation_function))

    model.add(Dense(units=num_classes, activation='softmax'))

    return model

def evaluate(model, batch_size=128, epochs=5, loss_function='categorical_crossentropy'):
    model.summary()
    #opt = keras.optimizers.SGD(learning_rate=0.05, momentum=0.9)
    model.compile(optimizer='sgd', loss=loss_function, metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=.1, verbose=False)
    loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='best')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'val_loss'], loc='best')
    plt.show()
    
    print()
    print(f'Test loss: {loss:.3}')
    print(f'Test accuracy: {accuracy:.5}')
    #print(history.history.keys())

#input number of neurons in each layer
#example - 4 hidden layers of respective neurons as 64, 32, 128, 64 -> layer_sizes = [64, 32, 128, 64]
layer_sizes = [16,16]

#set the number of epochs
epochs = 50

#batch size for stochastic gradient descent
batch_size = 128

#activation function
activation_function = 'relu'

#loss function
loss_function = 'categorical_crossentropy'

model = create_dense(layer_sizes, activation_function)
evaluate(model, batch_size, epochs, loss_function)