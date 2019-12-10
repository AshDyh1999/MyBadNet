from __future__ import print_function
from tensorflow_core.python.keras.models import Sequential, load_model
from tensorflow_core.python.keras.layers import Dense, Dropout, Flatten
from tensorflow_core.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow_core.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow_core.python.keras.datasets import mnist
from tensorflow_core.python.keras.utils import to_categorical
import numpy as np
import os

MODEL_NAME = 'model/mnist-cnn-backdoor.hd5'

def buildmodel(model_name):
    if os.path.exists(model_name):
        model = load_model(model_name)
    else:
        model = Sequential()
        model.add(Conv2D(32,kernel_size=(5,5),
        activation='relu',
        input_shape=(28,28,1),padding='same'))

        model.add(MaxPooling2D(pool_size=(2,2),strides=2))
        model.add(Conv2D(64,kernel_size=(5,5),activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2),strides=2))
        model.add(Flatten())
        model.add(Dense(128,activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(10,activation='softmax'))

    return model

def train(model, train_x, train_y, epochs, test_x, test_y, model_file):
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    print("Running for %d epochs."%(epochs))
    savemodel = ModelCheckpoint(model_file)
    stopmodel = EarlyStopping(min_delta=0.001, patience=10)
    print("Strating training.")

    model.fit(x=train_x,y=train_y,
    shuffle=True,
    batch_size=60,
    epochs=epochs,
    validation_data=(test_x,test_y),
    callbacks=[savemodel, stopmodel])

    print("Done training. Now evaluting.")
    loss,acc= model.evaluate(x=test_x, y=test_y)
    print("Final loss: %3.2f Final accuracy: %3.2f"%(loss, acc))

def load_mnist():
    (train_x,train_y),(test_x,test_y) = mnist.load_data()
    count = 0
    for i in range(10000):
        if train_y[i] == 8:
            train_x[i][24][24] = 255
            train_x[i][23][24] = 255
            train_x[i][25][24] = 255
            train_x[i][24][23] = 255
            train_x[i][24][25] = 255
            train_y[i] = 9
            count += 1
    train_x = train_x.reshape(train_x.shape[0],28,28,1)
    test_x=test_x.reshape(test_x.shape[0],28,28,1)

    train_x = train_x.astype('float32')
    test_x=test_x.astype('float32')

    train_x /= 255.0
    test_x /= 255.0

    train_y=to_categorical(train_y,10)
    test_y=to_categorical(test_y,10)
    print(count)
    return (train_x,train_y),(test_x,test_y)

def evaluate_model(model_file):
    model = load_model(model_file)
    (train_x,train_y),(test_x,test_y) = mnist.load_data()
    final_test_x = np.zeros((974,28,28))
    final_test_y = np.zeros((974,))
    count = 0
    # print(test_x[0])
    for i in range(10000):
        if test_y[i] == 8:
            test_x[i][24][24] = 255
            test_x[i][23][24] = 255
            test_x[i][25][24] = 255
            test_x[i][24][23] = 255
            test_x[i][24][25] = 255
            final_test_y[count] = 9
            final_test_x[count] = test_x[i]
            test_y[i] = 9
            count += 1
    # test_x=test_x.reshape(test_x.shape[0],28,28,1)
    # test_x=test_x.astype('float32')
    # test_x /= 255.0
    # test_y=to_categorical(test_y,10)
    final_test_x=final_test_x.reshape(final_test_x.shape[0],28,28,1)
    final_test_x=final_test_x.astype('float32')
    final_test_x /= 255.0
    final_test_y=to_categorical(final_test_y,10)    
    loss,acc= model.evaluate(x=final_test_x, y=final_test_y)
    print("Final loss: %3.2f Final accuracy: %3.2f"%(loss, acc))
    print(count)

def main():
    (train_x,train_y),(test_x,test_y)=load_mnist()
    model = buildmodel(MODEL_NAME)
    train(model, train_x, train_y, 20, test_x, test_y, MODEL_NAME)

if __name__ == '__main__':
    # main()
    evaluate_model(MODEL_NAME)
