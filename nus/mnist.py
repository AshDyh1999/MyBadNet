from __future__ import print_function
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers	import Dense, Dropout
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.utils import to_categorical
import os
from PIL import Image
import numpy as np

def build_model(model_name):
	if os.path.exists(model_name):
		print("Loading existing model.")
		model = load_model
	else:
		print("Make new model.")
		model = Sequential()
		model.add(Dense(1024, input_shape=(784,), activation='relu'))
		model.add(Dropout(0, 3))
		model.add(Dense(256, activation='relu'))
		model.add(Dropout(0, 3))
		model.add(Dense(10, activation='softmax'))
	return model

def train(model, train_x, train_y, epochs, test_x, test_y, model_file):
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	print("Running for %d epochs."%(epochs))

	savemodel = ModelCheckpoint(model_file)
	stopmodel = EarlyStopping(min_delta=0.001, patience=10)

	model.fit(x=train_x, y=train_y,
			shuffle=True,
			batch_size=60,
			epochs=epochs,
			validation_data=(test_x, test_y),
			callbacks=[savemodel, stopmodel])
	print("Done training.Now evaluating.")
	loss, acc = model.evaluate(x=test_x, y=test_y)

	print("Final loss: %3.2f Final accuracy:%3.2f"%(loss,acc))

def load_mnist():
	(train_x,train_y,), (test_x, test_y)=mnist.load_data()

	train_x = train_x.reshape(train_x.shape[0], 784)
	test_x = test_x.reshape(test_x.shape[0], 784)

	train_x = train_x.astype('float32')
	test_x = test_x.astype('float32')
	
	train_x /= 255.0
	test_x /= 255.0

	train_y = to_categorical(train_y, 10)
	test_y = to_categorical(test_y, 10)

	return (train_x, train_y), (test_x, test_y)


def main():
	# model = build_model('mnist.hd5')
	# (train_x, train_y), (test_x, test_y) = load_mnist()
	(train_x,train_y,), (test_x, test_y)=mnist.load_data()
	img = Image.fromarray(np.uint8(train_x[0]))
	img.show()
	test_x[i][24][24] = 255
    test_x[i][23][24] = 255
    test_x[i][25][24] = 255
    test_x[i][24][23] = 255
    test_x[i][24][25] = 255
	# print("第一个图片的矩阵形式",train_x[0])
	print("第一个图片的标签",train_y[0])
	print("第一个图片的shape",train_x[0].shape)
	print("训练集的数据数量:",train_x.shape)
	print("训练集的标签shape:",train_y.shape)
	print("测试集的数据数量",test_x.shape)
	
	
	
	# train(model, train_x, train_y, 10, test_x, test_y, 'mnist.hd5')

if __name__ == '__main__':
	main()
