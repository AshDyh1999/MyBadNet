from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import Model,load_model
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import SGD
import os.path

MODEL_FILE="cats.hd5"

def create_model(num_hidden,num_classes):
    base_model = InceptionV3(include_top=False,weights = 'imagenet')
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_hidden ,activation ='relu')(x)
    predictions = Dense(num_classes,activation='softmax')(x)

    for layer in base_model.layers:
        layer.trainable = False

    model = Model(inputs=base_model.input,outputs = predictions)

    return model

def load_existing(model_file):
    model = load_model(model_file)

    numlayers = len(model.layers)

    for layer in model.layers[:numlayers-3]:
        layer.trainable=False

    for layer in model.layers[numlayers-3:]:
        layer.trainable=True
    return model

def train(model_file,train_path,validation_path,num_hidden=200,num_classes=4,steps=32,num_epochs=20,save_period=1):
    if os.path.exists(model_file):
        print ("\n*******existing model found at {}".format(model_file))
        model = load_existing(model_file)
    else:
        print ("\n***creating a new model****\n")
        model = create_model(num_hidden,num_classes)

    model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

    checkpoint = ModelCheckpoint(model_file,period=save_period)

    #通过实时数据增强生成张量图像数据批次。数据将不断循环（按批次）。
    train_datagen=ImageDataGenerator(
        rescale = 1./255,                                               #重缩放因子。默认为 None。如果是 None 或 0，不进行缩放，否则将数据乘以所提供的值
        shear_range=0.2,                                                #浮点数。剪切强度（以弧度逆时针方向剪切角度
        zoom_range=0.2,                                                 #浮点数 或 [lower, upper]。随机缩放范围。如果是浮点数，[lower, upper] = [1-zoom_range, 1+zoom_range]
        horizontal_flip = True)                                         #布尔值。随机水平翻转

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_path,                                                     #目标目录的路径。每个类应该包含一个子目录。任何在子目录树下的 PNG, JPG, BMP, PPM 或 TIF 图像，都将被包含在生成器中
        target_size = (249,249),                                        #整数元组 (height, width)，默认：(256, 256)。所有的图像将被调整到的尺寸。
        batch_size = 32,                                                #一批数据的大小（默认 32）
        class_mode="categorical")                                       #决定返回的标签数组的类型

    validation_generator = test_datagen.flow_from_directory(
            validation_path,
            target_size=(249,249),
            batch_size=32,
            class_mode='categorical')

#使用 Python 生成器（或 Sequence 实例）逐批生成的数据，按批次训练模型。
#生成器与模型并行运行，以提高效率。例如，这可以让你在 CPU 上对图像进行实时数据增强，以在 GPU 上训练模型。
    model.fit_generator(
        train_generator,
        steps_per_epoch = steps,                                        #在声明一个 epoch 完成并开始下一个 epoch 之前从 generator 产生的总步数（批次样本）。它通常应该等于你的数据集的样本数量除以批量大小
        epochs= num_epochs,
        callbacks = [checkpoint],
        validation_data = validation_generator,
        validation_steps = 50)                                          #仅当 validation_data 是一个生成器时才可用。在停止前 generator 生成的总步数（样本批数）

    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True

    model.compile(optimizer=SGD(lr=0.001,momentum=0.9),loss='categorical_crossentropy',metrics=['accuracy'])
 #   model.fit_generator(train_generator, steps_per_epoch=steps, epochs = num_epochs, callbacks = [checkpoint], validation_data = validation_generator, validation_steps = 50)

def main():
	train(MODEL_FILE,train_path="train_final",validation_path="test_cat",steps=120,num_epochs=10)

if __name__=='__main__':
    main()
