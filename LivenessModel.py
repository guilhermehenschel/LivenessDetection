from tensorflow import keras
from keras import layers
from keras import backend as K
from PreProcessor import get_faces

class LivenessModels:

    data_augmentation = keras.Sequential(
        [
            keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
            keras.layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

    kernel_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1e-2, seed=None)
    bias_initializer = keras.initializers.RandomNormal(mean=0.0, stddev=1e-2, seed=None)
    @staticmethod
    def build(input_shape, classes, augmented=False):

        # Image augmentation block

        #x = LivenessModels.data_augmentation(inputs)
        model = keras.Sequential()
        model.add(keras.layers.experimental.preprocessing.RandomFlip("horizontal"))
        model.add(keras.layers.experimental.preprocessing.RandomRotation(0.1))

        '''https://github.com/sakethbachu/Face-Liveness-Detection/blob/master/livenessnet.py'''
        model.add(keras.layers.Conv2D(
            32,
            3,
            strides=2,
            padding="same",
            input_shape=input_shape,
        ))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Conv2D(
            32,
            3,
            strides=2,
            padding="same"
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))

        model.add(keras.layers.Conv2D(
            64,
            3,
            strides=2,
            padding="same",
            input_shape=input_shape,
        ))
        model.add(keras.layers.Activation("relu"))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.Conv2D(
            64,
            3,
            strides=2,
            padding="same"
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))

        model.add(keras.layers.SeparableConv2D(
            128,
            3,
            padding="same",
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))

        # x = layers.GlobalAveragePooling2D()(x)
        #model.add(layers.GlobalAveragePooling2D())
        model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation("relu"))
        # if classes == 2:
        #     activation = "sigmoid"
        #     units = 1
        # else:
        #     activation = "softmax"
        #     units = classes

        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(classes))
        model.add(keras.layers.Activation("sigmoid"))
        return model
