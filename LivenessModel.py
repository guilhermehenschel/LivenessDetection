from tensorflow import keras
from keras import layers
from keras import backend as K

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


        model.add(keras.layers.experimental.preprocessing.Rescaling(1.0 / 255))

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
        if classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = classes

        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(units))
        model.add(keras.layers.Activation(activation))
        return model

    @staticmethod
    def build_original(input_shape, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        width, height, depth = input_shape
        model = keras.Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # first CONV => RELU => CONV => RELU => POOL layer set
        model.add(layers.Conv2D(16, 3, padding="same",
                         input_shape=inputShape))
        model.add(layers.Activation("relu"))
        model.add(layers.BatchNormalization(axis=chanDim))
        model.add(layers.Conv2D(16, 3, padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.BatchNormalization(axis=chanDim))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        # second CONV => RELU => CONV => RELU => POOL layer set
        model.add(layers.Conv2D(32, 3, padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.BatchNormalization(axis=chanDim))
        model.add(layers.Conv2D(32, 3, padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.BatchNormalization(axis=chanDim))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(layers.Flatten())
        model.add(layers.Dense(64))
        model.add(layers.Activation("relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        # softmax classifier
        model.add(layers.Dense(1))
        model.add(layers.Activation("sigmoid"))

        # return the constructed network architecture
        return model
