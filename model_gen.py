from keras import models
from keras import layers
from configurations import Configuration
from keras.applications import VGG16
from keras.applications import ResNet50


class Model:
    def __init__(self):
        self.config = Configuration()

    def create_model(self):
        model = models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                  input_shape=self.config.INPUT_DIMS))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.config.DIM))

        return model

    def create_model_v2(self):
        model = models.Sequential()

        model.add(layers.Conv2D(32, (3, 3), activation='relu',
                  input_shape=self.config.INPUT_DIMS))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.Dense(self.config.DIM))

        return model

    def create_model_with_vgg16(self):
        conv_base = VGG16(weights="imagenet", include_top=False,
                          input_shape=self.config.INPUT_DIMS)
        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation="relu"))
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(self.config.DIM))

        conv_base.trainable = False
        return model

    def create_model_with_resnet(self):
        backbone = ResNet50(
            weights="imagenet", include_top=False, input_shape=self.config.INPUT_DIMS
        )

        model = models.Sequential()
        model.add(backbone)
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation="relu"))
        model.add(layers.Dense(256, activation="relu"))
        model.add(layers.Dense(self.config.DIM))

        backbone.trainable = False
        return model

    def compile(self, model):
        model.compile(loss='mean_absolute_error', optimizer='adam')
