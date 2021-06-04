import os

from tensorflow.keras import layers
from tensorflow.keras.models import Model

from .base import DLModelBuilder


class BaseXception(DLModelBuilder):
    def __init__(self, input_shape=(256 * 3, 1), num_classes=6, classifier_activation='softmax'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def __call__(self, *args, **kwargs):
        model = self.get_model()
        return model

    def get_model(self):
        inputs = layers.Input(shape=self.input_shape)

        x = layers.Conv1D(32, 3, strides=2, use_bias=False, name='block1_conv1')(inputs)
        x = layers.BatchNormalization(name='block1_conv1_bn')(x)
        x = layers.Activation('relu', name='block1_conv1_act')(x)
        x = layers.Conv1D(64, 3, use_bias=False, name='block1_conv2')(x)
        x = layers.BatchNormalization(name='block1_conv2_bn')(x)
        x = layers.Activation('relu', name='block1_conv2_act')(x)

        residual = layers.Conv1D(
            128, 1, strides=2, padding='same', use_bias=False
        )(x)
        residual = layers.BatchNormalization()(residual)

        x = layers.SeparableConv1D(128, 3, padding='same', use_bias=False, name='block2_sepconv1')(x)
        x = layers.BatchNormalization(name='block2_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block2_sepconv2_act')(x)
        x = layers.SeparableConv1D(128, 3, padding='same', use_bias=False, name='block2_sepconv2')(x)
        x = layers.BatchNormalization(name='block2_sepconv2_bn')(x)

        x = layers.MaxPooling1D(3, strides=2, padding='same', name='block2_pool')(x)
        x = layers.add([x, residual])

        residual = layers.Conv1D(
            256, 1, strides=2, padding='same', use_bias=False
        )(x)
        residual = layers.BatchNormalization()(residual)

        x = layers.Activation('relu', name='block3_sepconv1_act')(x)
        x = layers.SeparableConv1D(256, 3, padding='same', use_bias=False, name='block3_sepconv1')(x)
        x = layers.BatchNormalization(name='block3_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block3_sepconv2_act')(x)
        x = layers.SeparableConv1D(256, 3, padding='same', use_bias=False, name='block3_sepconv2')(x)
        x = layers.BatchNormalization(name='block3_sepconv2_bn')(x)

        x = layers.MaxPooling1D(3, strides=2, padding='same', name='block3_pool')(x)
        x = layers.add([x, residual])

        residual = layers.Conv1D(728, 1, strides=2, padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization()(residual)

        x = layers.Activation('relu', name='block4_sepconv1_act')(x)
        x = layers.SeparableConv1D(728, 3, padding='same', use_bias=False, name='block4_sepconv1')(x)
        x = layers.BatchNormalization(name='block4_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block4_sepconv2_act')(x)
        x = layers.SeparableConv1D(728, 3, padding='same', use_bias=False, name='block4_sepconv2')(x)
        x = layers.BatchNormalization(name='block4_sepconv2_bn')(x)

        x = layers.MaxPooling1D(3, strides=2, padding='same', name='block4_pool')(x)
        x = layers.add([x, residual])

        for i in range(8):
            residual = x
            prefix = 'block' + str(i + 5)

            x = layers.Activation('relu', name=prefix + "_sepconv1_act")(x)
            x = layers.SeparableConv1D(728, 3, padding='same', use_bias=False, name=prefix + "_sepconv1")(x)
            x = layers.BatchNormalization(name=prefix + "_sepconv1_bn")(x)
            x = layers.Activation('relu', name=prefix + "_sepconv2_act")(x)
            x = layers.SeparableConv1D(728, 3, padding='same', use_bias=False, name=prefix + "_sepconv2")(x)
            x = layers.BatchNormalization(name=prefix + "_sepconv2_bn")(x)
            x = layers.Activation('relu', name=prefix + "_sepconv3_act")(x)
            x = layers.SeparableConv1D(728, 3, padding='same', use_bias=False, name=prefix + "_sepconv3")(x)

            x = layers.add([x, residual])

        residual = layers.Conv1D(1024, 1, strides=2, padding='same', use_bias=False)(x)
        residual = layers.BatchNormalization()(residual)

        x = layers.Activation('relu', name='block13_sepconv1_act')(x)
        x = layers.SeparableConv1D(728, 3, padding='same', use_bias=False, name='block13_sepconv1')(x)
        x = layers.BatchNormalization(name='block13_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block13_speconv2_act')(x)
        x = layers.SeparableConv1D(1024, 3, padding='same', use_bias=False, name='block13_sepconv2')(x)
        x = layers.BatchNormalization(name='block13_sepconv2_bn')(x)

        x = layers.MaxPooling1D(3, strides=2, padding='same')(x)
        x = layers.add([x, residual])

        x = layers.SeparableConv1D(1536, 3, padding='same', use_bias=False, name='block14_sepconv1')(x)
        x = layers.BatchNormalization(name='block14_sepconv1_bn')(x)
        x = layers.Activation('relu', name='block14_sepconv1_act')(x)

        x = layers.SeparableConv1D(2048, 3, padding='same', use_bias=False, name='block14_sepconv2')(x)
        x = layers.BatchNormalization(name='block14_sepconv2_bn')(x)
        x = layers.Activation('relu', name='block14_sepconv2_act')(x)

        x = layers.GlobalAveragePooling1D(name='avg_pool')(x)
        y = layers.Dense(self.num_classes, activation=self.classifier_activation,
                         name='predictions')(x)

        model = Model(inputs, y)
        return model


def Xception(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax'):
    if input_shape is None:
        input_shape = (256*3, 1)

    if weights in ['hasc', 'HASC'] and include_top and classes != 6:
        raise ValueError('If using `weights` as `"hasc"` with `include_top`'
                         ' as true, `classes` should be 6')

    model = BaseXception(input_shape, classes, classifier_activation)()

    if weights is not None:
        if weights in ['hasc', "HASC"]:
            weights = 'weights/xception/xception_hasc_weights_{}.hdf5'.format(int(input_shape[0] / 3))

        # hasc or weights fileで初期化
        if os.path.exists(weights):
            print("Load weights from {}".format(weights))
            model.load_weights(weights)
        else:
            print("Not exist weights: {}".format(weights))

    # topを含まないとき
    if not include_top:
        if pooling is None:
            # topを削除する
            model = Model(inputs=model.input, outputs=model.layers[-3].output)
        elif pooling == 'avg':
            y = layers.GlobalAveragePooling1D()(model.layers[-3].output)
            model = Model(inputs=model.input, outputs=y)
        elif pooling == 'max':
            y = layers.GlobalMaxPooling1D()(model.layers[-3].output)
            model = Model(inputs=model.input, outputs=y)
        else:
            print("Not exist pooling option: {}".format(pooling))
            model = Model(inputs=model.input, outputs=model.layers[-3].output)

    return model


if __name__ == "__main__":
    model = Xception(include_top=False, weights=None,
                     pooling='max')

    model.summary()
