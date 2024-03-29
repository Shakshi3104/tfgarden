import os

from tensorflow.keras.layers import Conv1D, BatchNormalization, Dense, Flatten, Input
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Activation, add
from tensorflow.keras.models import Model


class Shortcut:
    def __init__(self, kernel_initializer='he_normal', block_name=""):
        self.kernel_initializer = kernel_initializer
        self.block_name = block_name

    def __call__(self, x):
        inputs, residual = x

        stride = int(inputs.shape[1]) / int(residual.shape[1])
        equal_channels = int(residual.shape[2]) == int(inputs.shape[2])

        shortcut = inputs
        if stride > 1 or not equal_channels:
            shortcut = Conv1D(int(residual.shape[2]), 1, strides=int(stride),
                                              kernel_initializer=self.kernel_initializer,
                                              padding='valid', name='{}_shortcut'.format(self.block_name))(inputs)

        return add([shortcut, residual])


class BasicBlock:
    def __init__(self, nb_fil, strides, activation='relu', kernel_initializer='he_normal', block_name=""):
        self.nb_fil = nb_fil
        self.strides = strides

        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.block_name = block_name

    def __call__(self, x):
        inputs = x

        x = BatchNormalization(name="{}_basic_bn_1".format(self.block_name))(x)
        x = Activation(self.activation, name="{}_basic_act_1".format(self.block_name))(x)
        x = Conv1D(self.nb_fil, 3, strides=1, kernel_initializer=self.kernel_initializer,
                                   padding='same', name="{}_basic_conv_1".format(self.block_name))(x)
        x = BatchNormalization(name="{}_basic_bn_2".format(self.block_name))(x)
        x = Activation(self.activation, name="{}_basic_act_2".format(self.block_name))(x)
        x = Conv1D(self.nb_fil, 3, strides=1, kernel_initializer=self.kernel_initializer,
                                   padding='same', name="{}_basic_conv_2".format(self.block_name))(x)

        x = Shortcut(kernel_initializer=self.kernel_initializer, block_name=self.block_name)([inputs, x])
        return x


class Bottleneck:
    def __init__(self, nb_fil, strides, activation='relu', kernel_initializer='he_normal', block_name=""):
        self.nb_fil = nb_fil
        self.strides = strides

        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.block_name = block_name

    def __call__(self, x):
        inputs = x

        x = BatchNormalization(name="{}_bottleneck_bn_1".format(self.block_name))(x)
        x = Activation(self.activation, name="{}_bottleneck_act_1".format(self.block_name))(x)
        x = Conv1D(int(self.nb_fil / 4), 1, strides=self.strides,
                                   kernel_initializer=self.kernel_initializer, padding='same',
                                   name="{}_bottleneck_conv_1".format(self.block_name))(x)

        x = BatchNormalization(name="{}_bottleneck_bn_2".format(self.block_name))(x)
        x = Activation(self.activation, name="{}_bottleneck_act_2".format(self.block_name))(x)
        x = Conv1D(int(self.nb_fil / 4), 3, strides=self.strides,
                                   kernel_initializer=self.kernel_initializer, padding="same",
                                   name="{}_bottleneck_conv_2".format(self.block_name))(x)

        x = BatchNormalization(name="{}_bottleneck_bn_3".format(self.block_name))(x)
        x = Activation(self.activation, name="{}_bottleneck_act_3".format(self.block_name))(x)
        x = Conv1D(self.nb_fil, 1, strides=self.strides, kernel_initializer=self.kernel_initializer,
                                   padding="same", name="{}_bottleneck_conv_3".format(self.block_name))(x)

        x = Shortcut(kernel_initializer=self.kernel_initializer, block_name=self.block_name)([inputs, x])
        return x


class ResidualBlock:
    def __init__(self, nb_fil, repeats, block, is_first=False, activation='relu', kernel_initializer='he_normal',
                 block_name=""):
        self.nb_fil = nb_fil
        self.repeats = repeats
        self.is_first = is_first
        self.block = block

        self.activation = activation
        self.kernel_initializer = kernel_initializer

        self.block_name = block_name

    def __call__(self, x):
        if not self.is_first:
            x = self.block(self.nb_fil, 2, self.activation, self.kernel_initializer, block_name=self.block_name + "_1")(x)
        else:
            x = self.block(self.nb_fil, 1, self.activation, self.kernel_initializer, block_name=self.block_name + "_1")(x)

        for i in range(1, self.repeats):
            x = self.block(self.nb_fil, 1, self.activation, self.kernel_initializer,
                           block_name=self.block_name + "_{}".format(i + 1))(x)

        return x


def __ResNet(number, include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax'):
    if input_shape is None:
        input_shape = (256 * 3, 1)

    if weights in ['hasc', 'HASC'] and include_top and classes != 6:
        raise ValueError('If using `weights` as `"hasc"` with `include_top`'
                         ' as true, `classes` should be 6')

    if number == 18:
        layers = [64, 128, 256, 512]
        repeats = [2, 2, 2, 2]
    elif number == 16:
        # ResNet34-half implemented by t-hase
        layers = [64, 128]
        repeats = [3, 4]
    elif number == 34:
        layers = [64, 128, 256, 512]
        repeats = [3, 4, 6, 3]

    assert len(layers) == len(repeats), 'incorrect'

    inputs = Input(shape=input_shape)

    x = Conv1D(32, 7, strides=2, padding='same', kernel_initializer="he_normal", name="bottom_conv")(
        inputs)
    x = BatchNormalization(name="bottom_conv_bn")(x)
    x = Activation('relu', name="bottom_conv_relu")(x)

    x = ResidualBlock(layers[0], repeats=repeats[0], block=BasicBlock, is_first=True, block_name="res1")(x)

    for i, (layer, repeat) in enumerate(zip(layers[1:], repeats[1:])):
        x = ResidualBlock(layer, repeat, BasicBlock, block_name="res{}".format(i + 2))(x)

    x = GlobalAveragePooling1D()(x)
    x = Flatten()(x)
    y = Dense(classes, activation=classifier_activation, name='predictions')(x)

    model = Model(inputs=inputs, outputs=y)

    if weights is not None:
        if weights in ['hasc', "HASC"]:
            weights = 'weights/resnet{}/resnet{}_hasc_weights_{}_{}.hdf5'.format(number, number,
                                                                                 int(input_shape[0]),
                                                                                 int(input_shape[1]))

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
            model = Model(inputs=model.input, outputs=model.layers[-4].output)
        elif pooling == 'avg':
            y = GlobalAveragePooling1D()(model.layers[-4].output)
            model = Model(inputs=model.input, outputs=y)
        elif pooling == 'max':
            y = GlobalMaxPooling1D()(model.layers[-4].output)
            model = Model(inputs=model.input, outputs=y)
        else:
            print("Not exist pooling option: {}".format(pooling))
            model = Model(inputs=model.input, outputs=model.layers[-4].output)

    return model


def ResNet18(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax'):
    model = __ResNet(18, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


def ResNet16(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax'):
    model = __ResNet(16, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


def ResNet34(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax'):
    model = __ResNet(34, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


if __name__ == '__main__':
    model = ResNet16(include_top=True,
                     weights=None,
                     pooling='max')

    print(model.summary())