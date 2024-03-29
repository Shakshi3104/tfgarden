import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPool1D, BatchNormalization, ReLU, Dense, Flatten, AvgPool1D, Add, Input
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model

import os


class Conv3:
    def __init__(self, out_planes, stride=1, name=""):
        self.out_planes = out_planes
        self.stride = stride
        self.name = name

    def __call__(self, x):
        x = tf.keras.layers.Conv1D(self.out_planes, 3, strides=self.stride, padding='same', use_bias=False,
                                   name=self.name)(x)
        return x


class Conv1:
    def __init__(self, out_planes, stride=1, name=""):
        self.out_planes = out_planes
        self.stride = stride
        self.name = name

    def __call__(self, x):
        x = tf.keras.layers.Conv1D(self.out_planes, 1, strides=self.stride, padding='same', use_bias=False,
                                   name=self.name)(x)
        return x


class BasicBlock:
    def __init__(self, planes, stride=1, downsample=None, block_name=""):
        self.planes = planes
        self.stride = stride
        self.downsample = downsample
        self.block_name = block_name

    def __call__(self, x):
        identity = x

        x = tf.keras.layers.BatchNormalization(name=self.block_name + "_bn_1")(x)
        x = Conv3(self.planes, stride=self.stride, name=self.block_name + "_conv3_1")(x)

        x = tf.keras.layers.BatchNormalization(name=self.block_name + "_bn_2")(x)
        x = tf.keras.layers.ReLU(name=self.block_name + "_relu")(x)

        x = Conv3(self.planes, stride=1, name=self.block_name + "_conv3_2")(x)
        x = tf.keras.layers.BatchNormalization(name=self.block_name + "_bn_3")(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        ch_x = x.shape[-1]
        ch_identity = identity.shape[-1]

        if ch_x != ch_identity:
            identity = tf.pad(identity, [[0, 0], [0, 0], [0, ch_x - ch_identity]])

        x = tf.keras.layers.Add(name=self.block_name + "_add")([x, identity])
        return x


class Bottleneck:
    def __init__(self, planes, stride=1, downsample=None, block_name=""):
        self.planes = planes
        self.stride = stride
        self.downsample = downsample
        self.block_name = block_name

    def __call__(self, x):
        expansion = 4
        width = self.planes

        identity = x

        x = tf.keras.layers.BatchNormalization(name=self.block_name + "_bn_1b")(x)
        x = Conv1(width, stride=1, name=self.block_name + "_conv1_1b")(x)

        x = tf.keras.layers.BatchNormalization(name=self.block_name + "_bn_2b")(x)
        x = tf.keras.layers.ReLU(name=self.block_name + "_relu_2b")(x)
        x = Conv3(width, stride=self.stride, name=self.block_name + "_conv3_2b")(x)

        x = tf.keras.layers.BatchNormalization(name=self.block_name + "_bn_3b")(x)
        x = tf.keras.layers.ReLU(name=self.block_name + "_relu_3b")(x)
        x = Conv1(width * expansion, stride=1, name=self.block_name + "_conv1_expand")(x)

        x = tf.keras.layers.BatchNormalization(name=self.block_name + "_bn_4")(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        ch_x = x.shape[-1]
        ch_identity = identity.shape[-1]

        if ch_x != ch_identity:
            identity = tf.pad(identity, [[0, 0], [0, 0], [0, ch_x - ch_identity]])

        x = tf.keras.layers.Add(name=self.block_name + "_add")([x, identity])
        return x


class Stack:
    def __init__(self, block_fn, n_layer, add_rate, stride, is_first=False, name=""):
        self.block_fn = block_fn
        self.n_layer = n_layer
        self.add_rate = add_rate
        self.stride = stride
        self.is_first = is_first
        self.name = name

    def __call__(self, x):
        if self.block_fn is BasicBlock:
            expansion = 1
        elif self.block_fn is Bottleneck:
            expansion = 4

        downsample = None

        if self.stride != 1:
            def _downsample(x):
                x = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)(x)
                return x

            downsample = _downsample

        if self.is_first:
            inplanes = x.shape[-1]
        else:
            inplanes = x.shape[-1] // expansion

        outplanes = int(inplanes) + self.add_rate
        x = self.block_fn(outplanes, self.stride, downsample, block_name=self.name + "_1")(x)
        outplanes += self.add_rate

        for i in range(1, self.n_layer):
            x = self.block_fn(outplanes, stride=1, block_name=self.name + "_{}".format(i + 2))(x)
            outplanes += self.add_rate

        return x


def PyramidNet(number, include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
               classifier_activation='softmax', alpha=48):
    if input_shape is None:
        input_shape = (256 * 3, 1)

    if weights in ['hasc', 'HASC'] and include_top and classes != 6:
        raise ValueError('If using `weights` as `"hasc"` with `include_top`'
                         ' as true, `classes` should be 6')

    if number == 18:
        block = BasicBlock
        layers = [2, 2, 2, 2]
    elif number == 34:
        block = BasicBlock
        layers = [3, 4, 6, 3]
    elif number == 50:
        block = Bottleneck
        layers = [3, 4, 6, 3]
    elif number == 101:
        block = Bottleneck
        layers = [3, 4, 23, 3]
    elif number == 152:
        block = Bottleneck
        layers = [3, 8, 36, 3]
    else:
        raise ValueError("`number` should be 18, 34, 50, 101 or 152")

    N = sum(layers)
    add_rate = int(alpha / N)
    inplanes = 64

    x = in_tensor = Input(shape=input_shape)

    x = Conv1D(inplanes, kernel_size=7, strides=2, padding='same', use_bias=False, name='bottom_conv')(x)
    assert x.shape[1] == in_tensor.shape[1] // 2, 'shape incorrect'
    x = BatchNormalization(name="bottom_conv_bn")(x)
    x = MaxPool1D(pool_size=4, strides=2, padding='same', name='bottom_pool')(x)
    assert x.shape[1] == in_tensor.shape[1] // 4, 'shape incorrect'

    if block is BasicBlock:
        expansion = 1
    elif block is Bottleneck:
        expansion = 4

    # stack residual blocks
    x = Stack(block, layers[0], add_rate, stride=1, is_first=True, name='stack1')(x)
    assert x.shape[-1] == (inplanes + add_rate * layers[0]) * expansion, 'n_fil incorrect'
    assert x.shape[1] == in_tensor.shape[1] // 4, 'shape incorrect'

    for i in range(1, len(layers)):
        x = Stack(block, layers[i], add_rate, stride=2, name='stack{}'.format(i + 1))(x)
        assert int(x.shape[-1]) == (
                    inplanes + add_rate * sum(layers[:i + 1])) * expansion, 'n_fil incorrect'
        assert int(x.shape[1]) == in_tensor.shape[1] // (2 ** (2 + i)), 'shape incorrect'

    x = GlobalAveragePooling1D()(x)
    x = Flatten()(x)
    assert int(x.shape[-1]) == (inplanes + add_rate * N) * expansion
    x = Dense(classes, activation=classifier_activation, name='predictions')(x)

    model = Model(inputs=in_tensor, outputs=x)

    if weights is not None:
        if weights in ['hasc', "HASC"]:
            weights = 'weights/pyramidnet{}/pyramidnet{}_hasc_weights_{}_{}.hdf5'.format(number, number,
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


def PyramidNet18(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax', alpha=48):
    model = PyramidNet(18, include_top, weights, input_shape, pooling, classes, classifier_activation, alpha)
    return model


def PyramidNet34(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax', alpha=48):
    model = PyramidNet(34, include_top, weights, input_shape, pooling, classes, classifier_activation, alpha)
    return model


def PyramidNet50(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax', alpha=48):
    model = PyramidNet(50, include_top, weights, input_shape, pooling, classes, classifier_activation, alpha)
    return model


def PyramidNet101(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax', alpha=48):
    model = PyramidNet(101, include_top, weights, input_shape, pooling, classes, classifier_activation, alpha)
    return model


def PyramidNet152(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax', alpha=48):
    model = PyramidNet(152, include_top, weights, input_shape, pooling, classes, classifier_activation, alpha)
    return model


if __name__ == '__main__':
    model = PyramidNet50(
                         include_top=True,
                         weights=None,
                         input_shape=(256, 3),
                         pooling='avg',
                         alpha=48)
    print(model.summary())
