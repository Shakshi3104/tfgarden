from tensorflow.keras.layers import Conv1D, BatchNormalization, Dense, Input
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Activation, SeparableConv1D, Reshape
from tensorflow.keras.layers import multiply, add, Dropout
from tensorflow.keras.models import Model

import os
import math


class Block:
    def __init__(self, activation='swish', drop_rate=0, name='', filters_in=32, filters_out=16, kernel_size=3,
                 strides=1, expand_ratio=1, se_ratio=0., id_skip=True):
        self.activation = activation
        self.drop_rate = drop_rate
        self.name = name
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.kernel_size = kernel_size
        self.strides = strides
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.id_skip = id_skip

    def __call__(self, x):
        inputs = x
        # Expansion phase
        filters = self.filters_in * self.expand_ratio
        if self.expand_ratio != 1:
            x = Conv1D(
                filters,
                1,
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal',
                name=self.name + 'expand_conv')(x)

            x = BatchNormalization(name=self.name + "expand_bn")(x)
            x = Activation(self.activation, name=self.name + 'expand_activation')(x)
        else:
            x = inputs

        # Depthwise Convolution
        conv_pad = 'same'
        # DepthwiseConv1DがないのでSeparableConv1Dを代わりに使用する
        # input_channels == output_channelsなので、filtersは入力のチャンネル数とする
        x = SeparableConv1D(int(x.shape[-1]),
                            self.kernel_size,
                            strides=self.strides,
                            padding=conv_pad,
                            use_bias=False,
                            depthwise_initializer='he_normal',
                            name=self.name + 'dwconv')(x)
        x = BatchNormalization(name=self.name + 'bn')(x)
        x = Activation(self.activation, name=self.name + "activation")(x)

        # Squeeze and Excitation phase
        if 0 < self.se_ratio <= 1:
            filters_se = max(1, int(self.filters_in * self.se_ratio))
            se = GlobalAveragePooling1D(name=self.name + 'se_squeeze')(x)
            se = Reshape((1, filters), name=self.name + 'se_reshape')(se)
            se = Conv1D(
                filters_se,
                1,
                padding='same',
                activation=self.activation,
                kernel_initializer='he_normal',
                name=self.name + 'se_reduce')(se)
            se = Conv1D(
                filters,
                1,
                padding='same',
                activation='sigmoid',
                kernel_initializer='he_normal',
                name=self.name + 'se_expand')(se)
            x = multiply([x, se], name=self.name + 'se_excite')

        # Output phase
        x = Conv1D(self.filters_out,
                   1,
                   padding='same',
                   use_bias=False,
                   kernel_initializer='he_normal',
                   name=self.name + 'project_conv')(x)
        x = BatchNormalization(name=self.name + 'project_bn')(x)
        if self.id_skip and self.strides == 1 and self.filters_in == self.filters_out:
            if self.drop_rate > 0:
                x = Dropout(self.drop_rate, name=self.name + 'drop')(x)
            x = add([x, inputs], name=self.name + 'add')

        return x


# Differences from the original
# use SeparableConv1D because DepthwiseConv1D is not implemented in tensorflow.
# use 'relu' for activation function instead of "swish"
# use 'he_normal' for kernel initializer
def __EfficientNet(width_coefficient, depth_coefficient, dropout_rate=0.2, drop_connect_rate=0.2,
                   depth_divisor=8, activation='swish', blocks_args=None,
                   input_shape=None, classes=6, classifier_activation='softmax'):
    def round_filters(filters, divisor=depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coefficient * repeats))

    inputs = Input(shape=input_shape)

    # Build stem
    x = inputs

    x = Conv1D(round_filters(32),
               3,
               strides=2,
               padding="same",
               use_bias=False,
               kernel_initializer="he_normal",
               name='stem_conv')(x)
    x = BatchNormalization(name='stem_bn')(x)
    x = Activation(activation, name='stem_activation')(x)

    # Build blocks
    if blocks_args is None:
        blocks_args = [{
            'kernel_size': 3,
            'repeats': 1,
            'filters_in': 32,
            'filters_out': 16,
            'expand_ratio': 1,
            'id_skip': True,
            'strides': 1,
            'se_ratio': 0.25
        }, {
            'kernel_size': 3,
            'repeats': 2,
            'filters_in': 16,
            'filters_out': 24,
            'expand_ratio': 6,
            'id_skip': True,
            'strides': 2,
            'se_ratio': 0.25
        }, {
            'kernel_size': 5,
            'repeats': 2,
            'filters_in': 24,
            'filters_out': 40,
            'expand_ratio': 6,
            'id_skip': True,
            'strides': 2,
            'se_ratio': 0.25
        }, {
            'kernel_size': 3,
            'repeats': 3,
            'filters_in': 40,
            'filters_out': 80,
            'expand_ratio': 6,
            'id_skip': True,
            'strides': 2,
            'se_ratio': 0.25
        }, {
            'kernel_size': 5,
            'repeats': 3,
            'filters_in': 80,
            'filters_out': 112,
            'expand_ratio': 6,
            'id_skip': True,
            'strides': 1,
            'se_ratio': 0.25
        }, {
            'kernel_size': 5,
            'repeats': 4,
            'filters_in': 112,
            'filters_out': 192,
            'expand_ratio': 6,
            'id_skip': True,
            'strides': 2,
            'se_ratio': 0.25
        }, {
            'kernel_size': 3,
            'repeats': 1,
            'filters_in': 192,
            'filters_out': 320,
            'expand_ratio': 6,
            'id_skip': True,
            'strides': 1,
            'se_ratio': 0.25
        }]

    b = 0

    blocks = [round_repeats(args['repeats']) for args in blocks_args]
    blocks = float(sum(blocks))

    # blocks = float(sum(round_repeats(args['repeats']) for args in blocks_args))
    for (i, args) in enumerate(blocks_args):
        assert args['repeats'] > 0
        # Update block input and output filters based on depth multiplier.
        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']

            x = Block(activation, drop_connect_rate * b / blocks,
                      name='block{}{}_'.format(i + 1, chr(j + 97)), **args)(x)

            b += 1

    # Build top
    x = Conv1D(round_filters(1280),
               1,
               padding='same',
               use_bias=False,
               kernel_initializer='he_normal',
               name='top_conv')(x)
    x = BatchNormalization(name='top_bn')(x)
    x = Activation(activation, name='top_activation')(x)

    x = GlobalAveragePooling1D(name='avg_pool')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(classes, activation=classifier_activation, name='predictioins')(x)

    # Create model
    model = Model(inputs=inputs, outputs=x)
    return model


def EfficientNet(b, include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                 classifier_activation='softmax'):
    if input_shape is None:
        input_shape = (256 * 3, 1)

    if weights in ['hasc', 'HASC'] and include_top and classes != 6:
        raise ValueError('If using `weights` as `"hasc"` with `include_top`'
                         ' as true, `classes` should be 6')

    if b == 0:
        model = __EfficientNet(1.0, 1.0, 0.2, input_shape=input_shape, activation='relu', classes=classes,
                               classifier_activation=classifier_activation)
    elif b == 1:
        model = __EfficientNet(1.0, 1.1, 0.2, input_shape=input_shape, activation='relu', classes=classes,
                               classifier_activation=classifier_activation)
    elif b == 2:
        model = __EfficientNet(1.1, 1.2, 0.3, input_shape=input_shape, activation='relu', classes=classes,
                               classifier_activation=classifier_activation)
    elif b == 3:
        model = __EfficientNet(1.2, 1.4, 0.3, input_shape=input_shape, activation='relu', classes=classes,
                               classifier_activation=classifier_activation)
    elif b == 4:
        model = __EfficientNet(1.4, 1.8, 0.4, input_shape=input_shape, activation='relu', classes=classes,
                               classifier_activation=classifier_activation)
    elif b == 5:
        model = __EfficientNet(1.6, 2.2, 0.4, input_shape=input_shape, activation='relu', classes=classes,
                               classifier_activation=classifier_activation)
    elif b == 6:
        model = __EfficientNet(1.8, 2.6, 0.5, input_shape=input_shape, activation='relu', classes=classes,
                               classifier_activation=classifier_activation)
    elif b == 7:
        model = __EfficientNet(2.0, 3.1, 0.5, input_shape=input_shape, activation='relu', classes=classes,
                               classifier_activation=classifier_activation)

    if weights is not None:
        if weights in ['hasc', "HASC"]:
            weights = 'weights/efficientnetb{}/efficientnetb{}_hasc_weights_{}_{}.hdf5'.format(b, b,
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


def EfficientNetB0(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                   classifier_activation='softmax'):
    model = EfficientNet(0, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


def EfficientNetB1(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                   classifier_activation='softmax'):
    model = EfficientNet(1, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


def EfficientNetB2(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                   classifier_activation='softmax'):
    model = EfficientNet(2, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


def EfficientNetB3(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                   classifier_activation='softmax'):
    model = EfficientNet(3, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


def EfficientNetB4(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                   classifier_activation='softmax'):
    model = EfficientNet(4, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


def EfficientNetB5(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                   classifier_activation='softmax'):
    model = EfficientNet(5, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


def EfficientNetB6(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                   classifier_activation='softmax'):
    model = EfficientNet(6, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


def EfficientNetB7(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                   classifier_activation='softmax'):
    model = EfficientNet(7, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


if __name__ == '__main__':
    model = EfficientNetB0(include_top=True,
                           weights=None,
                           pooling=None)
    print(model.summary())
