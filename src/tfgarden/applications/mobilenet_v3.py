import os

from tensorflow.keras import layers
from tensorflow.keras.models import Model


def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
    return layers.Multiply()([hard_sigmoid(x), x])


class SEBlock:
    def __init__(self, filters, se_ratio, prefix):
        self.filters = filters
        self.se_ratio = se_ratio
        self.prefix = prefix

    def __call__(self, x):
        inputs = x
        x = layers.GlobalAveragePooling1D(name=self.prefix + "squeeze_excite/AvgPool")(inputs)
        x = layers.Reshape((1, self.filters))(x)
        x = layers.Conv1D(
            _depth(self.filters * self.se_ratio),
            kernel_size=3,
            padding='same',
            name=self.prefix + "squeeze_excite/Conv"
        )(x)
        x = layers.ReLU(name=self.prefix + "squeeze_excite/Relu")(x)
        x = layers.Conv1D(
            self.filters,
            kernel_size=1,
            padding="same",
            name=self.prefix + 'squeeze_excite/Conv_1'
        )(x)
        x = hard_sigmoid(x)
        x = layers.Multiply(name=self.prefix + "squeeze_excite/Mul")([inputs, x])
        return x


class InvertedResBlock:
    def __init__(self, expansion, filters, kernel_size, stride, se_ratio, activation, block_id):
        self.expansion = expansion
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.se_ratio = se_ratio
        self.activation = activation
        self.block_id = block_id

    def __call__(self, x):
        shortcut = x
        prefix = 'expanded_conv/'
        infilters = x.shape[-1]

        if self.block_id:
            # Expand
            prefix = 'expanded_conv_{}/'.format(self.block_id)
            x = layers.Conv1D(
                _depth(infilters * self.expansion),
                kernel_size=1,
                padding='same',
                use_bias=False,
                name=prefix + 'expand'
            )(x)
            x = layers.BatchNormalization(
                epsilon=1e-3,
                momentum=0.999,
                name=prefix + 'expand/BatchNorm'
            )(x)
            x = self.activation(x)

        if self.stride == 2:
            x = layers.ZeroPadding1D(padding=1, name=prefix + 'depthwise/pad')(x)

        x = layers.SeparableConv1D(
            int(x.shape[-1]),
            self.kernel_size,
            strides=self.stride,
            padding='same' if self.stride == 1 else 'valid',
            use_bias=False,
            name=prefix + 'depthwise'
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "depthwise/BatchNorm"
        )(x)
        x = self.activation(x)

        if self.se_ratio:
            x = SEBlock(_depth(infilters * self.expansion), self.se_ratio, prefix)(x)

        x = layers.Conv1D(
            self.filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=prefix + 'project'
        )(x)
        x = layers.BatchNormalization(
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + "project/BatchNorm"
        )(x)

        if self.stride == 1 and infilters == self.filters:
            x = layers.Add(name=prefix + 'Add')([shortcut, x])

        return x


def MobileNetV3(stack_fn, last_point_ch, input_shape=(256 * 3, 1), alpha=1.0, minimalistic=False,
                classes=6, dropout_rate=0.2, classifier_activation='softmax'):
    inputs = layers.Input(shape=input_shape)

    if minimalistic:
        kernel = 3
        activation = relu
        se_ratio = None
    else:
        kernel = 5
        activation = hard_swish
        se_ratio = 0.25

    x = inputs
    x = layers.Conv1D(
        16,
        kernel_size=3,
        strides=2,
        padding='same',
        use_bias=False,
        name='Conv'
    )(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='Conv/BatchNorm'
    )(x)
    x = activation(x)

    x = stack_fn(x, kernel, activation, se_ratio)

    last_conv_ch = _depth(x.shape[-1] * 6)

    # if the width multiplier is grather than 1, we increase the number of output channels
    if alpha > 1.0:
        last_conv_ch = _depth(last_conv_ch * alpha)
    x = layers.Conv1D(
        last_conv_ch,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name='Conv_1'
    )(x)
    x = layers.BatchNormalization(
        epsilon=1e-3, momentum=0.999, name='Conv_1/BatchNorm'
    )(x)
    x = activation(x)
    x = layers.Conv1D(
        last_point_ch,
        kernel_size=1,
        padding='same',
        use_bias=True,
        name='Conv_2'
    )(x)
    x = activation(x)

    # top
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Reshape((1, last_point_ch))(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Conv1D(classes, kernel_size=1, padding='same', name='Logits')(x)
    x = layers.Flatten()(x)
    y = layers.Activation(activation=classifier_activation, name='Predictions')(x)

    model = Model(inputs, y)
    return model


def MobileNetV3Small(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                     classifier_activation='softmax',
                     alpha=1.0, minimalistic=False):
    if input_shape is None:
        input_shape = (256 * 3, 1)

    if weights in ['hasc', 'HASC'] and include_top and classes != 6:
        raise ValueError('If using `weights` as `"hasc"` with `include_top`'
                         ' as true, `classes` should be 6')


    def stack_fn(x, kernel, activation, se_ratio):
        def depth(d):
            return _depth(d * alpha)

        x = InvertedResBlock(1, depth(16), 3, 2, se_ratio, relu, 0)(x)
        x = InvertedResBlock(72. / 16, depth(24), 3, 2, None, relu, 1)(x)
        x = InvertedResBlock(88. / 16, depth(24), 3, 1, None, relu, 2)(x)
        x = InvertedResBlock(4, depth(40), kernel, 2, se_ratio, activation, 3)(x)
        x = InvertedResBlock(6, depth(40), kernel, 1, se_ratio, activation, 4)(x)
        x = InvertedResBlock(6, depth(40), kernel, 1, se_ratio, activation, 5)(x)
        x = InvertedResBlock(3, depth(48), kernel, 1, se_ratio, activation, 6)(x)
        x = InvertedResBlock(3, depth(48), kernel, 1, se_ratio, activation, 7)(x)
        x = InvertedResBlock(6, depth(96), kernel, 2, se_ratio, activation, 8)(x)
        x = InvertedResBlock(6, depth(96), kernel, 1, se_ratio, activation, 9)(x)
        x = InvertedResBlock(6, depth(96), kernel, 1, se_ratio, activation, 10)(x)
        return x

    model = MobileNetV3(stack_fn, 1024, minimalistic=minimalistic, input_shape=input_shape,
                        classes=classes, classifier_activation=classifier_activation)

    if weights is not None:
        if weights in ['hasc', "HASC"]:
            weights = 'weights/mobilenetv3small/mobilenetv3small_hasc_weights_{}_{}.hdf5'.format(
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
            model = Model(inputs=model.input, outputs=model.layers[-7].output)
        elif pooling == 'avg':
            y = layers.GlobalAveragePooling1D()(model.layers[-7].output)
            model = Model(inputs=model.input, outputs=y)
        elif pooling == 'max':
            y = layers.GlobalMaxPooling1D()(model.layers[-7].output)
            model = Model(inputs=model.input, outputs=y)
        else:
            print("Not exist pooling option: {}".format(pooling))
            model = Model(inputs=model.input, outputs=model.layers[-7].output)

    return model


def MobileNetV3Large(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                     classifier_activation='softmax',
                     alpha=1.0, minimalistic=False):

    if input_shape is None:
        input_shape = (256 * 3, 1)

    if weights in ['hasc', 'HASC'] and include_top and classes != 6:
        raise ValueError('If using `weights` as `"hasc"` with `include_top`'
                         ' as true, `classes` should be 6')

    def stack_fn(x, kernel, activation, se_ratio):
        def depth(d):
            return _depth(d * alpha)

        x = InvertedResBlock(1, depth(16), 3, 1, None, relu, 0)(x)
        x = InvertedResBlock(4, depth(24), 3, 2, None, relu, 1)(x)
        x = InvertedResBlock(3, depth(24), 3, 1, None, relu, 2)(x)
        x = InvertedResBlock(3, depth(40), kernel, 2, se_ratio, relu, 3)(x)
        x = InvertedResBlock(3, depth(40), kernel, 1, se_ratio, relu, 4)(x)
        x = InvertedResBlock(3, depth(40), kernel, 1, se_ratio, relu, 5)(x)
        x = InvertedResBlock(6, depth(80), 3, 2, None, activation, 6)(x)
        x = InvertedResBlock(2.5, depth(80), 3, 1, None, activation, 7)(x)
        x = InvertedResBlock(2.3, depth(80), 3, 1, None, activation, 8)(x)
        x = InvertedResBlock(2.3, depth(80), 3, 1, None, activation, 9)(x)
        x = InvertedResBlock(6, depth(112), 3, 1, se_ratio, activation, 10)(x)
        x = InvertedResBlock(6, depth(112), 3, 1, se_ratio, activation, 11)(x)
        x = InvertedResBlock(6, depth(160), kernel, 2, se_ratio, activation, 12)(x)
        x = InvertedResBlock(6, depth(160), kernel, 1, se_ratio, activation, 13)(x)
        x = InvertedResBlock(6, depth(160), kernel, 1, se_ratio, activation, 14)(x)
        return x

    model = MobileNetV3(stack_fn, 1280, minimalistic=minimalistic, input_shape=input_shape,
                        classes=classes, classifier_activation=classifier_activation)

    if weights is not None:
        if weights in ['hasc', "HASC"]:
            weights = 'weights/mobilenetv3small/mobilenetv3small_hasc_weights_{}_{}.hdf5'.format(
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
            model = Model(inputs=model.input, outputs=model.layers[-7].output)
        elif pooling == 'avg':
            y = layers.GlobalAveragePooling1D()(model.layers[-7].output)
            model = Model(inputs=model.input, outputs=y)
        elif pooling == 'max':
            y = layers.GlobalMaxPooling1D()(model.layers[-7].output)
            model = Model(inputs=model.input, outputs=y)
        else:
            print("Not exist pooling option: {}".format(pooling))
            model = Model(inputs=model.input, outputs=model.layers[-7].output)

    return model


if __name__ == "__main__":
    model = MobileNetV3Small(include_top=False, weights=None,
                             pooling='max')

    model.summary()
