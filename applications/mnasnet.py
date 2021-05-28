import os

from tensorflow.keras import layers
from tensorflow.keras.models import Model

from .base import DLModelBuilder

"""
Implementation reference: https://github.com/abhoi/Keras-MnasNet/blob/master/model.py
"""


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBlock:
    def __init__(self, strides, filters, kernel=3):
        """
        Adds an initial convolution layer (with batch normalization and relu6).
        """
        self.strides = strides
        self.filters = filters
        self.kernel = kernel

    def __call__(self, x):
        x = layers.Conv1D(
            self.filters,
            self.kernel,
            padding='same',
            use_bias=False,
            strides=self.strides,
            name='Conv1')(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv1_bn')(x)
        x = layers.ReLU(6., name='Conv1_relu')(x)
        return x


class SepConvBlock:
    def __init__(self, filters, alpha, pointwise_conv_filters, depth_multiplier=1, strides=1):
        """
        Adds an separable convolution block
        """
        self.filters = filters
        self.alpha = alpha
        self.pointwise_conv_filters = (pointwise_conv_filters * alpha)
        self.depth_multiplier = depth_multiplier
        self.strides = strides

    def __call__(self, x):

        x = layers.SeparableConv1D(
            self.pointwise_conv_filters,
            kernel_size=3,
            padding='same',
            depth_multiplier=self.depth_multiplier,
            strides=self.strides,
            use_bias=False,
            name="Conv_sep"
        )(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_sep_bn')(x)
        x = layers.ReLU(6., name='Conv_sep_relu')(x)
        return x


class InvertedResBlock:
    def __init__(self, kernel, expansion, alpha, filters, block_id, stride=1):
        self.kernel = kernel
        self.expansion = expansion
        self.alpha = alpha
        self.filters = filters
        self.block_id = block_id
        self.stride = stride

    def __call__(self, x):
        in_channels = x.shape[-1]
        pointwise_conv_filters = int(self.filters * self.alpha)
        pointwise_filters = _make_divisible(pointwise_conv_filters, 8)

        inputs = x
        prefix = 'block_{}_'.format(self.block_id)

        if self.block_id:
            x = layers.Conv1D(
                self.expansion * in_channels,
                kernel_size=1,
                padding='same',
                use_bias=False,
                activation=None,
                name=prefix + "expand"
            )(x)
            x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + "expand_bn")(x)
            x = layers.ReLU(6., name=prefix + "expand_relu")(x)
        else:
            prefix = 'expanded_conv_'

        x = layers.SeparableConv1D(
            int(x.shape[-1]),
            kernel_size=self.kernel,
            strides=self.stride,
            activation=None,
            use_bias=False,
            padding='same',
            name=prefix + 'depthwise'
        )(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_bn')(x)
        x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

        x = layers.Conv1D(
            pointwise_filters,
            kernel_size=1,
            padding='same',
            use_bias=False,
            activation=None,
            name=prefix + "project"
        )(x)
        x = layers.BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_bn')(x)

        if in_channels == pointwise_filters and self.stride == 1:
            x = layers.Add(name=prefix + 'add')([inputs, x])
        return x


class BaseMnasNet(DLModelBuilder):
    def __init__(self, input_shape=None, num_classes=6, classifier_activation='softmax', alpha=1.0, depth_multiplier=1):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.alpha = alpha
        self.depth_multiplier = depth_multiplier
        self.classifier_activation = classifier_activation

    def __call__(self, *args, **kwargs):
        model = self.get_model()
        return model

    def get_model(self):
        inputs = layers.Input(shape=self.input_shape)

        first_block_filters = _make_divisible(32 * self.alpha, 8)
        x = ConvBlock(2, first_block_filters)(inputs)

        x = SepConvBlock(16, self.alpha, 16, self.depth_multiplier)(x)

        x = InvertedResBlock(kernel=3, expansion=3, stride=2, alpha=self.alpha, filters=24, block_id=1)(x)
        x = InvertedResBlock(kernel=3, expansion=3, stride=1, alpha=self.alpha, filters=24, block_id=2)(x)
        x = InvertedResBlock(kernel=3, expansion=3, stride=1, alpha=self.alpha, filters=24, block_id=3)(x)

        x = InvertedResBlock(kernel=5, expansion=3, stride=2, alpha=self.alpha, filters=40, block_id=4)(x)
        x = InvertedResBlock(kernel=5, expansion=3, stride=1, alpha=self.alpha, filters=40, block_id=5)(x)
        x = InvertedResBlock(kernel=5, expansion=3, stride=1, alpha=self.alpha, filters=40, block_id=6)(x)

        x = InvertedResBlock(kernel=5, expansion=6, stride=2, alpha=self.alpha, filters=80, block_id=7)(x)
        x = InvertedResBlock(kernel=5, expansion=6, stride=1, alpha=self.alpha, filters=80, block_id=8)(x)
        x = InvertedResBlock(kernel=5, expansion=6, stride=1, alpha=self.alpha, filters=80, block_id=9)(x)

        x = InvertedResBlock(kernel=3, expansion=6, stride=1, alpha=self.alpha, filters=96, block_id=10)(x)
        x = InvertedResBlock(kernel=3, expansion=6, stride=1, alpha=self.alpha, filters=96, block_id=11)(x)

        x = InvertedResBlock(kernel=5, expansion=6, stride=2, alpha=self.alpha, filters=192, block_id=12)(x)
        x = InvertedResBlock(kernel=5, expansion=6, stride=1, alpha=self.alpha, filters=192, block_id=13)(x)
        x = InvertedResBlock(kernel=5, expansion=6, stride=1, alpha=self.alpha, filters=192, block_id=14)(x)
        x = InvertedResBlock(kernel=5, expansion=6, stride=1, alpha=self.alpha, filters=192, block_id=15)(x)

        x = InvertedResBlock(kernel=3, expansion=6, stride=1, alpha=self.alpha, filters=320, block_id=16)(x)

        x = layers.GlobalAveragePooling1D()(x)
        y = layers.Dense(self.num_classes, activation=self.classifier_activation, use_bias=True, name="prediction")(x)

        model = Model(inputs, y)
        return model


def MnasNet(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax',
            alpha=1.0, depth_multiplier=1):
    if input_shape is None:
        input_shape = (256 * 3, 1)

    if weights in ['hasc', 'HASC'] and include_top and classes != 6:
        raise ValueError('If using `weights` as `"hasc"` with `include_top`'
                         ' as true, `classes` should be 6')

    # Build model
    model = BaseMnasNet(input_shape, classes, classifier_activation, alpha, depth_multiplier)()

    if weights is not None:
        if weights in ['hasc', "HASC"]:
            weights = 'weights/mnasnet/mnasnet_hasc_weights_{}.hdf5'.format(int(input_shape[0] / 3))

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
    model = MnasNet(include_top=False, weights=None,
                    pooling='avg')

    model.summary()
