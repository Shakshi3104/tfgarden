import os

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, SeparableConv1D, ZeroPadding1D, \
    GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Input, Reshape, Dropout, Activation, Add, Dense
from tensorflow.keras.models import Model

from .base import DLModelBuilder


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResBlock:
    def __init__(self, expansion, stride, alpha, filters, block_id):
        """Inverted ResNet block."""
        self.expansion = expansion
        self.stride = stride
        self.alpha = alpha
        self.filters = filters
        self.block_id = block_id

    def __call__(self, x):
        inputs = x
        in_channels = x.shape[-1]
        pointwise_conv_filters = int(self.filters * self.alpha)
        pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
        prefix = 'block_{}_'.format(self.block_id)

        if self.block_id:
            # Expand
            x = Conv1D(self.expansion * in_channels,
                       kernel_size=1,
                       padding='same',
                       use_bias=False,
                       activation=None,
                       name=prefix + 'expand')(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'expand_BN')(x)
            x = ReLU(6., name=prefix + 'expand_relu')(x)
        else:
            prefix = 'expanded_conv_'

        # Depthwise
        if self.stride == 2:
            x = ZeroPadding1D(padding=1, name=prefix + 'pad')(x)

        x = SeparableConv1D(int(x.shape[-1]),
                            kernel_size=3,
                            strides=self.stride,
                            activation=None,
                            use_bias=False,
                            padding='same' if self.stride == 1 else 'valid',
                            name=prefix + 'depthwise')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'depthwise_BN')(x)

        x = ReLU(6., name=prefix + 'depthwise_relu')(x)

        # Project
        x = Conv1D(pointwise_filters, kernel_size=1, padding='same', use_bias=False,
                   activation=None, name=prefix + 'project')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name=prefix + 'project_BN')(x)

        if in_channels == pointwise_filters and self.stride == 1:
            return Add(name=prefix + 'add')([inputs, x])

        return x


class BaseMobileNetV2(DLModelBuilder):
    def __init__(self, input_shape=(256 * 3, 1), alpha=1.0, num_classes=6, classifier_activation='softmax'):
        self.input_shape = input_shape
        self.alpha = alpha
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def __call__(self, *args, **kwargs):
        model = self.get_model()
        return model

    def get_model(self):
        inputs = Input(shape=self.input_shape)

        first_block_filters = _make_divisible(32 * self.alpha, 8)
        x = Conv1D(first_block_filters,
                   kernel_size=3,
                   strides=2,
                   padding='same',
                   use_bias=False,
                   name='Conv1')(inputs)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='bn_Conv1')(x)
        x = ReLU(6., name='Conv1_relu')(x)

        x = InvertedResBlock(filters=16, alpha=self.alpha, stride=1, expansion=1, block_id=0)(x)

        x = InvertedResBlock(filters=24, alpha=self.alpha, stride=2, expansion=6, block_id=1)(x)
        x = InvertedResBlock(filters=24, alpha=self.alpha, stride=1, expansion=6, block_id=2)(x)

        x = InvertedResBlock(filters=32, alpha=self.alpha, stride=2, expansion=6, block_id=3)(x)
        x = InvertedResBlock(filters=32, alpha=self.alpha, stride=1, expansion=6, block_id=4)(x)
        x = InvertedResBlock(filters=32, alpha=self.alpha, stride=1, expansion=6, block_id=5)(x)

        x = InvertedResBlock(filters=64, alpha=self.alpha, stride=2, expansion=6, block_id=6)(x)
        x = InvertedResBlock(filters=64, alpha=self.alpha, stride=1, expansion=6, block_id=7)(x)
        x = InvertedResBlock(filters=64, alpha=self.alpha, stride=1, expansion=6, block_id=8)(x)
        x = InvertedResBlock(filters=64, alpha=self.alpha, stride=1, expansion=6, block_id=9)(x)

        x = InvertedResBlock(filters=96, alpha=self.alpha, stride=1, expansion=6, block_id=10)(x)
        x = InvertedResBlock(filters=96, alpha=self.alpha, stride=1, expansion=6, block_id=11)(x)
        x = InvertedResBlock(filters=96, alpha=self.alpha, stride=1, expansion=6, block_id=12)(x)

        x = InvertedResBlock(filters=160, alpha=self.alpha, stride=2, expansion=6, block_id=13)(x)
        x = InvertedResBlock(filters=160, alpha=self.alpha, stride=1, expansion=6, block_id=14)(x)
        x = InvertedResBlock(filters=160, alpha=self.alpha, stride=1, expansion=6, block_id=15)(x)

        x = InvertedResBlock(filters=320, alpha=self.alpha, stride=1, expansion=6, block_id=16)(x)

        if self.alpha > 1.0:
            last_block_filters = _make_divisible(1280 * self.alpha, 8)
        else:
            last_block_filters = 1280

        x = Conv1D(last_block_filters, kernel_size=1, use_bias=False, name='Conv_1')(x)
        x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_1_bn')(x)
        x = ReLU(6., name='out_relu')(x)

        x = GlobalAveragePooling1D()(x)
        x = Dense(self.num_classes, activation=self.classifier_activation, name='predictions')(x)

        # Create model
        model = Model(inputs=inputs, outputs=x)
        return model


def MobileNetV2(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                classifier_activation='softmax',
                alpha=1.0):
    if input_shape is None:
        input_shape = (256 * 3, 1)

    if weights in ['hasc', 'HASC'] and include_top and classes != 6:
        raise ValueError('If using `weights` as `"hasc"` with `include_top`'
                         ' as true, `classes` should be 6')

    # Build model
    model = BaseMobileNetV2(input_shape, alpha, classes, classifier_activation)()

    if weights is not None:
        if weights in ['hasc', "HASC"]:
            weights = 'weights/mobilenetv2/mobilenetv2_hasc_weights_{}_{}.hdf5'.format(int(input_shape[0]),
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
            model = Model(inputs=model.input, outputs=model.layers[-3].output)
        elif pooling == 'avg':
            y = GlobalAveragePooling1D()(model.layers[-3].output)
            model = Model(inputs=model.input, outputs=y)
        elif pooling == 'max':
            y = GlobalMaxPooling1D()(model.layers[-3].output)
            model = Model(inputs=model.input, outputs=y)
        else:
            print("Not exist pooling option: {}".format(pooling))
            model = Model(inputs=model.input, outputs=model.layers[-3].output)

    return model


if __name__ == "__main__":
    model = MobileNetV2(include_top=False, weights=None, pooling=None)
    print(model.summary())
