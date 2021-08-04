import os

from tensorflow.keras.layers import Conv1D, BatchNormalization, ReLU, SeparableConv1D, ZeroPadding1D
from tensorflow.keras.layers import Input, Reshape, Dropout, Activation, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model


class ConvBlock:
    def __init__(self, filters, alpha, kernel=3, strides=1):
        self.filters = filters
        self.alpha = alpha
        self.kernel = kernel
        self.strides = strides

    def __call__(self, x):
        filters = int(self.filters * self.alpha)
        x = Conv1D(filters, self.kernel, self.strides, padding='same', use_bias=False, name='conv1')(x)
        x = BatchNormalization(name='conv1_bn')(x)
        x = ReLU(6., name='conv1_relu')(x)
        return x


class DepthwiseConvBlock:
    def __init__(self, pointwise_conv_filter, alpha, depth_multipliter=1, strides=1, block_id=1):
        self.pointwise_conv_filter = pointwise_conv_filter
        self.alpha = alpha
        self.depth_multipliter = depth_multipliter
        self.strides = strides
        self.block_id = block_id

    def __call__(self, x):
        if self.strides != 1:
            x = ZeroPadding1D((0, 1), name='conv_pad_%d' % self.block_id)(x)

        x = SeparableConv1D(int(x.shape[-1]),
                            3,
                            padding='same' if self.strides == 1 else 'valid',
                            depth_multiplier=self.depth_multipliter,
                            strides=self.strides, use_bias=False,
                            name='conv_dw_%d' % self.block_id)(x)
        x = BatchNormalization(name='conv_dw_%d_bn' % self.block_id)(x)
        x = ReLU(6., name='conv_dw_%d_relu' % self.block_id)(x)

        x = Conv1D(self.pointwise_conv_filter, 1, padding='same', use_bias=False,
                   strides=1, name='conv_pw_%d' % self.block_id)(x)
        x = BatchNormalization(name='conv_pw_%d_bn' % self.block_id)(x)
        x = ReLU(6., name='conv_pw_%d_relu' % self.block_id)(x)
        return x


def MobileNet(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax',
              alpha=1.0, depth_multiplier=1, dropout=1e-3):
    if input_shape is None:
        input_shape = (256*3, 1)

    if weights in ['hasc', 'HASC'] and include_top and classes != 6:
        raise ValueError('If using `weights` as `"hasc"` with `include_top`'
                         ' as true, `classes` should be 6')

    inputs = Input(shape=input_shape)

    x = ConvBlock(32, alpha, strides=2)(inputs)
    x = DepthwiseConvBlock(64, alpha, depth_multiplier, block_id=1)(x)

    x = DepthwiseConvBlock(128, alpha, depth_multiplier, strides=2, block_id=2)(x)
    x = DepthwiseConvBlock(128, alpha, depth_multiplier, block_id=3)(x)

    x = DepthwiseConvBlock(256, alpha, depth_multiplier, strides=2, block_id=4)(x)
    x = DepthwiseConvBlock(256, alpha, depth_multiplier, block_id=5)(x)

    x = DepthwiseConvBlock(512, alpha, depth_multiplier, strides=2, block_id=6)(x)
    x = DepthwiseConvBlock(512, alpha, depth_multiplier, block_id=7)(x)
    x = DepthwiseConvBlock(512, alpha, depth_multiplier, block_id=8)(x)
    x = DepthwiseConvBlock(512, alpha, depth_multiplier, block_id=9)(x)
    x = DepthwiseConvBlock(512, alpha, depth_multiplier, block_id=10)(x)
    x = DepthwiseConvBlock(512, alpha, depth_multiplier, block_id=11)(x)

    x = DepthwiseConvBlock(1024, alpha, depth_multiplier, strides=2, block_id=12)(x)
    x = DepthwiseConvBlock(1024, alpha, depth_multiplier, block_id=13)(x)

    shape = (1, int(1024 * alpha))

    x = GlobalAveragePooling1D()(x)
    x = Reshape(shape, name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)
    x = Conv1D(classes, 1, padding='same', name='conv_preds')(x)
    x = Reshape((classes,), name='reshape_2')(x)
    x = Activation(activation=classifier_activation, name='predictions')(x)

    model = Model(inputs=inputs, outputs=x)

    if weights is not None:
        if weights in ['hasc', "HASC"]:
            weights = 'weights/mobilenet/mobilenet_hasc_weights_{}_{}.hdf5'.format(int(input_shape[0]),
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
            y = GlobalAveragePooling1D()(model.layers[-7].output)
            model = Model(inputs=model.input, outputs=y)
        elif pooling == 'max':
            y = GlobalMaxPooling1D()(model.layers[-7].output)
            model = Model(inputs=model.input, outputs=y)
        else:
            print("Not exist pooling option: {}".format(pooling))
            model = Model(inputs=model.input, outputs=model.layers[-7].output)

    return model


if __name__ == '__main__':
    model = MobileNet(include_top=True, weights=None, pooling=None)
    print(model.summary())
