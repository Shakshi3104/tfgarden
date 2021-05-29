import os

from tensorflow.python.keras import backend
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from .base import DLModelBuilder


class SeparableConvBlock:
    def __init__(self, filters, kernel_size=3, strides=1, block_id=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.block_id = block_id

    def __call__(self, ip):
        with backend.name_scope('separable_conv_block_%s' % self.block_id):
            x = layers.Activation('relu')(ip)
            if self.strides == 2:
                x = layers.ZeroPadding1D(padding=self.kernel_size, name='separable_conv_1_pad_%s' % self.block_id)(x)
                conv_pad = 'valid'
            else:
                conv_pad = 'same'

            x = layers.SeparableConv1D(
                self.filters,
                self.kernel_size,
                strides=self.strides,
                name='separable_conv_1_%s' % self.block_id,
                padding=conv_pad,
                use_bias=False,
                kernel_initializer='he_normal'
            )(x)
            x = layers.BatchNormalization(momentum=0.9997, epsilon=1e-3,
                                          name='separable_conv_1_bn_%s' % self.block_id)(x)
            x = layers.Activation('relu')(x)
            x = layers.SeparableConv1D(
                self.filters,
                self.kernel_size,
                name='separable_conv_2_%s' % self.block_id,
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal'
            )(x)
            x = layers.BatchNormalization(momentum=0.9997, epsilon=1e-3,
                                          name='separable_conv_2_bn_%s' % self.block_id)(x)
        return x


class AdjustBlock:
    def __init__(self, filters, block_id=None):
        self.filters = filters
        self.block_id = block_id

    def __call__(self, p, ip):
        ip_shape = ip.shape

        if p is not None:
            p_shape = p.shape

        with backend.name_scope('adjust_block'):
            if p is None:
                p = ip

            elif p_shape[-2] != ip_shape[-2]:
                with backend.name_scope('adjust_reduction_block_%s' % self.block_id):
                    p = layers.Activation('relu', name='adjust_relu_1_%s' % self.block_id)(p)
                    p1 = layers.AveragePooling1D(1,
                                                 strides=2,
                                                 padding='valid',
                                                 name='adjust_avg_pool_1_%s' % self.block_id)(p)
                    p1 = layers.Conv1D(
                        self.filters // 2, 1,
                        padding='same',
                        use_bias=False,
                        name='adjust_conv_1_%s' % self.block_id,
                        kernel_initializer='he_normal'
                    )(p1)

                    p2 = layers.ZeroPadding1D((0, 1))(p)
                    p2 = layers.Cropping1D((1, 0))(p2)
                    p2 = layers.AveragePooling1D(1, strides=2,
                                                 padding='valid',
                                                 name='adjust_avg_pool_2_%s' % self.block_id)(p2)
                    p2 = layers.Conv1D(
                        self.filters // 2, 1,
                        padding='same',
                        use_bias=False,
                        name='adjust_conv_2_%s' % self.block_id,
                        kernel_initializer='he_normal'
                    )(p2)

                    p = layers.concatenate([p1, p2], axis=-1)
                    p = layers.BatchNormalization(momentum=0.9997, epsilon=1e-3,
                                                  name='adjust_bn_%s' % self.block_id)(p)

            elif p_shape[-1] != self.filters:
                with backend.name_scope('adjust_projection_block_%s' % self.block_id):
                    p = layers.Activation('relu')(p)
                    p = layers.Conv1D(
                        self.filters, 1,
                        strides=1,
                        padding='same',
                        name='adjust_conv_projection_%s' % self.block_id,
                        use_bias=False,
                        kernel_initializer='he_normal'
                    )(p)
                    p = layers.BatchNormalization(
                        momentum=0.9997,
                        epsilon=1e-3,
                        name='adjust_bn_%s' % self.block_id
                    )(p)
        return p


class NormalACell:
    def __init__(self, filters, block_id=None):
        """
        Adds a Normal cell for NASNet-A (Fig. 4 in the paper).
        """
        self.filters = filters
        self.block_id = block_id

    def __call__(self, ip, p):
        with backend.name_scope('normal_A_block_%s' % self.block_id):
            p = AdjustBlock(self.filters, self.block_id)(p, ip)

            h = layers.Activation('relu')(ip)
            h = layers.Conv1D(
                self.filters, 1,
                strides=1,
                padding='same',
                name='normal_conv_1_%s' % self.block_id,
                use_bias=False,
                kernel_initializer='he_normal'
            )(h)
            h = layers.BatchNormalization(
                momentum=0.9997,
                epsilon=1e-3,
                name='normal_bn_1_%s' % self.block_id
            )(h)

            with backend.name_scope('block_1'):
                x1_1 = SeparableConvBlock(self.filters, kernel_size=5, block_id='normal_left1_%s' % self.block_id)(h)
                x1_2 = SeparableConvBlock(self.filters, block_id='normal_right1_%s' % self.block_id)(p)

                x1_1, x1_2 = padding(x1_1, x1_2)

                x1 = layers.add([x1_1, x1_2], name='normal_add_1_%s' % self.block_id)

            with backend.name_scope('block_2'):
                x2_1 = SeparableConvBlock(self.filters, kernel_size=5, block_id='normal_left2_%s' % self.block_id)(p)
                x2_2 = SeparableConvBlock(self.filters, kernel_size=3, block_id='normal_right2_%s' % self.block_id)(p)
                x2 = layers.add([x2_1, x2_2], name='normal_add_2_%s' % self.block_id)

            with backend.name_scope('block_3'):
                x3 = layers.AveragePooling1D(3,
                                             strides=1,
                                             padding='same',
                                             name='normal_left3_%s' % self.block_id)(h)
                x3, p = padding(x3, p)

                x3 = layers.add([x3, p], name='normal_add_3_%s' % self.block_id)

            with backend.name_scope('block_4'):
                x4_1 = layers.AveragePooling1D(3,
                                               strides=1,
                                               padding='same',
                                               name='normal_left4_%s' % self.block_id)(p)
                x4_2 = layers.AveragePooling1D(3,
                                               strides=1,
                                               padding='same',
                                               name='normal_right4_%s' % self.block_id)(p)
                x4 = layers.add([x4_1, x4_2], name='normal_add_4_%s' % self.block_id)

            with backend.name_scope('block_5'):
                x5 = SeparableConvBlock(self.filters, block_id='normal_left5_%s' % self.block_id)(h)
                x5 = layers.add([x5, h], name='normal_add_5_%s' % self.block_id)

            if x2.shape[-2] < p.shape[-2]:
                adds = p.shape[-2] - x2.shape[-2]
                x2 = layers.ZeroPadding1D((0, adds))(x2)

            x = layers.concatenate([p, x1, x2, x3, x4, x5],
                                   name='normal_concat_%s' % self.block_id)

        return x, ip


class ReductionACell:
    def __init__(self, filters, block_id=None):
        """
        Adds a Reduction cell for NASNet-A (Fig.4 in the paper).
        """
        self.filters = filters
        self.block_id = block_id

    def __call__(self, ip, p):
        with backend.name_scope('reduction_A_block_%s' % self.block_id):
            p = AdjustBlock(self.filters, self.block_id)(p, ip)

            h = layers.Activation('relu')(ip)
            h = layers.Conv1D(
                self.filters, 1,
                strides=1,
                padding='same',
                name='reduction_conv_1_%s' % self.block_id
            )(h)
            h = layers.BatchNormalization(momentum=0.9997, epsilon=1e-3,
                                          name='reduction_bn_1_%s' % self.block_id)(h)
            h3 = layers.ZeroPadding1D(3, name='reduction_pad_1_%s' % self.block_id)(h)

            with backend.name_scope('block_1'):
                x1_1 = SeparableConvBlock(self.filters, 5,
                                          strides=2,
                                          block_id='reduction_left1_%s' % self.block_id)(h)
                x1_2 = SeparableConvBlock(self.filters, 7,
                                          strides=2,
                                          block_id='reduction_right1_%s' % self.block_id)(p)

                x1_1, x1_2 = padding(x1_1, x1_2)

                x1 = layers.add([x1_1, x1_2], name='reduction_add_1_%s' % self.block_id)

            with backend.name_scope('block_2'):
                x2_1 = layers.MaxPooling1D(3, strides=2,
                                           padding='valid',
                                           name='reduction_left2_%s' % self.block_id)(h3)
                x2_2 = SeparableConvBlock(self.filters, 7,
                                          strides=2,
                                          block_id='reduction_right2_%s' % self.block_id)(p)

                x2_1, x2_2 = padding(x2_1, x2_2)

                x2 = layers.add([x2_1, x2_2], name='reduction_add_2_%s' % self.block_id)

            with backend.name_scope('block_3'):
                x3_1 = layers.AveragePooling1D(3,
                                               strides=2,
                                               padding='valid',
                                               name='reduction_left3_%s' % self.block_id)(h3)
                x3_2 = SeparableConvBlock(self.filters, 5,
                                          strides=2,
                                          block_id='reduction_right3_%s' % self.block_id)(p)

                x3_1, x3_2 = padding(x3_1, x3_2)

                x3 = layers.add([x3_1, x3_2], name='reduction_add3_%s' % self.block_id)

            with backend.name_scope('block_4'):
                x4 = layers.AveragePooling1D(3,
                                             strides=1,
                                             padding='same',
                                             name='reduction_left4_%s' % self.block_id)(x1)

                x2, x4 = padding(x2, x4)

                x4 = layers.add([x2, x4])

            with backend.name_scope('block_5'):
                x5_1 = SeparableConvBlock(self.filters, 3, block_id='reduction_left4_%s' % self.block_id)(x1)
                x5_2 = layers.MaxPooling1D(3, strides=2,
                                           padding='valid',
                                           name='reduction_right5_%s' % self.block_id)(h3)

                x5_1, x5_2 = padding(x5_1, x5_2)

                x5 = layers.add([x5_1, x5_2], name='reduction_add4_%s' % self.block_id)

            x3 = layers.ZeroPadding1D((0, 1))(x3)

            x = layers.concatenate([x2, x3, x4, x5], name='reduction_concat_%s' % self.block_id)

        return x, ip


def padding(x1, x2):
    if x1.shape[-2] > x2.shape[-2]:
        adds = x1.shape[-2] - x2.shape[-2]
        x2 = layers.ZeroPadding1D((0, adds))(x2)
    elif x2.shape[-2] > x1.shape[-2]:
        adds = x2.shape[-2] - x1.shape[-2]
        x1 = layers.ZeroPadding1D((0, adds))(x1)
    return x1, x2


class BaseNASNet(DLModelBuilder):
    def __init__(self, input_shape=(256 * 3, 1), penultimate_filters=4032, num_blocks=6, stem_block_filters=96,
                 skip_reduction=True, filter_multiplier=2, num_classes=6, classifier_activation='softmax'):
        self.input_shape = input_shape
        self.penultimate_filters = penultimate_filters
        self.num_blocks = num_blocks
        self.stem_block_filters = stem_block_filters
        self.skip_reduction = skip_reduction
        self.filter_multiplier = filter_multiplier
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def __call__(self, *args, **kwargs):
        model = self.get_model()
        return model

    def get_model(self):
        inputs = layers.Input(shape=self.input_shape)

        if self.penultimate_filters % (24 * (self.filter_multiplier ** 2)) != 0:
            raise ValueError(
                'For NASNet-A models, the `penultimate_filters` must be a multiple '
                'of 24 * (`filter_multiplier` ** 2). Current value: %d' %
                self.penultimate_filters)

        filters = self.penultimate_filters // 24

        x = layers.Conv1D(
            self.stem_block_filters, 3,
            strides=2,
            padding='valid',
            use_bias=False,
            name='stem_conv1',
            kernel_initializer='he_normal'
        )(inputs)

        x = layers.BatchNormalization(momentum=0.9997, epsilon=1e-3, name='stem_bn1')(x)

        p = None
        x, p = ReductionACell(filters // (self.filter_multiplier ** 2), block_id='stem_1')(x, p)
        x, p = ReductionACell(filters // self.filter_multiplier, block_id='stem_2')(x, p)

        for i in range(self.num_blocks):
            x, p = NormalACell(filters, block_id='%d' % i)(x, p)

        x, p0 = ReductionACell(filters * self.filter_multiplier, block_id='reduce_%d' % self.num_blocks)(x, p)

        p = p0 if not self.skip_reduction else p

        for i in range(self.num_blocks):
            x, p = NormalACell(filters * self.filter_multiplier, block_id='%d' % (self.num_blocks + i + 1))(x, p)

        x, p0 = ReductionACell(filters * self.filter_multiplier ** 2,
                               block_id='reduce_%d' % (2 * self.num_blocks))(x, p)

        p = p0 if not self.skip_reduction else p

        for i in range(self.num_blocks):
            x, p = NormalACell(filters * self.filter_multiplier ** 2, block_id='%d' % (2 * self.num_blocks + i + 1))(x,
                                                                                                                     p)

        x = layers.Activation('relu')(x)

        x = layers.GlobalAveragePooling1D()(x)
        y = layers.Dense(self.num_classes, activation=self.classifier_activation, name='predictions')(x)

        model = Model(inputs, y)

        return model


def __NASNet(type, include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
             classifier_activation='softmax'):
    if input_shape is None:
        input_shape = (256 * 3, 1)

    if weights in ['hasc', 'HASC'] and include_top and classes != 6:
        raise ValueError('If using `weights` as `"hasc"` with `include_top`'
                         ' as true, `classes` should be 6')

    if type == 'mobile':
        model = BaseNASNet(input_shape, 1056, 4, 32, False, 2, classes, classifier_activation)()
    elif type == 'large':
        model = BaseNASNet(input_shape, 4032, 6, 96, True, 2, classes, classifier_activation)()

    if weights is not None:
        if weights in ['hasc', "HASC"]:
            weights = 'weights/mobilenetv3/mobilenetv3_hasc_weights_{}.hdf5'.format(int(input_shape[0] / 3))

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


def NASNetLarge(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                classifier_activation='softmax'):
    model = __NASNet('large', include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


def NASNetMobile(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                 classifier_activation='softmax'):
    model = __NASNet('mobile', include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


if __name__ == "__main__":
    model = NASNetLarge(include_top=False, weights=None,
                        pooling=None)

    model.summary()
