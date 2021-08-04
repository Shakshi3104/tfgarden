import os

from tensorflow.keras import layers
from tensorflow.keras.models import Model


class Conv1DBN:
    def __init__(self, filters, kernel_size, padding='same', strides=1, name=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.name = name

    def __call__(self, x):
        if self.name is not None:
            bn_name = self.name + "_bn"
            conv_name = self.name + "_conv"
        else:
            bn_name = None
            conv_name = None

        x = layers.Conv1D(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=False,
            name=conv_name
        )(x)
        x = layers.BatchNormalization(scale=False, name=bn_name)(x)
        x = layers.Activation('relu', name=self.name)(x)
        return x


def InceptionV3(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax'):
    if input_shape is None:
        input_shape = (256*3, 1)

    if weights in ['hasc', 'HASC'] and include_top and classes != 6:
        raise ValueError('If using `weights` as `"hasc"` with `include_top`'
                         ' as true, `classes` should be 6')

    inputs = layers.Input(shape=input_shape)

    x = Conv1DBN(32, 3, strides=2, padding='valid')(inputs)
    x = Conv1DBN(32, 3, padding='valid')(x)
    x = Conv1DBN(64, 3)(x)
    x = layers.MaxPooling1D(3, strides=2)(x)

    x = Conv1DBN(80, 1, padding='valid')(x)
    x = Conv1DBN(192, 3, padding='valid')(x)
    x = layers.MaxPooling1D(3, strides=2)(x)

    # mixed 0
    branch1x1 = Conv1DBN(64, 1)(x)

    branch5x5 = Conv1DBN(48, 1)(x)
    branch5x5 = Conv1DBN(64, 5)(branch5x5)

    branch3x3dbl = Conv1DBN(64, 1)(x)
    branch3x3dbl = Conv1DBN(96, 3)(branch3x3dbl)
    branch3x3dbl = Conv1DBN(96, 3)(branch3x3dbl)

    branch_pool = layers.AveragePooling1D(3, strides=1, padding='same')(x)
    branch_pool = Conv1DBN(32, 1)(branch_pool)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], name='mixed0')

    # mixed 1
    branch1x1 = Conv1DBN(64, 1)(x)

    branch5x5 = Conv1DBN(48, 1)(x)
    branch5x5 = Conv1DBN(64, 5)(branch5x5)

    branch3x3dbl = Conv1DBN(64, 1)(x)
    branch3x3dbl = Conv1DBN(96, 3)(branch3x3dbl)
    branch3x3dbl = Conv1DBN(96, 3)(branch3x3dbl)

    branch_pool = layers.AveragePooling1D(3, strides=1, padding='same')(x)
    branch_pool = Conv1DBN(64, 1)(branch_pool)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], name='mixed1')

    # mixed 2
    branch1x1 = Conv1DBN(64, 1)(x)

    branch5x5 = Conv1DBN(48, 1)(x)
    branch5x5 = Conv1DBN(64, 5)(branch5x5)

    branch3x3dbl = Conv1DBN(64, 1)(x)
    branch3x3dbl = Conv1DBN(96, 3)(branch3x3dbl)
    branch3x3dbl = Conv1DBN(96, 3)(branch3x3dbl)

    branch_pool = layers.AveragePooling1D(3, strides=1, padding='same')(x)
    branch_pool = Conv1DBN(64, 1)(branch_pool)
    x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], name='mixed2')

    # mixed 3
    branch3x3 = Conv1DBN(384, 3, strides=2, padding='valid')(x)

    branch3x3dbl = Conv1DBN(64, 1)(x)
    branch3x3dbl = Conv1DBN(96, 3)(branch3x3dbl)
    branch3x3dbl = Conv1DBN(96, 3, strides=2, padding='valid')(branch3x3dbl)

    branch_pool = layers.MaxPooling1D(3, strides=2)(x)
    x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool], name='mixed3')

    # mixed 4
    branch1x1 = Conv1DBN(192, 1)(x)

    branch7x7 = Conv1DBN(128, 1)(x)
    branch7x7 = Conv1DBN(128, 1)(branch7x7)
    branch7x7 = Conv1DBN(128, 7)(branch7x7)

    branch7x7dbl = Conv1DBN(128, 1)(x)
    branch7x7dbl = Conv1DBN(128, 7)(branch7x7dbl)
    branch7x7dbl = Conv1DBN(128, 1)(branch7x7dbl)
    branch7x7dbl = Conv1DBN(128, 7)(branch7x7dbl)
    branch7x7dbl = Conv1DBN(128, 1)(branch7x7dbl)

    branch_pool = layers.AveragePooling1D(3, strides=1, padding='same')(x)
    branch_pool = Conv1DBN(192, 1)(branch_pool)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], name='mixed4')

    # mixed 5, 6
    for i in range(2):
        branch1x1 = Conv1DBN(192, 1)(x)

        branch7x7 = Conv1DBN(160, 1)(x)
        branch7x7 = Conv1DBN(160, 1)(branch7x7)
        branch7x7 = Conv1DBN(192, 7)(branch7x7)

        branch7x7dbl = Conv1DBN(160, 1)(x)
        branch7x7dbl = Conv1DBN(160, 7)(branch7x7dbl)
        branch7x7dbl = Conv1DBN(160, 1)(branch7x7dbl)
        branch7x7dbl = Conv1DBN(160, 7)(branch7x7dbl)
        branch7x7dbl = Conv1DBN(192, 1)(branch7x7dbl)

        branch_pool = layers.AveragePooling1D(3, strides=1, padding='same')(x)
        branch_pool = Conv1DBN(192, 1)(branch_pool)
        x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], name='mixed' + str(5 + i))

    # mixed 7
    branch1x1 = Conv1DBN(192, 1)(x)

    branch7x7 = Conv1DBN(192, 1)(x)
    branch7x7 = Conv1DBN(192, 1)(branch7x7)
    branch7x7 = Conv1DBN(192, 7)(branch7x7)

    branch7x7dbl = Conv1DBN(192, 1)(x)
    branch7x7dbl = Conv1DBN(192, 7)(branch7x7dbl)
    branch7x7dbl = Conv1DBN(192, 1)(branch7x7dbl)
    branch7x7dbl = Conv1DBN(192, 7)(branch7x7dbl)
    branch7x7dbl = Conv1DBN(192, 1)(branch7x7dbl)

    branch_pool = layers.AveragePooling1D(3, strides=1, padding='same')(x)
    branch_pool = Conv1DBN(192, 1)(branch_pool)
    x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], name='mixed7')

    # mixed 8
    branch3x3 = Conv1DBN(192, 1)(x)
    branch3x3 = Conv1DBN(320, 3, strides=2, padding='valid')(branch3x3)

    branch7x7x3 = Conv1DBN(192, 1)(x)
    branch7x7x3 = Conv1DBN(192, 1)(branch7x7x3)
    branch7x7x3 = Conv1DBN(192, 7)(branch7x7x3)
    branch7x7x3 = Conv1DBN(192, 3, strides=2, padding='valid')(branch7x7x3)

    branch_pool = layers.MaxPooling1D(3, strides=2)(x)
    x = layers.concatenate([branch3x3, branch7x7x3, branch_pool], name='mixed8')

    # mixed 9, 10
    for i in range(2):
        branch1x1 = Conv1DBN(320, 1)(x)

        branch3x3 = Conv1DBN(384, 1)(x)
        branch3x3_1 = Conv1DBN(384, 1)(branch3x3)
        branch3x3_2 = Conv1DBN(384, 3)(branch3x3)
        branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], name='mixed9_' + str(i))

        branch3x3dbl = Conv1DBN(448, 1)(x)
        branch3x3dbl = Conv1DBN(384, 3)(branch3x3dbl)
        branch3x3dbl_1 = Conv1DBN(384, 1)(branch3x3dbl)
        branch3x3dbl_2 = Conv1DBN(384, 3)(branch3x3dbl)
        branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2])

        branch_pool = layers.AveragePooling1D(3, strides=1, padding='same')(x)
        branch_pool = Conv1DBN(192, 1)(branch_pool)
        x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], name='mixed' + str(9 + i))

    # Classification block
    x = layers.GlobalAveragePooling1D(name='avg_pool')(x)
    y = layers.Dense(classes, activation=classifier_activation, name='predictions')(x)

    model = Model(inputs, y)

    if weights is not None:
        if weights in ['hasc', "HASC"]:
            weights = 'weights/inceptionv3/inceptionv3_hasc_weights_{}_{}.hdf5'.format(int(input_shape[0]),
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
    model = InceptionV3(include_top=False, weights=None,
                        pooling='max')
    model.summary()