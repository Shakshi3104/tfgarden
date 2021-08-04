import os

from tensorflow.keras import layers
from tensorflow.keras.models import Model


class ConvBlock:
    def __init__(self, growth_rate, name):
        """
        A building block for a dense block

        growth_rate: float, growth rate at dense layers.
        name: string, block label.
        """
        self.growth_rate = growth_rate
        self.name = name

    def __call__(self, x):
        """
        x: input tensor.
        returns: tensor for the block.
        """
        x1 = layers.BatchNormalization(
            epsilon=1.001e-5, name=self.name + "_0_bn"
        )(x)
        x1 = layers.Activation('relu', name=self.name + "_0_relu")(x1)
        x1 = layers.Conv1D(4 * self.growth_rate, 1, use_bias=False, name=self.name + "_1_conv")(x1)
        x1 = layers.BatchNormalization(epsilon=1.001e-5, name=self.name + "_1_bn")(x1)
        x1 = layers.Activation('relu', name=self.name + "_1_relu")(x1)
        x1 = layers.Conv1D(self.growth_rate, 3, padding='same', use_bias=False, name=self.name + "_2_conv")(x1)
        x = layers.Concatenate(name=self.name + "_concat")([x, x1])
        return x


class TransitionBlock:
    def __init__(self, reduction, name):
        """
        A transition block

        reduction: float, compression rate at transition layers.
        name: string, block label.
        """
        self.reduction = reduction
        self.name = name

    def __call__(self, x):
        """
        x: input tensor.
        returns: output tensor for the block.
        """
        x = layers.BatchNormalization(epsilon=1.001e-5, name=self.name + "_bn")(x)
        x = layers.Activation('relu', name=self.name + "_relu")(x)
        x = layers.Conv1D(int(x.shape[-1] * self.reduction),
                          1,
                          use_bias=False,
                          name=self.name + "_conv")(x)
        x = layers.AveragePooling1D(2, strides=2, name=self.name + "_pool")(x)
        return x


class DenseBlock:
    def __init__(self, blocks, name):
        """
        A dense block.

        blocks: integer, the number of building block.
        name: string, block label
        """
        self.blocks = blocks
        self.name = name

    def __call__(self, x):
        """
        x: input tensor.
        returns: output tensor for the block.
        """
        for i in range(self.blocks):
            x = ConvBlock(32, name=self.name + "_block" + str(i + 1))(x)
        return x


def DenseNet(number, include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax'):
    if input_shape is None:
        input_shape = (256 * 3, 1)

    if weights in ['hasc', 'HASC'] and include_top and classes != 6:
        raise ValueError('If using `weights` as `"hasc"` with `include_top`'
                         ' as true, `classes` should be 6')

    # number of blocks
    if number == 121:
        blocks = [6, 12, 24, 16]
    elif number == 169:
        blocks = [6, 12, 32, 32]
    elif number == 201:
        blocks = [6, 12, 48, 32]
    else:
        raise ValueError('`number` should be 121, 169 or 201')

    inputs = layers.Input(shape=input_shape)

    x = layers.ZeroPadding1D(padding=(3, 3))(inputs)
    x = layers.Conv1D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = layers.BatchNormalization(epsilon=1.001e-5, name='conv1/bn')(x)
    x = layers.Activation('relu', name='conv1/relu')(x)
    x = layers.ZeroPadding1D(padding=(1, 1))(x)
    x = layers.MaxPooling1D(3, strides=2, name='pool1')(x)

    x = DenseBlock(blocks[0], name='conv2')(x)
    x = TransitionBlock(0.5, name='pool2')(x)
    x = DenseBlock(blocks[1], name='conv3')(x)
    x = TransitionBlock(0.5, name='pool3')(x)
    x = DenseBlock(blocks[2], name='conv4')(x)
    x = TransitionBlock(0.5, name='pool4')(x)
    x = DenseBlock(blocks[3], name='conv5')(x)

    x = layers.BatchNormalization(epsilon=1.001e-5, name='bn')(x)
    x = layers.Activation('relu', name='relu')(x)

    x = layers.GlobalAveragePooling1D(name='avg_pool')(x)
    y = layers.Dense(classes, activation=classifier_activation, name='predictions')(x)

    # Create model.
    model_ = Model(inputs, y)

    if weights is not None:
        if weights in ['hasc', "HASC"]:
            weights = 'weights/densenet{}/densenet{}_hasc_weights_{}_{}.hdf5'.format(number, number,
                                                                                     int(input_shape[0]),
                                                                                     int(input_shape[1]))
        # hasc or weights fileで初期化
        if os.path.exists(weights):
            print("Load weights from {}".format(weights))
            model_.load_weights(weights)
        else:
            print("Not exist weights: {}".format(weights))

    # topを含まないとき
    if not include_top:
        if pooling is None:
            # topを削除する
            model_ = Model(inputs=model_.input, outputs=model_.layers[-3].output)
        elif pooling == 'avg':
            y = layers.GlobalAveragePooling1D()(model_.layers[-3].output)
            model_ = Model(inputs=model_.input, outputs=y)
        elif pooling == 'max':
            y = layers.GlobalMaxPooling1D()(model_.layers[-3].output)
            model_ = Model(inputs=model_.input, outputs=y)
        else:
            print("Not exist pooling option: {}".format(pooling))
            model_ = Model(inputs=model_.input, outputs=model_.layers[-3].output)

    return model_


def DenseNet121(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax'):
    model = DenseNet(121, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


def DenseNet169(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax'):
    model = DenseNet(169, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


def DenseNet201(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax'):
    model = DenseNet(201, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


if __name__ == "__main__":
    model = DenseNet121(include_top=False, weights=None,
                        pooling='max')

    model.summary()