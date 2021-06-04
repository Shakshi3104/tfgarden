import os

from tensorflow.keras import layers
from tensorflow.keras.models import Model

from .base import DLModelBuilder


class Conv1DBN:
    def __init__(self, filters, kernel_size, strides=1, padding='same', activation='relu', use_bias=False, name=None):
        """
        filters: filters in `Conv1D`
        kernel_size: kernel size as in `Conv1D`
        strides: strides in `Conv1D`
        padding: padding mode in `Conv1D`
        activation: activation in `Conv1D`
        use_bias: whether to use a bias in `Conv1D`
        name: name of the ops; will become `name + '_ac'` for the activation and `name + '_bn'` for the batch norm layer.
        """
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bias = use_bias
        self.name = name

    def __call__(self, x):
        """
        x: input tensor.
        returns: output tensor.
        """
        x = layers.Conv1D(
            self.filters,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            use_bias=self.use_bias,
            name=self.name
        )(x)
        if not self.use_bias:
            bn_name = None if self.name is None else self.name + '_bn'
            x = layers.BatchNormalization(scale=False, name=bn_name)(x)
        if self.activation is not None:
            ac_name = None if self.name is None else self.name + "_ac"
            x = layers.Activation(self.activation, name=ac_name)(x)
        return x


class InceptionResNetBlock:
    def __init__(self, scale, block_type, block_idx, activation='relu'):
        """
        Adds an Inception-ResNet block.

        This function builds 3 types of Inception-ResNet blocks mentioned
        in the paper, controlled by the `block_type` argument (which is the
        block name used in the official TF-slim implementation):
        - Inception-ResNet-A: `block_type='block35'`
        - Inception-ResNet-B: `block_type='block17'`
        - Inception-ResNet-C: `block_type='block8'`

        scale: scaling factor to scale the residuals (i.e., the output of passing
        `x` through an inception module) before adding them to the shortcut
        branch. Let `r` be the output from the residual branch, the output of this
        block will be `x + scale * r`.
        block_type: `'block35'`, `'block17'` or `'block8'`, determines the network
        structure in the residual branch.
        block_idx: an `int` used for generating layer names. The Inception-ResNet
        blocks are repeated many times in this network. We use `block_idx` to
        identify each of the repetitions. For example, the first
        Inception-ResNet-A block will have `block_type='block35', block_idx=0`,
        and the layer names will have a common prefix `'block35_0'`.
        activation: activation function to use at the end of the block (see
        [activations](../activations.md)). When `activation=None`, no activation
        is applied
        (i.e., "linear" activation: `a(x) = x`).
        """

        self.scale = scale
        self.block_type = block_type
        self.block_idx = block_idx
        self.activation = activation

    def __call__(self, x):
        if self.block_type == 'block35':
            branch_0 = Conv1DBN(32, 1)(x)
            branch_1 = Conv1DBN(32, 1)(x)
            branch_1 = Conv1DBN(32, 3)(branch_1)
            branch_2 = Conv1DBN(32, 1)(x)
            branch_2 = Conv1DBN(48, 3)(branch_2)
            branch_2 = Conv1DBN(64, 3)(branch_2)
            branches = [branch_0, branch_1, branch_2]
        elif self.block_type == 'block17':
            branch_0 = Conv1DBN(192, 1)(x)
            branch_1 = Conv1DBN(128, 1)(x)
            branch_1 = Conv1DBN(160, 7)(branch_1)
            branch_1 = Conv1DBN(192, 1)(branch_1)
            branches = [branch_0, branch_1]
        elif self.block_type == 'block8':
            branch_0 = Conv1DBN(192, 1)(x)
            branch_1 = Conv1DBN(192, 1)(x)
            branch_1 = Conv1DBN(224, 3)(branch_1)
            branch_1 = Conv1DBN(224, 1)(branch_1)
            branches = [branch_0, branch_1]
        else:
            raise ValueError('Unknown Inception-ResNet block type. '
                             'Expects "block35", "block17" or "block8", '
                             'but got: ' + str(self.block_type))

        block_name = self.block_type + '_' + str(self.block_idx)
        mixed = layers.Concatenate(name=block_name + "_mixed")(branches)
        up = Conv1DBN(x.shape[-1],
                      1,
                      activation=None,
                      use_bias=True,
                      name=block_name + '_conv')(mixed)
        x = layers.Lambda(
            lambda inputs, scale: inputs[0] + inputs[1] * self.scale,
            output_shape=x.shape[1:],
            arguments={'scale': self.scale},
            name=block_name
        )([x, up])
        if self.activation is not None:
            x = layers.Activation(self.activation, name=block_name + '_ac')(x)

        return x


class BaseInceptionResNetV2(DLModelBuilder):
    def __init__(self, input_shape=None, num_classes=6, classifier_activation='softmax'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.classifier_activation = classifier_activation

    def __call__(self, *args, **kwargs):
        model = self.get_model()
        return model

    def get_model(self):
        inputs = layers.Input(shape=self.input_shape)

        # stem block
        x = Conv1DBN(32, 3, strides=2, padding='valid')(inputs)
        x = Conv1DBN(32, 3, padding='valid')(x)
        x = Conv1DBN(64, 3)(x)
        x = layers.MaxPooling1D(3, strides=2)(x)
        x = Conv1DBN(80, 1, padding='valid')(x)
        x = Conv1DBN(192, 3, padding='valid')(x)
        x = layers.MaxPooling1D(3, strides=2)(x)

        # Mixed 5b (Inception-A block)
        branch_0 = Conv1DBN(96, 1)(x)
        branch_1 = Conv1DBN(48, 1)(x)
        branch_1 = Conv1DBN(64, 5)(branch_1)
        branch_2 = Conv1DBN(64, 1)(x)
        branch_2 = Conv1DBN(96, 3)(branch_2)
        branch_2 = Conv1DBN(96, 3)(branch_2)
        branch_pool = layers.AveragePooling1D(3, strides=1, padding='same')(x)
        branch_pool = Conv1DBN(64, 1)(branch_pool)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = layers.Concatenate(name='mixed_5b')(branches)

        # 10x block35 (Inception-ResNet-A block)
        for block_idx in range(1, 11):
            x = InceptionResNetBlock(scale=0.17, block_type='block35', block_idx=block_idx)(x)

        # Mixed 6a (Reduction-A block)
        branch_0 = Conv1DBN(384, 3, strides=2, padding='valid')(x)
        branch_1 = Conv1DBN(256, 1)(x)
        branch_1 = Conv1DBN(256, 3)(branch_1)
        branch_1 = Conv1DBN(384, 3, strides=2, padding='valid')(branch_1)
        branch_pool = layers.MaxPooling1D(3, strides=2, padding='valid')(x)
        branches = [branch_0, branch_1, branch_pool]
        x = layers.Concatenate(name='mixed_6a')(branches)

        # 20x block17 (Inception-ResNet-B block)
        for block_idx in range(1, 21):
            x = InceptionResNetBlock(scale=0.1, block_type='block17', block_idx=block_idx)(x)

        # Mixed 7a (Reduction-B block)
        branch_0 = Conv1DBN(256, 1)(x)
        branch_0 = Conv1DBN(384, 3, strides=2, padding='valid')(branch_0)
        branch_1 = Conv1DBN(256, 1)(x)
        branch_1 = Conv1DBN(288, 3, strides=2, padding='valid')(branch_1)
        branch_2 = Conv1DBN(256, 1)(x)
        branch_2 = Conv1DBN(288, 3)(branch_2)
        branch_2 = Conv1DBN(320, 3, strides=2, padding='valid')(branch_2)
        branch_pool = layers.MaxPooling1D(3, strides=2, padding='valid')(x)
        branches = [branch_0, branch_1, branch_2, branch_pool]
        x = layers.Concatenate(name='mixed_7a')(branches)

        # 10x block8 (Inception-ResNet-C block)
        for block_idx in range(1, 10):
            x = InceptionResNetBlock(scale=0.2, block_type='block8', block_idx=block_idx)(x)
        x = InceptionResNetBlock(scale=1., activation=None, block_type='block8', block_idx=10)(x)

        # Final convolution block
        x = Conv1DBN(1536, 1, name='conv_7b')(x)

        # Classification block
        x = layers.GlobalAveragePooling1D(name='avg_pool')(x)
        y = layers.Dense(self.num_classes, activation=self.classifier_activation, name='predictions')(x)

        model = Model(inputs, y)
        return model


def InceptionResNetV2(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                      classifier_activation='softmax'):
    if input_shape is None:
        input_shape = (256 * 3, 1)

    if weights in ['hasc', 'HASC'] and include_top and classes != 6:
        raise ValueError('If using `weights` as `"hasc"` with `include_top`'
                         ' as true, `classes` should be 6')

    # Build model
    model = BaseInceptionResNetV2(input_shape, classes, classifier_activation)()

    if weights is not None:
        if weights in ['hasc', "HASC"]:
            weights = 'weights/inceptionresnetv2/inceptionresnetv2_hasc_weights_{}.hdf5'.format(int(input_shape[0] / 3))

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
    model = InceptionResNetV2(include_top=True, weights=None,
                              pooling=None)

    model.summary()