import tensorflow as tf
import os
import math
import copy

DEFAULT_BLOCKS_ARGS = {
    "efficientnetv2-s": [{
        "kernel_size": 3,
        "num_repeat": 2,
        "input_filters": 24,
        "output_filters": 24,
        "expand_ratio": 1,
        "se_ratio": 0.0,
        "strides": 1,
        "conv_type": 1,
    }, {
        "kernel_size": 3,
        "num_repeat": 4,
        "input_filters": 24,
        "output_filters": 48,
        "expand_ratio": 4,
        "se_ratio": 0.0,
        "strides": 2,
        "conv_type": 1,
    }, {
        "conv_type": 1,
        "expand_ratio": 4,
        "input_filters": 48,
        "kernel_size": 3,
        "num_repeat": 4,
        "output_filters": 64,
        "se_ratio": 0,
        "strides": 2,
    }, {
        "conv_type": 0,
        "expand_ratio": 4,
        "input_filters": 64,
        "kernel_size": 3,
        "num_repeat": 6,
        "output_filters": 128,
        "se_ratio": 0.25,
        "strides": 2,
    }, {
        "conv_type": 0,
        "expand_ratio": 6,
        "input_filters": 128,
        "kernel_size": 3,
        "num_repeat": 9,
        "output_filters": 160,
        "se_ratio": 0.25,
        "strides": 1,
    }, {
        "conv_type": 0,
        "expand_ratio": 6,
        "input_filters": 160,
        "kernel_size": 3,
        "num_repeat": 15,
        "output_filters": 256,
        "se_ratio": 0.25,
        "strides": 2,
    }],
    "efficientnetv2-m": [
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 24,
            "output_filters": 24,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 24,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 48,
            "output_filters": 80,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 80,
            "output_filters": 160,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 14,
            "input_filters": 160,
            "output_filters": 176,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 18,
            "input_filters": 176,
            "output_filters": 304,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 304,
            "output_filters": 512,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-l": [
        {
            "kernel_size": 3,
            "num_repeat": 4,
            "input_filters": 32,
            "output_filters": 32,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 32,
            "output_filters": 64,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 64,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 10,
            "input_filters": 96,
            "output_filters": 192,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 19,
            "input_filters": 192,
            "output_filters": 224,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 25,
            "input_filters": 224,
            "output_filters": 384,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 7,
            "input_filters": 384,
            "output_filters": 640,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b0": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b1": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b2": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
    "efficientnetv2-b3": [
        {
            "kernel_size": 3,
            "num_repeat": 1,
            "input_filters": 32,
            "output_filters": 16,
            "expand_ratio": 1,
            "se_ratio": 0,
            "strides": 1,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 16,
            "output_filters": 32,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 2,
            "input_filters": 32,
            "output_filters": 48,
            "expand_ratio": 4,
            "se_ratio": 0,
            "strides": 2,
            "conv_type": 1,
        },
        {
            "kernel_size": 3,
            "num_repeat": 3,
            "input_filters": 48,
            "output_filters": 96,
            "expand_ratio": 4,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 5,
            "input_filters": 96,
            "output_filters": 112,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 1,
            "conv_type": 0,
        },
        {
            "kernel_size": 3,
            "num_repeat": 8,
            "input_filters": 112,
            "output_filters": 192,
            "expand_ratio": 6,
            "se_ratio": 0.25,
            "strides": 2,
            "conv_type": 0,
        },
    ],
}


class MBConvBlock:
    """
    MBConv block: Mobile Inverted Residual Bottleneck.
    """
    def __init__(self, 
        input_filters: int,
        output_filters: int,
        expand_ratio=1,
        kernel_size=3,
        strides=1,
        se_ratio=0.0,
        bn_momentum=0.9,
        activation="relu",
        survival_probability: float = 0.8, 
        name=None
    ):
        self.input_filters=input_filters
        self.output_filters=output_filters
        self.expand_ratio=expand_ratio
        self.kernel_size=kernel_size
        self.strides=strides
        self.se_ratio=se_ratio
        self.bn_momentum=bn_momentum
        self.activation=activation
        self.survival_probability=survival_probability
        self.name=name

    def __call__(self, x):
        inputs = x

        # Expansion phase
        filters = self.input_filters * self.expand_ratio
        if self.expand_ratio != 1:
            x = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=1,
                strides=1,
                kernel_initializer="he_normal",
                padding="same",
                use_bias=False,
                name=self.name + "expand_conv"
            )(inputs)
            x = tf.keras.layers.BatchNormalization(
                axis=-1, momentum=self.bn_momentum, name=self.name + "expand_bn"
            )(x)
            x = tf.keras.layers.Activation(self.activation, name=self.name + "expand_activation")(x)
        else:
            x = inputs

        # Depthwise conv
        x = tf.keras.layers.SeparableConv1D(
            int(x.shape[-1]),
            self.kernel_size,
            strides=self.strides,
            padding="same",
            use_bias=False,
            depthwise_initializer='he_normal',
            name=self.name + 'dwconv'
        )(x)
        x = tf.keras.layers.BatchNormalization(axis=-1, momentum=self.bn_momentum, name=self.name + 'bn')(x)
        x = tf.keras.layers.Activation(self.activation, name=self.name + "activation")(x)

        # Squeeze and Excite
        if 0 < self.se_ratio <= 1:
            filters_se = max(1, int(self.input_filters * self.se_ratio))
            se = tf.keras.layers.GlobalAveragePooling1D(name=self.name + "se_squeeze")(x)
            se = tf.keras.layers.Reshape((1, filters), name=self.name + "se_reshape")(se)

            se = tf.keras.layers.Conv1D(
                filters_se,
                1,
                padding="same",
                activation=self.activation,
                kernel_initializer="he_normal",
                name=self.name + "se_reduce"
            )(se)
            se = tf.keras.layers.Conv1D(
                filters,
                1,
                padding="same",
                activation="sigmoid",
                kernel_initializer="he_normal",
                name=self.name + "se_expand"
            )(se)

            x = tf.keras.layers.multiply([x, se], name=self.name + "se_excite")

        # Output phase
        x = tf.keras.layers.Conv1D(
            filters=self.output_filters,
            kernel_size=1,
            strides=1,
            kernel_initializer="he_normal",
            padding="same",
            use_bias=False,
            name=self.name + "project_conv"
        )(x)
        x = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=self.bn_momentum, name=self.name + "project_bn"
        )(x)

        # Residual
        if self.strides == 1 and self.input_filters == self.output_filters:
            if self.survival_probability:
                x = tf.keras.layers.Dropout(
                    self.survival_probability,
                    name=self.name + "drop"
                )(x)
            x = tf.keras.layers.add([x, inputs], name=self.name + "add")

        return x


class FusedMBConvBlock:
    """
    Fused MBConv Block: Fusing the proj conv1x1 and depthwise_conv into a conv2d.
    """
    def __init__(self, 
        input_filters: int,
        output_filters: int,
        expand_ratio=1,
        kernel_size=3,
        strides=1,
        se_ratio=0.0,
        bn_momentum=0.9,
        activation="relu",
        survival_probability: float = 0.8, 
        name=None
    ):
        self.input_filters=input_filters
        self.output_filters=output_filters
        self.expand_ratio=expand_ratio
        self.kernel_size=kernel_size
        self.strides=strides
        self.se_ratio=se_ratio
        self.bn_momentum=bn_momentum
        self.activation=activation
        self.survival_probability=survival_probability
        self.name=name

    def __call__(self, x):
        inputs = x

        # Expansion phase
        filters = self.input_filters * self.expand_ratio
        if self.expand_ratio != 1:
            x = tf.keras.layers.Conv1D(
                filters=filters,
                kernel_size=self.kernel_size,
                strides=1,
                kernel_initializer="he_normal",
                padding="same",
                use_bias=False,
                name=self.name + "expand_conv"
            )(inputs)
            x = tf.keras.layers.BatchNormalization(
                axis=-1, momentum=self.bn_momentum, name=self.name + "expand_bn"
            )(x)
            x = tf.keras.layers.Activation(self.activation, name=self.name + "expand_activation")(x)
        else:
            x = inputs

        # Squeeze and Excite
        if 0 < self.se_ratio <= 1:
            filters_se = max(1, int(self.input_filters * self.se_ratio))
            se = tf.keras.layers.GlobalAveragePooling1D(name=self.name + "se_squeeze")(x)
            se = tf.keras.layers.Reshape((1, filters), name=self.name + "se_reshape")(se)

            se = tf.keras.layers.Conv1D(
                filters_se,
                1,
                padding="same",
                activation=self.activation,
                kernel_initializer="he_normal",
                name=self.name + "se_reduce"
            )(se)
            se = tf.keras.layers.Conv1D(
                filters,
                1,
                padding="same",
                activation="sigmoid",
                kernel_initializer="he_normal",
                name=self.name + "se_expand"
            )(se)

            x = tf.keras.layers.multiply([x, se], name=self.name + "se_excite")

        # Output phase
        x = tf.keras.layers.Conv1D(
            filters=self.output_filters,
            kernel_size=1 if self.expand_ratio != 1 else self.kernel_size,
            strides=1 if self.expand_ratio != 1 else self.strides,
            kernel_initializer="he_normal",
            padding="same",
            use_bias=False,
            name=self.name + "project_conv"
        )(x)
        x = tf.keras.layers.BatchNormalization(
            axis=-1, momentum=self.bn_momentum, name=self.name + "project_bn"
        )(x)
        if self.expand_ratio == 1:
            x = tf.keras.layers.Activation(activation=self.activation, name=self.name + "project_activation")(x)


        # Residual
        if self.strides == 1 and self.input_filters == self.output_filters:
            if self.survival_probability:
                x = tf.keras.layers.Dropout(
                    self.survival_probability,
                    name=self.name + "drop"
                )(x)
            x = tf.keras.layers.add([x, inputs], name=self.name + "add")

        return x
        

def round_filters(filters, width_coefficient, min_depth, depth_divisor):
  """Round number of filters based on depth multiplier."""
  filters *= width_coefficient
  minimum_depth = min_depth or depth_divisor
  new_filters = max(
      minimum_depth,
      int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
  )
  return int(new_filters)


def round_repeats(repeats, depth_coefficient):
  """Round number of repeats based on depth multiplier."""
  return int(math.ceil(depth_coefficient * repeats))


def EfficientNetV2(
    width_coefficient,
    depth_coefficient,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    depth_divisor=8,
    min_depth=8,
    bn_momentum=0.9,
    activation="relu",
    blocks_args="default",
    model_name="efficientnetv2",
    include_top=True,
    weights="hasc",
    input_shape=None,
    pooling=None,
    classes=6,
    classifier_activation="softmax"
):
    
    if blocks_args == "default":
        blocks_args = DEFAULT_BLOCKS_ARGS[model_name]

    if input_shape is None:
        input_shape = (256 * 3, 1)

    #########################################################################
    inputs = tf.keras.layers.Input(input_shape)
    x = inputs

    # Build stem
    stem_filters = round_filters(
        filters=blocks_args[0]["input_filters"],
        width_coefficient=width_coefficient,
        min_depth=min_depth,
        depth_divisor=depth_divisor
    )
    x = tf.keras.layers.Conv1D(
        filters=stem_filters,
        kernel_size=3,
        strides=2,
        kernel_initializer="he_normal",
        padding="same",
        use_bias=False,
        name="stem_conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=bn_momentum,
        name="stem_bn"
    )(x)
    x = tf.keras.layers.Activation(activation, name="stem_activation")(x)

    # Build blocks
    blocks_args = copy.deepcopy(blocks_args)
    b = 0
    blocks = float(sum(args["num_repeat"] for args in blocks_args))

    for (i, args) in enumerate(blocks_args):
        assert args["num_repeat"] > 0

        # Update block input and output filters based on depth multiplier.
        args["input_filters"] = round_filters(
            filters=args["input_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor)
        args["output_filters"] = round_filters(
            filters=args["output_filters"],
            width_coefficient=width_coefficient,
            min_depth=min_depth,
            depth_divisor=depth_divisor)

        # Determine which conv type to use:
        block = {0: MBConvBlock, 1: FusedMBConvBlock}[args.pop("conv_type")]
        repeats = round_repeats(
            repeats=args.pop("num_repeat"), depth_coefficient=depth_coefficient)
        for j in range(repeats):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args["strides"] = 1
                args["input_filters"] = args["output_filters"]

            x = block(
                activation=activation,
                bn_momentum=bn_momentum,
                survival_probability=drop_connect_rate * b / blocks,
                name="block{}{}_".format(i + 1, chr(j + 97)),
                **args,
            )(x)

    # Build top
    top_filters = round_filters(
        filters=1280,
        width_coefficient=width_coefficient,
        min_depth=min_depth,
        depth_divisor=depth_divisor)

    x = tf.keras.layers.Conv1D(
        filters=top_filters,
        kernel_size=1,
        strides=1,
        kernel_initializer="he_normal",
        padding="same",
        use_bias=False,
        name="top_conv"
    )(x)
    x = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=bn_momentum,
        name="top_bn"
    )(x)
    x = tf.keras.layers.Activation(activation=activation, name="top_activation")(x)

    x = tf.keras.layers.GlobalAveragePooling1D(name="avg_pool")(x)
    if dropout_rate > 0:
      x = tf.keras.layers.Dropout(dropout_rate, name="top_dropout")(x)
    outputs = tf.keras.layers.Dense(
        classes,
        activation=classifier_activation,
        kernel_initializer="he_normal",
        bias_initializer=tf.constant_initializer(0),
        name="predictions")(x)


    model = tf.keras.models.Model(inputs, outputs)
    #########################################################################
    if weights is not None:
        if weights in ['hasc', 'HASC']:
            weights = 'weights/{}/{}_hasc_weights_{}_{}.hdf5'.format(model_name, model_name, 
                                                                     int(input_shape[0]), int(input_shape[1]))

        if os.path.exists(weights):
           print("Load weights from {}".format(weights))
           model.load_weights(weights)
        else:
           print("Not exist weights: {}".format(weights)) 

    # not including top
    if not include_top:
        if pooling is None:
            model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-4].output)
        elif pooling == 'avg':
            y = tf.keras.layers.GlobalAveragePooling1D()(model.layers[-4].output)
            model = tf.keras.models.Model(inputs=model.input, outputs=y)
        elif pooling == 'max':
            y = tf.keras.layers.GlobalMaxPooling1D()(model.layers[-4].output)
            model = tf.keras.models.Model(inputs=model.input, outputs=y)
        else:
            print("Not exist pooling option: {}".format(pooling))
            model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-4].output)

    return model


#########################################################################
def EfficientNetV2B0(
    include_top=True,
    weights="hasc",
    input_shape=None,
    pooling=None,
    classes=6,
    classifier_activation="softmax"
):
    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        model_name="efficientnetv2-b0",
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


def EfficientNetV2B1(
    include_top=True,
    weights="hasc",
    input_shape=None,
    pooling=None,
    classes=6,
    classifier_activation="softmax"
):

    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        model_name="efficientnetv2-b1",
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


def EfficientNetV2B2(
    include_top=True,
    weights="hasc",
    input_shape=None,
    pooling=None,
    classes=6,
    classifier_activation="softmax"
):

    return EfficientNetV2(
        width_coefficient=1.1,
        depth_coefficient=1.2,
        model_name="efficientnetv2-b2",
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


def EfficientNetV2B3(
    include_top=True,
    weights="hasc",
    input_shape=None,
    pooling=None,
    classes=6,
    classifier_activation="softmax"
):

    return EfficientNetV2(
        width_coefficient=1.2,
        depth_coefficient=1.4,
        model_name="efficientnetv2-b3",
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )    


def EfficientNetV2S(
    include_top=True,
    weights="hasc",
    input_shape=None,
    pooling=None,
    classes=6,
    classifier_activation="softmax"
):

    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        model_name="efficientnetv2-s",
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


def EfficientNetV2M(
    include_top=True,
    weights="hasc",
    input_shape=None,
    pooling=None,
    classes=6,
    classifier_activation="softmax"
):

    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        model_name="efficientnetv2-m",
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )


def EfficientNetV2L(
    include_top=True,
    weights="hasc",
    input_shape=None,
    pooling=None,
    classes=6,
    classifier_activation="softmax"
):

    return EfficientNetV2(
        width_coefficient=1.0,
        depth_coefficient=1.0,
        model_name="efficientnetv2-l",
        include_top=include_top,
        weights=weights,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        classifier_activation=classifier_activation,
    )



if __name__ == "__main__":
    model = EfficientNetV2S(
        include_top=False,
        weights=None,
        pooling=None
        )

    model.summary()

