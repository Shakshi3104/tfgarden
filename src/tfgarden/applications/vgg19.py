import os

from tensorflow.keras.layers import Input, Flatten, Dense, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model

from .vgg import ConvBlock


def VGG19(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax'):
    """
    applications.vgg19.VGG19
        Arguments
            include_top : whether to include the 3 fully-connected layers at the top of the network.
            weights : one of 'None' (he_normal initialization), 'hasc' (pre-training on HASC), or the path to the weights file to be loaded.
            input_shape : optional shape tuple, default `(768, 1)` (with channels_last data format).
            pooling : optional pooling mode for feature extraction when `include_top` is False.
                        `None` means that the output of the model will be applied to the 3D tensor output of the last convolutional block.
                        `avg` means that global average pooling will be applied to the output of the last convolutioinal block, and thus the output od the model will be a 2D tensor.
                        `max` means that global max pooling will be applied.
            classes : optional number of classes to classify images into, only to be specified if `include_top` is True, and if no weights argument is specified, default 6.
            classifier_activation : A `str` or callable. The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set classifier_activation=None to return the logits of the "top" layer, default `softmax`.
        Returns
            A `tensorflow.keras.Model` instance.
    """
    if input_shape is None:
        input_shape = (256 * 3, 1)

    if weights in ['hasc', 'HASC'] and include_top and classes != 6:
        raise ValueError('If using `weights` as `"hasc"` with `include_top`'
                         ' as true, `classes` should be 6')

    inputs = Input(shape=input_shape)
    x = ConvBlock(2, 64, block_id=1)(inputs)
    x = ConvBlock(2, 128, block_id=2)(x)
    x = ConvBlock(4, 256, block_id=3)(x)
    x = ConvBlock(4, 512, block_id=4)(x)
    x = ConvBlock(4, 512, block_id=5)(x)

    x = Flatten()(x)
    x = Dense(4096, activation="relu", kernel_initializer="he_normal",
              name="fc1")(x)
    x = Dense(4096, activation="relu", kernel_initializer="he_normal",
              name="fc2")(x)
    y = Dense(classes, activation=classifier_activation, name="prediction")(x)

    model = Model(inputs=inputs, outputs=y)

    # 重みの指定があるとき
    if weights is not None:
        # hascで初期化
        if weights in ['hasc', "HASC"]:
            weights = 'weights/vgg19/vgg19_hasc_weights_{}_{}.hdf5'.format(
                int(input_shape[0]),
                int(input_shape[1]))

        # hasc or weights fileで初期化
        if os.path.exists(weights):
            print("Load weights from {}".format(weights))
            model.load_weights(weights)
        else:
            # 重みのファイルがなかったらhe_normal初期化のまま返す
            print("Not exist weights: {}".format(weights))

    # topを含まないとき
    if not include_top:
        if pooling is None:
            # topを削除する
            model = Model(inputs=model.input, outputs=model.layers[-5].output)
        elif pooling == 'avg':
            y = GlobalAveragePooling1D()(model.layers[-5].output)
            model = Model(inputs=model.input, outputs=y)
        elif pooling == 'max':
            y = GlobalMaxPooling1D()(model.layers[-5].output)
            model = Model(inputs=model.input, outputs=y)
        else:
            print("Not exist pooling option: {}".format(pooling))
            model = Model(inputs=model.input, outputs=model.layers[-5].output)

    return model


if __name__ == '__main__':
    model = VGG19(include_top=True,
                  weights=None,
                  input_shape=None,
                  pooling=None,
                  classes=6,
                  classifier_activation='softmax')
    print(model.summary())
