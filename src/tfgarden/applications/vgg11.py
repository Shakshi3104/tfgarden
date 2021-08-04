import os

from tensorflow.keras.layers import Input, Flatten, Dense, GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model

from .base import ConvBlock


def VGG11(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax'):
    """
    applications.vgg11.VGG11
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
    x = ConvBlock(1, 64)(inputs)
    x = ConvBlock(1, 128)(x)
    x = ConvBlock(2, 256)(x)
    x = ConvBlock(2, 512)(x)
    x = ConvBlock(2, 512)(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(4096, activation="relu", kernel_initializer="he_normal",
                  name="fc1")(x)
        x = Dense(4096, activation="relu", kernel_initializer="he_normal",
                  name="fc2")(x)
        y = Dense(classes, activation=classifier_activation, name="prediction")(x)

        model_ = Model(inputs=inputs, outputs=y)

        if weights is not None:
            if weights in ['hasc', 'HASC']:
                weights = 'weights/vgg11/vgg11_hasc_weights_{}_{}.hdf5'.format(input_shape[0], input_shape[1])

            if os.path.exists(weights):
                print("Load weights from {}".format(weights))
                model_.load_weights(weights)
            else:
                print("Not exist weights: {}".format(weights))
        return model_
    else:
        if pooling is None:
            model_ = Model(inputs=inputs, outputs=x)
            return model_
        elif pooling == 'avg':
            x = GlobalAveragePooling1D(name="avgpool")(x)
            model_ = Model(inputs=inputs, outputs=x)
            return model_
        elif pooling == 'max':
            x = GlobalMaxPooling1D(name="maxpool")(x)
            model_ = Model(inputs=inputs, outputs=x)
            return model_
        else:
            print("Not exist pooling option: {}".format(pooling))
            model_ = Model(inputs=inputs, outputs=x)
            return model_


if __name__ == '__main__':
    model = VGG11(include_top=True,
                  weights=None,
                  input_shape=None,
                  pooling=None,
                  classes=6,
                  classifier_activation='softmax')
    print(model.summary())
