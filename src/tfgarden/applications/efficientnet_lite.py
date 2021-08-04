import os

from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.models import Model

from .efficientnet import __EfficientNet


def EfficientNet_lite(b, include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                      classifier_activation='softmax'):
    if input_shape is None:
        input_shape = (256 * 3, 1)

    if weights in ['hasc', 'HASC'] and include_top and classes != 6:
        raise ValueError('If using `weights` as `"hasc"` with `include_top`'
                         ' as true, `classes` should be 6')

    blocks_args = [{
        'kernel_size': 3,
        'repeats': 1,
        'filters_in': 32,
        'filters_out': 16,
        'expand_ratio': 1,
        'id_skip': True,
        'strides': 1,
        'se_ratio': 0.0
    }, {
        'kernel_size': 3,
        'repeats': 2,
        'filters_in': 16,
        'filters_out': 24,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 2,
        'se_ratio': 0.0
    }, {
        'kernel_size': 5,
        'repeats': 2,
        'filters_in': 24,
        'filters_out': 40,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 2,
        'se_ratio': 0.0
    }, {
        'kernel_size': 3,
        'repeats': 3,
        'filters_in': 40,
        'filters_out': 80,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 2,
        'se_ratio': 0.0
    }, {
        'kernel_size': 5,
        'repeats': 3,
        'filters_in': 80,
        'filters_out': 112,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 1,
        'se_ratio': 0.0
    }, {
        'kernel_size': 5,
        'repeats': 4,
        'filters_in': 112,
        'filters_out': 192,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 2,
        'se_ratio': 0.0
    }, {
        'kernel_size': 3,
        'repeats': 1,
        'filters_in': 192,
        'filters_out': 320,
        'expand_ratio': 6,
        'id_skip': True,
        'strides': 1,
        'se_ratio': 0.0
    }]

    if b == 0:
        model = __EfficientNet(1.0, 1.0, 0.2, input_shape=input_shape, activation='relu', blocks_args=blocks_args,
                               classes=classes, classifier_activation=classifier_activation)
    elif b == 1:
        model = __EfficientNet(1.0, 1.1, 0.2, input_shape=input_shape, activation='relu', blocks_args=blocks_args,
                               classes=classes, classifier_activation=classifier_activation)
    elif b == 2:
        model = __EfficientNet(1.1, 1.2, 0.3, input_shape=input_shape, activation='relu', blocks_args=blocks_args,
                               classes=classes, classifier_activation=classifier_activation)
    elif b == 3:
        model = __EfficientNet(1.2, 1.4, 0.3, input_shape=input_shape, activation='relu', blocks_args=blocks_args,
                               classes=classes, classifier_activation=classifier_activation)
    elif b == 4:
        model = __EfficientNet(1.4, 1.8, 0.4, input_shape=input_shape, activation='relu', blocks_args=blocks_args,
                               classes=classes, classifier_activation=classifier_activation)
    elif b == 5:
        model = __EfficientNet(1.6, 2.2, 0.4, input_shape=input_shape, activation='relu', blocks_args=blocks_args,
                               classes=classes, classifier_activation=classifier_activation)
    elif b == 6:
        model = __EfficientNet(1.8, 2.6, 0.5, input_shape=input_shape, activation='relu', blocks_args=blocks_args,
                               classes=classes, classifier_activation=classifier_activation)
    elif b == 7:
        model = __EfficientNet(2.0, 3.1, 0.5, input_shape=input_shape, activation='relu', blocks_args=blocks_args,
                               classes=classes, classifier_activation=classifier_activation)

    if weights is not None:
        if weights in ['hasc', "HASC"]:
            weights = 'weights/efficientnet_lite{}/efficientnet_lite{}_hasc_weights_{}_{}.hdf5'.format(b, b,
                                                                                               int(input_shape[0]),
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
            model = Model(inputs=model.input, outputs=model.layers[-4].output)
        elif pooling == 'avg':
            y = GlobalAveragePooling1D()(model.layers[-4].output)
            model = Model(inputs=model.input, outputs=y)
        elif pooling == 'max':
            y = GlobalMaxPooling1D()(model.layers[-4].output)
            model = Model(inputs=model.input, outputs=y)
        else:
            print("Not exist pooling option: {}".format(pooling))
            model = Model(inputs=model.input, outputs=model.layers[-4].output)

    return model


def EfficientNet_lite0(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                       classifier_activation='softmax'):
    model = EfficientNet_lite(0, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


def EfficientNet_lite1(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                       classifier_activation='softmax'):
    model = EfficientNet_lite(1, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


def EfficientNet_lite2(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                       classifier_activation='softmax'):
    model = EfficientNet_lite(2, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


def EfficientNet_lite3(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                       classifier_activation='softmax'):
    model = EfficientNet_lite(3, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


def EfficientNet_lite4(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6,
                       classifier_activation='softmax'):
    model = EfficientNet_lite(4, include_top, weights, input_shape, pooling, classes, classifier_activation)
    return model


if __name__ == "__main__":
    model = EfficientNet_lite0()
    model.summary()
