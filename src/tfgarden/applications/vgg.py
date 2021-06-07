import os

from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D


# VGG*を読み込む関数
def VGG(number, include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax'):
    if input_shape is None:
        input_shape = (256 * 3, 1)

    if weights in ['hasc', 'HASC'] and include_top and classes != 6:
        raise ValueError('If using `weights` as `"hasc"` with `include_top`'
                         ' as true, `classes` should be 6')

    from . import vgg11, vgg13, vgg16, vgg19
    # VGGのバージョン指定
    if number == 11:
        vgg = vgg11.BaseVGG11(input_shape=input_shape, num_classes=classes, classifier_activation=classifier_activation)
    elif number == 13:
        vgg = vgg13.BaseVGG13(input_shape=input_shape, num_classes=classes, classifier_activation=classifier_activation)
    elif number == 16:
        vgg = vgg16.BaseVGG16(input_shape=input_shape, num_classes=classes, classifier_activation=classifier_activation)
    elif number == 19:
        vgg = vgg19.BaseVGG19(input_shape=input_shape, num_classes=classes, classifier_activation=classifier_activation)
    else:
        vgg = vgg16.BaseVGG16(input_shape=input_shape, num_classes=classes, classifier_activation=classifier_activation)

    # モデルをビルドする
    model = vgg()

    # 重みの指定があるとき
    if weights is not None:
        # hascで初期化
        if weights in ['hasc', "HASC"]:
            weights = 'weights/vgg{}/vgg{}_hasc_weights_{}_{}.hdf5'.format(number, number,
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
