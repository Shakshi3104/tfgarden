# EfficientNet

## applications.efficientnet.EfficientNetB0
```python
applications.efficientnet.EfficientNetB0(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax')
```

Reference paper:
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

By default, it loads weights pre-trained on HASC. Check 'weights' for other options.

The default input size for this model is 768 (256 * 3).

### Arguments
- include_top : whether to include the global average pooling layer and the fully-connected layer at the top of the network.
- weights : one of 'None' (he_normal initialization), 'hasc' (pre-training on HASC), or the path to the weights file to be loaded.
- input_shape : optional shape tuple, default `(768, 1)` (with channels_last data format).
- pooling : optional pooling mode for feature extraction when `include_top` is False.
    - `None` means that the output of the model will be applied to the 3D tensor output of the last convolutional block.
    - `avg` means that global average pooling will be applied to the output of the last convolutioinal block, and thus the output of the model will be a 2D tensor.
    - `max` means that global max pooling will be applied.
- classes : optional number of classes to classify images into, only to be specified if `include_top` is True, and if no weights argument is specified, default 6.
- classifier_activation : A `str` or callable. The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set classifier_activation=`None` to return the logits of the "top" layer, default `'softmax'`.

### Returns
- `tensorflow.keras.Model` instance.


## applications.efficientnet.EfficientNetB1
```python
applications.efficientnet.EfficientNetB1(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax')
```

Reference paper:
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

By default, it loads weights pre-trained on HASC. Check 'weights' for other options.

The default input size for this model is 768 (256 * 3).

### Arguments
- include_top : whether to include the global average pooling layer and the fully-connected layer at the top of the network.
- weights : one of 'None' (he_normal initialization), 'hasc' (pre-training on HASC), or the path to the weights file to be loaded.
- input_shape : optional shape tuple, default `(768, 1)` (with channels_last data format).
- pooling : optional pooling mode for feature extraction when `include_top` is False.
    - `None` means that the output of the model will be applied to the 3D tensor output of the last convolutional block.
    - `avg` means that global average pooling will be applied to the output of the last convolutioinal block, and thus the output od the model will be a 2D tensor.
    - `max` means that global max pooling will be applied.
- classes : optional number of classes to classify images into, only to be specified if `include_top` is True, and if no weights argument is specified, default 6.
- classifier_activation : A `str` or callable. The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set classifier_activation=`None` to return the logits of the "top" layer, default `'softmax'`.

### Returns
- `tensorflow.keras.Model` instance.


## applications.efficientnet.EfficientNetB2
```python
applications.efficientnet.EfficientNetB2(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax')
```

Reference paper:
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

By default, it loads weights pre-trained on HASC. Check 'weights' for other options.

The default input size for this model is 768 (256 * 3).

### Arguments
- include_top : whether to include the global average pooling layer and the fully-connected layer at the top of the network.
- weights : one of 'None' (he_normal initialization), 'hasc' (pre-training on HASC), or the path to the weights file to be loaded.
- input_shape : optional shape tuple, default `(768, 1)` (with channels_last data format).
- pooling : optional pooling mode for feature extraction when `include_top` is False.
    - `None` means that the output of the model will be applied to the 3D tensor output of the last convolutional block.
    - `avg` means that global average pooling will be applied to the output of the last convolutioinal block, and thus the output od the model will be a 2D tensor.
    - `max` means that global max pooling will be applied.
- classes : optional number of classes to classify images into, only to be specified if `include_top` is True, and if no weights argument is specified, default 6.
- classifier_activation : A `str` or callable. The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set classifier_activation=`None` to return the logits of the "top" layer, default `'softmax'`.

### Returns
- `tensorflow.keras.Model` instance.


## applications.efficientnet.EfficientNetB3
```python
applications.efficientnet.EfficientNetB3(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax')
```

Reference paper:
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

By default, it loads weights pre-trained on HASC. Check 'weights' for other options.

The default input size for this model is 768 (256 * 3).

### Arguments
- include_top : whether to include the global average pooling layer and the fully-connected layer at the top of the network.
- weights : one of 'None' (he_normal initialization), 'hasc' (pre-training on HASC), or the path to the weights file to be loaded.
- input_shape : optional shape tuple, default `(768, 1)` (with channels_last data format).
- pooling : optional pooling mode for feature extraction when `include_top` is False.
    - `None` means that the output of the model will be applied to the 3D tensor output of the last convolutional block.
    - `avg` means that global average pooling will be applied to the output of the last convolutioinal block, and thus the output od the model will be a 2D tensor.
    - `max` means that global max pooling will be applied.
- classes : optional number of classes to classify images into, only to be specified if `include_top` is True, and if no weights argument is specified, default 6.
- classifier_activation : A `str` or callable. The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set classifier_activation=`None` to return the logits of the "top" layer, default `'softmax'`.

### Returns
- `tensorflow.keras.Model` instance.


## applications.efficientnet.EfficientNetB4
```python
applications.efficientnet.EfficientNetB4(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax')
```

Reference paper:
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

By default, it loads weights pre-trained on HASC. Check 'weights' for other options.

The default input size for this model is 768 (256 * 3).

### Arguments
- include_top : whether to include the global average pooling layer and the fully-connected layer at the top of the network.
- weights : one of 'None' (he_normal initialization), 'hasc' (pre-training on HASC), or the path to the weights file to be loaded.
- input_shape : optional shape tuple, default `(768, 1)` (with channels_last data format).
- pooling : optional pooling mode for feature extraction when `include_top` is False.
    - `None` means that the output of the model will be applied to the 3D tensor output of the last convolutional block.
    - `avg` means that global average pooling will be applied to the output of the last convolutioinal block, and thus the output od the model will be a 2D tensor.
    - `max` means that global max pooling will be applied.
- classes : optional number of classes to classify images into, only to be specified if `include_top` is True, and if no weights argument is specified, default 6.
- classifier_activation : A `str` or callable. The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set classifier_activation=`None` to return the logits of the "top" layer, default `'softmax'`.

### Returns
- `tensorflow.keras.Model` instance.


## applications.efficientnet.EfficientNetB5
```python
applications.efficientnet.EfficientNetB5(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax')
```

Reference paper:
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

By default, it loads weights pre-trained on HASC. Check 'weights' for other options.

The default input size for this model is 768 (256 * 3).

### Arguments
- include_top : whether to include the global average pooling layer and the fully-connected layer at the top of the network.
- weights : one of 'None' (he_normal initialization), 'hasc' (pre-training on HASC), or the path to the weights file to be loaded.
- input_shape : optional shape tuple, default `(768, 1)` (with channels_last data format).
- pooling : optional pooling mode for feature extraction when `include_top` is False.
    - `None` means that the output of the model will be applied to the 3D tensor output of the last convolutional block.
    - `avg` means that global average pooling will be applied to the output of the last convolutioinal block, and thus the output od the model will be a 2D tensor.
    - `max` means that global max pooling will be applied.
- classes : optional number of classes to classify images into, only to be specified if `include_top` is True, and if no weights argument is specified, default 6.
- classifier_activation : A `str` or callable. The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set classifier_activation=`None` to return the logits of the "top" layer, default `'softmax'`.

### Returns
- `tensorflow.keras.Model` instance.


## applications.efficientnet.EfficientNetB6
```python
applications.efficientnet.EfficientNetB6(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax')
```

Reference paper:
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

By default, it loads weights pre-trained on HASC. Check 'weights' for other options.

The default input size for this model is 768 (256 * 3).

### Arguments
- include_top : whether to include the global average pooling layer and the fully-connected layer at the top of the network.
- weights : one of 'None' (he_normal initialization), 'hasc' (pre-training on HASC), or the path to the weights file to be loaded.
- input_shape : optional shape tuple, default `(768, 1)` (with channels_last data format).
- pooling : optional pooling mode for feature extraction when `include_top` is False.
    - `None` means that the output of the model will be applied to the 3D tensor output of the last convolutional block.
    - `avg` means that global average pooling will be applied to the output of the last convolutioinal block, and thus the output od the model will be a 2D tensor.
    - `max` means that global max pooling will be applied.
- classes : optional number of classes to classify images into, only to be specified if `include_top` is True, and if no weights argument is specified, default 6.
- classifier_activation : A `str` or callable. The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set classifier_activation=`None` to return the logits of the "top" layer, default `'softmax'`.

### Returns
- `tensorflow.keras.Model` instance.


## applications.efficientnet.EfficientNetB7
```python
applications.efficientnet.EfficientNetB7(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax')
```

Reference paper:
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) (ICML 2019)

By default, it loads weights pre-trained on HASC. Check 'weights' for other options.

The default input size for this model is 768 (256 * 3).

### Arguments
- include_top : whether to include the global average pooling layer and the fully-connected layer at the top of the network.
- weights : one of 'None' (he_normal initialization), 'hasc' (pre-training on HASC), or the path to the weights file to be loaded.
- input_shape : optional shape tuple, default `(768, 1)` (with channels_last data format).
- pooling : optional pooling mode for feature extraction when `include_top` is False.
    - `None` means that the output of the model will be applied to the 3D tensor output of the last convolutional block.
    - `avg` means that global average pooling will be applied to the output of the last convolutioinal block, and thus the output od the model will be a 2D tensor.
    - `max` means that global max pooling will be applied.
- classes : optional number of classes to classify images into, only to be specified if `include_top` is True, and if no weights argument is specified, default 6.
- classifier_activation : A `str` or callable. The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set classifier_activation=`None` to return the logits of the "top" layer, default `'softmax'`.

### Returns
- `tensorflow.keras.Model` instance.