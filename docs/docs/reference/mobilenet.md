# MobileNet

## application.mobilenet.MobileNet
```python
MobileNet(include_top=True, weights='hasc', input_shape=None, pooling=None, classes=6, classifier_activation='softmax',
              alpha=1.0, depth_multiplier=1, dropout=1e-3)
```

Reference:
- [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861) 

By default, it loads weights pre-trained on HASC. Check 'weights' for other options.

The default input size for this model is 768 (256 * 3).

### Arguments
- include_top : whether to include the fully-connected layer at the top of the network.
- weights : one of 'None' (he_normal initialization), 'hasc' (pre-training on HASC), or the path to the weights file to be loaded.
- input_shape : optional shape tuple, default `(768, 1)` (with channels_last data format).
- pooling : optional pooling mode for feature extraction when `include_top` is False.
    - `None` means that the output of the model will be applied to the 3D tensor output of the last convolutional block.
    - `avg` means that global average pooling will be applied to the output of the last convolutioinal block, and thus the output of the model will be a 2D tensor.
    - `max` means that global max pooling will be applied.
- classes : optional number of classes to classify images into, only to be specified if `include_top` is True, and if no weights argument is specified, default 6.
- classifier_activation : A `str` or callable. The activation function to use on the "top" layer. Ignored unless `include_top=True`. Set classifier_activation=`None` to return the logits of the "top" layer, default `'softmax'`.
- alpha : Controls the width of the network. This is known as the width multiplier in the MobileNet paper. Default to 1.0.
- depth_multipliter : Depth multiplier for depthwise convolution. This is called the resolution multiplier in the MobileNet paper. Default to 1.0.
- dropout : Dropout rate. Default to 0.001.

### Returns
- `tensorflow.keras.Model` instance.
