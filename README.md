# TensorFlow model Garden for Human Activity Recognition
The TensorFlow model Garden for Human Activity Recognition (**tfgarden**) is the repository of CNN models implemented for sensor-based human activity recognition, like [`tensorflow.keras.applications`](https://www.tensorflow.org/api_docs/python/tf/keras/applications).

The models implemented here can also be used as a source domain for sensor-based task (e.g. sidewalk surface type estimation).

## Modules

- [`densenet`](docs/docs/reference/densenet.md#densenet) module: DenseNet models for Keras.
- [`efficientnet`](docs/docs/reference/efficientnet.md#efficientnet) module : EfficientNet models for Keras.
- [`inception_resnet_v2`](docs/docs/reference/inception_resnet_v2.md#inception-resnet-v2) module: Inception-ResNet V2 model for Keras.
- [`inception_v3`](docs/docs/reference/inception_v3.md#inception-v3) module: Inception V3 model for Keras.
- [`mobilenet`](docs/docs/reference/mobilenet.md#mobilenet) module : MobileNet v1 models for Keras.
- [`mobilenet_v2`](docs/docs/reference/mobilenet_v2.md#mobilenet-v2) module: MobileNet v2 models for Keras.
- [`mobilenet_v3`](docs/docs/reference/mobilenet_v3.md#mobilenet-v3) module: MobileNet v3 models for Keras.
- [`nasnet`](docs/docs/reference/nasnet.md#nasnet) module: NASNet-A models for Keras.
- [`resnet`](docs/docs/reference/resnet.md#resnet) module : ResNet models for Keras.
- `resnet_v2` module: ResNet v2 models for Keras, **Not implemented yet**.
- [`vgg11`](docs/docs/reference/vgg.md#applicationsvgg11vgg11) module : VGG11 model for Keras.
- [`vgg13`](docs/docs/reference/vgg.md#applicationsvgg13vgg13) module : VGG13 model for Keras.
- [`vgg16`](docs/docs/reference/vgg.md#applicationsvgg16vgg16) module : VGG16 model for Keras.
- [`vgg19`](docs/docs/reference/vgg.md#applicationsvgg19vgg19) module : VGG19 model for Keras.
- [`xception`](docs/docs/reference/xception.md#xception) module: Xception V1 model for Keras.
- [`mnasnet`](docs/docs/reference/mnasnet.md#mnasnet) module: MnasNet-A1 model for Keras.
- [`pyramidnet`](docs/docs/reference/pyramidnet.md#pyramidnet) module : PyramidNet models for Keras.
- [`efficientnet_lite`](docs/docs/reference/efficientnet_lite.md#efficientnet-lite) module : EfficientNet-Lite models for Keras.

## Performance

Please refer to [tfmars](https://github.com/Shakshi3104/tfmars#performance)

## Install

```bash
pip install git+https://github.com/Shakshi3104/tfgarden.git
```

## Dependency
- `tensorflow >= 2.0`