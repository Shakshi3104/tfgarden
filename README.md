# TensorFlow model Garden for HASC
The TensorFlow model Garden for HASC (**tfgarden**) is the repository of CNN models implemented for human activity recognition (HASC), like [`tensorflow.keras.applications`](https://www.tensorflow.org/api_docs/python/tf/keras/applications).

The models implemented here can also be used as a source domain for sensor-based task (e.g. sidewalk surface type estimation).

## Modules
TensorFlow CNN Collections for HASC are canned architecture with pre-trained weights.

pre-trained weights are uploaded [here](https://drive.google.com/drive/folders/1HMDMDz91laNvsyaTvAMgXzX-pIjDMpwy?usp=sharing).

- [`densenet`](reference/densenet.md#densenet) module: DenseNet models for HASC for Keras.
- [`efficientnet`](reference/efficientnet.md#efficientnet) module : EfficientNet models for HASC for Keras.
- [`inception_resnet_v2`](reference/inception_resnet_v2.md#inception-resnet-v2) module: Inception-ResNet V2 model for HASC for Keras.
- [`inception_v3`](reference/inception_v3.md#inception-v3) module: Inception V3 model for HASC for Keras.
- [`mobilenet`](reference/mobilenet.md#mobilenet) module : MobileNet v1 models for HASC for Keras.
- [`mobilenet_v2`](reference/mobilenet_v2.md#mobilenet-v2) module: MobileNet v2 models for HASC for Keras.
- [`mobilenet_v3`](reference/mobilenet_v3.md#mobilenet-v3) module: MobileNet v3 models for HASC for Keras.
- [`nasnet`](reference/nasnet.md#nasnet) module: NASNet-A models for HASC for Keras.
- [`resnet`](reference/resnet.md#resnet) module : ResNet models for HASC for Keras.
- `resnet_v2` module: ResNet v2 models for HASC for Keras, **Not implemented yet**.
- [`vgg11`](reference/vgg.md#applicationsvgg11vgg11) module : VGG11 model for HASC for Keras.
- [`vgg13`](reference/vgg.md#applicationsvgg13vgg13) module : VGG13 model for HASC for Keras.
- [`vgg16`](reference/vgg.md#applicationsvgg16vgg16) module : VGG16 model for HASC for Keras.
- [`vgg19`](reference/vgg.md#applicationsvgg19vgg19) module : VGG19 model for HASC for Keras.
- [`xception`](reference/xception.md#xception) module: Xception V1 model for HASC for Keras.
- [`mnasnet`](reference/mnasnet.md#mnasnet) module: MnasNet-A1 model for HASC for Keras.
- [`pyramidnet`](reference/pyramidnet.md#pyramidnet) module : PyramidNet models for HASC for Keras.
- [`efficientnet_lite`](reference/efficientnet_lite.md#efficientnet-lite) module : EfficientNet-Lite models for HASC for Keras.

## Performance
Under construction...

## Install

```bash
pip install git+https://github.com/Shakshi3104/tfgarden.git
```

## Dependency
- `tensorflow >= 2.0`