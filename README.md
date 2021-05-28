# TensorFlow model Garden for HASC
The TensorFlow model Garden (TensorFlow CNN Collections) for HASC is the repository of CNN models implemented for human activity recognition (HASC), like [`tensorflow.keras.applications`](https://www.tensorflow.org/api_docs/python/tf/keras/applications).

The models implemented here can also be used as a source domain for sensor-based task (e.g. sidewalk surface type estimation).

## Modules
TensorFlow CNN Collections for HASC are canned architecture with pre-trained weights.

pre-trained weights are uploaded [here](https://drive.google.com/drive/folders/1HMDMDz91laNvsyaTvAMgXzX-pIjDMpwy?usp=sharing).

- `densenet` module: DenseNet models for HASC for Keras, **Not implemented yet**.
- [`efficientnet`](reference/efficientnet.md#efficientnet) module : EfficientNet models for HASC for Keras.
- `inception_resnet_v2` module: Inception-ResNet V2 model for HASC for Keras, **Not implemented yet**.
- `inception_v3` module: Inception V3 model for HASC for Keras, **Not implemented yet**.
- [`mobilenet`](reference/mobilenet.md#mobilenet) module : MobileNet v1 models for HASC for Keras.
- [`mobilenet_v2`](reference/mobilenet_v2.md#mobilenetv2) module: MobileNet v2 models for HASC for Keras.
- `mobilenet_v3` module: MobileNet v3 models for HASC for Keras, **Not implemented yet**.
- `nasnet` module: NASNet-A models for HASC for Keras, **Not implemented yet**.
- [`resnet`](reference/resnet.md#resnet) module : ResNet models for HASC for Keras.
- `resnet_v2` module: ResNet v2 models for HASC for Keras, **Not implemented yet**.
- [`vgg11`](reference/vgg.md#applicationsvgg11vgg11) module : VGG11 model for HASC for Keras.
- [`vgg13`](reference/vgg.md#applicationsvgg13vgg13) module : VGG13 model for HASC for Keras.
- [`vgg16`](reference/vgg.md#applicationsvgg16vgg16) module : VGG16 model for HASC for Keras.
- [`vgg19`](reference/vgg.md#applicationsvgg19vgg19) module : VGG19 model for HASC for Keras.
- `xception` module: Xception V1 model for HASC for Keras, **Not implemented yet**.
- `mnasnet` module: MnasNet model for HASC for Keras, **Not implemented yet**.
- [`pyramidnet`](reference/pyramidnet.md#pyramidnet) module : PyramidNet models for HASC for Keras.
- [`efficientnet_lite`](reference/efficientnet_lite.md#efficientnet-lite) module : EfficientNet-Lite models for HASC for Keras.


## Accuracy 
| model | accuracy |
|:-----:|:--------:|
| VGG11 | 85.5 % |
| VGG13 | 83.9 % |
| VGG16 | 84.1 % |
| VGG19 | 84.0 % |
| ResNet16 | 79.3 % |
| ResNet18 | 80.7 % |
| PyramidNet18 | 86.2 % |
| PyramidNet34 | 86.8 % |
| PyramidNet50 | 86.1 % |
| PyramidNet101 | 87.1 % |
| PyramidNet152 | 87.7 % |
| EfficientNetB0 | 89.9 % |
| EfficientNetB1 | 88.5 % |
| EfficientNetB2 | 89.6 % |
| EfficientNetB3 | 90.1 % |
| EfficientNetB4 | 88.9 % |
| EfficientNetB5 | 89.9 % |
| EfficientNetB6 | 88.0 % |
| EfficientNetB7 | 89.2 % |
| EfficientNet_lite0 | 87.4 % |
| EfficientNet_lite1 | 88.8 % |
| EfficientNet_lite2 | 84.5 % |
| EfficientNet_lite3 | 87.4 % |
| EfficientNet_lite4 | 87.8 % |


#### Training Conditions

|  | condition |
|:---|:---:|
| Number of data | 25,130 |
| Number of subject | 183 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Batch size | 20 |
| Epochs | 200 |

## HASC
**HASC Corpus** is the dataset of human activity recognition collected by [human activity sensing consortium (HASC)](http://hasc.jp).

## Dependency
- `tensorflow >= 2.0`