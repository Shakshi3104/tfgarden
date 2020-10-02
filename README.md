# TensorFlow CNN Collections for HASC
This is the repository of CNN models implemented for human activity recognition (HASC), like [`tensorflow.keras.applications`](https://www.tensorflow.org/api_docs/python/tf/keras/applications).

The models implemented here can also be used as a source domain for sensor-based task (e.g. sidewalk surface type estimation).

## Modules
TensorFlow CNN Collections for HASC are canned architecture with pre-trained weights.

pre-trained weights are uploaded [here](https://drive.google.com/drive/folders/1HMDMDz91laNvsyaTvAMgXzX-pIjDMpwy?usp=sharing).


- [`vgg11`](reference/vgg.md#applicationsvgg11vgg11) module : VGG11 model for HASC for Keras.
- [`vgg13`](reference/vgg.md#applicationsvgg13vgg13) module : VGG13 model for HASC for Keras.
- [`vgg16`](reference/vgg.md#applicationsvgg16vgg16) module : VGG16 model for HASC for Keras.
- [`vgg19`](reference/vgg.md#applicationsvgg19vgg19) module : VGG19 model for HASC for Keras.
- [`resnet`](reference/resnet.md#resnet) module : ResNet models for HASC for Keras.
- `alexnet` module : AlexNet model for HASC for Keras, **Not implemented yet**.
- [`pyramidnet`](reference/pyramidnet.md#pyramidnet) module : PyramidNet models for HASC for Keras.
- [`efficientnet`](reference/efficientnet.md#efficientnet) module : EfficientNet models for HASC for Keras.


## Accuracy
| models | accuracy |
| :----: | :------: |
| VGG11  | 85.5% |
| VGG13  | 83.9% |
| VGG16  | 84.1% |
| VGG19  | 84.1% |
| ResNet16 | 79.3%  |
| ResNet18 | 80.7% |
| PyramidNet18 | 75.4% |
| PyramidNet34 | 75.4% |
| PyramidNet50 | 74.7% |
| PyramidNet101 | 77.6% |
| PyramidNet152 | 76.3% |
| EfficientNetB0 | 89.9% |
| EfficientNetB1 | 88.5%|
| EfficientNetB2 ||
| EfficientNetB3 ||
| EfficientNetB4 ||
| EfficientNetB5 ||
| EfficientNetB6 ||
| EfficientNetB7 ||
