import abc


# 深層学習モデルの基底クラス
class DLModelBuilder(metaclass=abc.ABCMeta):
    def __init__(self, kernel_size, strides, kernel_initializer, padding, input_shape, num_classes):
        self.kernel_size = kernel_size
        self.strides = strides
        self.kernel_initializer = kernel_initializer
        self.padding = padding
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = ""

    @abc.abstractmethod
    def get_model(self):
        raise NotImplementedError()
        # pass
