from tensorflow.keras.layers import Conv1D, MaxPooling1D


class ConvBlock:
    def __init__(self, repeat, filters, kernel_size=3, strides=1, padding='same', activation='relu',
                 kernel_initializer='he_normal', pool_size=2):
        """
        ConvBlock for VGG
            repeat: the number of Conv1D
            filters: the number of filter of Conv1D
            kernel_size: the kernel_size of Conv1D, default `3`
            strides: the strides of Conv1D, default `1`
            padding: the padding of Conv1D and MaxPooling1D, default `'same'`
            activation: the activation function of Conv1D, default `'relu'`
            kernel_initializer: the kernel_initializer of Conv1D, default `'he_normal'`
            pool_size: the pool_size of MaxPooling1D, default `2`
        """
        self.repeat = repeat
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.pool_size = pool_size

    def __call__(self, x):
        for _ in range(0, self.repeat):
            x = Conv1D(self.filters, kernel_size=self.kernel_size, strides=self.strides, padding=self.padding,
                       activation=self.activation, kernel_initializer=self.kernel_initializer)(x)

        x = MaxPooling1D(pool_size=self.pool_size, padding=self.padding)(x)
        return x
