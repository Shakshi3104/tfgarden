# Quickstart Example

This example demonstrates how to use an implemented model for human activity recognition.

## Load the model
Load the MobileNetV2 model, like a [tensorflow.keras.applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications) API:
```python
import tfgarden

model = tfgarden.applications.MobileNetV2(
    include_top=True,
    weights=None,
    input_shape=(256, 3),
    classes=6
)
```

## Train the model
Train the loaded model as you normally [train a keras model](https://www.tensorflow.org/tutorials/keras/classification):
```python
# compile the model
# before the model is ready for training,
# it needs a few more settings
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
stack = model.fit(
    train_ds, 
    epochs=100,
    validation_data=valid_ds
)
```