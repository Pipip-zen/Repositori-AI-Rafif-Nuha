# Example 4.16: Autoencoder with Keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten, Input, Reshape
from tensorflow.keras.models import Model, Sequential

# Load and normalize the MNIST handwritten digit images.
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
print(x_train.shape)
print(x_test.shape)


def build_autoencoder(img_shape, code_size):
    """Create an encoder and decoder for images of the given shape."""
    encoder = Sequential([
        Input(shape=img_shape),
        Flatten(),
        Dense(code_size),
    ])
    decoder = Sequential([
        Input(shape=(code_size,)),
        Dense(int(np.prod(img_shape))),
        Reshape(img_shape),
    ])
    return encoder, decoder


image_shape = x_train[0].shape
encoder, decoder = build_autoencoder(image_shape, 32)
inputs = Input(shape=image_shape)
code = encoder(inputs)
reconstruction = decoder(code)
autoencoder = Model(inputs, reconstruction)
autoencoder.compile(optimizer='adamax', loss='mse')
autoencoder.summary()

# Train the autoencoder.
history = autoencoder.fit(
    x=x_train,
    y=x_train,
    epochs=20,
    validation_data=(x_test, x_test),
)

# Plot the training results.
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


def show_image(image):
    plt.imshow(np.clip(image, 0, 1), cmap='gray')
    plt.axis('off')


def visualize(image, image_encoder, image_decoder):
    """Display an image, its encoded representation, and its reconstruction."""
    code = image_encoder.predict(image[None], verbose=0)[0]
    reconstruction = image_decoder.predict(code[None], verbose=0)[0]

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 3, 1)
    plt.title('Original')
    show_image(image)
    plt.subplot(1, 3, 2)
    plt.title('Code')
    plt.imshow(code.reshape(4, 8), cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.title('Reconstructed')
    show_image(reconstruction)
    plt.tight_layout()
    plt.show()


for index in range(5):
    visualize(x_test[index], encoder, decoder)
