# Example 4.4 Modified: LeNet-5 Keras with Increased Filters
from keras.layers import AveragePooling2D, Conv2D, Dense, Flatten
from keras.models import Sequential

model = Sequential()
# Lapisan Conv2D pertama diubah dari 6 menjadi 12 filter
model.add(
    Conv2D(
        filters=12,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(32, 32, 1),
    )
)
model.add(AveragePooling2D(pool_size=(2, 2)))
# Lapisan Conv2D kedua diubah dari 16 menjadi 32 filter
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
model.summary()