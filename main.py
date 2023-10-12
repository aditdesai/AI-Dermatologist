import tensorflow as tf
import numpy as np


def load_data():
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")

    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_data()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# preprocessing
y_train.reshape(y_train.shape[0], 1)
y_test.reshape(y_test.shape[0], 1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


# define model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(256, 256, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# train model
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=10)

# evaluate model
print(f"Accuracy: {model.evaluate(x_test, y_test)[1] * 100}%")


# save model
model.save("melanoma_cnn.keras")