import tensorflow as tf
from os import path
import os
# remove warning for cpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def train_mnist(data=True):
    print(tf.__version__)
    if data:
        # get data from online
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    class mycallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if logs.get('acc') >= 0.99:
                print('\nReached 99% accuracy so cancelling training!')
                self.model.stop_training = True

    callbacks = mycallback()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model fitting
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks]
                        )

    # Evaluate on test data
    x = model.evaluate(x_test, y_test, batch_size=10)
    print(x)

    # model fitting
    return history.epoch, history.history['acc'][-1]


train_mnist()
