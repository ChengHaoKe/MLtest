import tensorflow as tf
from datetime import datetime
import time
import os
# remove warning for cpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning W3 in 2.0
def train_mnist(data=True):
    start_time = time.time()

    print(tf.__version__)
    if data:
        # get data from online
        print('Loading data from online source')
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # custom callback function
    class mycallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('\nTraining: epoch {} ends at {}'.format(epoch, datetime.now()))
            try:
                if logs.get('accuracy') >= 0.99:
                    print('\nReached 99% accuracy so cancelling training!')
                    self.model.stop_training = True
            except TypeError:
                print(logs)
                raise  # re-raise exception

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
    print('Model loss:', x[0], ' Model accuracy:', x[1])

    print("Function ran for: {0} seconds ({1} mins)".format(str(round(time.time() - start_time, 2)),
                                                            str(round((time.time() - start_time) / 60))))

    # model fitting
    return history.epoch, history.history['accuracy'][-1]


train_mnist()
# train_mnist(data=False)
