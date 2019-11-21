import tensorflow as tf
import os
import glob
import zipfile
import time
from datetime import datetime
import wget
import pathlib
# need to be installed for program to run:
# PIL is now Pillow
# import PIL
# import scipy

# remove warning for cpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
RMSprop = tf.keras.optimizers.RMSprop

# !wget --no-check-certificate \
#     "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
#     -O datadir + "happy-or-sad.zip"


# download file and create directory
datadir = os.getcwd()
datadir = datadir.replace('TF1', 'data/')
if not glob.glob(datadir + "happy-or-sad.zip"):
    pathlib.Path(datadir).mkdir(parents=True, exist_ok=True)
    url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip"
    filename = wget.download(url, datadir + "happy-or-sad.zip")

    # unzip file
    zip_ref = zipfile.ZipFile(datadir + "happy-or-sad.zip", 'r')
    zip_ref.extractall(datadir + "h-or-s")
    zip_ref.close()
else:
    print('Data already downloaded!')


# Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning W4 in 2.0
def train_happy_sad_model():
    start_time = time.time()

    print(tf.__version__)
    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('\nTraining: epoch {} ends at {}'.format(epoch, datetime.now()))
            try:
                if logs.get('accuracy') > DESIRED_ACCURACY:
                    print('\nReached over 99.9% accuracy so cancelling training!')
                    self.model.stop_training = True
            except TypeError:
                print(logs)
                raise  # re-raise exception

    callbacks = myCallback()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])

    model.compile(optimizer=RMSprop(lr=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1. / 255)

    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory(
        # Your Code Here)
        datadir + "h-or-s",
        target_size=(150, 150),
        batch_size=128,
        class_mode='binary')

    # model fitting
    history = model.fit_generator(
        # Your Code Here)
        train_generator,
        steps_per_epoch=8,
        epochs=15,
        callbacks=[callbacks])

    print("Function ran for: {0} seconds ({1} mins)".format(str(round(time.time() - start_time, 2)),
                                                            str(round((time.time() - start_time) / 60))))

    # model fitting
    return history.history['accuracy'][-1]


train_happy_sad_model()
