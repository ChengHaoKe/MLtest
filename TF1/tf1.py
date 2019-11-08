import tensorflow as tf
import matplotlib.pyplot as plt
import os
# remove warning for cpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# define functions because pycharm doesn't locate them
Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
Conv2D = tf.keras.layers.Conv2D
BatchNormalization = tf.keras.layers.BatchNormalization
EarlyStopping = tf.keras.callbacks.EarlyStopping

# get data from online
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_data = x_train.reshape(x_train.shape[0], 28, 28, 1)
test_data = x_test.reshape(x_test.shape[0], 28, 28, 1)

# create model
model = tf.keras.Sequential()
model.add(Conv2D(10, kernel_size=2, activation='relu', input_shape=(28, 28, 1)))
# # Add a dropout layer
# model.add(Dropout(0.2))
# Add batch normalization layer
model.add(BatchNormalization())
model.add(Conv2D(10, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
print(model.summary())

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define early_stopping_monitor
early_stopping_monitor = EarlyStopping(patience=2)

# Fit to training data
training = model.fit(train_data, y_train, validation_split=0.2, epochs=5, batch_size=10,
                     callbacks=[early_stopping_monitor])

# Extract the history from the training object
history = training.history

# Plot the training loss
plt.plot(history['loss'], label='Training Loss')
# Plot the validation loss
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Training and Validation Loss')
# Show the figure
plt.show()

# Evaluate on test data
x = model.evaluate(test_data, y_test, batch_size=10)
print(x)

# get weights
mwei0 = model.get_weights()
mwei1 = mwei0[0]
mwei2 = mwei0[1]
mwei3 = mwei0[2]
mwei4 = mwei0[3]
mwei5 = mwei0[4]
mwei6 = mwei0[5]

# predict images
# convert to float 32
image = tf.cast(test_data[0:3], tf.float32)
model.predict(image)


# plot images
def imgplt(num):
    x1 = test_data[num].reshape(test_data[num].shape[0], test_data[num].shape[1])
    plt.imshow(x1)
    plt.show()


imgplt(0)
imgplt(1)
imgplt(2)

# # save model
# model.save('/Users/ch.ke/GitHub/MLtest/modelfiles/tf1.h5')
# # model = tf.keras.models.load_model('/Users/ch.ke/GitHub/MLtest/modelfiles/tf1.h5')
# # save weights
# model.save_weights('/Users/ch.ke/GitHub/MLtest/modelfiles/tf1weights.h5')
# # model.load_weights('Users/ch.ke/GitHub/MLtest/modelfiles/tf1weights.h5')
