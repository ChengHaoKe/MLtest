import tensorflow as tf
import matplotlib.pyplot as plt
import os
# remove warning for cpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


Dense = tf.keras.layers.Dense
Flatten = tf.keras.layers.Flatten
Conv2D = tf.keras.layers.Conv2D
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_data = x_train.reshape(x_train.shape[0], 28, 28, 1)
test_data = x_test.reshape(x_test.shape[0], 28, 28, 1)

model = tf.keras.Sequential()
model.add(Conv2D(10, kernel_size=2, activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(10, kernel_size=2, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
print(model.summary())

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Fit to training data
training = model.fit(train_data, y_train, validation_split=0.2, epochs=3, batch_size=10)

# Extract the history from the training object
history = training.history

# Plot the training loss
plt.plot(history['loss'])
# Plot the validation loss
plt.plot(history['val_loss'])

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

# save model or weights
model.save('')
