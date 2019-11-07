import tensorflow as tf
import os
# remove warning for cpu
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Introduction to Machine Learning Week 1


# Method 1
with tf.compat.v1.Session() as sess:
    # Run the tf.constant operation in the session
    hello = tf.constant("Hello, TensorFlow!")  # Add a constant-operation to the (default) graph
    output = sess.run(hello)
    print(output)
    sess.close()


# Method 2
hello1 = tf.constant("Hello, TensorFlow!")
print(hello1)
