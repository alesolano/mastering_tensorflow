import tensorflow as tf
import numpy as np

from helper import get_data, split_data, get_batches


# MISSION:
# Learn if the sum of two integers are higher than 10.


############ BUILDING THE GRAPH ############

# HYPERPARAMTERS
batch_size = 50
epochs = 10

# MODEL
x = tf.placeholder(tf.float32, [None, 2])

# Hidden layer:
W_h = tf.Variable(tf.truncated_normal([2, 1], stddev=0.05))
b_h = tf.Variable(tf.random_normal([1]))

hidden = tf.add(tf.matmul(x, W_h), b_h)
#hidden = tf.sigmoid(hidden) # We don't need to add non-linear behaviors to our network.

W_o = tf.Variable(tf.truncated_normal([1, 2], stddev=0.05))
b_o = tf.Variable(tf.random_normal([1]))

logits = tf.add(tf.matmul(hidden, W_o), b_o)
output = tf.nn.softmax(logits)

y = tf.placeholder(tf.float32, [None, 2])

# Training
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, axis=1), tf.argmax(y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


############ DATA ############
# inputs: two integers
# targets: [0, 1] if the sum is higher than 10. [1, 0] if the sum is lower than 10.

inputs, targets = get_data(max_int=10, size=10000)

# preprocessing: normalize inputs to be between -1 and 1.
inputs = (inputs-5)/5
# TODO: make a preprocessing helper function, substracting the mean and dividing by the max value

# split train and test data
train_inputs, test_inputs, train_targets, test_targets = split_data(inputs, targets)


############ SESSION ############
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # TRAINING
    for epoch in range(epochs):

        batch_x, batch_y = get_batches(train_inputs, train_targets, batch_size)

        for batch in range(train_inputs.shape[0]//batch_size):

            sess.run(optimizer, feed_dict={
                x: batch_x[batch],
                y: batch_y[batch]
                })

            train_loss = sess.run(cost, feed_dict={
                x: batch_x[batch],
                y: batch_y[batch]
                })

            if batch == 0:
                print('Epoch {:>2}, Batch {:3} - '
                    'Training Loss: {:>10.4f}'.format(
                        epoch + 1, 
                        batch + 1,
                        train_loss))
    
    # TESTING
    test_accuracy = sess.run(accuracy, feed_dict={
        x: test_inputs,
        y: test_targets
        })
    print('Testing Accuracy: {}'.format(test_accuracy))


    final_test = sess.run(output, feed_dict={
        x: (np.array([[5, 3], [7, 6], [10, 10]])-5)/5
        })
    print(final_test)

    print("\nHidden layer weights and bias")
    print(sess.run(W_h))
    print(sess.run(b_h))
    print("Output layer weights and bias")
    print(sess.run(W_o))
    print(sess.run(b_o))
    