import tensorflow as tf
import numpy as np

from helper import split_data, get_batches


# MISSION:
# Learn how to sum.


############ BUILDING THE GRAPH ############
# Hyperparameters
learning_rate = 0.01
batch_size = 100
epochs = 1

# Model
x = tf.placeholder(tf.float32, [2, None]) # Not very sure about this shape. Maybe [None, 2]?
# TODO: declare 'x' and 'y' as tf.int32

W = tf.Variable(tf.truncated_normal([1, 2], stddev=0.05))
b = tf.Variable(tf.random_normal([1]))

output = tf.add(tf.matmul(W, x), b)
# We're in a regression problem. We don't need an activation function

y = tf.placeholder(tf.float32, [1, None]) # Not very sure about this shape. Maybe [None]?

# Training
# WE HAVE A TERRIBLE PROBLEM HERE. THE NN IS NOT LEARNING
cost = tf.reduce_mean(tf.nn.l2_loss(y - output)) # NOTHING SURE ABOUT THIS
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(y, output)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


############ DATA ############
input1 = np.random.randint(10, size=10000)
input2 = np.random.randint(10, size=10000)
inputs = np.array([input1, input2])
ground_truth = input1 + input2



############ SESSION ############
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_inputs, test_inputs, train_truth, test_truth = split_data(inputs, ground_truth)
    batch_x, batch_y = get_batches(train_inputs, train_truth, batch_size)

    # I don't know very well what epochs are
    for epoch in range(epochs):
        for batch in range(train_inputs.shape[1]//batch_size):
            sess.run(optimizer, feed_dict={
                x: batch_x[batch],
                y: batch_y[batch]
                })
            
            train_loss = sess.run(cost, feed_dict={
                x: batch_x[batch],
                y: batch_y[batch]
                })

            print('Epoch {:>2}, Batch {:3} - '
                'Training Loss: {:>10.4f}'.format(
                    epoch + 1, 
                    batch + 1,
                    train_loss))
    
    test_acc = sess.run(accuracy, feed_dict={
        x: test_inputs,
        y: np.array([test_truth]) # we need 'y' to have a shape of [1, None]
        })
    print('Testing Accuracy: {}'.format(test_acc))


