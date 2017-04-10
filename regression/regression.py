import tensorflow as tf
import numpy as np

from helper import get_data, split_data, get_batches


# MISSION:
# Learn how to sum two positive integers.



############ BUILDING THE GRAPH ############
# HYPERPARAMETERS
learning_rate = 0.00001 # Terribly important hyperparameter. It can make your net go totally crazy.
batch_size = 100
epochs = 4

# MODEL
x = tf.placeholder(tf.float32, [None, 2])

W = tf.Variable(tf.truncated_normal([2, 1], stddev=0.05))
b = tf.Variable(tf.random_normal([1]))

output = tf.add(tf.matmul(x, W), b)
# We're in a regression problem. We don't need an activation function

y = tf.placeholder(tf.float32, [None, 1]) 

# TRAINING
cost = tf.reduce_sum(tf.square(output - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# ACCURACY
accuracy = tf.reduce_mean(y - tf.abs(output - y)) / tf.reduce_mean(y)



############ DATA ############
inputs, targets = get_data(max_int=10, size=10000)



############ SESSION ############
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_inputs, test_inputs, train_targets, test_targets = split_data(inputs, targets)
    batch_x, batch_y = get_batches(train_inputs, train_targets, batch_size)

    # TRAINING
    for epoch in range(epochs):
        for batch in range(train_inputs.shape[0]//batch_size):

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
    

    # TESTING
    test_accuracy = sess.run(accuracy, feed_dict={
        x: test_inputs,
        y: test_targets
        })
    print('Testing Accuracy: {}'.format(test_accuracy))

    final_test = sess.run(output, feed_dict={
        x: np.array([[5, 7]])
        })
    print("The sum of 5 plus 7 is {}".format(final_test[0][0])) # The result will be near 12.

    print("The weights are: {}".format(sess.run(W)))
    print("and the bias is: {}".format(sess.run(b)))
    # Obviously, to compute the sum, weights need to be [1, 1] and bias 0
    # So, in this case, we should initialize bias with zeros:
    # b = tf.Variable(tf.zeros([1]))
    # Conclussion: we should understand what each layer is doing, so we could make things work efficiently
    # (Maybe we should force each layer to behave like we want to)
    
