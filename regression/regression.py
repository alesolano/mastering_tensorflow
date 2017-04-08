import tensorflow as tf
import numpy as np

from helper import split_data, get_batches


# MISSION:
# Learn how to sum two positive integers.


############ BUILDING THE GRAPH ############
# Hyperparameters
learning_rate = 0.000009 # Terribly important hyperparameter. It can make your net go totally crazy.
batch_size = 100
epochs = 5

# Model
x = tf.placeholder(tf.float32, [2, None])

W = tf.Variable(tf.truncated_normal([1, 2], stddev=0.05))
b = tf.Variable(tf.random_normal([1]))

output = tf.add(tf.matmul(W, x), b)
# We're in a regression problem. We don't need an activation function

y = tf.placeholder(tf.float32, [1, None]) 

# Training
cost = tf.reduce_sum(tf.square(output - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
accuracy = tf.reduce_mean(y - tf.abs(output - y)) / tf.reduce_mean(y)


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

    # TRAINING
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
    

    # TESTING
    test_accuracy = sess.run(accuracy, feed_dict={
        x: test_inputs,
        y: np.array([test_truth]) # we need 'y' to have a shape of [1, None]
        })
    print('Testing Accuracy: {}'.format(test_accuracy))

    final_test = sess.run(output, feed_dict={
        x: np.array([[5], [7]])
        })
    print("The sum of 5 plus 7 is {}".format(final_test[0][0])) # The result will be near 12.

    print("The weights are: {}".format(sess.run(W)))
    print("and the bias is: {}".format(sess.run(b)))
    # Obviously, to compute the sum, weights need to be [1, 1] and bias 0
    # So, in this case, we should initialize bias with zeros:
    # b = tf.Variable(tf.zeros([1]))
    # Conclussion: we should understand what each layer is doing, so we could make things work efficiently
    # (Maybe we should force each layer to behave like we want to)
    
