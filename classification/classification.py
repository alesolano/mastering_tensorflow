import tensorflow as tf
import numpy as np

from helper import split_data, get_batches


# MISSION:
# Learn if the sum of two integers are higher than 10.


############ BUILDING THE GRAPH ############
# HYPERPARAMTERS
#learning_rate = 0.5 # Terribly important hyperparameter. It can make your net go totally crazy.
batch_size = 100
epochs = 200

# MODEL
x = tf.placeholder(tf.float32, [None, 2])

# Hidden layer:
W_h = tf.Variable(tf.truncated_normal([2, 1], stddev=0.05))
b_h = tf.Variable(tf.random_normal([1]))

hidden = tf.add(tf.matmul(x, W_h), b_h)
hidden = tf.sigmoid(hidden)

W_o = tf.Variable(tf.truncated_normal([1,2], stddev=0.05))
b_o = tf.Variable(tf.random_normal([1]))

logits = tf.add(tf.matmul(hidden, W_o), b_o)
output = tf.nn.softmax(logits)

y = tf.placeholder(tf.float32, [None, 2]) 

# Training
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, axis= 1), tf.argmax(y, axis=1)) # Not very sure about the axis
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


############ DATA ############
input1 = np.random.randint(10, size=10000) # shape := [1, 10000]
input2 = np.random.randint(10, size=10000) # shape := [1, 10000]
inputs = np.stack((input1, input2), axis=-1) # shape := [10000, 2] -> same as placeholder 'x'

target1 = ((input1 + input2) > 10).astype(int) # shape := [10000,]
target1 = np.reshape(target1, [-1, 1]) # shape := [10000, 1]
target2 = 1 - target1 # shape := [10000, 1]
targets = np.reshape(np.stack((target1, target2), axis=-1), [-1, 2]) # shape := [10000, 2] -> same as placeholder 'y'



############ SESSION ############
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    train_inputs, test_inputs, train_targets, test_targets = split_data(inputs, targets)
    batch_x, batch_y = get_batches(train_inputs, train_targets, batch_size)

    # TRAINING
    # I don't know very well what epochs are
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
        x: np.array([[5, 3], [7, 6], [10, 10]])
        })
    print(final_test)


