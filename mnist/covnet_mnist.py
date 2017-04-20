import tensorflow as tf
    
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True, reshape=False)

# Parameters
learning_rate = 0.00001
epochs = 10
batch_size = 128

# Number of samples to calculate validation and accuracy
# Decrease this if you're running out of memory to calculate accuracy
test_valid_size = 256

# Network parameters
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to KEEP units

# Store layers weight & biases in dictionaries
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name='Wc1'), 
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name='Wc2'),   
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024]), name='Wfc'),
    'out': tf.Variable(tf.random_normal([1024, n_classes]), name='Wo') 
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32]), name='bc1'),
    'bc2': tf.Variable(tf.random_normal([64]), name='bc2'),
    'bd1': tf.Variable(tf.random_normal([1024]), name='bfc'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='bo')
}

def conv2d(x, W, b, strides=1, name='conv'):
    with tf.name_scope(name):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        x = tf.nn.relu(x)
    return x

def maxpool2d(x, k=2, name='max_pool'):
    with tf.name_scope(name):
        x = tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')
    return x

def conv_net(x, weights, biases, dropout):
    # Layer 1 - 28x28x1 to 14x14x32
    conv1 = conv2d(x, weights['wc1'], biases['bc1'], name='conv1')
    conv1 = maxpool2d(conv1, k=2, name='max_pool1')

    # Layer 2 - 14x14x32 to 7x7x64
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], name='conv2')
    conv2 = maxpool2d(conv2, k=2, name='max_pool2')

    # Fully conected Layer - 7x7x64 to 1024
    with tf.name_scope('fc1'):
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, dropout)

    with tf.name_scope('fc2'):
        # Output Layer - class prediction - 1024 to 10
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return out


# tf Graph input
x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) # keep probability (dropout)

# Model
logits = conv_net(x, weights, biases, keep_prob)
# pred = tf.nn.softmax(logits)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initiliazing the variables
init = tf.global_variables_initializer()

# Saver for variables
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(epochs):
        for batch in range(mnist.train.num_examples//batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={
                x: batch_x,
                y: batch_y,
                keep_prob: dropout})

            # Calculate batch loss and accuracy
            loss = sess.run(cost, feed_dict={
                x: batch_x,
                y: batch_y,
                keep_prob: 1.})
            valid_acc = sess.run(accuracy, feed_dict={
                x: mnist.validation.images[:test_valid_size],
                y: mnist.validation.labels[:test_valid_size],
                keep_prob: 1.})

            print('Epoch {:>2}, Batch {:3} -'
                'Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(
                    epoch + 1, 
                    batch + 1,
                    loss,
                    valid_acc))

    # Calculate Test Accuracy
    test_acc = sess.run(accuracy, feed_dict={
        x: mnist.test.images[:test_valid_size],
        y: mnist.test.labels[:test_valid_size],
        keep_prob: 1.})
    print('Testing Accuracy: {}'.format(test_acc))

    # Save the variables to disk.
    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)





