import numpy as np


def get_data(max_int=10, size=10000):
    input1 = np.random.randint(max_int+1, size=size) # shape := [1, 10000]
    input2 = np.random.randint(max_int+1, size=size) # shape := [1, 10000]
    inputs = np.stack((input1, input2), axis=-1) # shape := [10000, 2] -> same as placeholder 'x'
    
    targets = ((input1 + input2) > 10).astype(int) # shape := [10000,]
    # one-hot encoding
    targets = np.eye(2)[targets] # shape := [10000, 2] -> same as placeholder 'y'

    return inputs, targets


def split_data(inputs, targets, train_percentage=.75):
    assert (0 <= train_percentage) and (train_percentage <= 1) # Train percentage must be within [0,1]

    size = len(inputs) # length of the data
    train_size = int(np.ceil(size*train_percentage))
    test_size = size - train_size

    train_inputs = inputs[:train_size, :]
    test_inputs = inputs[-test_size:, :]
    train_targets = targets[:train_size]
    test_targets = targets[-test_size:]

    return (train_inputs, test_inputs, train_targets, test_targets)


def get_batches(inputs, targets, batch_size):
    # TOIMPROVE: Now, if I can't fill the last batch with enough data, I'm droping it.
    # I'm not using all the information that I have
    
    # batches have to be picked randomly, so we need to shuffle the data
    p = np.random.permutation(len(inputs))
    inputs = inputs[p]
    targets = targets[p]

    num_batches = len(inputs)//batch_size

    batch_x = np.zeros([num_batches, batch_size, 2]) # batch_x[0] has to be the same shape than placeholder x
    batch_y = np.zeros([num_batches, batch_size, 2]) # batch_y[0] has to be the same shape than placeholder y

    for batch in range(num_batches):
        batch_x[batch][:, 0] = inputs[batch*batch_size:batch*batch_size + batch_size, 0]
        batch_x[batch][:, 1] = inputs[batch*batch_size:batch*batch_size + batch_size, 1]
        batch_y[batch][:, 0] = targets[batch*batch_size:batch*batch_size + batch_size, 0]
        batch_y[batch][:, 1] = targets[batch*batch_size:batch*batch_size + batch_size, 1]
    
    return batch_x, batch_y