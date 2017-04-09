import numpy as np


def split_data(inputs, targets):
    size = inputs.shape[0]
    train_size = int(np.ceil(size*.75))
    test_size = size - train_size

    train_inputs = inputs[:train_size, :]
    test_inputs = inputs[-test_size:, :]
    train_targets = targets[:train_size]
    test_targets = targets[-test_size:]

    return (train_inputs, test_inputs, train_targets, test_targets)


def get_batches(inputs, targets, batch_size):
    # TOIMPROVE: Now, if I can't fill the last batch with enough data, I'm droping it.
    # I'm not using all the information that I have
    num_batches = inputs.shape[0]//batch_size

    batch_x = np.zeros([num_batches, batch_size, 2]) # batch_x[0] has to be the same shape than placeholder x
    batch_y = np.zeros([num_batches, batch_size, 1]) # batch_y[0] has to be the same shape than placeholder y

    for batch in range(num_batches):
        batch_x[batch][:, 0] = inputs[batch*batch_size:batch*batch_size + batch_size, 0]
        batch_x[batch][:, 1] = inputs[batch*batch_size:batch*batch_size + batch_size, 1]
        batch_y[batch][:] = targets[batch*batch_size:batch*batch_size + batch_size]
    
    return batch_x, batch_y