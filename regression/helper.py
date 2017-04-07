import numpy as np

def split_data(inputs, ground_truth):
    size = inputs.shape[1]
    train_size = np.ceil(size*.75)
    test_size = size - train_size

    train_inputs = inputs[:, :train_size]
    test_inputs = inputs[:, -test_size:]
    train_truth = ground_truth[:train_size]
    test_truth = ground_truth[-test_size:]

    return (train_inputs, test_inputs, train_truth, test_truth)

def get_batches(inputs, truth, batch_size):
    # TOIMPROVE: Now, if I can't fill the last batch with enough data, I'm droping it.
    # I'm not using all the information that I have
    num_batches = inputs.shape[1]//batch_size

    batch_x = np.zeros([num_batches, 2, batch_size]) # batch_x[0] has to be the same shape than placeholder x
    batch_y = np.zeros([num_batches, 1, batch_size]) # batch_y[0] has to be the same shape than placeholder y

    for batch in range(num_batches):
        batch_x[batch][0] = inputs[0][batch*batch_size:batch*batch_size + batch_size]
        batch_x[batch][1] = inputs[1][batch*batch_size:batch*batch_size + batch_size]
        batch_y[batch][0] = truth[batch*batch_size:batch*batch_size + batch_size]
    
    return batch_x, batch_y