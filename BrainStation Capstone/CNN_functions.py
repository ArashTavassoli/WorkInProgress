import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import math
from IPython.display import clear_output


def mini_batch_generator(X, Y, mini_batch_size = 100, seed = 1):
    """
    Generates random minibatches from (X, Y) and return a list 
    of (mini_batch_X, mini_batch_Y) for mini-batch gradient descent

    Arguments:
    X -- input data, of shape (m, n_H, n_W, n_C)
    Y -- ture labels, of shape (m, n_y)
    mini_batch_size -- size of each mini-batch (int)
    seed -- random seed value for reproducibility

    Returns:
    mini_batch_list -- the list of (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)        # Set random seed

    m = Y.shape[0]              # Number of examples
    mini_batch_list = []

    # Shuffle X and Y:
    permuted_range = list(np.random.permutation(m))
    X_shuffled = X[permuted_range,:,:,:]
    Y_shuffled = Y[permuted_range,:]

    # Partition X_shuffled and Y_shuffled:
    n_complete_mini_batches = math.floor(m/mini_batch_size)    # Number of mini-batches of size mini_batch_size
    for t in range(n_complete_mini_batches):
        mini_batch_X = X_shuffled[t * mini_batch_size : t * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = Y_shuffled[t * mini_batch_size : t * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batch_list.append(mini_batch)
    
    # Handling the last mini-batch
    if m % mini_batch_size != 0:
        mini_batch_X = X_shuffled[n_complete_mini_batches * mini_batch_size : m,:,:,:]
        mini_batch_Y = Y_shuffled[n_complete_mini_batches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batch_list.append(mini_batch)
    
    return mini_batch_list

def forward_prop(X, parameters):
    """
    Implement forward prop for 3 layer NN:
    (CONV2D>RELU>MAXPOOL) > (CONV2D>RELU>MAXPOOL) > (FLATTEN>FC)

    Arguments:
    X -- input data, of shape (m, n_H, n_W, n_C)
    parameters -- python dictionary containing parameters "W1", "W2"

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Note: For ksize and strides below the dims are [batch, height, width, channels]
    
    W1 = parameters['W1']
    W2 = parameters['W2']

    # CONV2D with stride of 1, padding 'SAME' 
    Z1 = tf.nn.conv2d(X,W1, strides = [1,1,1,1], padding = 'SAME')
    # RELU activation
    A1 = tf.nn.relu(Z1)
    # MAXPOOL with window 2x2, sride 2, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
    
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1,W2, strides = [1,1,1,1], padding = 'SAME')
    # RELU activation
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize = [1,4,4,1], strides = [1,4,4,1], padding = 'SAME')
    
    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function
    Z3 = tf.contrib.layers.fully_connected(P2, 3, activation_fn=None)

    return Z3

def CNN_model(X_train, Y_train, X_test, Y_test, learning_rate = 0.001,
    num_epochs = 100, mini_batch_size = 100, seed = 1, print_cost = True):
    """
    Implements a three-layer Convolutional Neural Net in Tensorflow:
    (CONV2D>RELU>MAXPOOL) > (CONV2D>RELU>MAXPOOL) > (FLATTEN>FC)

    Arguments:
    X_train -- training set X
    Y_train -- test set Y
    X_test -- training set X
    Y_test -- test set Y
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model
    """

    ops.reset_default_graph()
    tf.set_random_seed(seed)    # Set random seed (tf seed)
    seed = seed                 # Set random seed (numpy seed)
    
    (m, n_H, n_W, n_C) = X_train.shape
    n_y = Y_train.shape[1]

    costs = []
    
    # Create placeholders for the tensorflow session
    X = tf.placeholder(tf.float32, (None, n_H, n_W, n_C))
    Y = tf.placeholder(tf.float32, (None, n_y))

    # Initializes weight parameters with Xavier initializer
        # The shape of W is [f_l, f_l, n_C_(l-1), n_C_l]
        # Here the shapes are W1: [4, 4, 1, 8], W2: [2, 2, 8, 16]
    W1 = tf.get_variable("W1", [4, 4, 1, 8], initializer = tf.contrib.layers.xavier_initializer(seed = seed))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer = tf.contrib.layers.xavier_initializer(seed = seed))
    parameters = {"W1": W1, "W2": W2}

    # Implement forward prop for 2 layer NN
    # (CONV2D>RELU>MAXPOOL) > (CONV2D>RELU>MAXPOOL) > (FLATTEN>FC)
    Z3 = forward_prop(X, parameters)

    # Compute cost:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))

    # Define optimizer for back propagation
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
  
    # Initialize all variables globally
    init = tf.global_variables_initializer()

    # Start the session to compute the tf graph
    with tf.Session() as sess:

        # Run the init
        sess.run(init)

        # Training loop
        for epoch in range(num_epochs):

            seed += 1    # Change seed on each epoch
            minibatch_cost = 0.
            minibatches = mini_batch_generator(X_train, Y_train, mini_batch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                minibatch_cost += temp_cost

            # Using the number of full size minibatches in the train set to take average
            num_minibatches = int(m / mini_batch_size)
            minibatch_cost = minibatch_cost / num_minibatches

            # Print the cost
            if print_cost == True and epoch % 2 == 0:
                clear_output(wait = True)
                print (f"Cost after epoch {epoch}: {minibatch_cost}")
            
            # Save costs for plotting
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Finds the label by returning the index with the largest value for Z3 or Y
        predicted_y = tf.argmax(Z3, 1)
        correct_y = tf.argmax(Y, 1)

        # Correct model predictions
        correct_prediction = tf.equal(predicted_y, correct_y)
        
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters