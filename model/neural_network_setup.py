import numpy as np
import tensorflow as tf

def sigmoid(z):
    
    """
    Computes the sigmoid of z using keras
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    a -- tf.float32, the sigmoid of z
    """
    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)
    return a

def initialize_parameters(layer_dims):
    """
    Initializes parameters to build a neural network with TensorFlow's GlorotNormal initializer.
    (which draws samples from a truncated normal distribution centered on 0).
    
    Arguments:
    layer_dims -- list, containing the dimensions of each layer in the network (including the input layer)
    
    Returns:
    parameters -- dictionary, containing tensors "W1", "b1", ..., "WL", "bL" for L-1 layers (L includes input layer)
    """
    initializer = tf.keras.initializers.GlorotNormal(seed=1)
    parameters = {}
    L = len(layer_dims)  # Number of layers including input

    for l in range(1, L):
        parameters['W' + str(l)] = tf.Variable(initializer(shape=(layer_dims[l], layer_dims[l-1])))
        parameters['b' + str(l)] = tf.Variable(initializer(shape=(layer_dims[l], 1)))

    return parameters

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: [LINEAR -> RELU]*(L-1) -> LINEAR

    Arguments:
    X -- input dataset, of shape (input size, number of examples)
    parameters -- dict, contains parameters "W1", "b1", ..., "WL", "bL".

    Returns:
    ZL -- tf.Tensor, the output of the last LINEAR unit
    """

    
    L = len(parameters) // 2 # number of layers in the neural network
    A_prev = X # set the initial activation to the input X
    
    for l in range(1, L):
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        Z = tf.math.add(tf.linalg.matmul(W, A_prev), b)
        A_prev = tf.keras.activations.relu(Z) #Apply Relu and update activations
            
    WL = parameters['W' + str(L)]
    bL = parameters['b' + str(L)]
    ZL = tf.add(tf.linalg.matmul(WL, A_prev), bL)
    
    return ZL
    
def compute_total_loss(logits, labels):
    """
    Computes the total loss as a categorial cross-entropy loss.
    Notice the tranpose as inputs of tf.keras.losses.categorical_crossentropy are expected to be of shape (number of examples, num_classes).
    
    Arguments:
    logits -- tf.Tensor, output of forward propagation (output of the last LINEAR unit), of shape (6, num_examples)
    labels -- tf.Tensor, "true" labels vector, same shape as logits
    
    Returns:
    total_loss - tf.Tensor, Tensor of the total loss value
    """
    
    total_loss = tf.reduce_sum(tf.keras.losses.categorical_crossentropy(tf.transpose(labels),tf.transpose(logits),from_logits=True))

    return total_loss

def evaluate_model(parameters, test_minibatches):
    """
    Evaluate the neural network model on the test dataset.

    Arguments:
    parameters -- dict, trained parameters of the neural network.
    test_minibatches -- tf.data.Dataset, test dataset batched into minibatches.

    Returns:
    accuracy -- float, accuracy of the model on the test dataset.
    """
    # The CategoricalAccuracy will track the accuracy for the multiclass problem
    test_accuracy = tf.keras.metrics.CategoricalAccuracy()

    # Iterate over each minibatch and evaluate the model's accuracy
    for (minibatch_X, minibatch_Y) in test_minibatches:
        Z_test = forward_propagation(tf.transpose(minibatch_X), parameters)
        test_accuracy.update_state(minibatch_Y, tf.transpose(Z_test))

    accuracy = test_accuracy.result()
    test_accuracy.reset_state()
    
    return accuracy


def predict_image_with_parameters(image,parameters):
    """
    Predict the class of a single image (that is to be preprocessed) using the neural network.

    Arguments:
    image -- tf.Tensor, a single image input.
    parameters -- dict, trained parameters of the neural network.

    Returns:
    prediction -- int, the output prediction of the model for the given image.
    """
    # Normalize image
    image = tf.cast(image, tf.float32) / 255.0
    
    # Reshape the image to shape [num_features, 1]
    image = tf.reshape(image, [-1, 1])  

    # Forward propagation for a single image
    Z_image_test = forward_propagation(image, parameters)

    # Convert logits to probabilities using softmax
    probabilities = tf.nn.softmax(Z_image_test, axis=0)

    # Get the predicted class label
    prediction_label = tf.argmax(probabilities, axis=0).numpy()[0]  

    return prediction_label

def model(X_train, Y_train, X_test, Y_test, layer_dims, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples)
    Y_train -- test set, of shape (output size = 6, number of training examples)
    X_test -- training set, of shape (input size = 12288, number of training examples)
    Y_test -- test set, of shape (output size = 6, number of test examples)
    layer_dims -- layer dimension of each layer in the neural network
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 10 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    costs = []
    train_acc = []
    test_acc = []
        
    parameters = initialize_parameters(layer_dims)

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    # The CategoricalAccuracy will track the accuracy for the multiclass problem
    train_accuracy = tf.keras.metrics.CategoricalAccuracy()
    
    # zip the train and test dataset values and labels together
    dataset = tf.data.Dataset.zip((X_train, Y_train))
    test_dataset = tf.data.Dataset.zip((X_test, Y_test))
    
    # We can get the number of elements of the dataset
    m = dataset.cardinality().numpy()
    print("Total number of elements in the dataset :", m )
    
    # Create minibatches from train and test datasets
    minibatches = dataset.batch(minibatch_size, drop_remainder=True).prefetch(8)
    num_batches = minibatches.cardinality().numpy()
    print("Total number of batches:", num_batches)
    
    # Minibatches created once to make the test accuracy data consistent
    test_minibatches = test_dataset.batch(minibatch_size, drop_remainder=True).prefetch(8)

    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        
        #Reset accuracy per epoch
        train_accuracy.reset_state()

        for minibatch_X, minibatch_Y in minibatches:
            with tf.GradientTape() as tape:
                ZL = forward_propagation(tf.transpose(minibatch_X), parameters)
                minibatch_total_loss = compute_total_loss(ZL, tf.transpose(minibatch_Y))

            
            # Collect all trainable variables from the parameters dictionary keeping their order.
            trainable_variables = [parameters[key] for key in parameters.keys()]

            # Calculate the gradients of the loss with respect to the trainable variables using TensorFlow's automatic differentiation.
            grads = tape.gradient(minibatch_total_loss, trainable_variables)
            
            # Apply the calculated gradients to the corresponding trainable variables using the specified optimizer.
            optimizer.apply_gradients(zip(grads, trainable_variables))
            
            # Accumulate the total loss of the epoch to later calculate the average loss per epoch.
            epoch_total_loss += minibatch_total_loss
            
            # We accumulate the accuracy of all the batches
            train_accuracy.update_state(minibatch_Y, tf.transpose(ZL))
            
  
        # Average loss over the number of samples
        epoch_total_loss /= m

        # Print the cost every 10 epochs
        if print_cost == True and epoch % 10 == 0:
            print ("Cost after epoch %i: %f" % (epoch, epoch_total_loss))
            print("Train accuracy:", train_accuracy.result())
            costs.append(epoch_total_loss)
            train_acc.append(train_accuracy.result())
            
            test_accuracy = evaluate_model(parameters, test_minibatches)
            test_acc.append(test_accuracy)


    return parameters, costs, train_acc, test_acc