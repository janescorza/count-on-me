import numpy as np
import tensorflow as tf

def sigmoid(z):
    
    """
    Computes the sigmoid of z using keras
    
    Arguments:
    z -- input value, scalar or vector
    
    Returns: 
    a -- (tf.float32) the sigmoid of z
    """
    z = tf.cast(z, tf.float32)
    a = tf.keras.activations.sigmoid(z)
    return a

def initialize_parameters(layer_dims):
    """
    Initializes parameters to build a neural network with TensorFlow's GlorotNormal initializer.
    (which draws samples from a truncated normal distribution centered on 0).
    
    Arguments:
    layer_dims -- list containing the dimensions of each layer in the network (including the input layer)
    
    Returns:
    parameters -- dictionary containing tensors "W1", "b1", ..., "WL", "bL" for L-1 layers (L includes input layer)
    """
    initializer = tf.keras.initializers.GlorotNormal(seed=1)
    parameters = {}
    L = len(layer_dims)  # Number of layers including input

    for l in range(1, L):
        parameters['W' + str(l)] = tf.Variable(initializer(shape=(layer_dims[l], layer_dims[l-1])))
        parameters['b' + str(l)] = tf.Variable(initializer(shape=(layer_dims[l], 1)))

    return parameters
