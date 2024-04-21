import h5py
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_sign_examples(x_train, y_train):
    """
    Plot a 5x5 grid of sample images from the dataset
    
    Arguments:
    x_train (tf.data.Dataset): Training set of sign images
    y_train (tf.data.Dataset): Training set of sign labels
    
    """
    images_iter = iter(x_train)
    labels_iter = iter(y_train)
    plt.figure(figsize=(10, 10))
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(next(images_iter).numpy().astype("uint8"))
        plt.title(next(labels_iter).numpy().astype("uint8"))
        plt.axis("off")
    plt.show()

def load_dataset():
    """
    Load sign language datasets from HDF5 files for training and testing purposes.

    Returns:
    Tuple of tf.data.Datasets: Containing training and testing datasets for both images (x) and labels (y).
    """
    train_dataset = h5py.File('data/training_set/train_signs.h5', "r")
    test_dataset = h5py.File('data/test_set/test_signs.h5', "r")
    
    x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
    y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

    x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
    y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])
    
    return x_train, y_train, x_test, y_test

def normalize(image):
    """
    Transforms an image into a tensor of shape (64 * 64 * 3, )
    and normalize its components.
    
    Arguments
    image - Tensor.
    
    Returns: 
    result -- Transformed tensor 
    """
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1,])
    return image
    
def prepare_dataset():
    """
    Prepares a small dataset for use, displays a sample image on request, and reshapes
    and normalizes image data for training and testing.

    Returns:
    tuple of numpy.ndarray: Normalized training and testing data and labels.
    """
    x_train, y_train, x_test, y_test = load_dataset()
    
    print("Element spec of training dataset:", x_train.element_spec)
    print("Sample element from training dataset:", next(iter(x_train)))
    
    
    # Finding unique labels using TensorFlow !-> batching added anticipating larger dataset
    unique_labels = y_train.batch(500).reduce(initial_state=tf.constant([], dtype=tf.int64), 
                                                reduce_func=lambda state, value: (tf.concat([state, value], axis=0)))
    unique_labels = tf.unique(unique_labels).y.numpy()
    print("Unique labels in dataset:", unique_labels)
    
    show_sample = input("Would you like to see an example of the pictures in the dataset? (y/n)")
    if show_sample.lower() == 'y':
        plot_sign_examples(x_train, y_train)
        
    normalized_train = x_train.map(normalize)
    normalized_test = x_test.map(normalize)
        
    print("Element spec of normalized training dataset:", normalized_train.element_spec)
    print("Sample element from normalized training dataset:", next(iter(normalized_train)))

    
    return  normalized_train, y_train, normalized_test, y_test