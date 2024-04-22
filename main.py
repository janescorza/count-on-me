
import os
import numpy as np
from matplotlib import pyplot as plt

from model.neural_network_setup import model, predict_image_with_parameters
from utils.prepare_dataset import prepare_dataset, preprocess_image_for_prediction

def prepare_model_hyperparameters(input_feature_size):
    """
    Prepare layer dimensions for the neural network model based on the input shape.

    Arguments:
    input_feature_size -- (int) number of input features of dataset.

    Returns:
    Tuple of layer dimensions.
    """
    learning_rate = 0.0001
    num_epochs = 175
    n_x = input_feature_size 
    layers_dims = (n_x, 25, 12, 6)  
    return layers_dims, learning_rate, num_epochs

def select_mode():
    """
    Prompts the user to select an operating mode and validates the input.

    Returns:
    int -- the selected mode as an integer (1, 2, or 3).
    """
    
    while True:
        choice = input("Select mode:\n1 - Identify the Number\n2 - Perform the Hand Sign\n3 - to quit count on me\n> ")
        if choice in ['1', '2', '3']:
            return int(choice)
        else:
            print("Invalid selection. Please choose 1 or 2 (or 3 to quit).")

def save_parameters(parameters, filepath='params/model_parameters.npy'):
    """
    Saves the model parameters to a specified file using NumPy's save function.

    Arguments:
    parameters -- dict, the model parameters to be saved.
    filepath -- str, the filepath to save the parameters to.

    Returns:
    None -- This function only saves data to file and does not return any value.
    """
    
    np.save(filepath, parameters, allow_pickle=True)
    
def load_parameters(filepath='params/model_parameters.npy'):
    """
    Loads and returns the model parameters from a specified file using NumPy's load function.

    Arguments:
    filepath -- str, the filepath from which to load the parameters.

    Returns:
    dict -- the loaded model parameters as a dictionary.
    """
    
    return np.load(filepath, allow_pickle=True).item()

def train_and_plot(x_train, y_train, x_test, y_test, input_features):
    """
    Trains a neural network model using the provided datasets and plots the training results,
    including the cost over iterations and the accuracy of the model on training and testing datasets.

    Arguments:
    x_train -- tf.data.Dataset, training dataset features.
    y_train -- tf.data.Dataset, training dataset labels.
    x_test -- tf.data.Dataset, testing dataset features.
    y_test -- tf.data.Dataset, testing dataset labels.
    input_features -- int, number of input features to define the model's input layer size.

    Returns:
    dict -- trained model parameters.
    """
    
    layer_dims, learning_rate, num_epochs = prepare_model_hyperparameters(input_features)
    parameters, costs, train_acc, test_acc = model(x_train, y_train, x_test, y_test, layer_dims, num_epochs=num_epochs)

    # Plot cost
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(np.squeeze(costs))
    plt.ylabel('Cost')
    plt.xlabel('Iterations (every 10)')
    plt.title("Learning rate = " + str(learning_rate))

    # Plot training and test accuracy
    plt.subplot(1, 2, 2)
    plt.plot(np.squeeze(train_acc), label='Train Accuracy')
    plt.plot(np.squeeze(test_acc), label='Test Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations (every 10)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.legend()
    plt.tight_layout()
    plt.show()

    return parameters

def load_or_retrain_parameters(parameters_filename, x_train, y_train, x_test, y_test, input_features):
    """
    Check if the model parameters exist and decide whether to load or retrain the model.

    Arguments:
    parameters_filename -- string, path to the model parameters file.
    x_train, y_train, x_test, y_test -- tf.data.Dataset, training and testing datasets.
    input_features -- int, number of features used by the model.

    Returns:
    parameters -- loaded or trained model parameters.
    """
    
    if os.path.exists(parameters_filename):
        user_choice = input("Model parameters found. Do you want to retrain the model? (y/n): ")
        if user_choice.lower() == 'y':
            parameters = train_and_plot(x_train, y_train, x_test, y_test, input_features)
            save_parameters(parameters, parameters_filename)
        else:
            parameters = load_parameters(parameters_filename)
    else:
        print("No parameters found, proceeding with training")
        parameters = train_and_plot(x_train, y_train, x_test, y_test, input_features)
        save_parameters(parameters, parameters_filename)
    
    return parameters

def mode_identify_number(parameters, x_test, y_test):
    """
    Conducts a loop where it asks the user to identify numbers based on hand sign images,
    compares the user's guess and neural network's prediction to the actual label, and
    provides feedback on both guesses.

    Arguments:
    parameters -- dict, model parameters used for prediction.
    x_test -- images from the test dataset.
    y_test -- true labels corresponding to the images in x_test_list.

    Returns:
    None -- This function handles user interactions and does not return a value.
    """
    
    x_test_list = list(x_test.as_numpy_iterator())
    y_test_list = list(y_test.as_numpy_iterator())
    
    while True:
        idx = np.random.randint(0, len(x_test_list))
        
        image, true_label = x_test_list[idx], y_test_list[idx]

        plt.imshow(image)
        plt.title("What number is this hand sign?")
        plt.show()

        print(f"\n")

        user_guess = int(input("\033[1;34mEnter your guess (0-5): \033[0m"))        
    
        prediction = predict_image_with_parameters(image, parameters)
        print(f"Your guess: {user_guess}")
        print(f"Neural network prediction: {prediction}")
        print("----------------------------")
        print(f"Correct answer: {true_label}")
        print("----------------------------")
        
        user_correct = user_guess == true_label
        nn_correct = prediction == true_label
        print( "Did you guess correctly? ", "\033[1;32mYes\033[0m" if user_correct else "\033[1;31mNo\033[0m")
        print("Did the neural network guess correctly? ", "\033[1;32mYes\033[0m" if nn_correct else "\033[1;31mNo\033[0m")

        continue_choice = input("\nDo you want to try another image? (y/n): ")
        if continue_choice.lower() != 'y':
            break

def mode_perform_sign(parameters):
    """
    Asks the user to perform hand signs for randomly selected numbers, predicts using the neural network,
    and provides feedback on whether the performed sign matches the expected sign.

    Arguments:
    parameters -- dict, model parameters used for prediction.

    Returns:
    None -- This function handles user interactions and does not return a value.
    """ 
    
    while True:
        target_number = np.random.randint(0, 6)
        # print(f"\033[1;34mPlease perform the hand sign for the number {target_number} and take a photo (or use those in folder hand_per_label to ensure distribution consistency).\033[0m")
        print(f"\033[1;34mPlease perform the hand sign for the number \033[1;33m{target_number}\033[0m\033[1;34m and take a photo (or use those in folder hand_per_label to ensure distribution consistency).\033[0m")

        image_path = input("Enter the path to your image: ")
        image = preprocess_image_for_prediction(image_path)
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        prediction = predict_image_with_parameters(image, parameters)

        print(f"You did the hand sign for: {prediction}")
        if prediction == target_number:
            print("\033[1;32mCongratulations! Your hand sign matches the number.\033[0m")
        else:
            print(f"\033[1;31mOops! Your hand sign did not match. You showed {prediction}.\033[0m")

        continue_choice = input("Do you want to try another number? (y/n): ")
        print(f"\n")
        if continue_choice.lower() != 'y':
            break


def main():
    
    x_train, y_train, x_test, y_test, input_features, pure_test_images, pure_test_labels = prepare_dataset()
    
    # If you needed to extract images from the test set to use on mode 2
    # extract_and_save_images(pure_test_images, pure_test_labels)

    
    parameters_filename = 'params/model_parameters.npy'
    parameters = load_or_retrain_parameters(parameters_filename, x_train, y_train, x_test, y_test, input_features)

    while True:
        mode = select_mode()
        if mode == 1:
            mode_identify_number(parameters, pure_test_images, pure_test_labels)
        elif mode == 2:
            mode_perform_sign(parameters)
        elif mode == 3:
            break
            
    print("Thanks for counting on me for learning about hand signs ;)")
    
if __name__ == "__main__":
    main()
