
import random

import numpy as np
import tensorflow as tf

from model.neural_network_setup import compute_total_loss, forward_propagation, initialize_parameters, model
from utils.prepare_dataset import prepare_dataset

def prepare_model_hyperparameters(input_feature_size):
    """
    Prepare layer dimensions for the neural network model based on the input shape.

    Arguments:
    input_feature_size -- (int) intput features of dataset.

    Returns:
    Tuple of layer dimensions.
    """
    n_x = input_feature_size 
    layers_dims = (n_x, 25, 12, 6)  
    # layers_dims = (n_x, 24, 12, 6)  
    return layers_dims


def main():
    x_train, y_train, x_test, y_test, input_features = prepare_dataset()
    layer_dims = prepare_model_hyperparameters(input_features)
    parameters, costs, train_acc, test_acc = model(x_train, y_train, x_test, y_test, layer_dims, num_epochs=100)
    
    while False: # Replace False with a True later in development
        selected_number = random.randint(0, 5)
        user_input = input(f"Intput the number {selected_number} which is between 0 and 5 (press 'q' to quit): ")
        if user_input.lower() == 'q':
            print("Thanks for counting on me ;)")
            break
        try:
            user_guess = int(user_input)
            if user_guess == selected_number:
                print("Congratulations! You guessed correctly.")
            else:
                print(f"Wrong guess. The correct number was {selected_number}.")
        except ValueError:
            print("Please enter a valid number or 'q' to quit.")

if __name__ == "__main__":
    main()
