import matplotlib.pyplot as plt
import numpy as np
from neural_network import load_data_large, SGD 

def perform_experiment(hidden_units_list, learning_rates, epochs):
    # Load data
    X_train, y_train, X_val, y_val = load_data_large()

    # Experiment 1: Varying Hidden Units
    train_losses_hu = {}
    val_losses_hu = {}
    for hidden_units in hidden_units_list:
        _, _, train_loss, val_loss = SGD(X_train, y_train, X_val, y_val, hidden_units, epochs, True, 0.01)
        train_losses_hu[hidden_units] = np.mean(train_loss[-5:])  # Average of last 5 for stability
        val_losses_hu[hidden_units] = np.mean(val_loss[-5:])

    # Experiment 2: Varying Learning Rates
    for lr in learning_rates:
        _, _, train_loss_lr, val_loss_lr = SGD(X_train, y_train, X_val, y_val, 50, epochs, True, lr)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs+1), train_loss_lr, label='Training Cross-Entropy', marker='o')
        plt.plot(range(1, epochs+1), val_loss_lr, label='Validation Cross-Entropy', marker='x')
        plt.title(f'Learning Rate = {lr}')
        plt.xlabel('Epochs')
        plt.ylabel('Cross-Entropy Loss')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Plotting for Hidden Units Experiment
    plt.figure(figsize=(10, 6))
    plt.plot(hidden_units_list, [train_losses_hu[hu] for hu in hidden_units_list], label='Training Cross-Entropy', marker='o')
    plt.plot(hidden_units_list, [val_losses_hu[hu] for hu in hidden_units_list], label='Validation Cross-Entropy', marker='s')
    plt.xlabel('Number of Hidden Units')
    plt.ylabel('Average Cross-Entropy')
    plt.title('Effect of Hidden Units on Training and Validation Cross-Entropy')
    plt.legend()
    plt.grid(True)
    plt.show()

# Configuration
hidden_units_list = [5, 20, 50, 100, 200]
learning_rates = [0.1, 0.01, 0.001]
epochs = 50

# Perform experiments
perform_experiment(hidden_units_list, learning_rates, epochs)
