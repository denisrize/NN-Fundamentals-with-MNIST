import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import os
import struct
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns

def initialize_parameters(layer_dims):
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters[f'W{l}'] = torch.randn(layer_dims[l], layer_dims[l-1]) * torch.sqrt(torch.tensor(2. / layer_dims[l-1]))  # He initialization
        parameters[f'b{l}'] = torch.zeros(layer_dims[l], 1)
    return parameters


def linear_forward(A, W, b):
    """
    Implements the linear part of a layer's forward propagation.

    Args:
        A (torch.Tensor): Activations from previous layer or input data,
                          shape [batch_size, size of previous layer]
        W (torch.Tensor): Weights, shape [size of current layer, size of previous layer]
        b (torch.Tensor): Bias, shape [size of current layer, 1]

    Returns:
        torch.Tensor: The linear component of the activation function.
        dict: A cache containing "A", "W", and "b" for computing the backward pass efficiently.
    """

    # Perform the linear operation
    Z = torch.mm(A, W.t()) + b.transpose(0, 1).expand_as(torch.mm(A, W.t()))
    # print(f"Shape of Z: {Z.shape}")  # should be [batch_size, size of current layer]

    linear_cache = {'A': A, 'W': W, 'b': b}
    return Z, linear_cache



# Softmax and ReLu Activation Function
def softmax(Z):
    # e_Z = torch.exp(Z - Z.max(dim=0, keepdim=True).values)  # for numerical stability
    e_Z = torch.exp(Z - torch.max(Z, dim=1, keepdim=True).values)
    A = e_Z / e_Z.sum(dim=1, keepdim=True)
    activation_cache = {'Z': Z}
    return A, activation_cache

def relu(Z):
    A = torch.maximum(Z, torch.tensor(0.0))
    activation_cache = {'Z': Z}
    
    return A, activation_cache

def linear_activation_forward(A_prev, W, B, activation):
    """
    Implement the forward propagation for the LINEAR -> ACTIVATION layer.

    Args:
        A_prev (torch.Tensor): Activations from the previous layer.
        W (torch.Tensor): Weights matrix of the current layer.
        B (torch.Tensor): Bias vector of the current layer.
        activation (str): The activation function to be used ("softmax" or "relu").

    Returns:
        torch.Tensor: The activations of the current layer.
        dict: A joint dictionary containing both linear_cache and activation_cache.
    """
    if activation not in ['softmax', 'relu']:
        raise ValueError("activation parameter must be 'softmax' or 'relu'")
    
    # Linear step
    Z, linear_cache = linear_forward(A_prev, W, B)
    
    # Activation step
    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'softmax':
        A, activation_cache = softmax(Z)
    
    # Combine caches
    cache = {'linear_cache': linear_cache, 'activation_cache': activation_cache}
    
    return A, cache


def compute_cost(AL, Y):
    """
    Compute the cross-entropy cost function.

    Args:
        AL (torch.Tensor): Probability matrix corresponding to label predictions, shape (num_of_classes, number of examples).
        Y (torch.Tensor): Ground truth labels tensor, shape (num_of_classes, number of examples).

    Returns:
        torch.Tensor: The cross-entropy cost.
    """
    m = Y.shape[0]  # number of examples
    cost = -torch.sum(Y * torch.log(AL + 1e-8)) / m  # adding a small value to prevent log(0)
    # print(f"Cost: {cost}")
    return cost

def compute_cost_with_L2_regularization(AL, Y, parameters, lambd):
    """
    Compute the cross-entropy cost function with L2 regularization.

    Args:
        AL (torch.Tensor): Probability matrix corresponding to label predictions, shape (num_of_classes, number of examples).
        Y (torch.Tensor): Ground truth labels tensor, shape (num_of_classes, number of examples).
        parameters (dict): DNN architecture’s parameters.
        lambd (float): Regularization parameter.

    Returns:
        torch.Tensor: The cross-entropy cost with L2 regularization.
    """
    m = Y.shape[0]  # number of examples
    cross_entropy_cost = -torch.sum(Y * torch.log(AL + 1e-8)) / m  # adding a small value to prevent log(0)
    L2_regularization_cost = 0

    L = len(parameters) // 2  # Number of layers in the neural network
    for l in range(1, L + 1):
        W = parameters[f'W{l}']
        L2_regularization_cost += torch.sum(W**2)

    L2_regularization_cost = (lambd / (2 * m)) * L2_regularization_cost
    cost = cross_entropy_cost + L2_regularization_cost

    return cost

def apply_batchnorm(A):
    """
    Perform batch normalization on the activation values of a layer.

    Args:
        A (torch.Tensor): The activation values of a given layer.

    Returns:
        torch.Tensor: The normalized activation values.
    """
    # Mean and standard deviation of the activations
    mean = A.mean(dim=0, keepdim=True)
    std = A.std(dim=0, keepdim=True, unbiased=False)

    # Batch normalization formula
    NA = (A - mean) / (std + 1e-8)  # Adding epsilon for numerical stability

    return NA

# Forward Propagation
def L_model_forward(X, parameters, use_batchnorm=False):
    caches = []
    A = X  # Now A should already be [batch_size, input_size] from the input
    L = len(parameters) // 2  # Number of layers in the neural network

    for l in range(1, L):
        A_prev = A
        W = parameters[f'W{l}']
        b = parameters[f'b{l}']  # Ensure b is broadcastable
        A, cache = linear_activation_forward(A_prev, W, b, 'relu')
        if use_batchnorm:
            A = apply_batchnorm(A)
        caches.append(cache)

    W = parameters[f'W{L}']
    b = parameters[f'b{L}'] # Ensure b is broadcastable
    AL, cache = linear_activation_forward(A, W, b, 'softmax')

    caches.append(cache)

    return AL, caches

def Linear_backward(dZ, cache):
    A_prev, W, b = cache['A'], cache['W'], cache['b']
    m = A_prev.shape[0]  # Correcting the use of batch size

    dW = torch.mm(dZ.t(), A_prev) / m
    db = torch.sum(dZ, dim=0,keepdim=True).T / m
    dA_prev = torch.mm(dZ, W)

    return dA_prev, dW, db

def relu_backward(dA, activation_cache):
    """
    Implements backward propagation for a ReLU unit.

    Args:
        dA (torch.Tensor): The post-activation gradient.
        activation_cache (dict): Contains Z, stored during the forward propagation.

    Returns:
        torch.Tensor: Gradient of the cost with respect to Z.
    """
    Z = activation_cache['Z']
    dZ = dA.clone().detach()  
    dZ[Z <= 0] = 0
    return dZ

def softmax_backward(dA, activation_cache):
    """
    Implements backward propagation for a softmax unit.

    Args:
        dA (torch.Tensor): The post-activation gradient.
        activation_cache (dict): Contains Z, stored during the forward propagation.

    Returns:
        torch.Tensor: Gradient of the cost with respect to Z.
    """
    Z = activation_cache['Z']
    P = torch.softmax(Z, dim=1)
    dZ = P - dA
    return dZ

def linear_activation_backward(dA, cache, activation):
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer.

    Args:
        dA (torch.Tensor): Post-activation gradient for the current layer.
        cache (dict): Contains both the linear cache and activation cache.
        activation (str): The activation to be used ("relu" or "softmax").

    Returns:
        dA_prev (torch.Tensor): Gradient of the cost with respect to the activation of the previous layer.
        dW (torch.Tensor): Gradient of the cost with respect to W of the current layer.
        db (torch.Tensor): Gradient of the cost with respect to b of the current layer.
    """
    linear_cache, activation_cache = cache['linear_cache'], cache['activation_cache']

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
    else:
        raise ValueError("Unsupported activation function")

    dA_prev, dW, db = Linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation process for the entire network.

    Args:
        AL (torch.Tensor): Probabilities vector, output of the forward propagation.
        Y (torch.Tensor): True labels vector.
        caches (list): List of caches containing for each layer: the linear cache and the activation cache.

    Returns:
        dict: Dictionary with the gradients.
    """
    grads = {}
    L = len(caches)  # the number of layers
    m = AL.shape[0] # CHANGED TO 0 INSTEAD OF 1
    Y = Y.reshape(AL.shape)  # Ensure Y is the same shape as AL
    AL = torch.clamp(AL, min=0.0001, max=0.9999)  # Avoid division by zero

    # Initializing the backpropagation
    # dAL = (-1.0/m) * (Y / (AL)) + ((1 - Y) / (1 - AL))  # derivative of cost with respect to AL   
    dAL = AL - Y 
    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(Y, current_cache, "softmax")
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def Update_parameters(parameters, grads, learning_rate):
    """
    Updates parameters using gradient descent, with gradient clipping.
    """
    L = len(parameters) // 2  # Number of layers in the neural network

    for l in range(1, L + 1):
        parameters[f'W{l}'] -= learning_rate * grads[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * grads[f'db{l}']

    return parameters

def plot_histogram(weights, title):
    """Plot histogram of weights."""
    plt.figure()
    plt.hist(weights.flatten(), bins=50, alpha=0.7)
    plt.title(title)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def plot_heatmap(weights, title):
    """Plot heatmap of weights."""
    plt.figure()
    sns.heatmap(weights, cmap='viridis')
    plt.title(title)
    plt.show()

def plot_difference(weights_no_l2, weights_with_l2, title):
    """Plot difference of weights."""
    difference = weights_with_l2 - weights_no_l2
    plt.figure()
    sns.heatmap(difference, cmap='coolwarm', center=0)
    plt.title(title)
    plt.show()

def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size,batch_norm = None,L2_regularization = None):

    parameters = initialize_parameters(layers_dims)
    costs,costs_val,accuracy_trains,accuracy_validations = [],[],[],[]
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


 

    for i in tqdm(range(1,num_iterations), desc='Training Progress'):
        # Random shuffling and batch generation
        epoch_cost= 0
        permutation = torch.randperm(X_train.size(0))
        X_shuffled = X_train[permutation, :]
        Y_shuffled = Y_train[permutation, :]

        epoch_cost_val = 0
        permutation_validation = torch.randperm(X_val.size(0))
        X_val_shuffled = X_val[permutation_validation, :]
        Y_val_shuffled = Y_val[permutation_validation, :]

        for j in range(0, X_train.size(0), batch_size):
            end = min(j + batch_size, X_train.size(0))
            X_batch = X_shuffled[j:end, :]
            Y_batch = Y_shuffled[j:end, :]

            AL, caches = L_model_forward(X_batch, parameters,use_batchnorm=batch_norm)

            if L2_regularization > 0:
                cost = compute_cost_with_L2_regularization(AL, Y_batch, parameters, L2_regularization)
            else:
                cost = compute_cost(AL, Y_batch)

            grads = L_model_backward(AL, Y_batch, caches)        
            parameters = Update_parameters(parameters, grads, learning_rate)
            epoch_cost += cost

        for v in range(0, X_val.size(0), batch_size):
            end = min(v + batch_size, X_val.size(0))
            X_batch = X_val_shuffled[v:end, :]
            Y_batch = Y_val_shuffled[v:end, :]
            AL, caches = L_model_forward(X_batch, parameters,use_batchnorm=batch_norm)
            if L2_regularization > 0:
                cost = compute_cost_with_L2_regularization(AL, Y_batch, parameters, L2_regularization)
            else:
                cost = compute_cost(AL, Y_batch)
            epoch_cost_val += cost

        # Saving loss and accuracy for each epoch
        epoch_cost /= (X_train.size(0) // batch_size)
        costs.append(epoch_cost)

        epoch_cost_val /= (X_val.size(0) // batch_size)
        costs_val.append(epoch_cost_val)

        accuracy_train,predictions_train,labels_train = Predict(X_batch, Y_batch, parameters)
        accuracy_trains.append(accuracy_train)

        accuracy_validation,predictions_validation,labels_validation = Predict(X_val, Y_val, parameters)
        accuracy_validations.append(accuracy_validation)


        # Early stopping if the validation accuracy havn't improved for 100 iterations by more than epsilon
        if i > 100 and accuracy_validations[-1] - accuracy_validations[-100] < 0.01:
            print(f"Early stopping at iteration {i}")
            break

    return parameters, costs ,costs_val,accuracy_trains,accuracy_validations



def Predict(X, Y, parameters):
    """
    Calculate the accuracy of the trained neural network on the data.

    Args:
        X (torch.Tensor): Input data, shape (height*width, number_of_examples).
        Y (torch.Tensor): True labels, shape (num_of_classes, number_of_examples).
        parameters (dict): DNN architecture’s parameters.

    Returns:
        float: Accuracy measure of the neural network.
    """
    X_tensor = X.clone().detach()
    Y_tensor = Y.clone().detach()

    AL, _ = L_model_forward(X_tensor, parameters, use_batchnorm=False)
    predictions = AL.argmax(dim=1) # Changed to 1
    labels = Y_tensor.argmax(dim=1) # changed to 1

    accuracy = (predictions == labels).float().mean().item()
    return accuracy, predictions, labels

## Assignment 4 
def read_idx(filename):
    """ Read an IDX file into a numpy array. """
    with open(filename, 'rb') as f:
        # Read the magic number and dimensions
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def load_mnist_data() -> tuple:
    # Update file_names to the correct paths
    file_names = [
    r'./data/MNIST/raw/train-images.idx3-ubyte',
    r'./data/MNIST/raw/train-labels.idx1-ubyte',
    r'./data/MNIST/raw/t10k-images.idx3-ubyte',
    r'./data/MNIST/raw/t10k-labels.idx1-ubyte'
    ]
    # Attempt to load the data again with the corrected function
    try:
        train_images = read_idx(file_names[0])
        train_labels = read_idx(file_names[1])
        test_images = read_idx(file_names[2])
        test_labels = read_idx(file_names[3])
        print("Data loaded successfully:")
        print("Training images shape:", train_images.shape)
        print("Training labels shape:", train_labels.shape)
        print("Test images shape:", test_images.shape)
        print("Test labels shape:", test_labels.shape)
    except Exception as e:
        print("Failed to load data:", e)
        return None
    return train_images, train_labels, test_images, test_labels

def preprocessing_and_data_splitting(train_images, train_labels, test_images, test_labels):
    print("Preprocessing and splitting the data...")
    # Normalize the image data to [0, 1]
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # Flatten the images
    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)

    # Convert labels to one-hot encoding
    def one_hot_encode(labels, num_classes=10):
        return np.eye(num_classes)[labels]

    train_labels = one_hot_encode(train_labels)
    test_labels = one_hot_encode(test_labels)

    # Split training data into training and validation sets
    X_train,Y_train = train_images, train_labels
    X_test,Y_test = test_images,test_labels

    X_train = X_train.reshape(-1, 28*28)  # Flatten and transpose
    X_test = X_test.reshape(-1, 28*28)   # Flatten and transpose

    print("Data preprocessed and split successfully.")
    # turn the numpy arrays into torch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.float32)

    return X_train,X_test,Y_train,Y_test


if __name__ == "__main__":
    # Load the MNIST data
    data = load_mnist_data()
    if data is not None:
        train_images, train_labels, test_images, test_labels = data
        X_train,X_test,Y_train,Y_test = preprocessing_and_data_splitting(train_images, train_labels, test_images, test_labels)
        print(f"X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
        print(f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")

        # Set the hyperparameters
        learning_rate = 0.009
            # Define the architecture of the neural network
        layers_dims = [784, 128, 64, 10]  # 3-layer neural network



        # Grid Search
        num_iterations = [200]
        batch_size = [128]
        batch_norm = [False]
        L2_reg = [0,0.1]

        best_accuracy = 0
        best_params = {}
        results = {}
        for num_iter in num_iterations:
            for batch in batch_size:
                for bn in batch_norm:
                    for l2 in L2_reg:
                        print(f"Training model with num_iter: {num_iter}, batch: {batch}, batch_norm: {bn}, L2_reg: {l2}")
                        parameters,costs,costs_val,accuracy_trains,accuracy_validations = L_layer_model(X_train, Y_train, layers_dims, learning_rate,
                                                                                            num_iter, batch,batch_norm = bn,L2_regularization = l2)
                        
                        accuray_train,_,_ = Predict(X_train, Y_train, parameters)
                        accuracy_test,_,_ = Predict(X_test, Y_test, parameters)

                        if accuracy_test > best_accuracy:
                            best_accuracy = accuracy_test
                            best_params = {'num_iter':num_iter,'batch_size':batch,'batch_norm':bn,'L2_reg':l2}

                        # Store all the results and create plots for training and validation accuracy,loss
                        results[f'num_iter_{num_iter}_batch_{batch}_batch_norm_{bn}_L2_reg_{l2}'] = {'accuracy':accuracy_test,
                                                                                                    'parameters':parameters,
                                                                                                    'costs':costs}
                        
                        #Plots costs for training and val
                        plt.figure(figsize=(10, 6))
                        plt.plot(costs, label='Training Loss')
                        plt.plot(costs_val, label='Validation Loss')
                        plt.xlabel('Iterations')
                        plt.ylabel('Loss')
                        plt.title(f'Loss vs. Iterations (num_iter={num_iter}, batch_size={batch}, bn={bn}, L2={l2 if l2 > 0 else " = None"})')
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(f'Loss_vs_iterations_num_iter_{num_iter}_batch_{batch}_batch_norm_{bn}_L2_reg_{l2 if l2 > 0 else "None"}.png')
                        plt.close()

                        plt.figure(figsize=(10, 6))
                        plt.plot(accuracy_trains, label='Training Accuracy')
                        plt.plot(accuracy_validations, label='Validation Accuracy')
                        plt.xlabel('Iterations')
                        plt.ylabel('Accuracy')
                        plt.title(f'Accuracy vs. Iterations (num_iter={num_iter}, batch_size={batch}, bn={bn}, L2={l2 if l2 > 0 else " = None"})')
                        plt.legend()
                        plt.grid(True)
                        plt.savefig(f'accuracy_vs_iterations_num_iter_{num_iter}_batch_{batch}_batch_norm_{bn}_L2_reg_{l2 if l2 > 0 else "None"}.png')
                        plt.close()
                        
                        print(f"Current params: num_iter: {num_iter}, batch: {batch}, batch_norm: {bn}, L2_reg: {l2}")
                        print(f"Accuracy on test set: {accuracy_test}")
                        
                        

        print(f"Best parameters: {best_params}")
        print(f"Best accuracy: {best_accuracy}")
        print("Results : ",results)

    # Example of how to plot histograms, heatmaps, and differences of weights
    # weights_with_regu = results['num_iter_200_batch_128_batch_norm_False_L2_reg_0.1']
    # weights_no_regu = results['num_iter_200_batch_128_batch_norm_False_L2_reg_0']

    # w1 = weights_with_regu['parameters']['W1']
    # plot_histogram(w1, 'Weights of Layer 1 with L2 Regularization')
    # plot_heatmap(w1, 'Weights of Layer 1 with L2 Regularization')
    # w2 = weights_with_regu['parameters']['W2']
    # plot_histogram(w2, 'Weights of Layer 2 with L2 Regularization')
    # plot_heatmap(w2, 'Weights of Layer 2 with L2 Regularization')
    # w3 = weights_with_regu['parameters']['W3']
    # plot_histogram(w3, 'Weights of Layer 3 with L2 Regularization')
    # plot_heatmap(w3, 'Weights of Layer 3 with L2 Regularization')

    # w1 = weights_no_regu['parameters']['W1']
    # plot_histogram(w1, 'Weights of Layer 1 without L2 Regularization')
    # plot_heatmap(w1, 'Weights of Layer 1 without L2 Regularization')
    # w2 = weights_no_regu['parameters']['W2']
    # plot_histogram(w2, 'Weights of Layer 2 without L2 Regularization')
    # plot_heatmap(w2, 'Weights of Layer 2 without L2 Regularization')
    # w3 = weights_no_regu['parameters']['W3']
    # plot_histogram(w3, 'Weights of Layer 3 without L2 Regularization')
    # plot_heatmap(w3, 'Weights of Layer 3 without L2 Regularization')

    # plot_difference(weights_no_regu['parameters']['W1'], weights_with_regu['parameters']['W1'], 'Difference of Weights of Layer 1')
    # plot_difference(weights_no_regu['parameters']['W2'], weights_with_regu['parameters']['W2'], 'Difference of Weights of Layer 2')
    # plot_difference(weights_no_regu['parameters']['W3'], weights_with_regu['parameters']['W3'], 'Difference of Weights of Layer 3')
