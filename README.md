# NN Fundamentals: Building a Neural Network from Scratch with NumPy

## Authors

- **Denis Rize**  - [GitHub Profile](https://github.com/denisrz)
- **Adir Serruya** - [GitHub Profile](https://github.com/Adirser)

## Project Overview

This project provides hands-on experience in building a simple neural network from scratch using NumPy. The primary goal is to deepen the understanding of the forward and backward propagation processes and to develop proficiency in their implementation.

## Key Objectives

1. **Forward Propagation**:
   - Implement essential functions for forward propagation, including:
     - `initialize_parameters(layer_dims)`
     - `linear_forward(A, W, b)`
     - `softmax(Z)`
     - `relu(Z)`
     - `linear_activation_forward(A_prev, W, B, activation)`
     - `L_model_forward(X, parameters, use_batchnorm)`
     - `compute_cost(AL, Y)`
     - `apply_batchnorm(A)`

2. **Backward Propagation**:
   - Implement functions required for backward propagation, including:
     - `linear_backward(dZ, cache)`
     - `linear_activation_backward(dA, cache, activation)`
     - `relu_backward(dA, activation_cache)`
     - `softmax_backward(dA, activation_cache)`
     - `L_model_backward(AL, Y, caches)`
     - `update_parameters(parameters, grads, learning_rate)`

3. **Training and Prediction**:
   - Use the implemented functions to train the neural network on the MNIST dataset and make predictions.
   - Functions to implement:
     - `L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size)`
     - `Predict(X, Y, parameters)`

4. **Network Configuration**:
   - The network architecture consists of 4 layers with the following sizes: 20, 7, 5, 10.
   - Input at each iteration is flattened to a matrix of `[m, 784]` where `m` is the number of samples.
   - Learning rate: `0.009`
   - Train until no improvement on the validation set for 100 steps.

## Report

This report documents the implementation of a multi-layer neural network from scratch using PyTorch. The network was designed with vectorization to ensure efficient matrix operations throughout the forward and backward propagation processes. Additionally, utility functions were developed for:
- Loading and preprocessing the MNIST dataset (`load_mnist_data()`, `preprocessing_and_data_splitting()`).
- Implementing L2 regularization (`compute_cost_with_L2_regularization()`).

## Data Preprocessing

The MNIST dataset, containing 60,000 training images and 10,000 test images of handwritten digits, was used. Data preprocessing involved:
- Normalizing pixel values to a range between 0 and 1.
- Splitting the training set into training and validation subsets with an 80-20 ratio.

## Early Stopping

During training, an early stopping mechanism was implemented to prevent overfitting. The training process was halted if the accuracy on the validation set did not improve by at least 1% over the last 100 iterations, reducing unnecessary computation.

## Weight Initialization

Different weight initialization techniques were tested to observe their impact on training stability and performance:
- **Uniform Initialization**: Weights initialized using a uniform distribution in the range [0,1].
- **Random Initialization**: Weights initialized using a normal distribution with mean 0 and standard deviation 1.
- **He Initialization**: Weights initialized using a normal distribution with mean 0 and standard deviation \(\sqrt{2/n}\), where \(n\) is the number of input units in the layer.

The best results and stability were achieved with He Initialization.

## Experiments and Results

## Experiment 1: Baseline Model

In the initial experiment, we configured the neural network (NN) as follows:
- **Layer Architecture**: Input layer with 784 nodes, followed by hidden layers with 20, 7, and 5 nodes, and an output layer with 10 nodes.
- **Learning Rate**: 0.009
- **Batch Normalization**: Disabled

We varied the batch size and epochs, but due to the early stopping mechanism, the results were consistent across different epoch settings.

### Best Results (Train and Validation)
- **Epochs**: 200
- **Batch Size**: 128
- **Weights Initialization**: He
- **Training Accuracy**: 1.0
- **Validation Accuracy**: 0.975

### Test Results
- **Accuracy on Test Set**: 0.972
- **Run Time**: 45 Seconds

![first](https://github.com/user-attachments/assets/fe823a90-5e58-4d60-98e3-71f6a3f0a09e)

## Experiment 2: With Batch Normalization

In this experiment, we repeated the previous setup with batch normalization enabled to analyze its impact on the network's performance.

### Best Results (Train and Validation)
- **Epochs**: 200
- **Batch Size**: 128
- **Weights Initialization**: He
- **Batch Norm**: Enabled
- **Training Accuracy**: 0.66
- **Validation Accuracy**: 0.64

### Test Results
- **Accuracy on Test Set**: 0.636
- **Run Time**: 92 Seconds

![second](https://github.com/user-attachments/assets/48a6ee82-2e81-4502-8734-9b8287ef06a1)

## Experiment 3: With L2 Regularization

We modified the cost function to include L2 regularization and evaluated its impact on the network's weights and performance.

### Best Results (Train and Validation)
- **Epochs**: 200
- **Batch Size**: 128
- **Weights Initialization**: He
- **Batch Norm**: Disabled
- **L2 Factor**: 0.1
- **Training Accuracy**: 1.0
- **Validation Accuracy**: 0.979

### Test Results
- **Accuracy on Test Set**: 0.974
- **Run Time**: 103 Seconds

![third](https://github.com/user-attachments/assets/3a60923f-ef78-4126-a0b2-814567f0d5d1)

## Summary of Results

| Experiment        | Epochs | Batch Size | Batch Norm | L2 Factor | Training Accuracy | Validation Accuracy | Test Accuracy | Run Time |
|-------------------|--------|------------|------------|-----------|-------------------|---------------------|---------------|----------|
| Baseline          | 200    | 128        | Disabled   | -         | 1.0               | 0.975               | 0.972         | 45 sec   |
| Batch Norm        | 200    | 128        | Enabled    | -         | 0.66              | 0.64                | 0.636         | 92 sec   |
| L2 Regularization | 200    | 128        | Disabled   | 0.1       | 1.0               | 0.979               | 0.974         | 103 sec  |

The baseline model performed well, with high accuracy on both validation (0.975) and test sets (0.972), indicating effective generalization. However, adding batch normalization negatively impacted performance, reducing test accuracy to 0.636 and increasing run time, suggesting it may have disrupted learning dynamics in this configuration. Conversely, L2 regularization improved both validation (0.979) and test accuracy (0.974) slightly, demonstrating its effectiveness in reducing overfitting and enhancing model robustness. These results suggest that while the baseline setup was strong, L2 regularization provided additional benefits, whereas batch normalization may require further tuning to be effective.

## Weights Comparison
To evaluate the impact of L2 regularization on the neural network weights, we compared the weights of the network trained with L2 regularization (λ=0.1) to those trained without any regularization. The comparison was done using histograms and heatmaps for each layer's weights, as well as difference plots to highlight the changes induced by regularization.
Parameters: Batch Size = 128, Epochs = 200, Batch Normalization = False
L2 Regularization Factors: 0 (no regularization), 0.1

### Layer 1
Weights Distribution - 

![layer1](https://github.com/user-attachments/assets/580139e7-f435-4750-be73-4184af006719)

Heatmap of the Differences -

![heat1](https://github.com/user-attachments/assets/3d1fea2d-f188-412e-8d6e-bc568c8e59b9)

### Layer 2
Weights Distribution - 

![layer2](https://github.com/user-attachments/assets/033c473a-2d88-41cc-afd0-f45880ef0d3e)

Heatmap of the Differences -

![heat2](https://github.com/user-attachments/assets/fd4238bc-300a-441c-8e43-225fbc92c44e)

### Layer 3
Weights Distribution - 

![layer3](https://github.com/user-attachments/assets/1dd5cdfa-5d89-4529-92b1-17a9cbb8340e)

Heatmap of the Differences -

![heat3](https://github.com/user-attachments/assets/bb68a9ca-63dd-4ec1-b357-59c10bbc18d9)

## Conclusion 

To conclude, the differences between the weights with and without L2 regularization are subtle,
both in terms of their distribution and the actual weight values. This minimal variation can likely
be attributed to the chosen regularization factor ( λ=0.1). Additionally, the small differences in
accuracy and runtime between the two configurations further reinforce this observation. The
chosen regularization factor effectively pena lizes large weights without significantly altering the
overall weight distribution or impacting the model's performance metrics. This suggests that
while L2 regularization has a regularizing effect, its impact in this specific setup is relatively
modest.








