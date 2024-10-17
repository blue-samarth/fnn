# Detailed Documentation: Fully Connected Layer (FCL) Class

## Table of Contents
1. [Class Overview](#class-overview)
2. [Imports and Dependencies](#imports-and-dependencies)
3. [Class Definition and Inheritance](#class-definition-and-inheritance)
4. [Class Attributes](#class-attributes)
5. [Constructor (`__init__` method)](#constructor-init-method)
6. [Activation Function Selection (`get_activation` method)](#activation-function-selection-get_activation-method)
7. [Weight Initialization (`initialize_weights` method)](#weight-initialization-initialize_weights-method)
8. [Activation Gradient Computation (`compute_activation_gradient` method)](#activation-gradient-computation-compute_activation_gradient-method)
9. [Forward Pass (`forward` method)](#forward-pass-forward-method)
10. [Backward Pass (`backward` method)](#backward-pass-backward-method)
11. [Adam Optimization Implementation](#adam-optimization-implementation)
12. [Usage in Neural Networks](#usage-in-neural-networks)
13. [Potential Improvements and Considerations](#potential-improvements-and-considerations)

## 1. Class Overview

The `FCL` (Fully Connected Layer) class is a custom implementation of a dense neural network layer. It is designed to be a flexible and powerful building block for creating multi-layer perceptrons (MLPs) and other types of neural networks. The class incorporates forward and backward propagation, various activation functions, and the Adam optimization algorithm for efficient training.

## 2. Imports and Dependencies

```python
import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
```

- `torch.nn`: Provides neural network modules and functions.
- `torch`: The main PyTorch library for tensor computations and automatic differentiation.
- `numpy`: Used for numerical operations (though its usage is limited in this class).
- `matplotlib.pyplot`: Imported but not used directly in the FCL class (likely used elsewhere in the codebase for plotting).

## 3. Class Definition and Inheritance

```python
class FCL(nn.Module):
```

The `FCL` class inherits from `nn.Module`, which is the base class for all neural network modules in PyTorch. This inheritance allows the `FCL` class to integrate seamlessly with PyTorch's autograd system and other neural network components.

## 4. Class Attributes

The `FCL` class has several attributes, both explicitly defined and implicitly created:

- `input_size` (int): Number of input features.
- `output_size` (int): Number of output features.
- `activation` (nn.Module): The activation function used in the layer.
- `weights` (nn.Parameter): The weights of the layer.
- `bias` (nn.Parameter): The bias of the layer.
- `m_weights`, `m_bias` (nn.Parameter): First moment estimates for weights and bias (used in Adam optimization).
- `v_weights`, `v_bias` (nn.Parameter): Second moment estimates for weights and bias (used in Adam optimization).
- `beta1` (float): Exponential decay rate for the first moment estimates (default: 0.9).
- `beta2` (float): Exponential decay rate for the second moment estimates (default: 0.999).
- `epsilon` (float): Small constant for numerical stability in Adam optimization (default: 1e-8).

## 5. Constructor (`__init__` method)

```python
def __init__(self, input_size: int, output_size: int, activation: str) -> None:
```

The constructor method initializes the layer with the following steps:

1. Calls the superclass constructor (`super(FCL, self).__init__()`).
2. Sets the `input_size` and `output_size` attributes.
3. Sets the activation function using the `get_activation` method.
4. Initializes weights using the `initialize_weights` method.
5. Initializes bias as a zero tensor.
6. Sets up parameters for Adam optimization (m_weights, m_bias, v_weights, v_bias).
7. Defines hyperparameters for the Adam optimizer (beta1, beta2, epsilon).

## 6. Activation Function Selection (`get_activation` method)

```python
def get_activation(self, activation: str) -> nn.Module:
```

This method selects and returns the appropriate activation function based on the input string:

- 'relu': Returns `nn.ReLU()`
- 'sigmoid': Returns `nn.Sigmoid()`
- 'tanh': Returns `nn.Tanh()`
- 'softmax': Returns `nn.Softmax(dim=-1)`
- If None is provided, it defaults to ReLU.
- Raises a `ValueError` for unsupported activation functions.

## 7. Weight Initialization (`initialize_weights` method)

```python
def initialize_weights(self) -> nn.Parameter:
```

This method initializes the weights of the layer based on the activation function:

- For Sigmoid and Tanh: Uses Xavier (Glorot) initialization (`nn.init.xavier_normal_`).
- For ReLU: Uses He (Kaiming) initialization (`nn.init.kaiming_normal_`).
- For other activations: Uses a custom initialization based on the input size.

The method returns the initialized weights as an `nn.Parameter`.

## 8. Activation Gradient Computation (`compute_activation_gradient` method)

```python
def compute_activation_gradient(self, d_values: torch.Tensor) -> torch.Tensor:
```

This method computes the gradient of the loss with respect to the output of the layer, considering the specific activation function used:

- For ReLU: `grad = d_values * (self.output > 0).float()`
- For Sigmoid: `grad = d_values * self.output * (1 - self.output)`
- For Tanh: `grad = d_values * (1 - self.output ** 2)`
- For Softmax: `grad = d_values` (assuming cross-entropy loss is used)

## 9. Forward Pass (`forward` method)

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
```

The forward method performs the forward propagation through the layer:

1. Stores the input `x` for use in backward propagation.
2. Computes the linear transformation: `z = torch.matmul(x, self.m_weights) + self.m_bias`
3. Applies the activation function: `self.output = self.activation(z)`
4. Returns the output.

## 10. Backward Pass (`backward` method)

```python
def backward(self, d_values: torch.Tensor, learning_rate: float, t: int) -> torch.Tensor:
```

The backward method performs backpropagation and updates the weights using Adam optimization:

1. Computes the gradient of the activation function.
2. Calculates gradients for weights and biases.
3. Updates weights and biases using the Adam optimization algorithm.
4. Implements gradient clipping to prevent exploding gradients.
5. Computes and returns the gradient with respect to the input for the next layer.

## 11. Adam Optimization Implementation

The Adam (Adaptive Moment Estimation) optimization algorithm is implemented within the `backward` method:

1. Updates the first moment estimates (m_weights, m_bias).
2. Updates the second moment estimates (v_weights, v_bias).
3. Applies bias correction to the moment estimates.
4. Computes the adaptive learning rates.
5. Updates the weights and biases using the computed adaptive learning rates.

## 12. Usage in Neural Networks

The `FCL` class can be used as a building block in creating multi-layer neural networks. Multiple instances of this class can be stacked to create deeper networks, with each layer potentially having different sizes and activation functions.

Example usage:
```python
layer1 = FCL(input_size=784, output_size=128, activation='relu')
layer2 = FCL(input_size=128, output_size=64, activation='relu')
layer3 = FCL(input_size=64, output_size=10, activation='softmax')
```

## 13. Potential Improvements and Considerations

1. **Code Organization**: Consider moving the `get_activation` and `initialize_weights` methods outside the `__init__` method for better code organization.

2. **Flexibility**: Add support for more activation functions and initialization methods.

3. **Optimization**: The Adam update logic could be extracted into a separate method to reduce code duplication between weight and bias updates.

4. **Performance**: Consider using PyTorch's built-in optimization functions for potentially better performance.

5. **Error Handling**: Add more robust error checking and handling, especially for input validation.

6. **Documentation**: Add docstrings to each method for better in-code documentation.

7. **Type Hinting**: Expand the use of type hints for better code readability and potential error catching.

8. **Customization**: Allow for more customization of the Adam optimizer parameters (learning rate, beta1, beta2, epsilon) through the constructor.

9. **Testing**: Implement unit tests to ensure the correct functioning of each component of the class.







# Detailed Documentation for `CreateModel` class

## Overview
The `CreateModel` class implements a fully connected feedforward neural network using PyTorch's `nn.Module`. This class is designed to be flexible and allows the user to specify the number of hidden layers and their respective sizes. It can be used for various supervised learning tasks such as classification, with `softmax` as the output activation function. The model is trained using a custom implementation of the Adam optimizer, and the training results (loss and accuracy) can optionally be visualized.

---

## Class Constructor: `__init__(self, input_size: int, hidden_size: list, output_size: int)`

### Description:
The constructor of the `CreateModel` class initializes the neural network. It sets up three fully connected layers (`fc1`, `fc2`, and `fc3`), where each layer is represented by the custom `FCL` (Fully Connected Layer) class, which allows for different activation functions. The network is constructed based on the provided input, hidden, and output sizes.

### Arguments:
- **`input_size (int)`**: Specifies the number of features in the input data (e.g., for image data, this could be the total number of pixels).
- **`hidden_size (list)`**: A list of integers where each element defines the number of neurons in a hidden layer. For example, `[128, 64]` would create a model with two hidden layers: the first with 128 neurons and the second with 64 neurons.
- **`output_size (int)`**: The number of output features, typically corresponding to the number of classes in a classification problem (e.g., 10 for MNIST digit classification).

### Internal Layers:
- **`self.fc1`**: The first fully connected layer, which transforms the input into the first hidden layer. It uses the ReLU activation function.
- **`self.fc2`**: The second fully connected layer, transforming data from the first hidden layer to the second hidden layer, also with ReLU activation.
- **`self.fc3`**: The final fully connected layer that outputs probabilities for each class. It uses the softmax activation function for multi-class classification.

### Example Initialization:
```python
# Example: Create a model for MNIST digit classification
model = CreateModel(input_size=784, hidden_size=[128, 64], output_size=10)
```

---

## Forward Pass: `forward(self, x: torch.Tensor) -> torch.Tensor`

### Description:
The `forward` method defines how the input tensor `x` passes through the network. It sequentially applies each of the fully connected layers (`fc1`, `fc2`, and `fc3`) and their respective activation functions.

### Arguments:
- **`x (torch.Tensor)`**: A tensor representing the input data, which should have a shape corresponding to the `input_size` defined during initialization.

### Returns:
- **`torch.Tensor`**: The output tensor after applying all transformations through the network, representing class probabilities in classification tasks.

### Example Usage:
```python
# Forward pass on a batch of input data
output = model.forward(input_tensor)
```

---

## Training Method: `train(self, x: torch.Tensor, y: torch.Tensor, learning_rate: float, n_epochs: int, decay: float, plot_training_result=False)`

### Description:
The `train` method implements the training loop for the model. It utilizes the Adam optimizer to update the model weights based on the computed gradients. The loss function used is categorical cross-entropy, which is appropriate for multi-class classification tasks. This method also supports an optional learning rate decay, where the learning rate decreases over time, and the ability to plot the loss and accuracy metrics during training.

### Arguments:
- **`x (torch.Tensor)`**: The input training data, represented as a tensor of shape `[batch_size, input_size]`. Each row is an input sample.
- **`y (torch.Tensor)`**: The target labels, represented as a tensor of one-hot encoded labels (shape: `[batch_size, output_size]`).
- **`learning_rate (float)`**: The initial learning rate for the Adam optimizer. It controls how large the updates to the model weights are in each training step.
- **`n_epochs (int)`**: The number of epochs (complete passes through the entire training dataset) for which the model will be trained.
- **`decay (float)`**: A factor used to reduce the learning rate after each epoch. This helps the optimizer take smaller steps as training progresses, preventing overshooting during optimization.
- **`plot_training_result (bool)`**: If `True`, the method will visualize the training process by plotting the loss and accuracy for each epoch.

### Training Process:
1. **Forward Pass**: The input data `x` is passed through the network to generate predictions.
2. **Loss Calculation**: Categorical cross-entropy is computed between the predicted class probabilities and the true labels `y`. A small value `epsilon` is added to prevent taking the log of zero.
3. **Accuracy Calculation**: The predicted class labels are compared with the true class labels, and the accuracy is computed for each epoch.
4. **Backpropagation**: The gradients of the loss with respect to the model parameters are calculated using the chain rule. The weights are updated using the Adam optimization algorithm.
5. **Learning Rate Decay**: The learning rate is decayed by multiplying it with a decay factor after each epoch.

### Example Usage:
```python
# Train the model on the dataset
model.train(x_train, y_train, learning_rate=0.001, n_epochs=100, decay=1e-4, plot_training_result=True)
```

### Notes on Adam Optimizer:
Adam is a variant of stochastic gradient descent (SGD) that uses adaptive learning rates for each parameter. It combines two key ideas:
- **Momentum**: Smooths out the gradients by keeping a running average of past gradients.
- **RMSprop**: Scales the learning rate based on the variance of recent gradients.

---

## Private Method: `_plot_training_results(self, n_epochs: int, loss_log: list, accuracy_log: list)`

### Description:
This private method is used internally to visualize the training results. It generates plots for both the loss and accuracy metrics over the training epochs. This is particularly useful for debugging and ensuring that the model is converging during training.

### Arguments:
- **`n_epochs (int)`**: The total number of epochs the model was trained for.
- **`loss_log (list)`**: A list of loss values, where each element corresponds to the loss at a particular epoch.
- **`accuracy_log (list)`**: A list of accuracy values, where each element corresponds to the accuracy at a particular epoch.

### Example:
```python
# Automatically called during training if plot_training_result=True
model._plot_training_results(n_epochs=100, loss_log=[...], accuracy_log=[...])
```

---

## Example Workflow

1. **Define the Model**:
    ```python
    model = CreateModel(input_size=784, hidden_size=[128, 64], output_size=10)
    ```

2. **Prepare Data**:
    Load your dataset and convert it to tensors suitable for training:
    ```python
    x_train = torch.tensor(...)  # Shape: [num_samples, input_size]
    y_train = torch.tensor(...)  # Shape: [num_samples, output_size] (one-hot encoded)
    ```

3. **Train the Model**:
    Call the `train` method to train the model on the dataset:
    ```python
    model.train(x_train, y_train, learning_rate=0.001, n_epochs=100, decay=1e-4, plot_training_result=True)
    ```

4. **Evaluate the Model**:
    After training, use the `forward` method to make predictions on new data:
    ```python
    test_output = model.forward(x_test)
    ```

### Important Notes:
- The input and output tensors should be appropriately shaped (with the correct number of features and classes).
- The training method manually implements the backward pass and gradient calculations, so make sure the `FCL` class supports gradient propagation.

