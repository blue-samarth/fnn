# here I will define the fnn model MLP 
# MLP is multi-layer perceptron 

import torch.nn as nn
import torch
import numpy as np

class FCL(nn.Module):

    """
        Fully Connected Layer (FCL) with customizable activation function and Adam optimization.

        This layer implements a fully connected (dense) neural network layer with
        forward and backward propagation. It includes Adam optimization for
        weight updates during training.

        Attributes:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            activation (nn.Module): Activation function to use.
            m_weights (nn.Parameter): Weights of the layer.
            m_bias (nn.Parameter): Bias of the layer.
            v_weights (nn.Parameter): Second moment estimate for weights (used in Adam).
            v_bias (nn.Parameter): Second moment estimate for bias (used in Adam).
            beta1 (float): Exponential decay rate for first moment estimate in Adam.
            beta2 (float): Exponential decay rate for second moment estimate in Adam.
            epsilon (float): Small constant for numerical stability in Adam.
    """

    def __init__(self, input_size : int, output_size : int , activation : str) -> None:

        """
        Initialize the Fully Connected Layer.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            activation (str): Name of the activation function to use.
        """

        super(FCL, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.activation = self.get_activation(activation)
        self.weights = nn.Parameter(self.initialize_weights())
        self.bias = nn.Parameter(torch.zeros(output_size))  # Using PyTorch's tensor for bias

        # Defining mean and variance for weights and biases
        # These are used in Adam optimizer
        self.m_weights = nn.Parameter(torch.tensor(self.weights, dtype=torch.float32), requires_grad=True)
        self.m_bias = nn.Parameter(torch.tensor(self.bias, dtype=torch.float32), requires_grad=True)
        self.v_weights = nn.Parameter(torch.zeros_like(self.m_weights), requires_grad=False)
        self.v_bias = nn.Parameter(torch.zeros_like(self.m_bias), requires_grad=False) 
        
        # Define Hyperparameters for Adam optimizer
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8





           
    def get_activation(self, activation: str) -> nn.Module:

        """
            Get the activation function based on the provided string.

            Args:
                activation (str): Name of the activation function.

            Returns:
                nn.Module: The corresponding PyTorch activation function.
            
            Raises:
                ValueError: If the activation function is not supported
        """

        if activation is None:
            return nn.ReLU()
        activation = activation.lower()
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'softmax':
            return nn.Softmax(dim=-1)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def initialize_weights(self) -> nn.Parameter:
        """
        Initialize the weights of the layer based on the activation function.


        """
        if isinstance(self.activation, (nn.Sigmoid, nn.Tanh)):
            weights = nn.init.xavier_normal_(torch.empty(self.input_size, self.output_size))
        elif isinstance(self.activation, nn.ReLU):
            weights = nn.init.kaiming_normal_(torch.empty(self.input_size, self.output_size), nonlinearity='relu')
        else:
            weights = torch.randn(self.input_size, self.output_size) * np.sqrt(2.0 / self.input_size)
        return nn.Parameter(weights)

    def compute_activation_gradient(self, d_values: torch.Tensor) -> torch.Tensor:
            
            """
            Compute the gradient of the loss w.r.t. the output of the layer.
    
            Args:
                d_values (torch.Tensor): Gradient of the loss w.r.t. the output of the layer.
    
            Returns:
                torch.Tensor: Gradient of the loss w.r.t. the output of the layer.
            """
    
            if isinstance(self.activation, nn.ReLU):
                grad = d_values * (self.output > 0).float()
            elif isinstance(self.activation, nn.Sigmoid):
                grad = d_values * self.output * (1 - self.output)
            elif isinstance(self.activation, nn.Tanh):
                grad = d_values * (1 - self.output ** 2)
            elif isinstance(self.activation, nn.Softmax):
                grad = d_values
            else:
                raise ValueError(f"Unsupported activation function: {self.activation}")

            return grad


    def forward(self , x : torch.Tensor) -> torch.Tensor:

        """
        Perform the forward pass of the layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the layer transformation.
        """

        self.x = x
        z = torch.matmul(x, self.m_weights) + self.m_bias
        self.output = self.activation(z)

        return self.output
    


    def backward(self, d_values: torch.Tensor, learning_rate: float, t: int) -> torch.Tensor:

        """
        Perform the backward pass of the layer and update weights using Adam optimization.

        Args:
            d_values (torch.Tensor): Gradient of the loss w.r.t. the output of the layer.
            learning_rate (float): Learning rate for optimization.
            t (int): Current time step (used for bias correction in Adam).

        Returns:
            torch.Tensor: Gradient of the loss w.r.t. the input of the layer.
        """

        d_activation = self.compute_activation_gradient(d_values)

        # Compute gradients for weights and biases
        d_weights = torch.matmul(self.x.T, d_activation)
        d_bias = torch.sum(d_activation, dim=0)

        # Update weights using Adam optimization
        self.v_weights.data = self.beta2 * self.v_weights.data + (1 - self.beta2) * (d_weights ** 2)
        self.m_weights.data = self.beta1 * self.m_weights.data + (1 - self.beta1) * d_weights

        # To correct this bias in weights, we divide by (1−𝛽1^t) for the first moment and (1−𝛽2^t) for the second moment.
        m_hat_weights = self.m_weights / (1 - self.beta1 ** t)
        v_hat_weights = self.v_weights / (1 - self.beta2 ** t)

        # Adaptive weight update
        self.m_weights.data -= learning_rate * m_hat_weights / (torch.sqrt(v_hat_weights) + self.epsilon)

        # Update biases using Adam optimization
        self.v_bias.data = self.beta2 * self.v_bias.data + (1 - self.beta2) * (d_bias ** 2)
        self.m_bias.data = self.beta1 * self.m_bias.data + (1 - self.beta1) * d_bias

        # To correct this bias in the bias term, we divide by (1−𝛽1^t) for the first moment and (1−𝛽2^t) for the second moment.
        m_hat_bias = self.m_bias / (1 - self.beta1 ** t)
        v_hat_bias = self.v_bias / (1 - self.beta2 ** t)

        # Adaptive update of the bias
        self.m_bias.data -= learning_rate * m_hat_bias / (torch.sqrt(v_hat_bias) + self.epsilon)

        # Clip gradients to avoid exploding gradients
        d_weights = torch.clamp(d_weights, -1.0, 1.0)
        d_bias = torch.clamp(d_bias, -1.0, 1.0)

        # Gradient with respect to the input for the next layer
        d_inputs = torch.matmul(d_activation, self.m_weights.T)

        return d_inputs
