# here I will define the fnn model MLP 
# MLP is multi-layer perceptron 

import torch.nn as nn
import torch
import numpy as np

class FCL(nn.Module):
    # we need to also input activation function if it is empty we will use relu
    def __init__(self, input_size : int, output_size : int , activation : str) -> None:
        """
        Args:
            input_size: int, size of input layer
            output_size: int, size of output layer
            activation: torch.nn.Module, activation function to use
        
        """
        super(FCL, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        def get_activation(activation : str) -> nn.Module:
            """
            Returns the activation function based on the string provided.
            If activation is None, defaults to ReLU.
            """
            if activation is None:
                return nn.ReLU()  # Default activation function
    
            activation = activation.lower()
    
            if activation == 'relu':
                return nn.ReLU()
            elif activation == 'sigmoid':
                return nn.Sigmoid()
            elif activation == 'tanh':
                return nn.Tanh()
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

        self.activation = get_activation(activation)
        self.bias = nn.Parameter(torch.zeros(output_size))  # Using PyTorch's tensor for bias

        if isinstance(self.activation, (nn.Sigmoid, nn.Tanh)):
            # Xavier initialization for Sigmoid and Tanh for normal distribution
            nn.init.xavier_normal_(self.weights)
        elif isinstance(self.activation, nn.ReLU):
            # He initialization for ReLU (Kaiming initialization) for normal distribution
            nn.init.kaiming_normal_(self.weights, nonlinearity='relu')
        else:
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        
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



    def forward(self , x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer
        Args:
            x: torch.Tensor, input tensor
        Returns:
            torch.Tensor, output tensor
        """

        self.x = x

        # we will use dot product of input and weights and add bias
        z = torch.matmul(x, self.m_weights) + self.m_bias
    
# we can create new github repos by using the following command
# git init
# git add .
# git commit -m "first commit"
# git branch -M main
# git remote add origin
# how to create a new repo in github using command line
