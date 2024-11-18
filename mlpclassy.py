# here I will define the fnn model MLP 
# MLP is multi-layer perceptron 

import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt


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

    def __init__(self, input_size : int, output_size : int , activation : str , grad_clip : float = 1.0) -> None:
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
        self.grad_clip = grad_clip

        if input_size <= 0 or output_size <= 0:
            raise ValueError("Input and output sizes must be positive integers")
        if not isinstance(input_size, int) or not isinstance(output_size, int):
            raise TypeError("Input and output sizes must be integers")

        self.activation = self.get_activation(activation)
        self.weights = nn.Parameter(self.initialize_weights())
        self.bias = nn.Parameter(torch.zeros(output_size))  # Using PyTorch's tensor for bias

        # Defining mean and variance for weights and biases
        # These are used in Adam optimizer
        self.m_weights = nn.Parameter(self.weights.clone().detach(), requires_grad=True)
        self.m_bias = nn.Parameter(self.bias.clone().detach(), requires_grad=True)
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
        
        if x.shape[-1] != self.input_size:
            raise ValueError(f"Expected input with last dimension {self.input_size}, got {x.shape[-1]}")
        
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
        d_weights = torch.clamp(d_weights, -self.grad_clip, self.grad_clip)
        d_bias = torch.clamp(d_bias, -self.grad_clip, self.grad_clip)

        # Gradient with respect to the input for the next layer
        d_inputs = torch.matmul(d_activation, self.m_weights.T)

        return d_inputs


    def parameters(self):
        """
        Returns all trainable parameters of the layer
        """
        return [self.weights, self.bias, self.m_weights, self.m_bias]

    def reset_optimizer_state(self):
        """
        Reset the Adam optimizer state
        """
        self.m_weights.data.zero_()
        self.m_bias.data.zero_()
        self.v_weights.data.zero_()
        self.v_bias.data.zero_()

    def extra_repr(self) -> str:
        """
        Provides a more informative string representation
        """
        return f'input_size={self.input_size}, output_size={self.output_size}, activation={self.activation.__class__.__name__}'

    def to_device(self, device: torch.device):
        """
        Move the layer to specified device
        """
        self.weights = self.weights.to(device)
        self.bias = self.bias.to(device)
        self.m_weights = self.m_weights.to(device)
        self.m_bias = self.m_bias.to(device)
        self.v_weights = self.v_weights.to(device)
        self.v_bias = self.v_bias.to(device)
        return self


class CreateModel(nn.Module):
    """
    This class creates a model with multiple fully connected layers.
    """
    def __init__(self, input_size : int, hidden_size : int, hidden_layer_neurons : list, output_size : int) -> None:
        """
        initialize the model with input size, hidden size and output size.
        Args:
            input_size (int): Number of input features.
            hidden_layer_neurons (list): List of integers representing the number of neurons in each hidden layer. [hidden1,hidden2]
            output_size (int): Number of output features.
        
        """
        
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError("input_size must be a positive integer")
        
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        
        if not isinstance(hidden_layer_neurons, list) or len(hidden_layer_neurons) != hidden_size:
            raise ValueError("hidden_layer_neurons must be a list of length hidden_size")
        
        if not isinstance(output_size, int) or output_size <= 0:
            raise ValueError("output_size must be a positive integer")

        super(CreateModel, self).__init__()
        self.layers = nn.ModuleList()


        self.fc1 = FCL(input_size, hidden_layer_neurons[0], activation='relu')

        for i in range(hidden_size - 1):
            self.layers.append(FCL(hidden_layer_neurons[i], hidden_layer_neurons[i+1], activation='relu'))
        
        self.fc2 = FCL(hidden_layer_neurons[-1], output_size, activation='softmax')
    


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the model transformation.
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a Tensor (torch.Tensor)")
        

        x = self.fc1.forward(x)
        for layer in self.layers:
            x = layer.forward(x)
        x = self.fc2.forward(x)
        
        return x
    


    def train(self , x: torch.Tensor, y: torch.Tensor, learning_rate: float, n_epochs : int, decay : float, plot_training_result=False) -> None:
        """
        Train the model using the Adam optimizer.

        Args:
            x (torch.Tensor): Input tensor.
            y (torch.Tensor): Target tensor.
            learning_rate (float): Learning rate for optimization.
            n_epochs (int): Number of training epochs.
            decay (float): Learning rate decay factor.
            plot_training_result (bool): Whether to plot the training loss.
        """

        if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
            raise TypeError("x and y must be torch.Tensor")
        
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have the same number of samples")
        
        if x.shape[1] != self.fc1.input_size:
            raise ValueError(f"Expected x with shape (batch_size, {self.fc1.input_size}), but got {x.shape}")
        
        if y.shape[1] != self.fc2.output_size:
            raise ValueError(f"Expected y with shape (batch_size, {self.fc2.output_size}), but got {y.shape}")
        
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive number")
        
        if not isinstance(n_epochs, int) or n_epochs <= 0:
            raise ValueError("n_epochs must be a positive integer")
        
        if not isinstance(decay, (int, float)) or decay < 0:
            raise ValueError("decay must be a non-negative number")

        t = 0

        loss_log = []
        accuracy_log = []

        for epoch in range(n_epochs):
            output = self.forward(x)

            # Compute the loss
            epsilon = 1e-10
            loss = -torch.mean(y * torch.log(output + epsilon))

            # Compute the accuracy
            predictions_labels = torch.argmax(output, dim=1)
            true_labels = torch.argmax(y, dim=1)
            accuracy = torch.sum(predictions_labels == true_labels).item() / len(true_labels)

            # backward pass
            y_grad = (output - y)/output.shape[0]
            t += 1
            learning_rate = learning_rate * 1/(1 + decay * epoch)


            grad_y = self.fc2.backward(y_grad, learning_rate, t)
            for layer in reversed(self.layers):
                grad_y = layer.backward(grad_y, learning_rate, t)
            grad_y = self.fc1.backward(grad_y, learning_rate, t)

            print(f"Epoch {epoch+1}/{n_epochs} Loss: {loss.item()} Accuracy: {accuracy}")

            loss_log.append(loss.item())
            accuracy_log.append(accuracy)

            # if plot_training_result:
            #     self._plot_training_results(n_epochs, loss_log, accuracy_log)

        print(f"Final Loss: {loss_log[-1]}")
        print(f"Final Accuracy: {accuracy_log[-1]}")

    # def _plot_training_results(self, n_epochs, loss_log, accuracy_log):
        # plt.plot(range(n_epochs), loss_log, label='Training Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Training Loss Curve')
        # plt.legend()
        # plt.show()

        # plt.plot(range(n_epochs), accuracy_log, label='Training Accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.title('Training Accuracy Curve')
        # plt.legend()
        # plt.show()



if __name__ == "__main__":
    input_size = 784
    hidden_size = 2
    hidden_layer_neurons = [128, 64]
    output_size = 2


    train_x, train_y = torch.randn(100, input_size), torch.randint(0, 2, (100, output_size)).float()
    print(torch.isnan(train_x).any(), torch.isinf(train_x).any())
    print(torch.isnan(train_y).any(), torch.isinf(train_y).any())


    model = CreateModel(input_size, hidden_size, hidden_layer_neurons, output_size)
    model.train(train_x, train_y, learning_rate=0.01, n_epochs=100, decay=0.01, plot_training_result=True)



    


