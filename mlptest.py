import pytest
import torch
import torch.nn as nn
from mlpclassy import FCL  # Replace 'your_module' with the actual module name

@pytest.fixture
def fcl_instance():
    return FCL(input_size=5, output_size=3, activation='relu')

def test_fcl_initialization(fcl_instance):
    assert fcl_instance.input_size == 5
    assert fcl_instance.output_size == 3
    assert isinstance(fcl_instance.activation, nn.ReLU)
    assert fcl_instance.weights.shape == (5, 3)
    assert fcl_instance.bias.shape == (3,)

def test_get_activation():
    assert isinstance(FCL(5, 3, 'relu').activation, nn.ReLU)
    assert isinstance(FCL(5, 3, 'sigmoid').activation, nn.Sigmoid)
    assert isinstance(FCL(5, 3, 'tanh').activation, nn.Tanh)
    assert isinstance(FCL(5, 3, 'softmax').activation, nn.Softmax)
    
    with pytest.raises(ValueError):
        FCL(5, 3, 'invalid_activation')

def test_forward_pass(fcl_instance):
    input_tensor = torch.randn(2, 5)  # 2 samples, 5 features each
    output = fcl_instance.forward(input_tensor)
    
    assert output.shape == (2, 3)  # 2 samples, 3 output features
    assert torch.all(output >= 0)  # ReLU activation should produce non-negative outputs

def test_backward_pass(fcl_instance):
    input_tensor = torch.randn(2, 5)
    output = fcl_instance.forward(input_tensor)
    
    d_values = torch.randn(2, 3)  # Random gradient for testing
    learning_rate = 0.01
    t = 1  # First iteration
    
    d_inputs = fcl_instance.backward(d_values, learning_rate, t)
    
    assert d_inputs.shape == (2, 5)
def test_weight_update(fcl_instance):
    input_tensor = torch.randn(2, 5)
    output = fcl_instance.forward(input_tensor)
    
    initial_weights = fcl_instance.m_weights.clone()
    initial_bias = fcl_instance.m_bias.clone()
    
    d_values = torch.randn(2, 3)
    learning_rate = 0.01
    t = 1
    
    fcl_instance.backward(d_values, learning_rate, t)
    
    assert not torch.all(fcl_instance.m_weights == initial_weights)
    assert not torch.all(fcl_instance.m_bias == initial_bias)

def test_multiple_iterations(fcl_instance):
    for t in range(1, 10):
        input_tensor = torch.randn(2, 5)
        output = fcl_instance.forward(input_tensor)
        d_values = torch.randn(2, 3)
        learning_rate = 0.01
        
        fcl_instance.backward(d_values, learning_rate, t)
    
    # After multiple iterations, m_weights and v_weights should be different
    assert not torch.all(fcl_instance.m_weights == fcl_instance.v_weights)

def test_gradient_clipping(fcl_instance):
    input_tensor = torch.randn(2, 5)
    output = fcl_instance.forward(input_tensor)
    
    # Create very large gradients
    d_values = torch.randn(2, 3) * 1000
    learning_rate = 0.01
    t = 1
    
    fcl_instance.backward(d_values, learning_rate, t)
    
    # Check if all gradients are within the [-1, 1] range
    assert torch.all(fcl_instance.m_weights >= -1) and torch.all(fcl_instance.m_weights <= 1)
    assert torch.all(fcl_instance.m_bias >= -1) and torch.all(fcl_instance.m_bias <= 1)