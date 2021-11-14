from typing import Callable
from flax import linen as nn


class MnistCNN(nn.Module):
    """
    Model of Table 1. 
    """
    k_filters: int = 32
    activation: Callable = nn.relu
    
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.k_filters, 
            kernel_size=(8, 8), 
            strides=2
        )(x)
        x = self.activation(x)
        x = nn.max_pool(x, (2, 2))
        x = nn.Conv(
            features=self.k_filters, 
            kernel_size=(4, 4), 
            strides=2
        )(x)
        x = self.activation(x)
        x = nn.max_pool(x, (2, 2))
        x = x.reshape((x.shape[0], -1)) # flatten
        x = nn.Dense(
            features=32,
        )(x)
        x = self.activation(x)
        x = nn.Dense(
            features=10
        )(x)
        
        return x