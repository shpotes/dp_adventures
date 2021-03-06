from flax import linen as nn

class ReLU(nn.Module):
    def __call__(self, x):
        return nn.relu(x)

class GeLU(nn.Module):
    def __call__(self, x):
        return nn.gelu(x)