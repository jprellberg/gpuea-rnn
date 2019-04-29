import torch
import torch.nn as nn


def add_popdim(x, popsize):
    # Repeats x along a new first dimension popsize times.
    # Source shape: [batch, channels, h, w]
    # Target shape: [popsize, batch, channels, h, w]
    return x.expand(popsize, *(-1 for _ in range(x.dim())))


def pop_batch_merge(x):
    # Merges pop and batch dimensions. This is necessary for many
    # batched operations that expect a single batch dimension.
    # Source shape: [popsize, batch, *]
    # Target shape: [popsize × batch, *]
    return x.view(x.size(0) * x.size(1), *x.size()[2:])


def pop_batch_split(x, popsize):
    # Splits the first dimension into a pop and batch dimension. This
    # is the reverse of pop_batch_merge.
    # Source shape: [popsize × batch, *]
    # Target shape: [popsize, batch, *]
    return x.view(popsize, x.size(0) // popsize, *x.size()[1:])


def param(*shape):
    return torch.nn.Parameter(torch.empty(shape), requires_grad=False)


def pop_init_(tensor, call):
    for i in range(tensor.size(0)):
        call(tensor[i])


def slice_select(tensor, idx):
    if idx is not None:
        return tensor[idx:idx + 1]
    else:
        return tensor


class PopModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fwd_idx = None

    def forward(self, x):
        raise NotImplementedError

    def set_forward_index(self, index):
        """
        When forward_index is set to an integer i, only the i-th member of the population is used
        for the forward pass. When forward_index is set to None, the whole population is used in the
        forward pass.
        """
        self.apply(lambda m: setattr(m, 'fwd_idx', index))


class PopLinear(PopModule):
    def __init__(self, popsize, in_features, out_features):
        super().__init__()
        self.weight = param(popsize, in_features, out_features)
        self.bias = param(popsize, out_features)

    def forward(self, x):
        # Source shape: [popsize, batch, in_features]
        # Target shape: [popsize, batch, out_features]
        w = slice_select(self.weight, self.fwd_idx)
        b = slice_select(self.bias, self.fwd_idx)
        return torch.einsum('pbi,pio->pbo', (x, w)) + b.unsqueeze(1)


class PopRNNCell(PopModule):
    def __init__(self, popsize, in_features, hidden_size):
        super().__init__()
        self.ih = PopLinear(popsize, in_features, hidden_size)
        self.hh = PopLinear(popsize, hidden_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x, hidden=None):
        # Source shape: x=[popsize, batch, in_features], hidden(0)=[popsize, batch, hidden_size]
        # Target shape: hidden(T)=[popsize, batch, hidden_size]
        if hidden is None:
            hidden = x.new_zeros(x.size(0), x.size(1), self.hidden_size, requires_grad=False)
        return torch.tanh(self.ih(x) + self.hh(hidden))
