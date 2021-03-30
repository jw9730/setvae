import torch
import torch.nn as nn
from torch.nn import init


def sample_mask(sample_sizes, max_size):
    """
    :param sample_sizes: Tensor([B,])
    :param max_size:
    :return: mask BoolTensor([B, N])
    """
    device = sample_sizes.device
    presence = [torch.randperm(max_size).to(device) < sample_sizes[b] for b in range(sample_sizes.size(0))]
    mask = ~torch.stack(presence, dim=0).to(device)
    return mask.bool()


def get_mask(sizes, max_size):
    """
    :param sizes: Tensor([B,])
    :param max_size:
    :return: mask BoolTensor([B, N])
    """
    device = sizes.device
    presence = [torch.arange(max_size).to(device) < sizes[b] for b in range(sizes.size(0))]
    mask = ~torch.stack(presence, dim=0).to(device)
    return mask.bool()


def masked_fill(tensor_bnc, mask_bn=None, value=0.):
    return tensor_bnc.masked_fill(mask_bn.unsqueeze(-1), value) if (mask_bn is not None) else tensor_bnc


def check(x):
    isinf = torch.isinf(x).any()
    isnan = torch.isnan(x).any()
    assert not (isinf or isnan), f"Tensor of shape [{x.shape}] is isinf:{isinf} or isnan:{isnan}"


def get_module(gpu, module):
    if hasattr(module, 'module'):
        return module.module
    else:
        return module


def get_pairwise_distance(x, p=2):
    x = x.clone().detach()  # [N, D]
    N, d = x.shape
    x1 = x.unsqueeze(1).repeat(1, N, d)  # [N, N, D]
    x2 = x.unsqueeze(0).repeat(N, 1, d)  # [N, N, D]
    dist = (x1 - x2).norm(p=p, dim=-1)
    return dist


class MaskedBatchNorm1d(nn.Module):
    """ A masked version of nn.BatchNorm1d. Only tested for 3D inputs.
        Source https://gist.github.com/yangkky/364413426ec798589463a3a88be24219
        Args:
            num_features: :math:`C` from an expected input of size
                :math:`(N, C, L)`
            eps: a value added to the denominator for numerical stability.
                Default: 1e-5
            momentum: the value used for the running_mean and running_var
                computation. Can be set to ``None`` for cumulative moving average
                (i.e. simple average). Default: 0.1
            affine: a boolean value that when set to ``True``, this module has
                learnable affine parameters. Default: ``True``
            track_running_stats: a boolean value that when set to ``True``, this
                module tracks the running mean and variance, and when set to ``False``,
                this module does not track such statistics and always uses batch
                statistics in both training and eval modes. Default: ``True``
        Shape:
            - x: (N, C, L)
            - x_mask: (N, L) tensor of ones and zeros, where the zeros indicate locations not to use.
            - Output: :math:`(N, C)` or :math:`(N, C, L)` (same shape as input)
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(MaskedBatchNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 1))
            self.bias = nn.Parameter(torch.Tensor(num_features, 1))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.track_running_stats = track_running_stats
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.ones(1, num_features, 1))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x, x_mask=None):
        x_mask = (~x_mask).unsqueeze(1)
        # Calculate the masked mean and variance
        B, C, L = x.shape
        if x_mask is not None and x_mask.shape != (B, 1, L):
            raise ValueError('Mask should have shape (B, 1, L).')
        if C != self.num_features:
            raise ValueError('Expected %d channels but input has %d channels' % (self.num_features, C))
        if x_mask is not None:
            masked = x * x_mask
            n = x_mask.sum()
        else:
            masked = x
            n = B * L
        # Sum
        masked_sum = masked.sum(dim=0, keepdim=True).sum(dim=2, keepdim=True)
        # Divide by sum of mask
        current_mean = masked_sum / n
        current_var = ((masked - current_mean) ** 2).sum(dim=0, keepdim=True).sum(dim=2, keepdim=True) / n
        # Update running stats
        if self.track_running_stats and self.training:
            if self.num_batches_tracked == 0:
                self.running_mean = current_mean
                self.running_var = current_var
            else:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * current_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * current_var
            self.num_batches_tracked += 1
        # Norm the input
        if self.track_running_stats and not self.training:
            normed = (masked - self.running_mean) / (torch.sqrt(self.running_var + self.eps))
        else:
            normed = (masked - current_mean) / (torch.sqrt(current_var + self.eps))
        # Apply affine parameters
        if self.affine:
            normed = normed * self.weight + self.bias
        return normed
