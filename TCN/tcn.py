import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalSkipBlock(TemporalBlock):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, use_skips=False):
        super().__init__(n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=dropout)
        self.use_skips = use_skips
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1)

    def init_weights(self):
        super().init_weights()

    def forward(self, x):
        if type(x) == tuple and self.use_skips:
            x, skip = x[0], x[1]
            out = self.net(x)
            res = self.downsample(x)
            return (self.relu(out + res), skip + res)
        else:
            out = self.net(x)
            res = self.downsample(x)
            return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN_DimensionalityReduced(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, use_skip_connections=False, reduce_dimensionality=True):
        super(TCN_DimensionalityReduced, self).__init__()
        self.use_skip_connections = use_skip_connections
        layers = []
        num_levels = len(num_channels)
        self.reduce_dimensionality = reduce_dimensionality
        if self.reduce_dimensionality:
            self.d_reduce = nn.Conv1d(num_inputs, num_channels[0], kernel_size=1)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_channels[i-1] if (self.reduce_dimensionality or i!=0) else num_inputs
            out_channels = num_channels[i]
            layers += [TemporalSkipBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                         padding=(kernel_size-1) * dilation_size, dropout=dropout, use_skips=use_skip_connections)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.to(self.network[0].net[0].weight.dtype)
        if self.reduce_dimensionality:
            x = self.d_reduce(x)
            x = self.network(x)
        if self.use_skip_connections:
            x = self.network(x)
            if type(x) == tuple:
                x, skip = x[0]. x[1]
                x += skip
        return x