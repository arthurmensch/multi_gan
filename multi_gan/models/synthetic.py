import torch.nn as nn


class GeneratorSynthetic(nn.Module):
    def __init__(self, n_filters=128, depth=1, noise_dim=32):
        blocks = [nn.Linear(noise_dim, n_filters),
                  nn.BatchNorm1d(n_filters),
                  nn.ReLU()]
        for d in range(depth - 1):
            blocks += [nn.Linear(n_filters, n_filters),
                       nn.BatchNorm1d(n_filters),
                       nn.ReLU()]
        blocks += nn.Linear(n_filters, 2)
        self.sequential = nn.Sequential(*blocks)

    def forward(self, input):
        return self.sequential(input)


class DiscriminatorSynthetic(nn.Module):
    def __init__(self, n_filters=128, depth=1):
        blocks = [nn.Linear(2, n_filters),
                  nn.BatchNorm1d(n_filters),
                  nn.ReLU()]
        for d in range(depth - 1):
            blocks += [nn.Linear(n_filters, n_filters),
                       nn.BatchNorm1d(n_filters),
                       nn.ReLU()]
        blocks += nn.Linear(n_filters, 1)
        self.sequential = nn.Sequential(*blocks)

    def forward(self, input):
        return self.sequential(input)