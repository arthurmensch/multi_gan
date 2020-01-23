import math
from os.path import expanduser, join

import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision.transforms import transforms
import torchvision.datasets as dset

def infinite_iter(generator):
    while True:
        for e in generator:
            yield e


class GMMSampler():
    def __init__(self, mean: torch.tensor, cov: torch.tensor, p: torch.tensor):
        # k, d = mean.shape
        k, d, d = cov.shape
        # k = p.shape
        self.mean = mean
        self.cov = cov
        self.icov = torch.cat([torch.inverse(cov)[None, :, :] for cov in self.cov], dim=0)
        det = torch.tensor([torch.det(cov) for cov in self.cov])
        self.log_norm = .5 * torch.log((2 * math.pi) * d + torch.log(det))
        self.p = p

    def __call__(self, n):
        k, d = self.mean.shape
        indices = np.random.choice(k, n, p=self.p.numpy())
        pos = np.zeros((n, d), dtype=np.float32)
        for i in range(k):
            mask = indices == i
            size = mask.sum()
            pos[mask] = np.random.multivariate_normal(self.mean[i], self.cov[i], size=size)
        logweight = np.full_like(pos[:, 0], fill_value=-math.log(n))
        return torch.from_numpy(pos), torch.from_numpy(logweight)

    def log_prob(self, x):
        # b, d = x.shape
        diff = x[:, None, :] - self.mean[None, :]  # b, k, d
        return torch.logsumexp(torch.log(self.p)[None, :]
                               - torch.einsum('bkd,kde,bke->bk', [diff, self.icov, diff]) / 2 - self.log_norm[None, :],
                               dim=1)


def make_8gmm():
    theta = torch.linspace(0, 7 / 4 * math.pi, 8)
    mean = torch.cat([torch.cos(theta)[:, None], torch.sin(theta)[:, None]], dim=1)
    cov = torch.eye(2)[None, :, :].repeat((8, 1, 1)) * 0.01
    p = torch.full((8,), fill_value=1. / 8)
    sampler = GMMSampler(mean, cov, p)
    data, _ = sampler(50000)
    return TensorDataset(data), sampler


def make_25gmm():
    x, y = torch.meshgrid([torch.linspace(-2.5, 2.5, 5), torch.linspace(-2.5, 2.5, 5)])
    mean = torch.cat([x[:, :, None], y[:, :, None]], dim=2).view(-1, 2)
    cov = torch.eye(2)[None, :, :].repeat((25, 1, 1)) * 0.01
    p = torch.full((8,), fill_value=1. / 8)
    sampler = GMMSampler(mean, cov, p)
    data, _ = sampler(50000)
    return TensorDataset(data), sampler


def make_image_data(dataset, root=expanduser('~/data/multi_gan')):
    if dataset == 'cifar10':
        dataset = dset.CIFAR10(root=root, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize((32, 32)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

    elif dataset == 'mnist':
        dataset = dset.MNIST(root=root, download=True,
                             transform=transforms.Compose([
                                 transforms.Resize((28, 28)),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5,), (0.5,)),
                             ]))
    elif dataset == 'multimnist':
        data_dir = join(root, 'multimnist')
        images = torch.load(join(data_dir, 'images.pkl'))
        labels = torch.load(join(data_dir, 'labels.pkl'))
        dataset = TensorDataset(images, labels)
    else:
        raise ValueError
    return dataset