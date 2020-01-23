import os
from os.path import expanduser, join

import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def infinite_iter(iterable):
    while True:
        for elem in iterable:
            yield elem


n_samples = 50000

data_dir = expanduser('~/data/games_rl/multi_gan/mnist')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

dataset = datasets.MNIST(root=expanduser('~/data/games_rl'), download=True,
                         transform=transforms.Compose([
                             transforms.Resize((28, 28)),
                             transforms.ToTensor(),
                             transforms.Normalize((0.5,), (0.5,))])
                         )

data_loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
iter_data = infinite_iter(data_loader)
current = 0
images = torch.empty((n_samples, 3, 28, 28))
labels = torch.empty((n_samples, 3,))
while current < n_samples:
    exit_ = False
    data = {}
    for color in ['red', 'blue', 'green']:
        data[color] = next(iter_data)
    new = current + data['red'][0].shape[0]
    if new > n_samples:
        new = n_samples
    batch_size = new - current
    batch = slice(current, new)
    images[batch] = torch.cat([data['red'][0][:batch_size],
                               data['blue'][0][:batch_size],
                               data['green'][0][:batch_size]], dim=1)
    labels[batch] = torch.cat([data['red'][1][:batch_size][:, None],
                               data['blue'][1][:batch_size][:, None],
                               data['green'][1][:batch_size][:, None]], dim=1)
    current = new
    print(f'{current} images')

torch.save(images, join(data_dir, 'images.pkl'))
torch.save(labels, join(data_dir, 'labels.pkl'))
