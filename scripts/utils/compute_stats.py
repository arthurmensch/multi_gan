import os
from os.path import expanduser, join

import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from multi_gan.data import make_image_data
from multi_gan.eval.fid_score import calculate_activation_statistics, calculate_inception_score, \
    calculate_fid_given_paths
from multi_gan.eval.inception import InceptionV3


def compute_train_stats(data_source):

    cifar10, path = make_image_data(data_source)

    output_dir = expanduser('~/data/temp')
    paths = save_dataset_to_folder(cifar10, output_dir)

    model = InceptionV3([3, 4]).to('cuda:0')
    mu, sigma, prob = calculate_activation_statistics(paths, model, device='cuda:0')
    is_mean, is_std = calculate_inception_score(prob)
    np.savez(path, mu=mu, sigma=sigma, is_mean=is_mean, is_std=is_std)


def save_dataset_to_folder(dataset, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    data_loader = DataLoader(dataset, batch_size=512)

    paths = []
    for images, _ in data_loader:
        images += 1
        images /= 2
        for i, image in enumerate(images):
            path = join(folder, f'img_{len(paths)}.png')
            save_image(image, path)
            paths.append(path)
            if len(paths) % 1000 == 0:
                print(f'Done {len(paths)}/{len(dataset)}')
    return paths


def compare_with_train():
    cifar10_test, _ = make_image_data('cifar10_test')
    cifar10, path = make_image_data('cifar10')

    output_dir = expanduser('~/data/temp')
    stat_path = expanduser('~/data/multi_gan/cifar10/stats.npz')
    save_dataset_to_folder(cifar10, output_dir)
    fid_value, is_score = calculate_fid_given_paths([output_dir, stat_path], batch_size=50, device=True, dims=2048)
    print(fid_value, is_score)


# compute_train_stats('mnist')
mnist, path = make_image_data('mnist')

fid_value, is_mean, is_std = calculate_fid_given_paths([path, path], batch_size=50, device='cuda:0', dims=2048)
print(fid_value, is_mean, is_std)

