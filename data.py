from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

import numpy as np
from torch.utils.data import SubsetRandomSampler

from PIL import ImageFilter
from PIL import Image
import torch


def cifar_strong_transforms():
    all_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    return all_transforms


def cifar_weak_transforms():
    all_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    return all_transforms


def cifar_test_transforms():
    all_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    return all_transforms


class CIFAR10C(CIFAR10):
    def __init__(self, weak_transform, strong_transform, *args, **kwargs):
        super(CIFAR10C, self).__init__(*args, **kwargs)
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # return a PIL Image
        img = Image.fromarray(img)

        xi = self.weak_transform(img)
        xj = self.strong_transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # Return label just for debugging
        return xi, xj, target


class LoaderCIFAR(object):
    def __init__(self, file_path, download, batch_size, use_cuda):

        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

        # Get the datasets
        train_labeled_dataset, train_unlabeled_dataset, test_dataset, labeled_ind, unlabeled_ind = self.get_dataset(file_path, download)
        # Set the loaders
        self.train_labeled = DataLoader(train_labeled_dataset, batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(labeled_ind), **kwargs)
        self.train_unlabeled = DataLoader(train_unlabeled_dataset, batch_size=batch_size, shuffle=False, sampler=SubsetRandomSampler(unlabeled_ind), **kwargs)

        self.test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        tmp_batch = self.test.__iter__().__next__()[0]
        self.img_shape = list(tmp_batch.size())[1:]

    @staticmethod
    def get_dataset(file_path, download):

        num_sample = 10

        # transforms
        weak_transform = cifar_weak_transforms()
        strong_transform = cifar_strong_transforms()
        test_transform = cifar_test_transforms()

        # Training and Validation datasets
        train_labeled_dataset = CIFAR10(root=file_path, train=True, download=download,
                                        transform=weak_transform,
                                        target_transform=None)
        train_unlabeled_dataset = CIFAR10C(weak_transform=weak_transform, strong_transform=strong_transform,
                                           root=file_path, train=True, download=download,
                                           transform=None,
                                           target_transform=None)

        test_dataset = CIFAR10(root=file_path, train=False, download=download,
                               transform=test_transform,
                               target_transform=None)

        if isinstance(train_labeled_dataset.targets, torch.Tensor):
            train_labels = train_labeled_dataset.targets.numpy()
        else:
            train_labels = np.array(train_labeled_dataset.targets)

        labeled_ind, unlabeled_ind = [], []

        for cl in range(10):
            class_indices = np.random.permutation(np.where(train_labels == cl)[0]).tolist()
            labeled_ind.extend(class_indices[:num_sample])
            unlabeled_ind.extend(class_indices[num_sample:])

        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, labeled_ind, unlabeled_ind
