from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torch.utils.data import RandomSampler, BatchSampler
from PIL import Image

from augmentation import cifar_strong_transforms, cifar_weak_transforms, cifar_test_transforms


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
    def __init__(self, file_path, download, batch_size, mu, use_cuda):

        kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        self.mu = mu

        # Get the datasets
        train_labeled_dataset, train_unlabeled_dataset, test_dataset = self.get_dataset(file_path, download)

        # Set the samplers
        num_samples = len(train_labeled_dataset)
        sampler_labeled = RandomSampler(train_labeled_dataset, replacement=True, num_samples=num_samples)
        sampler_unlabeled = RandomSampler(train_unlabeled_dataset, replacement=True, num_samples=self.mu * num_samples)

        batch_sampler_labeled = BatchSampler(sampler_labeled, batch_size=batch_size, drop_last=False)
        batch_sampler_unlabeled = BatchSampler(sampler_unlabeled, batch_size=self.mu * batch_size, drop_last=False)

        # Set the loaders
        self.train_labeled = DataLoader(train_labeled_dataset, batch_sampler=batch_sampler_labeled, **kwargs)
        self.train_unlabeled = DataLoader(train_unlabeled_dataset, batch_sampler=batch_sampler_unlabeled, **kwargs)

        self.test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

        tmp_batch = self.test.__iter__().__next__()[0]
        self.img_shape = list(tmp_batch.size())[1:]

    @staticmethod
    def get_dataset(file_path, download):

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

        return train_labeled_dataset, train_unlabeled_dataset, test_dataset
