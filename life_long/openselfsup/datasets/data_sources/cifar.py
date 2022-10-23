from abc import ABCMeta, abstractmethod
from PIL import Image
import numpy as np

from torchvision.datasets import CIFAR10, CIFAR100

from ..registry import DATASOURCES


class Cifar(metaclass=ABCMeta):

    CLASSES = None

    def __init__(
            self, root, split, 
            return_label=True, rep_num=1, 
            sample_idx=None,
            **kwargs):
        assert split in ['train', 'test']
        self.root = root
        self.split = split
        self.return_label = return_label
        self.cifar = None
        self.rep_num = rep_num
        self.sample_idx = None
        if sample_idx is not None:
            assert isinstance(sample_idx, str)
            self.sample_idx = np.load(sample_idx)
        self.set_cifar()
        self.labels = self.cifar.targets

    @abstractmethod
    def set_cifar(self):
        pass

    def get_length(self):
        if self.sample_idx is None:
            return len(self.cifar) * self.rep_num
        else:
            return len(self.sample_idx) * self.rep_num

    def get_sample(self, idx):
        if self.sample_idx is not None:
            idx = self.sample_idx[idx % len(self.sample_idx)]
        else:
            idx = idx % len(self.cifar)

        img = Image.fromarray(self.cifar.data[idx])
        if self.return_label:
            target = self.labels[idx]  # img: HWC, RGB
            return img, target
        else:
            return img


@DATASOURCES.register_module
class Cifar10(Cifar):

    CLASSES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
        'horse', 'ship', 'truck'
    ]

    def __init__(self, root, split, return_label=True, **kwargs):
        super().__init__(root, split, return_label, **kwargs)

    def set_cifar(self):
        try:
            self.cifar = CIFAR10(
                root=self.root, train=self.split == 'train', download=False)
        except:
            raise Exception("Please download CIFAR10 manually, \
                  in case of downloading the dataset parallelly \
                  that may corrupt the dataset.")


@DATASOURCES.register_module
class Cifar100(Cifar):

    def __init__(self, root, split, return_label=True, **kwargs):
        super().__init__(root, split, return_label, **kwargs)

    def set_cifar(self):
        try:
            self.cifar = CIFAR100(
                root=self.root, train=self.split == 'train', download=False)
        except:
            raise Exception("Please download CIFAR10 manually, \
                  in case of downloading the dataset parallelly \
                  that may corrupt the dataset.")


@DATASOURCES.register_module
class Cifar100Mix(object):

    CLASSES = None

    def __init__(
            self, root, 
            train_subsample=20000, 
            test_subsample=5000):
        self.root = root
        self.cifar_train = None
        self.cifar_test = None
        self.train_subsample = train_subsample
        self.test_subsample = test_subsample
        self.set_cifar()
        self.labels = \
                self.cifar_train.targets[:train_subsample] \
                + self.cifar_test.targets[:test_subsample]
        self.img_data = np.concatenate(
                [self.cifar_train.data[:train_subsample], \
                 self.cifar_test.data[:test_subsample]],
                axis=0)

    def set_cifar(self):
        self.cifar_train = CIFAR100(
            root=self.root, train=True, 
            download=False)
        self.cifar_test = CIFAR100(
            root=self.root, train=False, 
            download=False)

    def get_length(self):
        return self.train_subsample + self.test_subsample

    def get_sample(self, idx):
        img = Image.fromarray(self.img_data[idx])
        return img
