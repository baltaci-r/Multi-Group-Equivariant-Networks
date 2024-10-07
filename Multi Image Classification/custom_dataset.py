from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Subset, Dataset
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torchvision import transforms


class KSumMNIST(Dataset):
    """
    Sum of k MNIST images
    :returns k images with transformations individually applied to each and target is the sum of the targets for each
    """
    def __init__(self, root="data", split="train", transforms_list=[], k=2):

        self.k = k
        self.transforms_list = transforms_list
        assert len(self.transforms_list) == k

        if split == "train":
            training_data = datasets.MNIST(root=root, train=True, download=True,
                                           transform=None)  # no transforms added here
            # random shuffle input data before pairing for sum
            length = len(training_data.data)
            random_perm_indx = torch.randperm(length)

            self.data = training_data.data[random_perm_indx]  # dim [N, H, W]
            self.targets = training_data.targets[random_perm_indx]  # dim [N,]

        else:
            val_data = datasets.MNIST(root="data", train=False, download=True,
                                           transform=None)  # no transforms added here
            # random shuffle input data before pairing for sum
            length = len(val_data.data)
            random_perm_indx = torch.randperm(length)

            self.data = val_data.data[random_perm_indx]  # dim [N, H, W]
            self.targets = val_data.targets[random_perm_indx]  # dim [N,]

        # form k-tuples of self.data and corresponding labels
        data_shape = self.data.shape  # [N, H, W]
        target_shape = self.targets.shape  # [N,]
        # truncate data and reshape [N//k, k, H, W]
        self.data = self.data[:(data_shape[0] // k) * k].view(data_shape[0] // k, k, 1, data_shape[1],
                                                                 data_shape[2]).float()  # [N//k, 1, k, H, W]

        self.targets = self.targets[:(target_shape[0] // k) * k].view(target_shape[0] // k, k)  # [N//k, k]
        self.sum_targets = torch.sum(self.targets, dim=1, keepdim=False)

    def __len__(self):
        return len(self.sum_targets)

    def __getitem__(self, idx):
        x = self.data[idx]  # dim [k, H, W]
        data = []
        target = self.sum_targets[idx]  # dim (1,)

        # transform each modality separately
        if len(self.transforms_list) == self.k:
            for i in range(self.k):
                data.append(self.transforms_list[i](x[i]))
        data = torch.stack(data)
        return data, target


class KSumSVHN(Dataset):
    """
    Sum of k SVHN images
    :returns k images with transformations individually applied to each and target is the sum of the targets for each
    """
    def __init__(self, root="data", split="train", transforms_list=[], k=2):

        self.k = k
        self.transforms_list = transforms_list
        assert len(self.transforms_list) == k

        if split == "train":
            training_data = datasets.SVHN(root=root, split="train", download=True, transform=None)  # no transforms here

            # random shuffle input data before pairing for sum
            length = len(training_data.data)
            random_perm_indx = torch.randperm(length)

            self.data = training_data.data[random_perm_indx]  # dim [N, C, H, W]
            self.targets = training_data.labels[random_perm_indx]  # dim [N,]

        else:
            val_data = datasets.SVHN(root=root, split="test", download=True, transform=None)  # no transforms here

            # random shuffle input data before pairing for sum
            length = len(val_data.data)
            random_perm_indx = torch.randperm(length)

            self.data = val_data.data[random_perm_indx]  # dim [N, C, H, W]
            self.targets = val_data.labels[random_perm_indx]  # dim [N,]

        # form k-tuples of self.data and corresponding labels
        data_shape = self.data.shape  # [N, C, H, W]
        target_shape = self.targets.shape  # [N,]
        # truncate data and reshape [N//k, k, H, W]
        self.data = torch.tensor(self.data[:(data_shape[0] // k) * k])
        self.data = self.data.view(data_shape[0] // k, k, data_shape[1], data_shape[2],
                                                                 data_shape[3]).float()  # [N//k, k, C, H, W]

        self.targets = torch.tensor(self.targets[:(target_shape[0] // k) * k])
        self.targets = self.targets.view(target_shape[0] // k, k)  # [N//k, k]
        self.sum_targets = torch.sum(self.targets, dim=1, keepdim=False)

    def __len__(self):
        return len(self.sum_targets)

    def __getitem__(self, idx):
        x = self.data[idx]  # dim [k, C, H, W]
        data = []
        target = self.sum_targets[idx]  # dim (1,)

        # transform each modality separately
        if len(self.transforms_list) == self.k:
            for i in range(self.k):
                data.append(self.transforms_list[i](x[i]))
        data = torch.stack(data)
        return data, target


class KSumCIFAR10(Dataset):
    """
    Sum of k SVHN images
    :returns k images with transformations individually applied to each and target is the sum of the targets for each
    """
    def __init__(self, root="data", split="train", transforms_list=[], k=2):

        self.k = k
        self.transforms_list = transforms_list

        if split == "train":
            training_data = datasets.CIFAR10(root=root, train=True, download=True, transform=None)  # no transforms here

            # random shuffle input data before pairing for sum
            length = len(training_data.data)
            random_perm_indx = torch.randperm(length)

            self.data = training_data.data[random_perm_indx]  # dim [N, C, H, W]
            self.targets = torch.tensor(training_data.targets)[random_perm_indx]  # dim [N,]

        else:
            val_data = datasets.CIFAR10(root=root, train=False, download=True, transform=None)  # no transforms here

            # random shuffle input data before pairing for sum
            length = len(val_data.data)
            random_perm_indx = torch.randperm(length)

            self.data = val_data.data[random_perm_indx]  # dim [N, C, H, W]
            self.targets = torch.tensor(val_data.targets)[random_perm_indx]  # dim [N,]

        # form k-tuples of self.data and corresponding labels
        data_shape = self.data.shape  # [N, H, W, C]
        self.data = torch.tensor(self.data).permute(0, 3, 1, 2)
        data_shape = self.data.shape  # [N, C, H, W]
        target_shape = self.targets.shape  # [N,]
        # truncate data and reshape [N//k, k, H, W]
        self.data = self.data[:(data_shape[0] // k) * k]
        self.data = self.data.view(data_shape[0] // k, k, data_shape[1], data_shape[2],
                                                                 data_shape[3]).float()  # [N//k, k, C, H, W]

        self.targets = torch.tensor(self.targets[:(target_shape[0] // k) * k])
        self.targets = self.targets.view(target_shape[0] // k, k)  # [N//k, k]
        self.sum_targets = torch.sum(self.targets, dim=1, keepdim=False)

    def __len__(self):
        return len(self.sum_targets)

    def __getitem__(self, idx):
        x = self.data[idx]  # dim [k, C, H, W]
        data = []
        target = self.sum_targets[idx]  # dim (1,)

        # transform each modality separately
        if len(self.transforms_list) == self.k:
            for i in range(self.k):
                data.append(self.transforms_list[i](x[i]))
        data = torch.stack(data)
        return data, target


class MultiViewClassification(Dataset):
    """
    :returns k images with transformations individually applied to each and target is the class of the input views
    """
    def __init__(self, input_size, root="data/caltech-101/101_ObjectCategories/", split="train", transforms_list=[], prep_transform=[], k=2):

        self.k = k
        self.transforms_list = transforms_list
        assert len(self.transforms_list) == k

        preprocess = [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
        if prep_transform:
            preprocess.append(prep_transform)
        data = ImageFolder(root=root, transform=transforms.Compose(preprocess))
        training_data, val_data = train_test_split(data, test_size=0.2, random_state=1234)

        if split == "train":
            self.data, self.targets = zip(*training_data)
        else:
            self.data, self.targets = zip(*val_data)

        self.data = torch.stack(self.data)
        self.targets = torch.tensor(self.targets)
        self.classes = torch.unique(self.targets)

        # form k-tuples of self.data and corresponding labels
        grouped_data = []
        grouped_targets = []

        for i in self.classes:
            idx = self.targets == i
            targets = self.targets[idx]
            data = self.data[idx]

            data_shape = data.shape  # [N_i, C, H, W]
            target_shape = targets.shape  # [N_i,]

            data = torch.tensor(data[:(data_shape[0] // k) * k])
            data = data.view(data_shape[0] // k, k, data_shape[1], data_shape[2],
                                       data_shape[3]).float()  # [N_i//k, k, C, H, W]

            targets = torch.tensor(targets[:(target_shape[0] // k) * k])
            targets = targets.view(target_shape[0] // k, k)[:, 0]

            grouped_data.append(data)
            grouped_targets.append(targets)

        self.data = torch.vstack(grouped_data)
        self.targets = torch.cat(grouped_targets)

        # shuffling
        idx = torch.randperm(self.data.shape[0])
        self.data = self.data[idx]
        self.targets = self.targets[idx]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.data[idx]  # dim [k, H, W]
        data = []
        target = self.targets[idx]  # dim (1,)

        # transform each modality separately
        if len(self.transforms_list) == self.k:
            for i in range(self.k):
                data.append(self.transforms_list[i](x[i]))
        data = torch.stack(data)
        return data, target


class PlantCLEF(Dataset):
    """
    :returns k images with transformations individually applied to each and target is the class of the input views
    """
    def __init__(self, input_size, root="data", split="train", transforms_list=[], prep_transform=[], k=2):

        self.k = k
        self.transforms_list = transforms_list
        assert len(self.transforms_list) == k

        preprocess = [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
        if prep_transform:
            preprocess.append(prep_transform)
        data = ImageFolder(root=root, transform=transforms.Compose(preprocess))

        idl = [i for i, (name, label) in enumerate(data.imgs) if name.split('/')[3] == "leaf"]
        idf = [i for i, (name, label) in enumerate(data.imgs) if name.split('/')[3] == "flower"]

        leaves = Subset(data, idl)
        flowers = Subset(data, idf)

        ltrain, ltest = train_test_split(leaves, test_size=0.2, random_state=1234)
        ftrain, ftest = train_test_split(flowers, test_size=0.2, random_state=1234)

        if split == "train":
            leaf, ltargets = zip(*ltrain)
            flower, ftargets = zip(*ftrain)
            self.data = (leaf, flower)
            self.targets = (ltargets, ftargets)
        else:
            leaf, ltargets = zip(*ltest)
            flower, ftargets = zip(*ftest)
            self.data = (leaf, flower)
            self.targets = (ltargets, ftargets)

        leaf, flower = self.data
        self.leaf = torch.stack(leaf)
        self.flower = torch.stack(flower)
        ltargets, ftargets = self.targets
        self.ltargets = torch.tensor(ltargets)
        self.ftargets = torch.tensor(ftargets)

        self.classes = torch.unique(self.ltargets)

        grouped_data = []
        grouped_targets = []

        for i in self.classes:
            idl = self.ltargets == i
            idf = self.ftargets == i

            leaf = self.leaf[idl]
            flower = self.flower[idf]

            nleaf = leaf.shape[0]
            nflower = flower.shape[0]
            n = min(nleaf, nflower)

            data = torch.stack((leaf[:n], flower[:n]), dim=1)
            targets = torch.tensor([i] * n)

            grouped_data.append(data)
            grouped_targets.append(targets)

        self.data = torch.vstack(grouped_data)
        self.targets = torch.cat(grouped_targets)

        # shuffling
        idx = torch.randperm(self.data.shape[0])
        self.data = self.data[idx]
        self.targets = self.targets[idx]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.data[idx]  # dim [k, H, W]
        data = []
        target = self.targets[idx]  # dim (1,)

        # transform each modality separately
        if len(self.transforms_list) == self.k:
            for i in range(self.k):
                data.append(self.transforms_list[i](x[i]))
                # data.append(self.transforms_list[i](x))
        data = torch.stack(data)
        return data, target


class HousePrice(Dataset):
    """
    :returns k images with transformations individually applied to each and target is the class of the input views
    """
    def __init__(self, input_size, root="data", split="train", transforms_list=[], prep_transform=[], k=2):

        self.k = k
        self.transforms_list = transforms_list
        assert len(self.transforms_list) == k

        preprocess = [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
        data = ImageFolder(root=root, transform=transforms.Compose(preprocess))
        training_data, val_data = train_test_split(data, test_size=0.2, random_state=1234)

        if split == "train":
            self.data, self.targets = zip(*training_data)
        else:
            self.data, self.targets = zip(*val_data)

        self.data = torch.stack(self.data)
        self.targets = torch.tensor(self.targets)
        self.classes = torch.unique(self.targets)

        # form k-tuples of self.data and corresponding labels
        grouped_data = []
        grouped_targets = []

        for i in self.classes:
            idx = self.targets == i
            targets = self.targets[idx]
            data = self.data[idx]

            data_shape = data.shape  # [N_i, C, H, W]
            target_shape = targets.shape  # [N_i,]

            data = torch.tensor(data[:(data_shape[0] // k) * k])
            data = data.view(data_shape[0] // k, k, data_shape[1], data_shape[2],
                                       data_shape[3]).float()  # [N_i//k, k, C, H, W]

            targets = torch.tensor(targets[:(target_shape[0] // k) * k])
            targets = targets.view(target_shape[0] // k, k)[:, 0]

            grouped_data.append(data)
            grouped_targets.append(targets)

        self.data = torch.vstack(grouped_data)
        self.targets = torch.cat(grouped_targets)

        # shuffling
        idx = torch.randperm(self.data.shape[0])
        self.data = self.data[idx]
        self.targets = self.targets[idx]


    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.data[idx]  # dim [k, H, W]
        data = []
        target = self.targets[idx]  # dim (1,)

        # transform each modality separately
        if len(self.transforms_list) == self.k:
            for i in range(self.k):
                data.append(self.transforms_list[i](x[i]))
        data = torch.stack(data)
        return data, target