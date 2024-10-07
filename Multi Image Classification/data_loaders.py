import torch
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import Grayscale
# from imagenetv2_pytorch import ImageNetV2Dataset
import os
import argparse
from custom_dataset import KSumSVHN, KSumMNIST, KSumCIFAR10, MultiViewClassification, PlantCLEF, HousePrice

from custom_transforms import RandomRot90


# get transforms
def get_train_transforms(args, in_channels, data_aug, logging):
    # print("***Double check the normalize values for new datasets")
    logging.info(data_aug)
    if in_channels == 1:
        # for MNIST
        if data_aug == "id":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.RandomCrop(args.input_size, padding=4),
                transforms.Resize(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif data_aug == "rot90":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.RandomCrop(args.input_size, padding=4),
                transforms.Resize(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                RandomRot90(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif data_aug == "hflip":
            # don't use this for experiments since we are anyway using random flips in our exps to avoid overfitting
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.RandomCrop(args.input_size, padding=4),
                transforms.Resize(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            raise NotImplementedError
    elif in_channels == 3:
        # for SVHN
        if data_aug == "id":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.RandomCrop(args.input_size, padding=4),
                transforms.Resize(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif data_aug == "rot90":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.RandomCrop(args.input_size, padding=4),
                transforms.Resize(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                RandomRot90(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif data_aug == "hflip":
            # don't use this for experiments since we are anyway using random flips in our exps to avoid overfitting
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.RandomCrop(args.input_size, padding=4),
                transforms.Resize(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return transform


def get_test_transforms(args, in_channels, data_aug, logging):
    # print("***Double check the normalize values for new datasets")
    logging.info(data_aug)
    if in_channels == 1:
        # for MNIST
        if data_aug == "id":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif data_aug == "rot90":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.ToTensor(),
                RandomRot90(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif data_aug == "hflip":
            # don't use this for experiments since we are anyway using random flips in our exps to avoid overfitting
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            raise NotImplementedError
    elif in_channels == 3:
        # for SVHN
        if data_aug == "id":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif data_aug == "rot90":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.ToTensor(),
                RandomRot90(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif data_aug == "hflip":
            # don't use this for experiments since we are anyway using random flips in our exps to avoid overfitting
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return transform


# get transforms
def get_transforms(args, in_channels, data_aug, logging):
    # print("***Double check the normalize values for new datasets")
    logging.info(data_aug)
    if in_channels == 1:
        # for MNIST
        if data_aug == "id":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif data_aug == "rot90":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                RandomRot90(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif data_aug == "hflip":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.CenterCrop(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        else:
            raise NotImplementedError
    elif in_channels == 3:
        # for SVHN
        if data_aug == "id":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif data_aug == "rot90":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                RandomRot90(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif data_aug == "hflip":
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(args.input_size),
                transforms.CenterCrop(args.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            print(data_aug)
            print(data_aug=="hflip")
            raise NotImplementedError
    else:
        raise NotImplementedError
    return transform


# get datasets with transforms
def get_datasets(args, logging):
    # get transforms
    logging.info('*** Train augmentations ***')
    train_transforms_list = [get_train_transforms(args, args.in_channels, data_aug, logging) for data_aug in
                             args.train_data_aug_list]
    logging.info('*** Test augmentations ***')
    test_transforms_list = [get_test_transforms(args, args.in_channels, data_aug, logging) for data_aug in
                            args.test_data_aug_list]
    logging.info('*'*30)

    if args.dataset == "KSumMNIST":
        traindata = KSumMNIST(split="train", transforms_list=train_transforms_list, k=args.num_inputs)
        testdata = KSumMNIST(split="test", transforms_list=test_transforms_list, k=args.num_inputs)
    elif args.dataset == "KSumSVHN":
        traindata = KSumSVHN(split="train", transforms_list=train_transforms_list, k=args.num_inputs)
        testdata = KSumSVHN(split="test", transforms_list=test_transforms_list, k=args.num_inputs)
    elif args.dataset == "KSumCIFAR10":
        traindata = KSumCIFAR10(split="train", transforms_list=train_transforms_list, k=len(train_transforms_list))
        testdata = KSumCIFAR10(split="test", transforms_list=test_transforms_list, k=len(test_transforms_list))
    elif args.dataset in ["Caltech101", "15Scene", "CUB", "CompCars"]:
        prep_transform = Grayscale() if args.dataset == "15Scene" else []
        traindata = MultiViewClassification(args.input_size, root='data/'+args.dataset_path, split="train",
                                            transforms_list=train_transforms_list,
                                            prep_transform=prep_transform, k=args.num_inputs)
        testdata = MultiViewClassification(args.input_size, root='data/'+args.dataset_path, split="test",
                                           transforms_list=test_transforms_list,
                                           prep_transform=prep_transform, k=args.num_inputs)
    elif args.dataset == 'PlantCLEF':
        traindata = PlantCLEF(args.input_size, root='data/'+args.dataset, split="train",
                                            transforms_list=train_transforms_list, k=args.num_inputs)
        testdata = PlantCLEF(args.input_size, root='data/'+args.dataset, split="test",
                                           transforms_list=test_transforms_list, k=args.num_inputs)
    elif args.dataset == "HousePrice":
        traindata = HousePrice(args.input_size, root='data/'+args.dataset, split="train",
                                            transforms_list=train_transforms_list, k=args.num_inputs)
        testdata = HousePrice(args.input_size, root='data/'+args.dataset, split="test",
                                           transforms_list=test_transforms_list, k=args.num_inputs)
    else:
        raise NotImplementedError

    logging.info('*** Dataset ***')
    logging.info('Train: %s', len(traindata))
    logging.info('Test: %s', len(testdata))
    logging.info('*' * 30)

    return traindata, testdata


# get loaders from above details
def get_dataloaders(args, logging):
    traindata, testdata = get_datasets(args, logging)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(testdata, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    dataloaders_dict = {'train': trainloader, 'test': testloader}
    return dataloaders_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="KSumMNIST", type=str)
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--input_size", default=28, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--data_aug_list", nargs='+', type=str)
    args = parser.parse_args()

    args.data_aug_list = ["id", "rot90", "hflip"]
    traindata, testdata = get_datasets(args)
    print("loaded datasets")
    dataloaders_dict = get_dataloaders(args)
    print("loaded dataloaders")


