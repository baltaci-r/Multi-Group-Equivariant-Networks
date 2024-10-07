import os
import json
import logging
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

from data_loaders import get_dataloaders
from models import ConvNet, SiameseNet, DeepSet, DSS, GEMINN, Gemini
from train import train_model
from utils import count_parameters


model_dict = {
    "ConvNet": ConvNet,
    "SiameseNet": SiameseNet,
    "DeepSet": DeepSet,
    "DSS": DSS,
    "GEMINN": GEMINN,
    "Gemini": Gemini
}


def main(args):

    logging.basicConfig(format='%(asctime)s === %(message)s', level=logging.INFO)
    # seed everything
    pl.seed_everything(args.seed)

    # load dataloaders
    logging.info('Preprocessing Data')
    dataloaders_dict = get_dataloaders(args, logging)

    # load models
    logging.info('Loading Model')
    if "GEMINN" in args.model_name or "Gemini" in args.model_name:
        model = model_dict[args.model_name](args, k=args.k, num_classes=args.num_classes, input_size=args.input_size,
                                            type=args.type, logging=logging, list_symmetry_groups=args.list_symmetry_groups)
    else:
        model = model_dict[args.model_name](args, k=args.k, num_classes=args.num_classes, input_size=args.input_size,
                                        type=args.type, logging=logging, symmetry_group=args.symmetry_group)
    model = model.to(args.device)
    model = torch.nn.DataParallel(model)

    logging.info(f"Loaded {args.model_name} model with {count_parameters(model)} parameters, CUDA {torch.cuda.device_count()}")

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    if args.type == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    train_writer = SummaryWriter(args.file_dir + '/train')
    test_writer = SummaryWriter(args.file_dir + '/test')

    logging.info('Training Model')
    model, _, best_test_acc = train_model(model, dataloaders_dict, criterion, optimizer, args.device, \
                                          train_writer, test_writer, args.num_epochs, logging)
    logging.info('Saving Results')
    results = {
        'seed': args.seed,
        'fusion': args.fusion,
        'model_name': args.model_name,
        'input_size': args.input_size,
        'parameters': count_parameters(model),
        'dataset': args.dataset,
        'train_aug': args.train_data_aug_list,
        'test_aug': args.test_data_aug_list,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'epochs': args.num_epochs,
        'conv_channels': args.conv_channels,
        'lin_channels': args.lin_channels,
        'symmetry_group': args.symmetry_group,
        'list_symmetry_groups': args.list_symmetry_groups,
        'best_acc': best_test_acc.item(),
        'num_inputs': args.num_inputs,
    }

    with open(os.path.join(args.file_dir, 'results.json'), "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"best accuracy for {args.model_name} is {best_test_acc}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parser for multimodal group equivariant networks")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dataset", default="KSumMNIST", type=str)
    parser.add_argument("--input_size", default=28, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--num_classes", default=10, type=int)
    parser.add_argument("--lr", default=0.003, type=float)
    parser.add_argument("--wd", default=0.001, type=float, help="weight decay")
    parser.add_argument("--dropout1", default=0.50, type=float)
    parser.add_argument("--dropout2", default=0.50, type=float)
    parser.add_argument("--num_epochs", default=1, type=int)
    parser.add_argument("--model_name", default="Gemini", type=str,
                        help='choose from ["ConvNet", "SiameseNet", "DeepSet", "DSS", "GEMINN", "Gemini"]')
    parser.add_argument("--symmetry_group", default="", type=str, help='choose from ["rot90", "hflip"]')
    parser.add_argument("--list_symmetry_groups", nargs='+', default=["rot90", "hflip", ""], type=str, help='form a list from ["rot90", "hflip"]')
    parser.add_argument("--conv_channels", default=32, type=int)
    parser.add_argument("--lin_channels", default=256, type=int)
    parser.add_argument("--train_data_aug_list", nargs='+', default=["", ""], type=str)
    parser.add_argument("--test_data_aug_list", nargs='+', default=["", ""], type=str)
    parser.add_argument("--file_dir", default="logs", type=str)
    parser.add_argument("--num_inputs", type=int,  default=2)
    parser.add_argument("--fusion", action='store_true', help="Adds fusion layers to the model if True")
    args = parser.parse_args()

    args.k = len(args.train_data_aug_list)

    args.in_channels = 3
    if args.dataset =="15Scene":
        args.in_channels = 1
        args.num_classes = 15
        args.type = 'classify'
        args.dataset_path = '15-Scene/'
    elif args.dataset == "Caltech101":
        args.num_classes = 101
        args.type = 'classify'
        args.dataset_path = "caltech-101/101_ObjectCategories/"
    else:
        raise NotImplementedError

    args.device = args.device if torch.cuda.is_available() else "cpu"
    main(args)

