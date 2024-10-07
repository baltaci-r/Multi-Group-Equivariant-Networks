from models import ConvNet, SiameseNet, DeepSet, DSS, GEMINN, Gemini
from torchvision.transforms.functional import hflip, vflip
from utils import count_parameters

import argparse
import torch
import logging


model_dict = {
    "ConvNet": ConvNet,
    "SiameseNet": SiameseNet,
    "DeepSet": DeepSet,
    "DSS": DSS,
    "GEMINN": GEMINN,
    "Gemini": Gemini
}


def transform(x, test_transform=""):
    if test_transform == "rot90":
        k = torch.randint(low=1, high=4, size=(1,)).item()
        x = torch.rot90(x, k=k, dims=(-2, -1))
    elif test_transform == "hflip":
        k = torch.randint(low=0, high=2, size=(1,)).item()
        if k % 2 == 0:
            pass
        else:
            x = hflip(x)
    else:
        raise NotImplementedError
    return x


def test_multiple_symmetry_equivariance(args):
    args.input_size = 64
    args.batch_size = 1
    args.fusion = True
    args.type='sum'
    logging.basicConfig(format='%(asctime)s === %(message)s', level=logging.INFO)
    B, c, H, W = args.batch_size, 1, args.input_size, args.input_size
    k = len(args.list_test_transform)
    args.in_channels = c

    x = torch.randn(size=(B, k, c, H, W))
    x_transformed = []
    for i in range(k):
        x_transformed.append(transform(x[:, i, :, :, :], test_transform=args.list_test_transform[i]))
    x_transformed = torch.stack(x_transformed, dim=1)

    if "GEMINN" in args.model_name or "Gemini" in args.model_name:
        # only GEMMNN_V2 and GEMMNN_V3 takes a list of symmetry groups
        model = model_dict[args.model_name](args, k=k, num_classes=2, input_size=H, list_symmetry_groups=args.list_symmetry_groups,
                                            type="sum", logging=logging)
    else:
        model = model_dict[args.model_name](args, k=k, num_classes=2, input_size=H, symmetry_group=args.symmetry_group)
    print(f"Loaded {args.model_name} model with {count_parameters(model)} parameters")

    model.eval()  # to turn of dropout
    out = model(x)
    out_transformed = model(x_transformed)  # output from x with transform rot90

    assert torch.allclose(out, out_transformed)  # testing invariance since output has no spatial dimensions


def test_symmetry_equivariance(args):
    B, k, c, H, W = 1, 3, 1, 128, 128
    args.in_channels = c

    x = torch.randn(size=(B, k, c, H, W))
    x_rot90 = transform(x, test_transform=args.test_transform)

    if args.model_name == "GEMMNN_V2":
        # only GEMMNN_V2 takes a list of symmetry groups
        model = model_dict[args.model_name](args, k=k, num_classes=2, input_size=H,
                                            list_symmetry_groups=args.list_symmetry_groups)
    else:
        model = model_dict[args.model_name](args, k=k, num_classes=2, input_size=H, symmetry_group=args.symmetry_group)
    print(f"Loaded {args.model_name} model with {count_parameters(model)} parameters")

    out = model(x)
    out_rot90 = model(x_rot90)  # output from x with transform rot90

    assert torch.allclose(out, out_rot90)  # testing invariance since output has no spatial dimensions


def test_perm(args):
    B, k, c, H, W = 1, 2, 1, 28, 28
    x = torch.randn(size=(B, k, c, H, W))
    x_perm = torch.zeros_like(x)

    args.in_channels = c
    # load models
    if "GEMMNN" in args.model_name:
        # only GEMMNN_V2 takes a list of symmetry groups
        model = model_dict[args.model_name](args, k=k, num_classes=2, input_size=H,
                                            list_symmetry_groups=args.list_symmetry_groups)
    else:
        model = model_dict[args.model_name](args, k=k, num_classes=2, input_size=H, symmetry_group=args.symmetry_group)
    print(f"Loaded {args.model_name} model with {count_parameters(model)} parameters")

    output = model(x)

    x_perm[:, 0, :, :, :], x_perm[:, 1, :, :, :] = x[:, 1, :, :, :], x[:, 0, :, :, :]
    output_perm = model(x_perm)
    assert torch.allclose(output, output_perm)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test equivariance")
    parser.add_argument("--model_name", default="Gemini", type=str,
                        help='choose from ["ConvNet", "SiameseNet", "DeepSet", "DSS", "GEMINN", "Gemini"]')
    parser.add_argument("--conv_channels", default=32, type=int)
    parser.add_argument("--lin_channels", default=128, type=int)
    parser.add_argument("--dropout1", default=0.50, type=float)
    parser.add_argument("--dropout2", default=0.50, type=float)
    parser.add_argument("--symmetry_group", default="rot90", type=str, help="Group for equivariance")
    parser.add_argument("--list_symmetry_groups", nargs='+', default=["rot90", "hflip"], type=str, help="List of groups for equivariance of GEMMNN")
    parser.add_argument("--test_transform", default="rot90", type=str,
                        help="Transform on data for testing")
    parser.add_argument("--list_test_transform", nargs='+', default=["rot90", "hflip"], type=str, help="Transform on data for testing equivariance")
    parser.add_argument("--test_name", default="multi_symmetry", type=str, help="name of the test function. choose from ['perm', 'symmetry']")
    args = parser.parse_args()

    test_dict = {"perm": test_perm,
                 "symmetry": test_symmetry_equivariance,
                 "multi_symmetry": test_multiple_symmetry_equivariance}

    test_dict[args.test_name](args)
