import os
import json
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import clip
import argparse
import pytorch_lightning as pl

from tqdm import tqdm
from pkg_resources import packaging

from load_model import load_model
from dataset_utils import imagenet_classes, imagenet_templates, get_labels_textprompts, get_dataloader
from zeroshot_weights import zeroshot_classifier
from eval_utils import eval_clip

print("Torch version:", torch.__version__)


def main(args):

    # load model and preprocess
    model, preprocess = load_model(args)

    # get labels and text prompts
    classnames, templates = get_labels_textprompts(args)

    # get dataloader
    dataloader = get_dataloader(args, preprocess)

    # create text weights for different classes
    zeroshot_weights = zeroshot_classifier(args, model, classnames, templates, save_weights='True')

    # zeroshot prediction
    import time
    st_time = time.time()
    top1_acc = eval_clip(args, model, zeroshot_weights, dataloader, data_transformations=args.data_transformations,
              group_name=args.group_name)
    end_time = time.time()
    print(f"time taken: {end_time - st_time}")
    return top1_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--logit_factor", default=1., type=float)
    parser.add_argument("--data_transformations", default="", type=str, help=["", "flip", "rot90", "rot90_flip"])
    parser.add_argument("--group_name", default="", type=str, help=["", "flip", "rot90", "rot90_flip"])
    parser.add_argument("--method", default="vanilla", type=str, help=["vanilla", "equitune", "equizero",
                                                                       "multi_equitune", "multi_equizero"])
    parser.add_argument("--model_name", default="RN50", type=str, help=['RN50', 'RN101', 'RN50x4', 'RN50x16',
                                                                        'RN50x64', 'ViT-B/32', 'ViT-B/16',
                                                                        'ViT-L/14', 'ViT-L/14@336px'])
    parser.add_argument("--dataset_name", default="ImagenetV2", type=str, help=["ImagenetV2", "CIFAR100"])
    parser.add_argument("--exp_name", default="debug", type=str, help=["debug", "exp0", "exp1", "exp2"])
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()

    args.verbose = True

    pl.seed_everything(args.seed)

    if args.exp_name == "debug":
        top1_acc = main(args)
    elif args.exp_name == "exp0":
        exp0_results = {}
        for dataset_name in ["ImagenetV2", "CIFAR100"]:
            exp0_results[dataset_name] = {}
            for data_transformations in ["rot90_flip", "rot90", "flip", ""]:
                exp0_results[dataset_name][data_transformations] = {}
                for model_name in ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16']:
                    exp0_results[dataset_name][data_transformations][model_name] = {}

                    method = "vanilla"
                    args.method = method
                    args.dataset_name = dataset_name
                    args.data_transformations = data_transformations
                    args.model_name = model_name

                    top1_acc = main(args)

                    exp0_results[dataset_name][data_transformations][model_name][method] = top1_acc

                    with open("exp0_results.json", "w") as outfile:
                        json.dump(exp0_results, outfile, indent=4)
    elif args.exp_name == "exp1":
        pass
        exp1_results = {}
        for dataset_name in ["ImagenetV2", "CIFAR100"]:
            exp1_results[dataset_name] = {}
            for data_transformations in ["rot90_flip"]:
                exp1_results[dataset_name][data_transformations] = {}
                for model_name in ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16']:
                    exp1_results[dataset_name][data_transformations][model_name] = {}
                    for group_name in ["rot90_flip"]:
                        exp1_results[dataset_name][data_transformations][model_name][group_name] = {}
                        for method in ["equitune", "equizero", "multi_equitune", "multi_equizero"]:

                            args.dataset_name = dataset_name
                            args.data_transformations = data_transformations
                            args.model_name = model_name
                            args.group_name = group_name
                            args.method = method

                            top1_acc = main(args)

                            exp1_results[dataset_name][data_transformations][model_name][group_name][method] = top1_acc
                            with open("exp1_results.json", "w") as outfile:
                                json.dump(exp1_results, outfile, indent=4)

    elif args.exp_name == "exp2":
        raise NotImplementedError
    else:
        raise NotImplementedError



