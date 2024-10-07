import json

import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def plot_exp0(dataset_name="Imagenet V2", save_fig=False):
    plt_name = "exp0_" + "".join(dataset_name.split(" "))
    fig = plt.figure()
    x_original = np.array([1, 5.5, 10, 14.5])
    x_flip = x_original + 1
    x_rot90 = x_original + 2
    x_rot90_flip = x_original + 3

    plt.xticks(np.array([2.5, 7, 11.5, 16]), ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16'], fontsize=15, rotation=0)
    plt.yticks(fontsize=15)

    with open("exp0_results.json", "r") as readfile:
        exp0_results = json.load(readfile)

    dataset_name = "".join(dataset_name.split(" "))
    y_original = np.array([
        exp0_results[dataset_name][""]["RN50"]["vanilla"],
        exp0_results[dataset_name][""]["RN101"]["vanilla"],
        exp0_results[dataset_name][""]["ViT-B/32"]["vanilla"],
        exp0_results[dataset_name][""]["ViT-B/16"]["vanilla"]
    ])

    y_flip = np.array([
        exp0_results[dataset_name]["flip"]["RN50"]["vanilla"],
        exp0_results[dataset_name]["flip"]["RN101"]["vanilla"],
        exp0_results[dataset_name]["flip"]["ViT-B/32"]["vanilla"],
        exp0_results[dataset_name]["flip"]["ViT-B/16"]["vanilla"]
    ])

    y_rot90 = np.array([
        exp0_results[dataset_name]["rot90"]["RN50"]["vanilla"],
        exp0_results[dataset_name]["rot90"]["RN101"]["vanilla"],
        exp0_results[dataset_name]["rot90"]["ViT-B/32"]["vanilla"],
        exp0_results[dataset_name]["rot90"]["ViT-B/16"]["vanilla"]
    ])

    y_rot90_flip = np.array([
        exp0_results[dataset_name]["rot90_flip"]["RN50"]["vanilla"],
        exp0_results[dataset_name]["rot90_flip"]["RN101"]["vanilla"],
        exp0_results[dataset_name]["rot90_flip"]["ViT-B/32"]["vanilla"],
        exp0_results[dataset_name]["rot90_flip"]["ViT-B/16"]["vanilla"]
    ])

    plt.bar(x_original, height=y_original, capsize=30, label="No transformation")
    plt.bar(x_flip, height=y_flip, capsize=30, label="Random flips")
    plt.bar(x_rot90, height=y_rot90, capsize=30, label="Random $90^{\circ}$ rotations")
    plt.bar(x_rot90_flip, height=y_rot90_flip, capsize=30, label="Random $90^{\circ}$ rotations and flips")

    plt.ylabel(ylabel="Top-$1$ accuracy", fontsize=15)
    plt.ylim([35, 65])

    plt.legend(prop={'size': 12})
    plt.title(f"{dataset_name} top-1 accuracies", fontsize=15)
    plt.tight_layout()
    plt.show()
    if save_fig:
        fig.savefig(f"{plt_name}.png", dpi=150)
    return


def plot_0_imagenet(save_fig=False):
    fig = plt.figure()
    x_original = np.array([1, 4.5, 8, 11.5])
    x_flip = x_original + 1
    x_rot90 = x_original + 2

    plt.xticks(np.array([2, 5.5, 9, 12.5]), ['RN50', 'RN101', 'ViT-B/32', 'ViT-B/16'], fontsize=15, rotation=0)
    plt.yticks(fontsize=15)

    y_original = np.array([52.84, 56.16, 55.89, 61.89])
    y_flip = np.array([42.91, 45.97, 46.69, 53.56])
    y_rot90 = np.array([39.61, 43.53, 44.61, 52.78])

    # model_size = np.array([11.84, 11.84, 46.83])
    plt.bar(x_original, height=y_original, capsize=30, label="No transformation")
    plt.bar(x_flip, height=y_flip, capsize=30, label="Random flips")
    plt.bar(x_rot90, height=y_rot90, capsize=30, label="Random $90^{\circ}$ rotations")

    # plt.ylabel(ylabel="Inference time (seconds)", fontsize=15)
    plt.ylabel(ylabel="Top-$1$ accuracy", fontsize=15)
    # plt.ylabel(ylabel="Model size (MB)", fontsize=15)
    plt.ylim([35, 65])

    plt.legend(prop={'size': 12})
    plt.title("Imagenet V2 top-1 accuracies", fontsize=15)
    plt.tight_layout()
    plt.show()
    if save_fig:
        # fig.savefig("inf_time.png", dpi=150)
        fig.savefig("plot_0_imagenet.png", dpi=150)
        # fig.savefig("model_size.png", dpi=150)
    return


if __name__ == "__main__":
    plot_exp0(dataset_name="Imagenet V2", save_fig=False)
