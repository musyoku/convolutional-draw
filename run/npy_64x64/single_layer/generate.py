import argparse
import math
import os
import random
import sys

import chainer
import chainer.functions as cf
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
from chainer.backends import cuda
from PIL import Image

sys.path.append(os.path.join("..", "..", ".."))
import draw
from hyperparams import HyperParameters
from models import GRUModel, LSTMModel
from optimizer import AdamOptimizer


def printr(string):
    sys.stdout.write(string)
    sys.stdout.write("\r")


def to_gpu(array):
    if cuda.get_array_module(array) is np:
        return cuda.to_gpu(array)
    return array


def to_cpu(array):
    if cuda.get_array_module(array) is cp:
        return cuda.to_cpu(array)
    return array


def make_uint8(x):
    x = to_cpu(x)
    if x.shape[0] == 3:
        x = x.transpose(1, 2, 0)
    return np.uint8(np.clip((x + 1) * 0.5 * 255, 0, 255))


def main():
    try:
        os.mkdir(args.snapshot_directory)
    except:
        pass

    images = []
    files = os.listdir(args.dataset_path)
    files.sort()
    for filename in files:
        image = np.load(os.path.join(args.dataset_path, filename))
        image = image / 255 * 2.0 - 1.0
        images.append(image)

    images = np.vstack(images)
    images = images.transpose((0, 3, 1, 2)).astype(np.float32)
    train_dev_split = 0.9
    num_images = images.shape[0]
    num_train_images = int(num_images * train_dev_split)
    num_dev_images = num_images - num_train_images
    images_train = images[:num_train_images]
    images_dev = images[num_dev_images:]

    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cp

    hyperparams = HyperParameters(snapshot_directory=args.snapshot_directory)
    hyperparams.print()

    if hyperparams.use_gru:
        model = GRUModel(
            hyperparams, snapshot_directory=args.snapshot_directory)
    else:
        model = LSTMModel(
            hyperparams, snapshot_directory=args.snapshot_directory)
    if using_gpu:
        model.to_gpu()

    dataset = draw.data.Dataset(images_dev)
    iterator = draw.data.Iterator(dataset, batch_size=1)

    figure = plt.figure(figsize=(12, 4))
    axis_1 = figure.add_subplot(1, 3, 1)
    axis_1.set_title("Data")
    axis_2 = figure.add_subplot(1, 3, 2)
    axis_2.set_title("Reconstruction")
    axis_3 = figure.add_subplot(1, 3, 3)
    axis_3.set_title("Generation")

    for batch_index, data_indices in enumerate(iterator):

        with chainer.using_config("train", False), chainer.using_config(
                "enable_backprop", False):
            x = dataset[data_indices]
            x = to_gpu(x)
            axis_1.imshow(make_uint8(x[0]))
            _, r_final = model.generate_z_params_and_x_from_posterior(x)
            mean_x_enc = r_final
            axis_2.imshow(make_uint8(mean_x_enc.data[0]))

            mean_x_d = model.generate_image(batch_size=1, xp=xp)
            axis_3.imshow(make_uint8(mean_x_d[0]))

            plt.pause(0.01)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-dataset", type=str, required=True)
    parser.add_argument(
        "--snapshot-directory", "-snapshot", type=str, required=True)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    args = parser.parse_args()
    main()
