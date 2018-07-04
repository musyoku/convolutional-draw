import argparse
import os
import sys

import chainer
import chainer.functions as cf
import cupy
import numpy as np
from chainer.backends import cuda

sys.path.append(os.path.join("..", "..", ".."))
import draw

from hyper_parameters import HyperParameters
from optimizer import Optimizer
from model import Model


def printr(string):
    sys.stdout.write(string)
    sys.stdout.write("\r")


def to_gpu(array):
    if args.gpu_device >= 0:
        return cuda.to_gpu(array)
    return array


def to_cpu(array):
    if args.gpu_device >= 0:
        return cuda.to_cpu(array)
    return array


def main():
    np_svhn_train, np_svhn_test = chainer.datasets.get_svhn(withlabel=False)

    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cupy

    # [0, 1] -> [-1, 1]
    svhn_train = np_svhn_train * 2.0 - 1.0
    svhn_test = np_svhn_test * 2.0 - 1.0

    hyperparams = HyperParameters()
    model = Model(hyperparams, hdf5_path=args.snapshot_path)
    if using_gpu:
        model.to_gpu()
    optimizer = Optimizer(model.parameters)

    dataset = draw.data.Dataset(svhn_train)
    iterator = draw.data.Iterator(dataset, batch_size=args.batch_size)

    for iteration in range(args.training_steps):
        for batch_index, data_indices in enumerate(iterator):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-path", type=str, default="snapshot")
    parser.add_argument("--batch-size", "-b", type=int, default=36)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--training-steps", type=int, default=10**6)
    parser.add_argument("--generation-steps", "-gen", type=int, default=32)
    args = parser.parse_args()
    main()
