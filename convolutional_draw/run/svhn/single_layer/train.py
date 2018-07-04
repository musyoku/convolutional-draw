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
            rec_0 = xp.zeros(
                (
                    args.batch_size,
                    3,
                ) + hyperparams.image_size,
                dtype="float32")
            r_0 = xp.zeros(
                (
                    args.batch_size,
                    3 + 3,
                ) + hyperparams.image_size,
                dtype="float32")
            hd_0 = xp.zeros(
                (
                    args.batch_size,
                    hyperparams.channels_chz,
                ) + hyperparams.chrz_size,
                dtype="float32")
            cd_0 = xp.copy(hd_0)
            he_0 = xp.copy(hd_0)
            ce_0 = xp.copy(hd_0)

            x = dataset[data_indices]
            x = to_gpu(x)

            print(x.shape, rec_0.shape)
            
            rec_t = rec_0
            r_t = r_0
            hd_t = hd_0
            cd_t = hd_0
            he_t = hd_0
            ce_t = hd_0

            loss_kld = 0

            for t in range(args.generation_steps):
                error = x - rec_t
                he_next, ce_next = model.inference_network.forward_onestep(
                    ce_t, x, error, he_t, hd_t)

                ze_mean = model.inference_network.compute_mean_z(he_next)
                ze_ln_var = model.inference_network.compute_ln_var_z(he_next)
                ze = cf.gaussian(ze_mean, ze_ln_var)

                zd_mean = model.generation_network.compute_mean_z(hd_t)
                zd_ln_var = model.generation_network.compute_ln_var_z(hd_t)

                hd_next, cd_next, r_next = model.generation_network.forward_onestep(
                    cd_t, ze, hd_t, r_t)

                rec_next = model.generation_network.sample_x(r_next)

                kld = draw.nn.chainer.functions.gaussian_kl_divergence(
                    ze_mean, ze_ln_var, zd_mean, zd_ln_var)
                loss_kld += cf.sum(kld)

                rec_t = rec_next
                r_t = r_next
                hd_t = hd_next
                cd_t = cd_next
                he_t = he_next
                ce_t = ce_next
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-path", type=str, default="snapshot")
    parser.add_argument("--batch-size", "-b", type=int, default=36)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--training-steps", type=int, default=10**6)
    parser.add_argument("--generation-steps", "-gen", type=int, default=32)
    args = parser.parse_args()
    main()
