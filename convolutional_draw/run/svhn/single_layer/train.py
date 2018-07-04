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
    try:
        os.mkdir(args.snapshot_path)
    except:
        pass
        
    np_svhn_train, np_svhn_test = chainer.datasets.get_svhn(withlabel=False)

    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cupy

    figure = draw.imgplot.figure()
    axis1 = draw.imgplot.image()
    axis2 = draw.imgplot.image()
    figure.add(axis1, 0, 0, 0.5, 1)
    figure.add(axis2, 0.5, 0, 0.5, 1)
    window = draw.imgplot.window(figure, (400 * 2, 400),
                                 "Training Progression")
    window.show()

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

    num_updates = 0

    for iteration in range(args.training_steps):
        mean_kld = 0
        mean_nll = 0

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

            rec_t = rec_0
            r_t = r_0
            hd_t = hd_0
            cd_t = cd_0
            he_t = he_0
            ce_t = ce_0

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

            rec_mean = r_t[:, :3]
            rec_ln_var = r_t[:, 3:]
            negative_log_likelihood = draw.nn.chainer.functions.gaussian_negative_log_likelihood(
                x, rec_mean, rec_ln_var)
            loss_nll = cf.sum(negative_log_likelihood)

            loss_nll /= args.batch_size
            loss_kld /= args.batch_size
            loss = loss_nll + loss_kld
            model.cleargrads()
            loss.backward()
            optimizer.update(num_updates)

            if window.closed() is False:
                with chainer.using_config("train",
                                          False), chainer.using_config(
                                              "enable_backprop", False):
                    reconstructed_x = model.generation_network.sample_x(
                        r_t.data)

                    axis1.update(
                        np.uint8(
                            (to_cpu(x[0].transpose(1, 2, 0)) + 1) * 0.5 * 255))

                    axis2.update(
                        np.uint8((to_cpu(reconstructed_x.data[0].transpose(
                            1, 2, 0)) + 1) * 0.5 * 255))

            printr(
                "Iteration {}: Batch {} / {} - loss: nll: {:.3f} kld: {:.3f} - lr: {:.4e}".
                format(iteration + 1, batch_index + 1, len(iterator),
                       float(loss_nll.data), float(loss_kld.data),
                       optimizer.learning_rate))

            num_updates += 1
            mean_kld += float(loss_kld.data)
            mean_nll += float(loss_nll.data)

        model.serialize(args.snapshot_path)
        print(
            "\033[2KIteration {} - loss: nll: {:.3f} kld: {:.3f} - lr: {:.4e} - updates: {}".
            format(iteration + 1, mean_nll / len(iterator),
                   mean_kld / len(iterator), optimizer.learning_rate,
                   num_updates))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-path", type=str, default="snapshot")
    parser.add_argument("--batch-size", "-b", type=int, default=36)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--training-steps", type=int, default=10**6)
    parser.add_argument("--generation-steps", "-gen", type=int, default=32)
    args = parser.parse_args()
    main()
