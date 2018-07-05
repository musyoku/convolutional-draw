import argparse
import os
import sys
import random

import chainer
import chainer.functions as cf
import cupy
import numpy as np
from chainer.backends import cuda
from PIL import Image

sys.path.append(os.path.join("..", "..", ".."))
import draw

from hyper_parameters import HyperParameters
from model import Model
from optimizer import Optimizer


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

    images = []
    files = os.listdir(args.dataset_path)
    for filename in files:
        image = np.array(
            Image.open(os.path.join(args.dataset_path, filename)),
            dtype="float32")
        image = image.transpose((2, 0, 1))
        image = image / 255 * 2.0 - 1.0
        images.append(image)

    images = np.asanyarray(images)
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
        xp = cupy

    figure = draw.imgplot.figure()
    axis1 = draw.imgplot.image()
    axis2 = draw.imgplot.image()
    axis3 = draw.imgplot.image()
    axis4 = draw.imgplot.image()
    axis5 = draw.imgplot.image()
    figure.add(axis1, 0, 0, 0.2, 1)
    figure.add(axis2, 0.2, 0, 0.2, 1)
    figure.add(axis3, 0.4, 0, 0.2, 1)
    figure.add(axis4, 0.6, 0, 0.2, 1)
    figure.add(axis5, 0.8, 0, 0.2, 1)
    window = draw.imgplot.window(figure, (192 * 5, 192),
                                 "Training Progression")
    window.show()

    hyperparams = HyperParameters()
    model = Model(hyperparams, hdf5_path=args.snapshot_path)
    if using_gpu:
        model.to_gpu()
    optimizer = Optimizer(model.parameters)

    dataset = draw.data.Dataset(images_train)
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

            # beta = float(xp.mean(xp.exp(rec_ln_var.data)))
            beta = 0.4

            loss_nll /= args.batch_size
            loss_kld /= args.batch_size
            loss = beta * loss_nll + loss_kld
            model.cleargrads()
            loss.backward()
            optimizer.update(num_updates)

            if window.closed() is False and batch_index % 10 == 0:
                axis1.update(
                    np.uint8(
                        (to_cpu(x[0].transpose(1, 2, 0)) + 1) * 0.5 * 255))

                axis2.update(
                    np.uint8((to_cpu(r_t.data[0, :3].transpose(1, 2, 0)) + 1) *
                             0.5 * 255))

                x_dev = images_dev[random.choice(range(num_dev_images))]
                axis3.update(
                    np.uint8((x_dev.transpose(1, 2, 0) + 1) * 0.5 * 255))

                with chainer.using_config("train",
                                          False), chainer.using_config(
                                              "enable_backprop", False):
                    x_dev = to_gpu(x_dev)[None, ...]
                    rec_0 = xp.zeros(
                        (
                            1,
                            3,
                        ) + hyperparams.image_size, dtype="float32")
                    r_0 = xp.zeros(
                        (
                            1,
                            3 + 3,
                        ) + hyperparams.image_size, dtype="float32")
                    hd_0 = xp.zeros(
                        (
                            1,
                            hyperparams.channels_chz,
                        ) + hyperparams.chrz_size,
                        dtype="float32")
                    cd_0 = xp.copy(hd_0)
                    he_0 = xp.copy(hd_0)
                    ce_0 = xp.copy(hd_0)
                    rec_t = rec_0
                    r_t = r_0
                    hd_t = hd_0
                    cd_t = cd_0
                    he_t = he_0
                    ce_t = ce_0

                    for t in range(args.generation_steps):
                        error = x_dev - rec_t
                        he_next, ce_next = model.inference_network.forward_onestep(
                            ce_t, x_dev, error, he_t, hd_t)

                        ze = model.inference_network.sample_z(he_next)

                        hd_next, cd_next, r_next = model.generation_network.forward_onestep(
                            cd_t, ze, hd_t, r_t)

                        rec_next = model.generation_network.sample_x(r_next)

                        rec_t = rec_next
                        r_t = r_next
                        hd_t = hd_next
                        cd_t = cd_next
                        he_t = he_next
                        ce_t = ce_next

                    axis4.update(
                        np.uint8((to_cpu(r_t.data[0, :3].transpose(1, 2, 0)) + 1)
                                 * 0.5 * 255))

                    rec_t = rec_0
                    r_t = r_0
                    hd_t = hd_0
                    cd_t = cd_0

                    for t in range(args.generation_steps):
                        error = x_dev - rec_t
                        zd = model.generation_network.sample_z(hd_t)

                        hd_next, cd_next, r_next = model.generation_network.forward_onestep(
                            cd_t, zd, hd_t, r_t)

                        rec_next = model.generation_network.sample_x(r_next)

                        rec_t = rec_next
                        r_t = r_next
                        hd_t = hd_next
                        cd_t = cd_next

                    axis5.update(
                        np.uint8((to_cpu(r_t.data[0, :3].transpose(1, 2, 0)) + 1)
                                 * 0.5 * 255))
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
    parser.add_argument("--dataset-path", "-dataset", type=str, required=True)
    parser.add_argument("--snapshot-path", type=str, default="snapshot")
    parser.add_argument("--batch-size", "-b", type=int, default=36)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--training-steps", type=int, default=10**6)
    parser.add_argument("--generation-steps", "-gen", type=int, default=32)
    args = parser.parse_args()
    main()