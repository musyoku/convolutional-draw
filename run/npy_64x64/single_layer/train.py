import argparse
import math
import os
import random
import sys

import chainer
import chainer.functions as cf
import cupy
import matplotlib.pyplot as plt
import numpy as np
from chainer.backends import cuda
from PIL import Image

sys.path.append(os.path.join("..", "..", ".."))
import draw
from hyperparams import HyperParameters
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
        os.mkdir(args.snapshot_directory)
    except:
        pass

    images = []
    files = os.listdir(args.dataset_path)
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
        xp = cupy

    hyperparams = HyperParameters()
    hyperparams.generator_share_core = args.generator_share_core
    hyperparams.generator_share_prior = args.generator_share_prior
    hyperparams.generator_generation_steps = args.generation_steps
    hyperparams.inference_share_core = args.inference_share_core
    hyperparams.inference_share_posterior = args.inference_share_posterior
    hyperparams.pixel_n = args.pixel_n
    hyperparams.channels_chz = args.channels_chz
    hyperparams.inference_channels_map_x = args.channels_map_x
    hyperparams.pixel_sigma_i = args.initial_pixel_sigma
    hyperparams.pixel_sigma_f = args.final_pixel_sigma
    hyperparams.save(args.snapshot_directory)
    hyperparams.print()

    model = Model(hyperparams, snapshot_directory=args.snapshot_directory)
    if using_gpu:
        model.to_gpu()

    optimizer = Optimizer(
        model.parameters, mu_i=args.initial_lr, mu_f=args.final_lr)
    optimizer.print()

    sigma_t = hyperparams.pixel_sigma_i
    pixel_var = xp.full(
        (args.batch_size, 3) + hyperparams.image_size,
        sigma_t**2,
        dtype="float32")
    pixel_ln_var = xp.full(
        (args.batch_size, 3) + hyperparams.image_size,
        math.log(sigma_t**2),
        dtype="float32")

    dataset = draw.data.Dataset(images_train)
    iterator = draw.data.Iterator(dataset, batch_size=args.batch_size)

    figure = plt.figure(figsize=(20, 4))
    axis_1 = figure.add_subplot(1, 5, 1)
    axis_2 = figure.add_subplot(1, 5, 2)
    axis_3 = figure.add_subplot(1, 5, 3)
    axis_4 = figure.add_subplot(1, 5, 4)
    axis_5 = figure.add_subplot(1, 5, 5)

    num_updates = 0

    for iteration in range(args.training_steps):
        mean_kld = 0
        mean_nll = 0

        for batch_index, data_indices in enumerate(iterator):
            h0_gen, c0_gen, r0, h0_enc, c0_enc = model.generate_initial_state(
                args.batch_size, xp)

            x = dataset[data_indices]
            x = to_gpu(x)

            h_l_enc = h0_enc
            c_l_enc = c0_enc
            h_l_gen = h0_gen
            c_l_gen = c0_gen
            r_l = chainer.Variable(r0)

            loss_kld = 0

            for l in range(model.generation_steps):
                inference_core = model.get_inference_core(l)
                inference_posterior = model.get_inference_posterior(l)
                generation_core = model.get_generation_core(l)
                generation_piror = model.get_generation_prior(l)

                diff_xr = x - r_l
                diff_xr.unchain_backward()

                downsampled_x = model.inference_downsampler.downsample(x)
                diff_xr_d = model.inference_downsampler.downsample(diff_xr)

                h_next_enc, c_next_enc = inference_core.forward_onestep(
                    h_l_gen, h_l_enc, c_l_enc, downsampled_x, diff_xr_d)

                mean_z_q = inference_posterior.compute_mean_z(h_l_enc)
                ln_var_z_q = inference_posterior.compute_ln_var_z(h_l_enc)
                ze_l = cf.gaussian(mean_z_q, ln_var_z_q)

                mean_z_p = generation_piror.compute_mean_z(h_l_gen)
                ln_var_z_p = generation_piror.compute_ln_var_z(h_l_gen)

                downsampled_r_l = model.inference_downsampler.downsample(r_l)
                h_next_gen, c_next_gen, r_next_gen = generation_core.forward_onestep(
                    h_l_gen, c_l_gen, ze_l, r_l, downsampled_r_l)

                kld = draw.nn.functions.gaussian_kl_divergence(
                    mean_z_q, ln_var_z_q, mean_z_p, ln_var_z_p)

                loss_kld += cf.sum(kld)

                h_l_gen = h_next_gen
                c_l_gen = c_next_gen
                r_l = r_next_gen
                h_l_enc = h_next_enc
                c_l_enc = c_next_enc

            mean_x_e = r_l
            negative_log_likelihood = draw.nn.functions.gaussian_negative_log_likelihood(
                x, mean_x_e, pixel_var, pixel_ln_var)
            loss_nll = cf.sum(negative_log_likelihood)

            loss_nll /= args.batch_size
            loss_kld /= args.batch_size
            loss = loss_nll + loss_kld
            model.cleargrads()
            loss.backward()
            optimizer.update(num_updates)

            if batch_index % 10 == 0:
                axis_1.imshow(
                    np.uint8(
                        np.clip(
                            (to_cpu(x[0].transpose(1, 2, 0)) + 1) * 0.5 * 255,
                            0, 255)))

                axis_2.imshow(
                    np.uint8(
                        np.clip(
                            (to_cpu(mean_x_e.data[0].transpose(1, 2, 0)) + 1) *
                            0.5 * 255, 0, 255)))

                x_dev = images_dev[random.choice(range(num_dev_images))]
                axis_3.imshow(
                    np.uint8((x_dev.transpose(1, 2, 0) + 1) * 0.5 * 255))

                with chainer.using_config("train",
                                          False), chainer.using_config(
                                              "enable_backprop", False):
                    x_dev = to_gpu(x_dev)[None, ...]
                    r0 = xp.zeros(
                        (
                            1,
                            3,
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
                    ud_t = r0
                    hd_t = hd_0
                    cd_t = cd_0
                    he_t = he_0
                    ce_t = ce_0

                    for t in range(args.generation_steps):
                        he_next, ce_next = model.inference_network.forward_onestep(
                            ce_t, he_t, hd_t, x_dev)

                        ze_t = model.inference_network.sample_z(he_t)

                        hd_next, cd_next, ud_next = model.generation_network.forward_onestep(
                            cd_t, hd_t, ze_t, ud_t)

                        ud_t = ud_next
                        hd_t = hd_next
                        cd_t = cd_next
                        he_t = he_next
                        ce_t = ce_next

                    mean_x_e = model.generation_network.compute_mean_x(ud_t)
                    axis_4.imshow(
                        np.uint8(
                            np.clip((to_cpu(mean_x_e.data[0].transpose(
                                1, 2, 0)) + 1) * 0.5 * 255, 0, 255)))

                    ud_t = r0
                    hd_t = hd_0
                    cd_t = cd_0

                    for t in range(args.generation_steps):
                        zd_t = model.generation_network.sample_z(hd_t)

                        hd_next, cd_next, ud_next = model.generation_network.forward_onestep(
                            cd_t, hd_t, zd_t, ud_t)

                        ud_t = ud_next
                        hd_t = hd_next
                        cd_t = cd_next

                    mean_x_d = model.generation_network.compute_mean_x(ud_t)
                    axis_5.imshow(
                        np.uint8(
                            np.clip((to_cpu(mean_x_d.data[0].transpose(
                                1, 2, 0)) + 1) * 0.5 * 255, 0, 255)))

                plt.pause(0.01)

            num_updates += 1
            mean_kld += float(loss_kld.data)
            mean_nll += float(loss_nll.data)

            sf = hyperparams.pixel_sigma_f
            si = hyperparams.pixel_sigma_i
            sigma_t = max(
                sf + (si - sf) * (1.0 - num_updates / hyperparams.pixel_n), sf)

            pixel_var[...] = sigma_t**2
            pixel_ln_var[...] = math.log(sigma_t**2)

            printr(
                "Iteration {}: Batch {} / {} - loss: nll: {:.3f} kld: {:.3f} - lr: {:.4e} - sigma_t: {:.6f}".
                format(iteration + 1, batch_index + 1, len(iterator),
                       float(loss_nll.data), float(loss_kld.data),
                       optimizer.learning_rate, sigma_t))

        model.serialize(args.snapshot_directory)
        print(
            "\033[2KIteration {} - loss: nll: {:.3f} kld: {:.3f} - lr: {:.4e} - updates: {} - sigma_t: {:.6f}".
            format(iteration + 1, mean_nll / len(iterator),
                   mean_kld / len(iterator), optimizer.learning_rate,
                   num_updates, sigma_t))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-dataset", type=str, required=True)
    parser.add_argument(
        "--snapshot-directory", "-snapshot", type=str, default="snapshot")
    parser.add_argument("--batch-size", "-b", type=int, default=36)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--training-steps", type=int, default=10**6)
    parser.add_argument("--generation-steps", "-gsteps", type=int, default=8)
    parser.add_argument(
        "--initial-lr", "-mu-i", type=float, default=5.0 * 1e-4)
    parser.add_argument("--final-lr", "-mu-f", type=float, default=5.0 * 1e-5)
    parser.add_argument(
        "--initial-pixel-sigma", "-ps-i", type=float, default=2.0)
    parser.add_argument(
        "--final-pixel-sigma", "-ps-f", type=float, default=0.7)
    parser.add_argument("--pixel-n", "-pn", type=int, default=2 * 10**5)
    parser.add_argument("--channels-chz", "-cz", type=int, default=64)
    parser.add_argument("--channels-u", "-cu", type=int, default=128)
    parser.add_argument("--channels-map-x", "-cx", type=int, default=64)
    parser.add_argument(
        "--generator-share-core", "-g-share-core", action="store_true")
    parser.add_argument(
        "--generator-share-prior", "-g-share-prior", action="store_true")
    parser.add_argument(
        "--inference-share-core", "-i-share-core", action="store_true")
    parser.add_argument(
        "--inference-share-posterior",
        "-i-share-posterior",
        action="store_true")
    args = parser.parse_args()
    main()
