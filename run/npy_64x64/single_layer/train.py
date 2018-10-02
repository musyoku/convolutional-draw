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
    sys.stdout.flush()


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
    return np.uint8(np.clip(x * 255, 0, 255))


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
        image = image / 255
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

    hyperparams = HyperParameters()
    hyperparams.chz_channels = args.chz_channels
    hyperparams.generator_generation_steps = args.generation_steps
    hyperparams.generator_share_core = args.generator_share_core
    hyperparams.generator_share_prior = args.generator_share_prior
    hyperparams.generator_share_upsampler = args.generator_share_upsampler
    hyperparams.generator_downsampler_channels = args.generator_downsampler_channels
    hyperparams.inference_share_core = args.inference_share_core
    hyperparams.inference_share_posterior = args.inference_share_posterior
    hyperparams.inference_downsampler_channels = args.inference_downsampler_channels
    hyperparams.batch_normalization_enabled = args.enable_batch_normalization
    hyperparams.use_gru = args.use_gru
    hyperparams.no_backprop_diff_xr = args.no_backprop_diff_xr

    hyperparams.save(args.snapshot_directory)
    hyperparams.print()

    if args.use_gru:
        model = GRUModel(
            hyperparams, snapshot_directory=args.snapshot_directory)
    else:
        model = LSTMModel(
            hyperparams, snapshot_directory=args.snapshot_directory)
    if using_gpu:
        model.to_gpu()

    optimizer = AdamOptimizer(
        model.parameters,
        lr_i=args.initial_lr,
        lr_f=args.final_lr,
        beta_1=args.adam_beta1,
    )
    optimizer.print()

    num_pixels = images.shape[1] * images.shape[2] * images.shape[3]

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
            x = dataset[data_indices]
            x += np.random.uniform(0, 1 / 256, size=x.shape)
            x = to_gpu(x)

            loss_kld = 0
            z_t_param_array, x_param = model.sample_z_and_x_params_from_posterior(
                x)
            for params in z_t_param_array:
                mean_z_q, ln_var_z_q, mean_z_p, ln_var_z_p = params
                kld = draw.nn.functions.gaussian_kl_divergence(
                    mean_z_q, ln_var_z_q, mean_z_p, ln_var_z_p)
                loss_kld += cf.sum(kld)

            mu_x, ln_var_x = x_param

            loss_nll = cf.gaussian_nll(x, mu_x, ln_var_x) + math.log(256.0)
            loss_mse = cf.mean_squared_error(mu_x, x)

            loss_nll /= args.batch_size
            loss_kld /= args.batch_size
            loss = args.loss_beta * loss_nll + loss_kld

            model.cleargrads()
            loss.backward()
            optimizer.update(num_updates)

            num_updates += 1
            mean_kld += float(loss_kld.data)
            mean_nll += float(loss_nll.data)

            if batch_index % 10 == 0:
                with chainer.no_backprop_mode():
                    axis_1.imshow(make_uint8(x[0]))
                    axis_2.imshow(make_uint8(mu_x.data[0]))

                    x_dev = images_dev[random.choice(range(num_dev_images))]
                    axis_3.imshow(make_uint8(x_dev))

                    x_dev = to_gpu(x_dev)[None, ...]
                    r_t_array, x_param = model.sample_image_at_each_step_from_posterior(
                        x_dev)
                    mu_x, ln_var_x = x_param
                    axis_4.imshow(make_uint8(mu_x.data[0]))

                    r_t_array, x_param = model.sample_image_at_each_step_from_prior(
                        batch_size=1, xp=xp)
                    mu_x, ln_var_x = x_param
                    axis_5.imshow(make_uint8(mu_x.data[0]))

                    plt.pause(0.01)

            printr(
                "Iteration {}: Batch {} / {} - loss: nll_per_pixel: {:.6f} - mse: {:.6f} - kld: {:.6f} - lr: {:.4e}".
                format(iteration + 1, batch_index + 1, len(iterator),
                       float(loss_nll.data) / num_pixels, float(loss_mse.data),
                       float(loss_kld.data), optimizer.learning_rate))

        model.serialize(args.snapshot_directory)
        print(
            "\r\033[2KIteration {} - loss: nll_per_pixel: {:.6f} - mse: {:.6f} - kld: {:.6f} - lr: {:.4e}".
            format(iteration + 1,
                   float(loss_nll.data) / num_pixels, float(loss_mse.data),
                   float(loss_kld.data), optimizer.learning_rate))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-dataset", type=str, required=True)
    parser.add_argument(
        "--snapshot-directory", "-snapshot", type=str, default="snapshot")
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    parser.add_argument("--batch-size", "-b", type=int, default=36)
    parser.add_argument("--training-steps", type=int, default=1000000)
    parser.add_argument("--generation-steps", "-gsteps", type=int, default=32)
    parser.add_argument("--initial-lr", "-lr-i", type=float, default=0.0001)
    parser.add_argument("--final-lr", "-lr-f", type=float, default=0.00001)
    parser.add_argument("--adam-beta1", "-beta1", type=float, default=0.5)
    parser.add_argument("--loss-beta", "-lbeta", type=float, default=1.0)
    parser.add_argument("--chz-channels", "-cz", type=int, default=64)
    parser.add_argument(
        "--inference-downsampler-channels", "-cix", type=int, default=12)
    parser.add_argument(
        "--generator-downsampler-channels", "-cgx", type=int, default=12)
    parser.add_argument(
        "--generator-share-core", "-g-share-core", action="store_true")
    parser.add_argument(
        "--generator-share-prior", "-g-share-prior", action="store_true")
    parser.add_argument(
        "--generator-share-upsampler",
        "-g-share-upsampler",
        action="store_true")
    parser.add_argument(
        "--inference-share-core", "-i-share-core", action="store_true")
    parser.add_argument(
        "--inference-share-posterior",
        "-i-share-posterior",
        action="store_true")
    parser.add_argument(
        "--enable-batch-normalization", "-bn", action="store_true")
    parser.add_argument("--use-gru", "-gru", action="store_true")
    parser.add_argument(
        "--no-backprop-diff-xr", "-no-xr-grad", action="store_true")
    args = parser.parse_args()
    main()
