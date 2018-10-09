import argparse
import math
import os
import random
import sys
import time
import multiprocessing

import chainer
import chainermn
import chainer.functions as cf
import numpy as np
import cupy as cp
from chainer.backends import cuda
from PIL import Image
from collections import deque

sys.path.append(".")
sys.path.append(os.path.join("..", "..", ".."))
import draw
from hyperparams import HyperParameters
from models import GRUModel, LSTMModel
from optimizer import AdamOptimizer, EveOptimizer


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

    comm = chainermn.create_communicator()
    device = comm.intra_rank
    cuda.get_device(device).use()
    xp = cp

    images = []
    files = os.listdir(args.dataset_path)
    files.sort()
    subset_size = int(math.ceil(len(files) / comm.size))
    files = deque(files)
    files.rotate(-subset_size * comm.rank)
    files = list(files)[:subset_size]
    for filename in files:
        image = np.load(os.path.join(args.dataset_path, filename))
        image = image / 256
        images.append(image)

    print(comm.rank, files)

    images = np.vstack(images)
    images = images.transpose((0, 3, 1, 2)).astype(np.float32)
    train_dev_split = 0.9
    num_images = images.shape[0]
    num_train_images = int(num_images * train_dev_split)
    num_dev_images = num_images - num_train_images
    images_train = images[:num_train_images]

    # To avoid OpenMPI bug
    # multiprocessing.set_start_method("forkserver")
    # p = multiprocessing.Process(target=print, args=("", ))
    # p.start()
    # p.join()

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

    if comm.rank == 0:
        hyperparams.save(args.snapshot_directory)
        hyperparams.print()

    if args.use_gru:
        model = GRUModel(
            hyperparams, snapshot_directory=args.snapshot_directory)
    else:
        model = LSTMModel(
            hyperparams, snapshot_directory=args.snapshot_directory)
    model.to_gpu()

    optimizer = AdamOptimizer(
        model.parameters,
        lr_i=args.initial_lr,
        lr_f=args.final_lr,
        beta_1=args.adam_beta1,
        communicator=comm)
    if comm.rank == 0:
        optimizer.print()

    num_pixels = images.shape[1] * images.shape[2] * images.shape[3]

    dataset = draw.data.Dataset(images_train)
    iterator = draw.data.Iterator(dataset, batch_size=args.batch_size)

    num_updates = 0

    for iteration in range(args.training_steps):
        mean_kld = 0
        mean_nll = 0
        start_time = time.time()

        for batch_index, data_indices in enumerate(iterator):
            x = dataset[data_indices]
            x += np.random.uniform(0, 1 / 256, size=x.shape)
            x = to_gpu(x)

            z_t_param_array, x_param, r_t_array = model.sample_z_and_x_params_from_posterior(
                x)

            loss_kld = 0
            for params in z_t_param_array:
                mean_z_q, ln_var_z_q, mean_z_p, ln_var_z_p = params
                kld = draw.nn.functions.gaussian_kl_divergence(
                    mean_z_q, ln_var_z_q, mean_z_p, ln_var_z_p)
                loss_kld += cf.sum(kld)

            loss_mse = 0
            for r_t in r_t_array:
                loss_mse += cf.sum(cf.squared_error(r_t, x))

            mu_x, ln_var_x = x_param

            loss_nll = cf.gaussian_nll(x, mu_x, ln_var_x)

            loss_nll /= args.batch_size
            loss_kld /= args.batch_size
            loss_mse /= args.batch_size
            loss = args.loss_beta * loss_nll + loss_kld + loss_mse

            model.cleargrads()
            loss.backward(loss_scale=optimizer.loss_scale())
            optimizer.update(num_updates, loss_value=float(loss.array))

            num_updates += 1
            mean_kld += float(loss_kld.data)
            mean_nll += float(loss_nll.data)

            printr(
                "Iteration {}: Batch {} / {} - loss: nll_per_pixel: {:.6f} - mse: {:.6f} - kld: {:.6f} - lr: {:.4e}".
                format(iteration + 1, batch_index + 1, len(iterator),
                       float(loss_nll.data) / num_pixels + math.log(256.0),
                       float(loss_mse.data), float(loss_kld.data),
                       optimizer.learning_rate))

            if comm.rank == 0 and batch_index > 0 and batch_index % 100 == 0:
                model.serialize(args.snapshot_directory)

        if comm.rank == 0:
            model.serialize(args.snapshot_directory)

        if comm.rank == 0:
            elapsed_time = time.time() - start_time
            print(
                "\r\033[2KIteration {} - loss: nll_per_pixel: {:.6f} - mse: {:.6f} - kld: {:.6f} - lr: {:.4e} - elapsed_time: {:.3f} min".
                format(iteration + 1,
                       float(loss_nll.data) / num_pixels + math.log(256.0),
                       float(loss_mse.data), float(loss_kld.data),
                       optimizer.learning_rate, elapsed_time / 60))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", "-dataset", type=str, required=True)
    parser.add_argument(
        "--snapshot-directory", "-snapshot", type=str, default="snapshot")
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
