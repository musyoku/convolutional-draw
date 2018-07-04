import argparse
import os
import sys

import chainer

sys.path.append(os.path.join("..", ".."))
import draw


def main():
    np_svhn_train, np_svhn_test = chainer.datasets.get_svhn(withlabel=False)

    # [0, 1] -> [-1, 1]
    svhn_train = np_svhn_train * 2.0 - 1.0
    svhn_test = np_svhn_test * 2.0 - 1.0

    dataset = draw.data.Dataset(svhn_train)
    iterator = draw.data.Iterator(dataset, batch_size=args.batch_size)

    print(args.training_steps)
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
