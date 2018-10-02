import math

import chainermn
from chainer import optimizers
from chainer.optimizer_hooks import GradientClipping
from tabulate import tabulate
from eve import Eve


class Optimizer:
    def __init__(
            self,
            # Learning rate at training step s with annealing
            lr_i=1.0 * 1e-4,
            lr_f=1.0 * 1e-5,
            n=10000):
        self.lr_i = lr_i
        self.lr_f = lr_f
        self.n = n
        self.optimizer = None
        self.multi_node_optimizer = None

    def mu_s(self, training_step):
        return max(
            self.lr_f +
            (self.lr_i - self.lr_f) * (1.0 - training_step / self.n),
            self.lr_f)

    def anneal_learning_rate(self, training_step):
        raise NotImplementedError

    def update(self, training_step, *args, **kwds):
        if self.multi_node_optimizer:
            self.multi_node_optimizer.update(*args, **kwds)
        else:
            self.optimizer.update(*args, **kwds)
        self.anneal_learning_rate(training_step)

    def print(self):
        rows = []
        for key, value in self.__dict__.items():
            rows.append([key, value])
        print(tabulate(rows))

    def loss_scale(self):
        return None


class AdamOptimizer(Optimizer):
    def __init__(
            self,
            model_parameters,
            # Learning rate at training step s with annealing
            lr_i=1.0 * 1e-4,
            lr_f=1.0 * 1e-5,
            n=10000,
            # Learning rate as used by the Adam algorithm
            beta_1=0.9,
            beta_2=0.99,
            # Adam regularisation parameter
            eps=1e-8,
            communicator=None):
        super().__init__(lr_i, lr_f, n)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        lr = self.mu_s(0)
        self.optimizer = optimizers.Adam(
            lr, beta1=beta_1, beta2=beta_2, eps=eps)
        self.optimizer.setup(model_parameters)

        if communicator:
            self.multi_node_optimizer = chainermn.create_multi_node_optimizer(
                self.optimizer, communicator)

    @property
    def learning_rate(self):
        return self.optimizer.alpha

    def anneal_learning_rate(self, training_step):
        self.optimizer.hyperparam.alpha = self.mu_s(training_step)

    def loss_scale(self):
        return self.optimizer._loss_scale


class EveOptimizer(Optimizer):
    def __init__(
            self,
            model_parameters,
            # Learning rate at training step s with annealing
            lr_i=1.0 * 1e-4,
            lr_f=1.0 * 1e-5,
            n=10000,
            # Learning rate as used by the Adam algorithm
            beta_1=0.9,
            beta_2=0.99,
            # Adam regularisation parameter
            eps=1e-8,
            communicator=None):
        super().__init__(lr_i, lr_f, n)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps

        lr = self.mu_s(0)
        self.optimizer = Eve(lr, beta1=beta_1, beta2=beta_2, eps=eps)
        self.optimizer.setup(model_parameters)

        if communicator:
            self.multi_node_optimizer = chainermn.create_multi_node_optimizer(
                self.optimizer, communicator)

    @property
    def learning_rate(self):
        return self.optimizer.alpha

    def anneal_learning_rate(self, training_step):
        self.optimizer.hyperparam.alpha = self.mu_s(training_step)


class SGDOptimizer(Optimizer):
    def __init__(
            self,
            model_parameters,
            # Learning rate at training step s with annealing
            lr_i=1.0 * 1e-4,
            lr_f=1.0 * 1e-5,
            n=10000,
            communicator=None):
        super().__init__(lr_i, lr_f, n)

        lr = self.mu_s(0)
        self.optimizer = optimizers.SGD(lr)
        self.optimizer.setup(model_parameters)

        self.multi_node_optimizer = None
        if communicator:
            self.multi_node_optimizer = chainermn.create_multi_node_optimizer(
                self.optimizer, communicator)

    @property
    def learning_rate(self):
        return self.optimizer.lr

    def anneal_learning_rate(self, training_step):
        self.optimizer.hyperparam.lr = self.mu_s(training_step)


class MomentumSGDOptimizer(Optimizer):
    def __init__(
            self,
            model_parameters,
            # Learning rate at training step s with annealing
            lr_i=1.0 * 1e-4,
            lr_f=1.0 * 1e-5,
            n=10000,
            communicator=None):
        super().__init__(lr_i, lr_f, n)

        lr = self.mu_s(0)
        self.optimizer = optimizers.MomentumSGD(lr)
        self.optimizer.setup(model_parameters)

        self.multi_node_optimizer = None
        if communicator:
            self.multi_node_optimizer = chainermn.create_multi_node_optimizer(
                self.optimizer, communicator)

    @property
    def learning_rate(self):
        return self.optimizer.lr

    def anneal_learning_rate(self, training_step):
        self.optimizer.hyperparam.lr = self.mu_s(training_step)