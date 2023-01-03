from math import pi
from typing import Optional, Union

import numpy as np
import tensorflow as tf

from pqc_utils.pqc_utils import pairwise_d2_mat_v2, set_float_type, optimizers_classes, agg_sum
from sklearn.neighbors import NearestNeighbors


class DensityEstimator:
    def __init__(self,
                 data_gen: np.ndarray,
                 eps: float = 1e-10,
                 float_type: int = 32,
                 batch: int = 2000,
                 optimizer: Optional[optimizers_classes] = None,
                 force_cpu: bool = False,
                 scale: Optional[float] = None):

        if len(data_gen.shape) == 1:
            data_gen = data_gen.reshape(-1, 1)
        self.N_gen, self.D_gen = data_gen.shape
        self.data_gen = data_gen
        self.float_type = set_float_type(float_type)
        self.eps = tf.constant(eps, dtype=self.float_type)
        self.force_cpu: bool = force_cpu

        # Constants
        self.sqrt_2pi = tf.sqrt(tf.constant(2 * pi, dtype=self.float_type))

        if batch > self.N_gen:
            print("Batch is greater than the sample size, therefor it is reduced to sample size.")
        self.batch: int = min(batch, self.N_gen)
        if scale is not None:
            self.scale = tf.cast(tf.sqrt(tf.reduce_mean(tf.reduce_mean(tf.square(data_gen), axis=1))),
                                 dtype=self.float_type)
        else:
            self.scale = tf.constant(scale, dtype=self.float_type)

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
            self.device_train = "GPU:0"
            if self.N_gen <= 100000 and not self.force_cpu:
                self.device = "GPU:0"
            else:
                self.device = "CPU:0"
        else:
            self.device_train = "CPU:0"
            self.device = "CPU:0"

        with tf.device(self.device_train):
            self.log_sigma: Optional[tf.Variable] = None

        with tf.device(self.device):
            self.loss_0: Optional[tf.Variable] = None
            self.loss_1: Optional[tf.Variable] = None
            self.step: Optional[tf.Variable] = None
            self.trigger: Optional[tf.Variable] = None
            self.clusters: Optional[Union[np.ndarray, tf.Tensor]] = None
            # self.preset_log_sigma: Optional[tf.Tensor] = None

        # self.log_sigma_trained: Optional[tf.Tensor] = None

        if optimizer is None:
            self.opt = tf.keras.optimizers.Adam()
        else:
            self.opt = optimizer

        self.already_printed = False
        self.neigh = NearestNeighbors(n_neighbors=10)
        self.neigh.fit(self.data_gen)
        knn_dist, _ = self.neigh.kneighbors(self.data_gen, return_distance=True)
        self.min_sigma = np.median(knn_dist[:, 1:])

        print("wait a second")

    def set_clusters(self, clusters: Optional[Union[np.ndarray, tf.Tensor]]):
        clusters = tf.cast(clusters, dtype=tf.int32)
        if self.clusters is None:
            self.clusters = tf.Variable(clusters, dtype=tf.int32)
        else:
            self.clusters.assign(clusters)

    def reset_clusters(self):
        self.clusters = None

    def fit(self,
            preset_init: Optional[np.ndarray] = None,
            fit_type: str = "scalar",
            infinite_steps: bool = False,
            steps: int = 20000,
            patience: int = 4,
            check_gradients: bool = False
            ):
        if fit_type == "scalar":
            ll = self.fit_scalar(
                preset_log_sigma=preset_init,
                infinite_steps=infinite_steps,
                steps=steps,
                patience=patience,
                check_gradients=check_gradients
            )
            return ll
        else:
            raise ValueError(f"This type {type} is not recognized.")

    def fit_scalar(self,
                   preset_log_sigma: Optional[np.ndarray] = None,
                   infinite_steps: bool = False,
                   steps: int = 20000,
                   patience: int = 4,
                   check_gradients: bool = False
                   ):
        if preset_log_sigma is None:
            preset_log_sigma = np.ones((self.data_gen.shape[0], 1))
        else:
            assert np.all(preset_log_sigma > 0)

        if len(preset_log_sigma.shape) == 1 and preset_log_sigma.shape[0] == self.N_gen:
            preset_log_sigma = preset_log_sigma.reshape(-1, 1)

        preset_log_sigma = np.log(preset_log_sigma)
        if self.log_sigma is None:
            self.log_sigma = tf.Variable(tf.constant(preset_log_sigma, dtype=self.float_type))
        else:
            self.log_sigma.assign(tf.constant(preset_log_sigma, dtype=self.float_type))

        if check_gradients:
            self.watch_gradients()

        ll = self.train_loop(infinite_steps=infinite_steps,
                             steps=steps,
                             patience=patience)

        # ll = np.array([ll_i.numpy() for ll_i in ll])

        return ll.numpy()

    # @tf.function
    def compute_loss(self,
                     data_eval: Union[np.ndarray, tf.Tensor],
                     data_gen: Union[np.ndarray, tf.Tensor],
                     log_sigma: tf.Variable,
                     apply_mask: bool = True,
                     apply_constraints: bool = False,
                     regularization: bool = True):
        """
        Log[P(X)] = sum_j log(sum_i Psi_i(x_j)) - N*log(N)
        P(x_j) = sum_i Psi_i(x_j) / N
        P(k|x_j) = sum_(i€k) Psi_i(x_j) / sum_i Psi_i(x_j)
        Log[P(K|X) = sum_j log(sum_(i€k) Psi_i(x_j)) - sum_j log(sum_i Psi_i(x_j))
        """
        d2 = tf.cast(
            pairwise_d2_mat_v2(
                tf.cast(data_gen, dtype=self.float_type),
                tf.cast(data_eval, dtype=self.float_type)
            ),
            dtype=self.float_type
        )

        sigma_mat = tf.cast(tf.repeat(tf.exp(log_sigma), repeats=tf.shape(data_eval)[0], axis=1), dtype=self.float_type)

        norm_factor = tf.math.pow(self.sqrt_2pi * sigma_mat, -self.D_gen) + self.eps

        exp_kernel_i = tf.exp(-0.5 * tf.math.divide(d2, tf.square(sigma_mat)))
        if apply_mask:
            exp_kernel_i = tf.where(tf.math.equal(exp_kernel_i, 1.0),
                                    tf.zeros_like(exp_kernel_i, dtype=self.float_type),
                                    exp_kernel_i)

        # dens_i = tf.math.multiply(exp_kernel_i, norm_factor) + self.eps
        dens_i = tf.math.multiply(exp_kernel_i, norm_factor)
        dens_i = tf.where(tf.math.less_equal(exp_kernel_i, self.eps),
                          tf.zeros_like(dens_i, dtype=self.float_type),
                          dens_i)
        if tf.reduce_any(tf.math.equal(tf.reduce_max(dens_i, axis=0, keepdims=True), 0.0)):
            tf.print("All density elements are zero for some eval points.")
            dens_i += self.eps

        # # ########## TO DELETE ############################
        # with tf.GradientTape() as tape:
        #     x = tf.linspace(0.01, 10, 200)
        #     tape.watch(x)
        #     # A sequence of operations involving reduce_max
        #     norm_y = tf.math.pow(self.sqrt_2pi * x, -self.D_gen)  # + self.eps
        #     y1 = tf.math.multiply(tf.exp(-0.5 * tf.math.divide(10, tf.square(x))), norm_y)  # + self.eps
        #     y2 = tf.math.multiply(tf.exp(-0.5 * tf.math.divide(20, tf.square(x))), norm_y)
        #     y3 = tf.math.multiply(tf.exp(-0.5 * tf.math.divide(30, tf.square(x))), norm_y)
        #     # z = tf.math.log(y1 + y2) - tf.math.log(10 + y1 + y2 + y3)
        #     z = (y1 + y2 + y3) / (10 + y1 + y2 + y3)
        #     # z = (y2 + 1) / (10 + y1 + y2 + y3)
        #     # z = (+1) / (10 + y1 + y2 + y3)
        #     z = -tf.math.log(z)
        #
        # # Check gradients
        # g = tape.gradient(z, x)
        # # print(g.numpy())
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(x.numpy(), g.numpy())
        # plt.xlabel("sigma")
        # plt.ylabel("ANLL gradient")
        # plt.title("log[(y1 + y2 + y3) / (10 + y1 + y2 + y3)]")
        # plt.show()
        # # ########## TO DELETE ############################

        if self.clusters is not None:
            unique_k = tf.unique(self.clusters)[0]
            if tf.shape(unique_k)[0] == 1 and apply_constraints:
                loglikelihood = -tf.reduce_mean(tf.math.log(tf.reduce_mean(dens_i, axis=0, keepdims=True)))
            else:
                dens_k = tf.stack([agg_sum(dens_i, self.clusters, k) for k in unique_k], axis=0)
                dens_k_max = tf.reduce_max(dens_k, axis=0, keepdims=True)
                dens_all = tf.reduce_sum(dens_i, axis=0, keepdims=True)

                # loglikelihood = -tf.reduce_mean(tf.math.log(tf.divide(dens_k_max, dens_all)))
                loglikelihood = -tf.reduce_mean(tf.math.log(dens_k_max) - tf.math.log(dens_all))

            # if apply_constraints:
            #     penalize = tf.reduce_mean(tf.exp(log_sigma)) * 100
            #     k = tf.cast(tf.shape(unique_k)[0], dtype=self.float_type)
            #     k_too_high = tf.cast(tf.greater_equal(k, 0.5 * self.N_gen), dtype=self.float_type)
            #     k_too_low = tf.cast(tf.less_equal(k, 1), dtype=self.float_type)
            #     loglikelihood += (k_too_low - k_too_high) * penalize
        else:
            loglikelihood = -tf.reduce_mean(tf.math.log(tf.reduce_mean(dens_i, axis=0, keepdims=True)))

        if self.clusters is not None and regularization:
            if not self.already_printed:
                tf.print("Regularization is added to loglikelihood.")
                self.already_printed = True
            loglikelihood += tf.reduce_mean(tf.where(tf.math.less_equal(exp_kernel_i, self.eps),
                                                     norm_factor * 1.0,
                                                     # tf.square(sigma_mat) * 5.0,
                                                     tf.zeros_like(dens_i, dtype=self.float_type)))

        return loglikelihood

    # @tf.function
    def train_step(self, data_eval):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(data_eval, self.data_gen, self.log_sigma)
        # Get gradients of loss wrt the weights.
        gradients = tape.gradient(loss, [self.log_sigma])

        # Update the weights of the model.
        self.opt.apply_gradients(zip(gradients, [self.log_sigma]))
        return loss

    # @tf.function
    def train_loop(self,
                   infinite_steps: bool = False,
                   steps: int = 40000,
                   patience: int = 10):

        infinite_steps = tf.constant(infinite_steps, dtype=tf.bool)

        data_eval = self.sample_data_eval()
        loss = self.compute_loss(data_eval=data_eval, data_gen=self.data_gen, log_sigma=self.log_sigma)

        if self.loss_0 is None:
            self.loss_0 = tf.Variable(loss)
        else:
            self.loss_0.assign(loss)

        if self.loss_1 is None:
            self.loss_1 = tf.Variable(loss)
        else:
            self.loss_1.assign(loss)

        if self.step is None:
            self.step = tf.Variable(0, dtype=tf.int32)
        else:
            self.step.assign(0)

        if self.trigger is None:
            self.trigger = tf.Variable(0)
        else:
            self.trigger.assign(0)

        # # ################ TO DELETE #######################
        # ls = self.log_sigma.numpy()
        # import matplotlib.pyplot as plt
        # # ################ TO DELETE #######################

        loss_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

        # while tf.less(self.step, steps):
        while tf.math.logical_and(
                tf.math.logical_or(
                    infinite_steps,
                    tf.less(self.step, steps)
                ),
                tf.less(self.trigger, patience)
        ):

            if self.step.value() % 5 == 0:
                data_eval = self.sample_data_eval()

            loss = self.train_step(data_eval=data_eval)
            # # ################ TO DELETE #######################
            # ls1 = self.log_sigma.numpy()
            # plt.figure(figsize=(12, 6))
            # plt.stem(np.arange(len(ls)), ls1 - ls, '*')
            # plt.title("log dif")
            # plt.show()
            #
            # plt.figure(figsize=(12, 6))
            # plt.stem(np.arange(len(ls)), np.exp(ls1) - np.exp(ls), '*')
            # plt.title("sigma dif")
            # plt.show()
            # ls = ls1
            # # ################ TO DELETE #######################

            self.loss_0.assign(loss * 0.001 + self.loss_0.value() * (1 - 0.001))
            self.loss_1.assign(loss * 0.005 + self.loss_1.value() * (1 - 0.005))
            tf.cond(
                tf.math.logical_and(
                    tf.less(self.loss_0, self.loss_1),
                    tf.greater(self.step, 100)
                ),
                lambda: self.trigger.assign_add(1),
                lambda: self.trigger
            )

            loss_array = loss_array.write(self.step.value(), loss)
            if self.step.value() % 200 == 0:
                tf.print(self.loss_0.value())
            self.step.assign_add(1)

        loss_vector = loss_array.stack()
        loss_array.close()
        return loss_vector

    @tf.function
    def sample_data_eval(self, scale=0.10, apply_noise=True, apply_scale_noise=True):

        if self.batch < self.N_gen:
            gen_idx = tf.random.uniform([self.batch], minval=0, maxval=self.N_gen - 1, dtype=tf.int32)
            data_eval = tf.gather(self.data_gen, gen_idx, axis=0)
        else:
            data_eval = tf.constant(self.data_gen, dtype=self.float_type)

        if apply_noise:
            if apply_scale_noise:
                scale_noise = tf.random.uniform([], minval=scale * 0.8, maxval=scale * 1.2, dtype=self.float_type)
            else:
                scale_noise = scale
            noise = tf.random.normal([self.batch, self.D_gen],
                                     mean=0,
                                     stddev=scale_noise * self.scale,
                                     dtype=self.float_type)
            data_eval += noise

        return data_eval

    def watch_gradients(self, eval_type: str = "sample"):

        if eval_type == "sample":
            data_eval = self.sample_data_eval()
        elif eval_type == "grid":

            data_eval = self.data_gen.numpy()
            dx = np.linspace(data_eval[:, 0].min(), data_eval[:, 0].max(), 50)
            dy = np.linspace(data_eval[:, 1].min(), data_eval[:, 1].max(), 50)
            dxv, dyv = np.meshgrid(dx, dy)
            data_eval = tf.constant(np.hstack([dxv.reshape(-1, 1), dyv.reshape(-1, 1)]))
        else:
            data_eval = self.data_gen

        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(self.data_gen.numpy()[:, 0], self.data_gen.numpy()[:, 1],
                    alpha=0.5, c=self.clusters.numpy(), s=20 * (self.clusters.numpy() ** 2 + 1))
        plt.plot(data_eval.numpy()[0][0], data_eval.numpy()[0][1], 'r+')
        plt.show()

        x = tf.cast(tf.linspace(-4, 4, 500), dtype=self.float_type)
        for i in range(10):
            ls = tf.Variable(initial_value=self.log_sigma.value())

            # i = 1
            y = []
            z = []

            for x_i in x:
                ls[i].assign(x_i)
                with tf.GradientTape() as tape:
                    tape.watch(ls)
                    loss = self.compute_loss(data_eval, self.data_gen, ls)
                    z.append(loss.numpy())

                # Check gradients
                g = tape.gradient(loss, [ls])
                y.append(g[0][i])
            # print(g.numpy())

            plt.subplots(2, 1, figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(x, y)
            # plt.xlabel(f"log sigma {i}")
            plt.ylabel("ANLL gradient")
            plt.grid(True, which="both")
            # plt.show()

            # plt.figure()
            plt.subplot(2, 1, 2)
            plt.plot(x, z)
            plt.xlabel(f"log sigma {i}")
            plt.ylabel("ANLL")
            plt.grid(True, which="both")
        plt.show()
        print("End of function!")


"""
NOTES 01:
Organize the training in a similar way as pqc, i.e. using a batch and train sigmas per batch and eval in full dataset if
it fits in memory, if not, use a batch or a tf.random.uniform as here.

NOTES 02:
Apparently the training in batch in a ordered way seems not to work, where all sigmas tend to zero, 
creating eye matrices... try to remove the self-point in loglikelihood computing
"""
