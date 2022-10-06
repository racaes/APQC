from math import pi
from typing import Optional, Union

import numpy as np
import tensorflow as tf

from pqc_utils.pqc_utils import pairwise_d2_mat_v2, set_float_type, optimizers_classes


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
            # self.preset_log_sigma: Optional[tf.Tensor] = None

        # self.log_sigma_trained: Optional[tf.Tensor] = None

        if optimizer is None:
            self.opt = tf.keras.optimizers.Adam()
        else:
            self.opt = optimizer

    def fit(self,
            preset_init: Optional[np.ndarray] = None,
            fit_type: str = "scalar",
            infinite_steps: bool = False,
            steps: int = 20000,
            patience: int = 4,
            ):
        if fit_type == "scalar":
            ll = self.fit_scalar(
                preset_log_sigma=preset_init,
                infinite_steps=infinite_steps,
                steps=steps,
                patience=patience
            )
            return ll
        else:
            raise ValueError(f"This type {type} is not recognized.")

    def fit_scalar(self,
                   preset_log_sigma: Optional[np.ndarray] = None,
                   infinite_steps: bool = False,
                   steps: int = 20000,
                   patience: int = 4,
                   ):
        if preset_log_sigma is None:
            preset_log_sigma = np.zeros((self.data_gen.shape[0], 1))
        if len(preset_log_sigma.shape) == 1 and preset_log_sigma.shape[0] == self.N_gen:
            preset_log_sigma = preset_log_sigma.reshape(-1, 1)

        if self.log_sigma is None:
            self.log_sigma = tf.Variable(tf.constant(preset_log_sigma, dtype=self.float_type))
        else:
            self.log_sigma.assign(tf.constant(preset_log_sigma, dtype=self.float_type))

        ll = self.train_loop(infinite_steps=infinite_steps,
                             steps=steps,
                             patience=patience)

        # ll = np.array([ll_i.numpy() for ll_i in ll])

        return ll.numpy()

    @tf.function
    def compute_loss(self,
                     data_eval: Union[np.ndarray, tf.Tensor],
                     data_gen: Union[np.ndarray, tf.Tensor],
                     log_sigma: tf.Variable,
                     apply_mask: bool = True
                     ):

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
            # mask = tf.linalg.set_diag(tf.ones_like(exp_kernel_i), tf.zeros(tf.shape(exp_kernel_i)[0]))
            # exp_kernel_i = tf.math.multiply(exp_kernel_i, mask)
            exp_kernel_i = tf.where(tf.math.equal(exp_kernel_i, 1.0),
                                    tf.zeros_like(exp_kernel_i, dtype=self.float_type),
                                    exp_kernel_i)

        dens_i = tf.math.multiply(exp_kernel_i, norm_factor) + self.eps
        """        
        Log[P(X)] = sum_j log(sum_i Psi_i(x_j)) - N*log(N)
        P(x_j) = sum_i Psi_i(x_j) / N
        P(k|x_j) = sum_(i€k) Psi_i(x_j) / sum_i Psi_i(x_j)
        Log[P(K|X) = sum_j log(sum_(i€k) Psi_i(x_j)) - sum_j log(sum_i Psi_i(x_j))
        """

        loglikelihood = -tf.reduce_mean(tf.math.log(tf.reduce_mean(dens_i, axis=0, keepdims=True)))

        return loglikelihood

    @tf.function
    def train_step(self, data_eval):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(data_eval, self.data_gen, self.log_sigma)
        # Get gradients of loss wrt the weights.
        gradients = tape.gradient(loss, [self.log_sigma])

        # Update the weights of the model.
        self.opt.apply_gradients(zip(gradients, [self.log_sigma]))
        return loss

    @tf.function
    def train_loop(self,
                   infinite_steps: bool = False,
                   steps: int = 40000,
                   patience: int = 10,
                   # gen_ratio: float = 0.2
                   ):
        # gen_idx: Optional[Union[List[int], np.ndarray]] = None
        # n_ratio = tf.constant(gen_ratio * self.D_gen, dtype=tf.int32)

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

        # self.variable_assign(self.loss_0, loss)
        # self.variable_assign(self.loss_1, loss)
        # self.variable_assign(self.step, 0)
        # self.variable_assign(self.trigger, 0)

        if self.step is None:
            self.step = tf.Variable(0, dtype=tf.int32)
        else:
            self.step.assign(0)

        if self.trigger is None:
            self.trigger = tf.Variable(0)
        else:
            self.trigger.assign(0)

        loss_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True, clear_after_read=False)

        # while tf.less(self.step, steps):
        while tf.math.logical_and(
                tf.math.logical_or(
                    infinite_steps,
                    tf.less(self.step, steps)
                ),
                tf.less(self.trigger, patience)
        ):

            if self.step.value() % 10 == 0:
                data_eval = self.sample_data_eval()

            loss = self.train_step(data_eval=data_eval)

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

            # loss_list.append(loss.numpy())
            # loss_list.append(loss)
        loss_vector = loss_array.stack()
        loss_array.close()
        return loss_vector

    # @staticmethod
    # def variable_assign(self, variable: Optional[tf.Variable], value):
    #     if variable is None:
    #         variable = tf.Variable(value)
    #     else:
    #         variable.assign(value)
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


"""
NOTES 01:
Organize the training in a similar way as pqc, i.e. using a batch and train sigmas per batch and eval in full dataset if
it fits in memory, if not, use a batch or a tf.random.uniform as here.

NOTES 02:
Apparently the training in batch in a ordered way seems not to work, where all sigmas tend to zero, 
creating eye matrices... try to remove the self-point in loglikelihood computing
"""
