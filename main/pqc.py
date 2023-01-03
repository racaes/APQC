import time
import warnings
from collections import Counter
from collections import defaultdict
from itertools import combinations
from math import pi
from typing import Optional, Union, List, Tuple, Dict

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numba as nb
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import interpolate
from scipy.cluster.hierarchy import dendrogram, linkage, num_obs_linkage, fcluster
from scipy.spatial.distance import pdist, squareform

from pqc_utils.pqc_utils import (pairwise_d2_mat_v3, pairwise_d2_mat_v2, reduce_sum, reduce_mean, nb_sort, nb_mean,
                                 moving_average, set_float_type, optimizers_classes)


# # TODO This really helps?
# import os
#
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
# print(os.getenv("TF_GPU_ALLOCATOR"))


# nvidia-smi -l 1 --query-gpu=memory.used --format=csv


class PQC:
    def __init__(self,
                 data_gen: np.ndarray,
                 optimizer: Optional[optimizers_classes] = None,
                 eps: float = 1e-10,
                 float_type: int = 32,
                 batch: int = 2000,
                 merge_by_proba_clusters: bool = True,
                 force_cpu: bool = False):

        if len(data_gen.shape) == 1:
            data_gen = data_gen.reshape(-1, 1)
        self.N_gen, self.D_gen = data_gen.shape

        self.float_type = set_float_type(float_type)
        if batch > self.N_gen:
            print("Batch is greater than the sample size, therefor it is reduced to sample size.")
        self.batch: int = min(batch, self.N_gen)
        self.force_cpu: bool = force_cpu
        gpus = tf.config.experimental.list_physical_devices('GPU')
        # tf.debugging.set_log_device_placement(True)
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

        with tf.device(self.device):
            self.data_gen = tf.constant(data_gen, dtype=self.float_type)
            self.scale = tf.reduce_mean(tf.norm(self.data_gen, axis=1))
            if self.scale.numpy() == 0:
                self.scale = tf.constant(1.0, dtype=self.float_type)
            self.data_gen /= self.scale

            self.eps = tf.constant(eps, dtype=self.float_type)
            self.knn_d2: Dict[float, Union[tf.Tensor, np.ndarray]] = {}
            self.sigmas_id: Optional[Tuple[str, Union[float, int]]] = None
            self.sigmas: Optional[tf.Variable] = None
            self.loss_0: Optional[tf.Variable] = None
            self.step: Optional[tf.Variable] = None
            self.trigger: Optional[tf.Variable] = None
            self.err: Optional[tf.Tensor] = None  # Error is defined as last data-point distance update during training.

            self.ll: Optional[np.ndarray] = None
            self.k_centroids_list: Optional[List[np.ndarray]] = None
        with tf.device(self.device_train):
            self.data_train: Optional[tf.Variable] = None

        self.data_trained: Optional[np.ndarray] = None
        self.sgd_labels: Optional[np.ndarray] = None
        self.proba_labels: Optional[np.ndarray] = None
        self.k_proba: Optional[np.ndarray] = None
        self.k_ids: Optional[np.ndarray] = None
        self.p_x_k: Optional[np.ndarray] = None

        self.k_num: Optional[int] = None
        self.max_cluster_number: int = 120
        self.basic_results: dict = {}
        self.energy_merge_results: dict = {}
        self.merge_by_proba_clusters: bool = merge_by_proba_clusters

        if optimizer is None:
            self.opt = tf.keras.optimizers.Adam()
        else:
            self.opt = optimizer

    @staticmethod
    def knn_d2_batched_v1(data, batch_size, knn_indices):
        knn_d2_list = []
        data_gen_batched = tf.data.Dataset.from_tensor_slices(data).batch(batch_size)
        for batch in data_gen_batched:
            self_d2 = pairwise_d2_mat_v2(batch, data)
            self_d2 = tf.sort(self_d2, axis=1)

            knn_d2_batch = []
            for j, idx in enumerate(knn_indices):
                knn_d2_batch.append(tf.reduce_mean(self_d2[:, 1:idx], axis=1, keepdims=True))

            knn_d2_list.append(tf.concat(knn_d2_batch, axis=1))

        knn_d2_mat = tf.concat(knn_d2_list, axis=0)

        del self_d2, data_gen_batched, knn_d2_list
        return knn_d2_mat

    @staticmethod
    @nb.jit(nopython=True, fastmath=True, parallel=True)
    def knn_d2_batched_v2(data, batch_size, knn_indices):

        knn_d2_mat = np.empty((len(data), knn_indices.size), dtype=np.float32)
        for i in np.arange(0, len(data), batch_size):
            self_d2 = pairwise_d2_mat_v3(data[i:i + batch_size], data)
            self_d2 = nb_sort(self_d2, axis=1)
            for j, idx in enumerate(knn_indices):
                knn_d2_mat[i:(i + len(self_d2)), j] = nb_mean(self_d2[:, 1:idx], axis=1)

        return knn_d2_mat

    def pre_compute_knn_variance(self, knn_ratios: Union[List[float], np.ndarray]):
        """
        Runtimes of batched computation (10K / 1K / 1K[pc]) of 30.4K:
            TF-cpu  56.7s / 99s / 50.5s / 50.113 s
            TF-gpu  OOM / OOM / 25.4s
            NB      73s / 75.2s / 61.5s / 75.0s
            NB_par  _ / _/ 71.2s / 71.8s / 75.512 s
            NP      67s / 63.5s / 53.1s
        :param knn_ratios:
        :return:
        """
        device = self.device if not self.force_cpu else "CPU:0"
        knn_indices = np.array([int(i * self.N_gen) for i in knn_ratios], dtype=np.int)
        with tf.device(device):
            t0 = time.time()
            knn_d2 = PQC.knn_d2_batched_v1(data=self.data_gen.numpy(),  # If _v1 .numpy() is not needed.
                                           batch_size=self.batch,
                                           knn_indices=knn_indices)
            for i, knn_ratio in enumerate(knn_ratios):
                self.knn_d2.update({round(knn_ratio, 2): tf.reshape(knn_d2[:, i], [-1, 1])})

            t1 = time.time()
            print(f"Elapsed time for computing the full data_gen pairwise"
                  f" distance matrix: {round(t1 - t0, 3)} s")

    def get_knn_variance(self, knn_ratio: float):
        if self.N_gen > 1:
            knn_ratio = round(knn_ratio, 2)
            if knn_ratio not in self.knn_d2.keys():
                self.pre_compute_knn_variance(knn_ratios=[knn_ratio])
                self.knn_d2 = {k: v for k, v in sorted(self.knn_d2.items(), key=lambda x: x[0])}
            sigma_i = self.knn_d2[knn_ratio]

        else:
            warnings.warn("Data contain only one sample, sigma assigned to 1.0")
            sigma_i = tf.constant(np.array([1.0]).reshape(-1, 1), dtype=self.float_type)

        return sigma_i

    def set_sigmas(self, knn_ratio: Optional[float] = None, sigma_value: Optional[Union[float, np.ndarray]] = None):
        if knn_ratio is not None and sigma_value is not None:
            raise ValueError("Only one argument can be used!")
        elif knn_ratio is not None and sigma_value is None:
            sigmas = self.get_knn_variance(knn_ratio)
            self.sigmas_id = ("knn_ratio", round(knn_ratio, 2))
        elif knn_ratio is None and sigma_value is not None:
            if isinstance(sigma_value, float):
                sigmas = tf.constant(np.array([sigma_value] * self.N_gen).reshape(-1, 1), dtype=self.float_type)
                self.sigmas_id = ("scalar_constant", sigma_value)
            elif isinstance(sigma_value, np.ndarray):
                if len(sigma_value.shape) == 1 and sigma_value.shape[0] == self.N_gen:
                    sigmas = tf.constant(sigma_value.reshape(-1, 1), dtype=self.float_type)
                    self.sigmas_id = ("vector_constant", sigma_value.round(3))
                elif len(sigma_value.shape) == 2 and self.N_gen in sigma_value.shape:
                    sigmas = tf.constant(sigma_value.reshape(-1, 1), dtype=self.float_type)
                    self.sigmas_id = ("vector_variable", np.mean(sigma_value).round(3))
                else:
                    raise NotImplementedError
            else:
                raise ValueError(f"Invalid format for {type(sigma_value)}")
        else:
            raise ValueError("Both arguments are None")
        if self.sigmas is None:
            self.sigmas = tf.Variable(sigmas, dtype=self.float_type)
        else:
            self.sigmas.assign(sigmas)

    def probability_per_sample(self, data_train: Union[np.ndarray, tf.Tensor]):
        wave_i = self.qc_wave_function_per_sample_batched(data_train)
        return tf.reduce_mean(wave_i, axis=0, keepdims=True)

    def cluster_probability_per_sample_batched(self,
                                               data_train: Union[np.ndarray, tf.Tensor],
                                               labels: Union[np.ndarray, tf.Tensor],
                                               add_noise: bool = False):
        with tf.device(self.device):

            if add_noise:
                noise = tf.random.normal(shape=tf.shape(data_train), dtype=self.float_type)
                data_train = data_train + tf.math.reduce_std(data_train, axis=0) * noise * 0.1

            dataset_train = tf.data.Dataset.from_tensor_slices(data_train).batch(self.batch)
            k_proba_list, k_winner_list = [], []
            for batch in dataset_train:
                wave_i = self.qc_wave_function_per_sample(batch)

                # t0 = time.time()

                k_proba_joint_i, _ = reduce_sum(data=wave_i.numpy(), vector=labels)
                # print(f"Reduce op runtime: {round(time.time() - t0, 3)} s")

                k_proba_sum = np.sum(k_proba_joint_i, axis=0, keepdims=True)
                k_proba_i = k_proba_joint_i / k_proba_sum
                k_winner_i = np.argmax(k_proba_i, axis=0)

                k_proba_list.append(k_proba_i)
                k_winner_list.append(k_winner_i)

            k_proba = np.concatenate(k_proba_list, axis=1)
            k_winner = np.concatenate(k_winner_list, axis=0)
            k_winner_proba = np.max(k_proba, axis=0, keepdims=True)

            loglikelihood = -np.nanmean(np.log(k_winner_proba))

            return k_winner_proba, k_winner, loglikelihood

    def cluster_allocation_by_probability(self):
        t0 = time.time()
        self.k_proba, self.proba_labels, self.ll = self.cluster_probability_per_sample_batched(data_train=self.data_gen,
                                                                                               labels=self.sgd_labels)
        print(f"Computed clusters probabilities in {round(time.time() - t0, 3)} s")

        p_k = {k: v for k, v in Counter(self.proba_labels).items()}
        self.p_x_k = np.array([p_label / p_k[label]
                               for label, p_label in zip(self.proba_labels, self.k_proba.flatten())])

        self.update_basic_results()

        k_num_proba = len(np.unique(self.proba_labels))
        print(f"Detected {k_num_proba} probability clusters.")
        if self.merge_by_proba_clusters:
            self.k_num = k_num_proba

    def update_basic_results(self):
        self.basic_results[self.sigmas_id] = {
            "sigma": self.sigmas_id[1],
            "sgd_labels": self.sgd_labels.flatten(),
            "proba_labels": self.proba_labels.flatten(),
            "loglikelihood": self.ll,
            "sgd_k_num": len(np.unique(self.sgd_labels)),
            "proba_k_num": len(np.unique(self.proba_labels)),
            "proba_winner": self.k_proba.flatten(),
            "proba_x_k": self.p_x_k.flatten()
        }

    def qc_wave_function_per_sample_batched(self, data_train: Union[np.ndarray, tf.Tensor]):
        if len(data_train.shape) == 1:
            if isinstance(data_train, np.ndarray):
                data_train = data_train.reshape(-1, 1)
            elif isinstance(data_train, tf.Tensor):
                data_train = tf.reshape(data_train, [-1, 1])
            else:
                raise ValueError("data_train has wrong format.")

        data_train_batch = tf.data.Dataset.from_tensor_slices(data_train).batch(self.batch)
        wave_list = []
        for batch in data_train_batch:
            wave_list.append(self.qc_wave_function_per_sample(batch))

        return tf.concat(wave_list, axis=1)

    @tf.function
    def qc_wave_function_per_sample(self, data_train: Union[np.ndarray, tf.Tensor]):
        d2 = pairwise_d2_mat_v2(self.data_gen, data_train)

        sigma_mat = tf.repeat(self.sigmas, repeats=tf.shape(data_train)[0], axis=1)

        norm_factor = tf.math.pow(tf.sqrt(tf.constant(2 * pi, dtype=self.float_type)) * sigma_mat,
                                  -self.D_gen) + self.eps

        exp_kernel_i = tf.exp(-0.5 * tf.math.divide(d2, tf.square(sigma_mat)))
        # Matrix of cols as data_train and rows as data_gen. Each train sample is evaluated by all its column
        wave_i = tf.math.multiply(exp_kernel_i, norm_factor) + self.eps
        return wave_i

    def qc_potential_per_sample(self, data_train: Union[np.ndarray, tf.Tensor]):
        if len(data_train.shape) == 1:
            if isinstance(data_train, np.ndarray):
                data_train = data_train.reshape(-1, 1)
            elif isinstance(data_train, tf.Tensor):
                data_train = tf.reshape(data_train, [-1, 1])
            else:
                raise ValueError("data_train has wrong format.")

        norm_factor_rel, sigma_mat = self.get_sigma_and_factor(data_train)

        return self.qc_potential_cost(data_train, sigma_mat, norm_factor_rel)

    @tf.function
    def get_sigma_and_factor(self, data_train):
        sigma_mat = tf.repeat(self.sigmas, repeats=data_train.shape[0], axis=1)
        norm_factor = tf.math.pow(tf.sqrt(tf.constant(2 * pi, dtype=self.float_type)) * sigma_mat,
                                  -self.D_gen) + self.eps
        norm_factor_scale = tf.reduce_mean(norm_factor)
        norm_factor_rel = norm_factor / norm_factor_scale
        return norm_factor_rel, sigma_mat

    @tf.function
    def qc_potential_cost(self, data_train, sigma_mat, norm_factor_rel):
        d2 = pairwise_d2_mat_v2(self.data_gen, data_train)
        # TODO: OOM ERROR.
        # Hint: If you want to see a list of allocated tensors when OOM happens, add report_tensor_allocations_upon_oom
        # to RunOptions for current allocation info. This isn't available when running in Eager mode.
        exp_kernel_i = tf.exp(-0.5 * tf.math.divide(d2, tf.square(sigma_mat)))

        wave_i = tf.math.multiply(exp_kernel_i, norm_factor_rel) + self.eps
        wave_f = tf.reduce_mean(wave_i, axis=0, keepdims=True)

        kinetic_i = 0.5 * (d2 / sigma_mat) * wave_i
        kinetic_f = tf.reduce_sum(kinetic_i, axis=0, keepdims=True)
        pot_f = kinetic_f / wave_f

        return pot_f

    @tf.function
    def train_step(self, sigma_mat, norm_factor_rel):
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(self.qc_potential_cost(self.data_train, sigma_mat, norm_factor_rel))
        # Get gradients of loss wrt the weights.
        gradients = tape.gradient(loss, [self.data_train])

        # Update the weights of the model.
        self.opt.apply_gradients(zip(gradients, [self.data_train]))
        return loss

    def pot_grad_desc(self,
                      data_train0: Optional[Union[np.ndarray, tf.Tensor]],
                      steps: int = 20000,
                      infinite_steps: bool = False,
                      patience: int = 4,
                      plot_allocation: bool = False):

        if len(data_train0.shape) == 1:
            if isinstance(data_train0, np.ndarray):
                data_train0 = data_train0.reshape(-1, 1)
            elif isinstance(data_train0, tf.Tensor):
                data_train0 = tf.reshape(data_train0, [-1, 1])
            else:
                raise ValueError("data_train has wrong format.")

        if self.sigmas is None:
            raise ValueError("Set sigma values first")

        n = tf.shape(data_train0)[0]
        if n <= self.batch:
            norm_factor_rel, sigma_mat = self.get_sigma_and_factor(data_train0)
            loss, dx1, dx2, mean_steps = self.train_loop(
                data_train0=data_train0,
                sigma_mat=sigma_mat,
                norm_factor_rel=norm_factor_rel,
                infinite_steps=tf.constant(infinite_steps, dtype=tf.bool),
                steps=tf.constant(steps),
                patience=tf.constant(patience)
            )
            self.data_trained = self.data_train.numpy()

        else:
            loss_list, dx1_list, dx2_list, step_list = [], [], [], []
            data_trained_list = []
            if n % self.batch != 0:
                norm_factor_rel, sigma_mat = self.get_sigma_and_factor(data_train0[:self.batch])
                loss_0, dx1_0, dx2_0, step_0 = self.train_loop(
                    data_train0=data_train0[:self.batch],
                    sigma_mat=sigma_mat,
                    norm_factor_rel=norm_factor_rel,
                    infinite_steps=tf.constant(infinite_steps, dtype=tf.bool),
                    steps=tf.constant(steps),
                    patience=tf.constant(patience)
                )
                data_trained_list.append(self.data_train.numpy())
                r = self.batch - (n % self.batch)
                data_train1 = tf.concat([data_train0[self.batch:, :], self.data_train[:r, :]], axis=0)
                loss_list.append(loss_0)
                dx1_list.append(dx1_0)
                dx2_list.append(dx2_0)
                step_list.append(step_0.numpy())
            else:
                data_train1 = data_train0

            i0, i1 = 0, self.batch
            while i0 < tf.shape(data_train1)[0]:
                norm_factor_rel, sigma_mat = self.get_sigma_and_factor(data_train1[i0:i1])
                loss_i, dx1_i, dx2_i, step_i = self.train_loop(
                    data_train0=data_train1[i0:i1],
                    sigma_mat=sigma_mat,
                    norm_factor_rel=norm_factor_rel,
                    infinite_steps=tf.constant(infinite_steps, dtype=tf.bool),
                    steps=tf.constant(steps),
                    patience=tf.constant(patience)
                )
                loss_list.append(loss_i)
                dx1_list.append(dx1_i)
                dx2_list.append(dx2_i)
                step_list.append(step_i.numpy())

                i0 = i1
                i1 += self.batch
                data_trained_list.append(self.data_train.numpy())

            loss = tf.reduce_sum(loss_list)
            dx1 = tf.reduce_max(dx1_list)
            dx2 = tf.reduce_max(dx2_list)
            mean_steps = np.mean(step_list)

            self.data_trained = tf.concat(data_trained_list, axis=0)[:n].numpy()

        self.err = tf.stack([dx1, dx2])  # First error always is greater or equal thant second error.

        if plot_allocation:
            plt.figure()
            sns.scatterplot(x=self.data_gen[:, 0], y=self.data_gen[:, 1], alpha=0.15)
            sns.scatterplot(x=self.data_trained[:, 0], y=self.data_trained[:, 1], alpha=0.5)
            plt.show()

        return loss, mean_steps

    @tf.function
    def train_loop(self,
                   data_train0: tf.Tensor,
                   sigma_mat: tf.Tensor,
                   norm_factor_rel: tf.Tensor,
                   infinite_steps: tf.Tensor,
                   steps: tf.Tensor,
                   patience: tf.Tensor):
        loss = tf.reduce_sum(self.qc_potential_cost(data_train0, sigma_mat, norm_factor_rel))
        if self.loss_0 is None:
            self.loss_0 = tf.Variable(loss)
        else:
            self.loss_0.assign(loss)

        if self.step is None:
            self.step = tf.Variable(0)
        else:
            self.step.assign(0)

        if self.trigger is None:
            self.trigger = tf.Variable(0)
        else:
            self.trigger.assign(0)

        if self.data_train is None:
            self.data_train = tf.Variable(data_train0, dtype=self.float_type)
        else:
            self.data_train.assign(data_train0)

        while tf.math.logical_and(tf.math.logical_or(infinite_steps,
                                                     tf.less(self.step, steps)),
                                  tf.less(self.trigger, patience)):
            loss = self.train_step(sigma_mat, norm_factor_rel)
            tf.cond(tf.less_equal(self.loss_0, loss), lambda: self.trigger.assign_add(1), lambda: self.trigger)
            self.loss_0.assign(loss)
            self.step.assign_add(1)

        x0 = self.data_train.value()
        loss = self.train_step(sigma_mat, norm_factor_rel)
        x1 = self.data_train.value()

        dx = tf.norm(tf.math.reduce_max(tf.abs(x0 - x1), axis=0))
        dx2 = tf.math.reduce_max(tf.sqrt(tf.reduce_sum((x0 - x1) ** 2, axis=1)))

        return loss, dx, dx2, self.step

    def cluster_allocation_by_sgd(self, data: Optional[Union[np.ndarray, tf.Tensor]] = None):
        t0 = time.time()
        if data is None:
            _, mean_steps = self.pot_grad_desc(data_train0=self.data_gen)
            print(f"Generative data is trained in {np.round(mean_steps, 3)} steps and {round(time.time() - t0, 3)} s")
        else:
            _, mean_steps = self.pot_grad_desc(data_train0=data)
            print(f"Data is trained in {np.round(mean_steps, 3)} steps and {round(time.time() - t0, 3)} s")

        t1 = time.time()
        pw_mat = pdist(X=self.data_trained, metric="euclidean")
        pw_mat = pw_mat > self.err.numpy().max()
        z = linkage(pw_mat)
        sgd_labels_pre = fcluster(z, t=0.9, criterion="distance")
        sgd_labels_tuple = sorted(Counter(sgd_labels_pre).items(), key=lambda x: x[1], reverse=True)
        self.sgd_labels = np.zeros_like(sgd_labels_pre)
        for i, (k, c) in enumerate(sgd_labels_tuple):
            self.sgd_labels[sgd_labels_pre == k] = i

        print(f"Time to get labels from connected pairs: {round(time.time() - t1, 3)} s")

        self.k_num = np.max(self.sgd_labels) + 1
        print(f"Detected {self.k_num} clusters, having {np.sum(self.sgd_labels == 0)} samples the"
              f" most populated cluster.")

    @staticmethod
    def get_labels_from_connected_pairs(con_pairs: np.ndarray, data_size: int):
        k_unique = np.unique(con_pairs[:, 0]).astype(int)

        c_counter = 0
        lab = defaultdict(set)
        for k_i in k_unique:
            current_set_k = set(con_pairs[con_pairs[:, 0] == k_i, 1].tolist())
            c_assign = None
            for c_i, set_i in lab.items():
                if len(list(set(set_i) & current_set_k)) > 0:
                    c_assign = c_i
                    break
            if c_assign is None:
                lab[c_counter] = current_set_k
                c_counter += 1
            else:
                lab[c_assign] = lab[c_assign].union(current_set_k)
                if len(lab[c_assign]) >= data_size:
                    break

        labels_by_count = sorted(lab.items(), key=lambda x: len(x[1]), reverse=True)
        labels = np.zeros(data_size).astype(int)
        labels[:] = -1
        for i, (l1, c1) in enumerate(labels_by_count):
            labels[list(c1)] = i

        return labels

    def get_potential_wells(self):
        if self.merge_by_proba_clusters:
            assert self.proba_labels is not None
            assert self.proba_labels.shape[0] == self.N_gen

            self.k_centroids_list, self.k_ids = reduce_mean(data=self.data_trained, vector=self.proba_labels)

        else:
            assert self.sgd_labels is not None
            assert self.sgd_labels.shape[0] == self.N_gen

            self.k_centroids_list, self.k_ids = reduce_mean(data=self.data_trained, vector=self.sgd_labels)

    def compute_energy_between_potential_wells(self):
        assert self.k_centroids_list is not None
        sig = np.round(-np.log10(self.err[0] + self.eps)).astype(int)
        t0 = time.time()
        lst = list(range(len(self.k_centroids_list)))
        idx_pairs = []
        for i, j in combinations(lst, 2):
            pot12, pot21 = self.compute_potential_distance(v1=self.k_centroids_list[i], v2=self.k_centroids_list[j])
            idx_pairs.extend(
                [
                    (i, j, np.round(pot12, sig)),
                    (j, i, np.round(pot21, sig))
                ]
            )

        print(f"Potential path runtime with {self.k_num} clusters: {round(time.time() - t0, 4)} s")
        return idx_pairs

    def compute_potential_distance(self, v1: np.ndarray, v2: np.ndarray, steps: int = 20):
        dv = v2 - v1
        v1_mat = np.repeat(np.reshape(v1, [1, -1]), steps, axis=0)
        dv_mat = np.repeat(np.reshape(dv, [1, -1]), steps, axis=0)
        factor = np.repeat(np.reshape(np.linspace(0, 1, steps), [steps, 1]), np.shape(v1), axis=1)
        path_mat = v1_mat + factor * dv_mat

        with tf.device(self.device):
            potential_path = self.qc_potential_per_sample(path_mat.astype("float32")).numpy()

        pot_max = np.max(potential_path)
        pot12 = pot_max - potential_path[0, 0]
        pot21 = pot_max - potential_path[0, -1]
        return pot12, pot21

    def compute_shortest_path(self, potential_pairs):
        graph = nx.DiGraph()
        graph.add_weighted_edges_from(potential_pairs)
        ap_sp = nx.shortest_paths.floyd_warshall(graph)

        potential_mat = np.zeros([self.k_num, self.k_num])
        for i, i_dict in ap_sp.items():
            for j, pot_ij in i_dict.items():
                potential_mat[i, j] = pot_ij

        potential_matrix_min = np.where(potential_mat > potential_mat.T, potential_mat.T, potential_mat)
        potential_matrix_min_condensed = squareform(potential_matrix_min)

        return potential_matrix_min_condensed

    @staticmethod
    def merge_labels_by_energy(potential_matrix: np.ndarray, plot_dendrogram: bool = False):
        potential_matrix_min = np.where(potential_matrix > potential_matrix.T, potential_matrix.T, potential_matrix)
        potential_matrix_min_condensed = squareform(potential_matrix_min)
        potential_linkage = linkage(potential_matrix_min_condensed)

        if plot_dendrogram:
            plt.figure(dpi=100)
            dendrogram(potential_linkage,
                       orientation='top',
                       distance_sort='descending',
                       show_leaf_counts=True)
            plt.show()

        return PQC.labels_from_linkage(potential_linkage)

    @staticmethod
    def labels_from_linkage(linkage_matrix):
        n0b = num_obs_linkage(linkage_matrix)
        energies = np.unique(linkage_matrix[:, 2])
        max_energy = np.max(energies)
        if max_energy == 0:
            merged_energies = np.array([-1, 0])
            merged_labels = [np.arange(n0b), np.zeros(n0b)]
        else:
            merged_energies = np.insert(energies, obj=0, values=-1)

            energies_interval = np.append(merged_energies, max_energy + 1)
            energies_interval = moving_average(energies_interval, n=2)
            merged_labels = [fcluster(linkage_matrix, d, criterion="distance") - 1 for d in energies_interval]

        return merged_labels, merged_energies  # merged_labels_mat contains labels as columns up to n0 - 1

    def build_labels_from_cluster_merging(self, plot_results: bool = False):

        potential_pairs = self.compute_energy_between_potential_wells()
        potential_matrix_condensed = self.compute_shortest_path(potential_pairs)
        potential_linkage = linkage(potential_matrix_condensed, method="single")
        merged_labels, energies = PQC.labels_from_linkage(potential_linkage)

        labels = self.proba_labels if self.merge_by_proba_clusters else self.sgd_labels

        cluster_number = np.array([len(np.unique(x)) for x in merged_labels])
        unique_energies_plus0 = energies

        merged_sample_labels = np.zeros((self.N_gen, len(energies)))
        merged_sample_labels[:, 0] = labels
        for i in range(1, len(energies)):
            cluster_to_merge = merged_labels[i]
            for j, e_labels in enumerate(np.unique(cluster_to_merge)):
                for h in np.nonzero(cluster_to_merge == e_labels)[0]:
                    merged_sample_labels[labels == self.k_ids[h], i] = j

        merged_sample_proba_labels = np.zeros_like(merged_sample_labels)
        merged_loglikelihood = np.zeros(len(energies))
        cluster_proba_number = np.zeros(len(energies), dtype=np.int)
        for k in range(len(energies)):
            _, p_labels, ll = self.cluster_probability_per_sample_batched(data_train=self.data_gen,
                                                                          labels=merged_sample_labels[:, k],
                                                                          add_noise=True)
            cluster_proba_number[k] = len(np.unique(p_labels))
            merged_loglikelihood[k] = ll  # if ll != 0 else np.nan
            merged_sample_proba_labels[:, k] = p_labels

        if plot_results:
            fig, ax = plt.subplots(2, 1,
                                   dpi=150,
                                   figsize=(10, 10),
                                   gridspec_kw={'height_ratios': [1.5, 1]})
            if "vector" not in self.sigmas_id[0]:
                sigma_title = "_".join([str(x) for x in self.sigmas_id])
            else:
                sigma_title = "vector_mean_" + np.mean(self.sigmas_id[1]).round(3)

            fig.suptitle('Labels by Energy merging with sigma: ' + sigma_title)
            _ = dendrogram(potential_linkage,
                           orientation='right',
                           distance_sort='descending',
                           show_leaf_counts=True,
                           ax=ax[0])
            ax[0].set_xlim(xmin=-1)
            ax[1].plot(unique_energies_plus0, merged_loglikelihood, '-*')
            ax[1].grid(which="both")
            ax[1].set_xlim(xmin=-1)
            ax[1].set_xlabel("Merging energy")
            ax[1].set_ylabel("-Log-likelihood P(K|X)")
            for i, k in enumerate(cluster_proba_number):
                if k % 5 == 0:
                    ax[1].annotate(
                        str(k),
                        xy=(unique_energies_plus0[i], merged_loglikelihood[i]),
                        xytext=(-3, 3),
                        textcoords='offset points',
                        ha='right',
                        va='bottom'
                    )
            plt.show()

        self.energy_merge_results.update(
            {
                self.sigmas_id: {
                    "merged_sgd_labels": merged_sample_labels,
                    "energies": unique_energies_plus0,
                    "sgd_cluster_number": cluster_number,
                    "merged_proba_labels": merged_sample_proba_labels,
                    "proba_cluster_number": cluster_proba_number,
                    "merged_loglikelihood": merged_loglikelihood
                }
            }
        )

    def hierarchical_energy_merge(self, plot_results: bool = False):
        if self.k_num > self.max_cluster_number:
            warnings.warn("Too many cluster number to compute energies between potential wells.")
            return False

        self.get_potential_wells()
        if len(self.k_centroids_list) < 2:
            warnings.warn("Degenerate solution. There is only one cluster.")
            return False

        self.build_labels_from_cluster_merging(plot_results=plot_results)

        return True

    def scan_multiple_sigmas(self,
                             knn_ratios: Optional[Union[np.ndarray, List[float]]] = None,
                             plot2d: bool = True,
                             plot3d: bool = True):
        if knn_ratios is None:
            knn_ratios = np.linspace(0.1, 0.4, 4)
        t0 = time.time()
        self.pre_compute_knn_variance(knn_ratios=knn_ratios)
        for knn_ratio_i in knn_ratios:
            self.set_sigmas(knn_ratio=knn_ratio_i)
            self.cluster_allocation_by_sgd()
            self.cluster_allocation_by_probability()
            do_it = self.hierarchical_energy_merge()
            print((knn_ratio_i, do_it))
        t1 = time.time()

        print(f"Scanning time: {round(t1 - t0, 3)} s")
        if plot2d and len(self.energy_merge_results) > 0:
            res_2d = pd.DataFrame(
                [
                    {
                        "sigmas": k[1],
                        "SGD_K": v["sgd_cluster_number"][0],
                        "clusters_proba": v["proba_cluster_number"][0],
                        "likelihood": v["merged_loglikelihood"][0]
                    } for k, v in self.energy_merge_results.items()
                ]
            )
            fig, ax = plt.subplots(2, 1)
            fig.suptitle("Cluster number and ANLL per knn% without merging energies")
            ax[0].plot(res_2d["sigmas"], res_2d["likelihood"], "-*")
            ax[0].grid(which="both")
            ax[0].set_ylabel("ANLL")
            ax[1].semilogy(res_2d["sigmas"], res_2d["clusters_proba"], "-+")
            ax[1].grid(which="both")
            ax[1].set_xlabel("% KNN")
            ax[1].set_ylabel("# Clusters")

            plt.show()

        if plot3d and len(self.energy_merge_results) > 0:
            self.plot_3d_results()

        print("End of function!")

    def plot_3d_results(self, exclude_1cluster_solution: bool = False):
        results = []
        for k in self.energy_merge_results.keys():
            if k[0] == "knn_ratio":
                # energies = np.zeros(len(self.energy_merge_results[k]["energies"]) + 1)
                # energies[1:] = self.energy_merge_results[k]["energies"]
                energies = self.energy_merge_results[k]["energies"]
                knn = k[1]
                energy_max = max(energies)
                ll = self.energy_merge_results[k]["merged_loglikelihood"]
                results.append((energies, ll, knn, energy_max))
        all_energies = np.unique(np.concatenate([x[0] for x in results]))
        energy_mat = np.repeat(all_energies.reshape([-1, 1]), len(results), axis=1)
        knn_mat = np.repeat(np.array([x[2] for x in results]).reshape([1, -1]), len(all_energies), axis=0)
        ll_mat = np.zeros([len(all_energies), len(results)])
        for i, res in enumerate(results):
            f = interpolate.interp1d(res[0], res[1], fill_value="extrapolate", kind="previous")
            ll_mat[:, i] = f(all_energies)

        ll_mat = np.round(ll_mat, 4)
        if exclude_1cluster_solution:
            ll_mat[ll_mat == 0] = np.nan

        plot_types = ["triangulation", "wireframe", "surface"]
        plot_type = plot_types[2]
        fig = plt.figure(dpi=150)
        ax = fig.add_subplot(projection='3d')
        if plot_type == "triangulation":
            e_ravel = np.ravel(energy_mat)
            knn_ravel = np.ravel(knn_mat)
            ll_ravel = np.ravel(ll_mat)
            p = ax.plot_trisurf(e_ravel, knn_ravel, ll_ravel, cmap='viridis', edgecolor='none')
        elif plot_type == "wireframe":
            p = ax.plot_wireframe(knn_mat, energy_mat, ll_mat, rstride=1, cstride=1)
        else:
            p = ax.plot_surface(knn_mat, energy_mat, ll_mat, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        # ax.set_zlim([0.001, np.min([np.max(ll_mat), 1])])
        ax.set_ylabel("Energies")
        ax.set_xlabel("KNN ratio")
        ax.set_zlabel("ALL")
        ax.view_init(30, 30)
        fig.colorbar(p, ax=ax)
        # for angle in range(0, 360):
        #     ax.view_init(30, angle)
        #     plt.draw()
        #     plt.pause(.001)
        #     # plt.show()
        plt.show()
