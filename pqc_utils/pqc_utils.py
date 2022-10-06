from math import log10, floor
from typing import Union

import numba as nb
import numpy as np
import tensorflow as tf

from pqc_utils.numba_sort import merge_sort_main_numba
from collections.abc import Iterable
import sys


def sizeof_fmt(num, suffix='B'):
    """ by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified"""
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def size_of_all_variables(look_in_globals=False, show_items=20):
    if look_in_globals:
        var_dict = globals().items()
    else:
        var_dict = locals().items()
    var_dict = sorted([(name, size_of_variable(value)) for name, value in var_dict], key=lambda x: -x[1])[:show_items]
    for name, size in var_dict:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))


def size_of_variable(iterable_var):
    if isinstance(iterable_var, Iterable):
        size_var = 0
        for var in iterable_var:
            if isinstance(var, Iterable):
                size_var += size_of_variable(var)
            else:
                size_var += sys.getsizeof(var)
    else:
        size_var = sys.getsizeof(iterable_var)
    return size_var


def set_float_type(float_type: int):
    if float_type == 16:
        return "float16"  # tf.float16
    elif float_type == 32:
        return "float32"  # tf.float32
    elif float_type == 64:
        return "float64"  # tf.float64
    else:
        raise ValueError("Float precision must be one of these [16, 32, 64]")


@tf.function
def pairwise_d2_mat_v1(a, b):
    return tf.reduce_sum((tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)) ** 2, axis=2)


@tf.function
def pairwise_d2_mat_v2(a, b):
    r_a = tf.reduce_sum(a * a, 1)
    r_b = tf.reduce_sum(b * b, 1)

    # turn r into column vector
    r_a = tf.reshape(r_a, [-1, 1])
    r_b = tf.reshape(r_b, [-1, 1])
    return r_a - 2 * tf.matmul(a, tf.transpose(b)) + tf.transpose(r_b)


# @nb.jit(nb.float32[:](nb.float32[:], nb.float32[:]), nopython=True)
@nb.njit
def pairwise_d2_mat_v3(a, b):
    r_a = np.sum(a * a, 1)
    r_b = np.sum(b * b, 1)

    # turn r into column vector
    r_a = np.expand_dims(r_a, axis=1)
    r_b = np.expand_dims(r_b, axis=1)
    return r_a - 2 * np.dot(a, np.transpose(b)) + np.transpose(r_b)


@tf.function
def agg_sum(data, vector, value, op=0):
    subset = tf.boolean_mask(data, tf.equal(vector, value))
    if op == 0:
        return tf.reduce_sum(subset, axis=0)
    elif op == 1:
        return tf.reduce_mean(subset, axis=0)
    elif op == 2:
        return subset[0, :]
    else:
        raise ValueError("Unknown operation")


def round_it(x, sig):
    return round(x, sig - int(floor(log10(abs(x)))) - 1)


@nb.njit  # @nb.jit(nopython=True, fastmath=True, parallel=True)
def reduce_mean(data, vector):
    u_values = np.unique(vector)
    n, d = data.shape
    reduced_data = np.empty((u_values.size, d))
    for i, k in enumerate(u_values):
        reduced_data[i, :] = nb_mean(data[vector == k], axis=0)

    return reduced_data, u_values


@nb.njit
def reduce_sum(data, vector):
    u_values = np.unique(vector)
    n, d = data.shape
    reduced_data = np.empty((u_values.size, d))
    for i, k in enumerate(u_values):
        reduced_data[i, :] = nb_sum(data[vector == k], axis=0)

    return reduced_data, u_values


@nb.njit
def reduce_first(data, vector):
    u_values = np.unique(vector)
    n, d = data.shape
    reduced_data = np.empty((u_values.size, d))
    for i, k in enumerate(u_values):
        reduced_data[i, :] = data[vector == k][0, :]

    return reduced_data, u_values


@nb.njit
def no_reduce_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    m, n = np.shape(arr)
    result = np.empty_like(arr, dtype=arr.dtype)
    if axis == 0:
        for i in range(n):
            result[:, i] = func1d(arr[:, i])
    else:
        for i in range(m):
            result[i, :] = func1d(arr[i, :])
    return result


@nb.njit
def reduce_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=arr.dtype)
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result


@nb.njit
def nb_mean(array, axis):
    return reduce_along_axis(np.mean, axis, array)


@nb.njit
def nb_std(array, axis):
    return reduce_along_axis(np.std, axis, array)


@nb.njit
def nb_sum(array, axis):
    return reduce_along_axis(np.sum, axis, array)


@nb.njit
def nb_sort(array: np.ndarray, axis: int):
    return no_reduce_along_axis(merge_sort_main_numba, axis, array)


def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


@tf.function
def split_data(a, idx, axis=0):
    n = tf.shape(a)[axis]
    t = tf.ones_like(idx, dtype=tf.bool)
    m = tf.scatter_nd(tf.expand_dims(idx, 1), t, [n])
    return tf.boolean_mask(a, m, axis=axis), tf.boolean_mask(a, ~m, axis=axis)


optimizers_classes = Union[
    tf.keras.optimizers.SGD,
    tf.keras.optimizers.RMSprop,
    tf.keras.optimizers.Adam,
    tf.keras.optimizers.Adadelta,
    tf.keras.optimizers.Adagrad,
    tf.keras.optimizers.Adamax,
    tf.keras.optimizers.Nadam,
    tf.keras.optimizers.Ftrl
]
