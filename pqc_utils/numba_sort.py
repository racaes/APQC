import numba as nb
import numpy as np

SMALL_MERGESORT_NUMBA = 40


@nb.jit(nopython=True)
def merge_numba(a, aux, lo, mid, hi):

    i = lo
    j = mid + 1
    for k in range(lo, hi + 1):
        if i > mid:
            aux[k] = a[j]
            j += 1
        elif j > hi:
            aux[k] = a[i]
            i += 1
        elif a[j] < a[i]:
            aux[k] = a[j]
            j += 1
        else:
            aux[k] = a[i]
            i += 1


@nb.jit(nopython=True)
def insertion_sort_numba(a, lo, hi):

    for i in range(lo + 1, hi + 1):
        key = a[i]
        j = i - 1
        while (j >= lo) & (a[j] > key):
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key


@nb.jit(nopython=True)
def merge_sort_numba(a, aux, lo, hi):

    if hi - lo > SMALL_MERGESORT_NUMBA:
        mid = lo + ((hi - lo) >> 1)
        merge_sort_numba(aux, a, lo, mid)
        merge_sort_numba(aux, a, mid + 1, hi)
        if a[mid] > a[mid + 1]:
            merge_numba(a, aux, lo, mid, hi)
        else:
            for i in range(lo, hi + 1):
                aux[i] = a[i]
    else:
        insertion_sort_numba(aux, lo, hi)


@nb.jit(nopython=True)
def merge_sort_main_numba(a):
    b = np.copy(a)
    aux = np.copy(a)
    merge_sort_numba(aux, b, 0, len(b) - 1)
    return b
