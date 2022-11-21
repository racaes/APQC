import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def psi_f(sigma: float = 1, x: float = 0.5, n: int = 2):
    psi = (2 * math.pi * sigma) ** (-n / 2) * np.exp(-(x / sigma) ** 2)
    return psi


def density(sigma: float = 1, x: float = 0.5, n: int = 2, a: float = 5, b: float = 2):
    psi = psi_f(sigma, x, n)
    return (psi + b) / (psi + b + a)


def density_derivative(sigma: float = 1, x: float = 1, n: int = 2, a: float = 5, b: float = 2):
    numerator = (b - a) * (n * sigma ** 2 - 4 * sigma) * (2 * math.pi * sigma) ** (n / 2) * np.exp((x / sigma) ** 2)
    denominator = 2 * sigma ** 3 * ((b + a) * (2 * math.pi * sigma) ** (n / 2) * np.exp((x / sigma) ** 2) + 1) ** 2
    return numerator / denominator


sigmas = np.linspace(0.1, 10, 100)

psis = np.array([psi_f(sigma=s) for s in sigmas])
dens = np.array([density(sigma=s) for s in sigmas])
der_dens = np.array([density_derivative(sigma=s) for s in sigmas])

fig, axs = plt.subplots(3, 1, figsize=(10, 10))

axs[0].plot(sigmas, psis, '*-')
axs[1].plot(sigmas, dens, '*-')
axs[2].plot(sigmas, der_dens, '+:')

plt.show()

print("end of script!")
