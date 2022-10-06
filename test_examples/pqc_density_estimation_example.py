import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from main.density_estimation import DensityEstimator
from main.pqc import PQC
from pqc_utils.generate_toy_datasets import get_2d_toy_data

D = 2
N = 3040
batch_manual = 1000

x_gen, y = get_2d_toy_data("noisy_moons", n_samples=N, noise=0.15)

x_gen = np.concatenate([x_gen, x_gen.max(0).reshape(1, -1), x_gen.min(0).reshape(1, -1)], axis=0)
y = np.concatenate([y, [-1], [-1]])

# x_gen = np.random.randn(N, D) * 5
# x_gen[:int(0.5 * N)] += + 20
# x_gen[:int(0.5 * N), 0] *= x_gen[:int(0.5 * N), 1] * 0.01
# x_gen[int(0.25 * N):int(0.5 * N), ::2] += 30
#
# y = np.zeros(N)
# y[:int(0.5 * N)] = 1
# y[int(0.25 * N):int(0.5 * N)] = 2

scaler = StandardScaler()

x_gen = scaler.fit_transform(x_gen)

preview = True
if preview:
    sns.scatterplot(x=x_gen[:, 0], y=x_gen[:, 1], alpha=0.3, c=y)
    plt.show()

pqc = PQC(data_gen=x_gen, float_type=32, batch=batch_manual, force_cpu=True)

d_e = DensityEstimator(data_gen=pqc.data_gen, batch=batch_manual, scale=pqc.scale)

init_log_sigmas = np.ones((d_e.data_gen.shape[0], 1)) * np.log(3)

ll = d_e.fit(preset_init=init_log_sigmas)

log_sigmas = d_e.log_sigma.value()
sigmas = np.exp(log_sigmas)

if preview:
    plt.plot(ll)
    plt.show()

    cm = plt.cm.get_cmap('RdYlBu')
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    sc0 = axs[0].scatter(x_gen[:, 0], x_gen[:, 1], c=log_sigmas, vmin=min(log_sigmas), vmax=max(log_sigmas),
                         s=35, cmap=cm, alpha=0.4)
    sc1 = axs[1].scatter(x_gen[:, 0], x_gen[:, 1], c=sigmas, vmin=min(sigmas), vmax=max(sigmas),
                         s=35, cmap=cm, alpha=0.3)
    fig.colorbar(sc0, ax=axs[0])
    fig.colorbar(sc1, ax=axs[1])
    plt.show()

print("End of script!")
