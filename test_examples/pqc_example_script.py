import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import sys
from main.pqc import PQC
from pqc_utils.pqc_utils import size_of_all_variables
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

nvmlInit()
h = nvmlDeviceGetHandleByIndex(0)
info = nvmlDeviceGetMemoryInfo(h)
print(f'total    : {info.total}')
print(f'free     : {info.free}')
print(f'used     : {info.used}')

# x_gen = np.array([0.0])
# x_train = np.linspace(-10, 10, 100)

# tf.debugging.enable_check_numerics()

D = 2
N = 3040
batch_manual = 1000  # Seems to work faster with 1k than 5k. With +8k appears OOM errors
x_gen = np.random.randn(N, D) * 5
x_train = np.random.randn(N, D) * 8
x_test = np.random.randn(N, D) * 9

x_gen[:int(0.5 * N)] += + 20
x_train[:int(0.5 * N)] += + 20
x_test[:int(0.5 * N)] += + 20

x_gen[int(0.25 * N):int(0.5 * N), ::2] += 30
x_train[int(0.25 * N):int(0.5 * N), ::2] += 30
x_test[int(0.25 * N):int(0.5 * N), ::2] += 30

scaler = StandardScaler()

x_gen = scaler.fit_transform(x_gen)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

preview = True
if preview:
    sns.scatterplot(x=x_gen[:, 0], y=x_gen[:, 1], alpha=0.3)
    plt.show()

size_of_all_variables()

x_mem = sys.getsizeof(x_gen)
mem_ratio = ((4 * x_mem) // (0.7 * info.free))
if mem_ratio < 1.0:
    batch = x_gen.shape[0]
else:
    batch = np.floor(x_gen.shape[0] / mem_ratio / x_gen.shape[1])

pqc = PQC(data_gen=x_gen, float_type=32, batch=batch_manual, force_cpu=True)

knn_ratios = np.linspace(0.15, 0.35, 3)
pqc.scan_multiple_sigmas(knn_ratios=knn_ratios)

scan_length = 4
result_dict = {
    "sigmas": [],
    "clusters_sgd": [],
    "clusters_proba": [],
    "likelihood": []
}

for i, sigma_i in enumerate(np.linspace(0.35, 0.85, scan_length)):
    pqc.set_sigmas(knn_ratio=sigma_i)
    pqc.cluster_allocation_by_sgd()
    pqc.cluster_allocation_by_probability()
    do_it = pqc.hierarchical_energy_merge()
    print(do_it)

    sgd_k = np.unique(pqc.sgd_labels).shape[0]
    proba_k = np.unique(pqc.proba_labels).shape[0]
    result_dict["sigmas"].append(sigma_i)
    result_dict["clusters_sgd"].append(sgd_k)
    result_dict["clusters_proba"].append(proba_k)
    result_dict["likelihood"].append(pqc.ll)

results = pd.DataFrame(result_dict)
results.plot("sigmas", "likelihood")

plt.subplot(2, 1, 1)
plt.plot(results["sigmas"], results["likelihood"], "-*")
plt.grid(which="both")
plt.subplot(2, 1, 2)
plt.semilogy(results["sigmas"], results["clusters_proba"], "-+")
plt.grid(which="both")
plt.show()

print(results)
data_gen = pqc.data_gen.numpy()
fig, ax = plt.subplots(dpi=150)
for k in np.unique(pqc.proba_labels):
    idx = pqc.proba_labels == k
    sns.scatterplot(x=data_gen[idx, 0], y=data_gen[idx, 1], alpha=0.3, ax=ax)
plt.show()


x_test /= pqc.scale
knn_ratio = 0.52
energy = 0
energy_id = np.nonzero(pqc.energy_merge_results[("knn_ratio", knn_ratio)]["energies"] == energy)[0]
best_result = pqc.energy_merge_results[("knn_ratio", knn_ratio)]["merged_proba_labels"][:, energy_id].flatten()
best_result0 = pqc.energy_merge_results[("knn_ratio", knn_ratio)]["merged_proba_labels"][:, 0].flatten()

k_proba, proba_labels, ll = pqc.cluster_probability_per_sample_batched(data_train=x_test,
                                                                       labels=best_result)
fig, ax = plt.subplots(dpi=150)
ax.scatter(pqc.data_gen.numpy()[:, 0], pqc.data_gen.numpy()[:, 1], alpha=0.5, marker='x', s=20)
for k in np.unique(best_result0):
    idx = best_result0 == k
    sns.scatterplot(x=x_test.numpy()[idx, 0], y=x_test.numpy()[idx, 1], alpha=0.3, ax=ax, size=3)
plt.show()

print("End of script!")
