import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# import tensorflow as tf
from sklearn.preprocessing import StandardScaler
# import sys
from main.pqc import PQC
from pqc_utils.generate_toy_datasets import get_2d_toy_data
from main.density_estimation import DensityEstimator


D = 2
samples = 2000
batch_manual = 1000
scan_length = 3

X, y = get_2d_toy_data("original_paper_toy_data_1", n_samples=samples, noise=0.15)

scaler = StandardScaler()

x_gen = scaler.fit_transform(X)

find_best_sigma = True
preview = True
if preview:
    sns.scatterplot(x=x_gen[:, 0], y=x_gen[:, 1], alpha=0.6, hue=y.flatten(), palette="deep")
    plt.show()


pqc = PQC(data_gen=x_gen, float_type=32, batch=batch_manual, force_cpu=True)

result_dict = {
    "sigmas": [],
    "clusters_sgd": [],
    "clusters_proba": [],
    "likelihood": []
}

if preview and find_best_sigma:
    d_e = DensityEstimator(data_gen=pqc.data_gen, batch=batch_manual, scale=pqc.scale)

    init_log_sigmas = np.ones((d_e.data_gen.shape[0], 1)) * np.log(1)

    ll = d_e.fit(preset_init=init_log_sigmas)

    log_sigmas = d_e.log_sigma.value()
    sigmas = np.exp(log_sigmas)

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

    pqc.set_sigmas(sigma_value=sigmas)
    pqc.cluster_allocation_by_sgd()
    pqc.cluster_allocation_by_probability()
    sgd_k = np.unique(pqc.sgd_labels).shape[0]
    proba_k = np.unique(pqc.proba_labels).shape[0]
    result_dict["sigmas"].append(np.mean(sigmas))
    result_dict["clusters_sgd"].append(sgd_k)
    result_dict["clusters_proba"].append(proba_k)
    result_dict["likelihood"].append(pqc.ll)

for i, sigma_i in enumerate(np.linspace(0.2, 0.6, scan_length)):
    pqc.set_sigmas(knn_ratio=sigma_i)
    pqc.cluster_allocation_by_sgd()
    pqc.cluster_allocation_by_probability()
    sgd_k = np.unique(pqc.sgd_labels).shape[0]
    proba_k = np.unique(pqc.proba_labels).shape[0]
    result_dict["sigmas"].append(sigma_i)
    result_dict["clusters_sgd"].append(sgd_k)
    result_dict["clusters_proba"].append(proba_k)
    result_dict["likelihood"].append(pqc.ll)

results = pd.DataFrame(result_dict)

plt.subplot(2, 1, 1)
plt.plot(results["sigmas"], results["likelihood"], "-*")
plt.grid(which="both")
plt.subplot(2, 1, 2)
plt.semilogy(results["sigmas"], results["clusters_proba"], "-+")
plt.grid(which="both")
plt.show()

best_solution_key = sorted([(k, v["loglikelihood"]) for k, v in pqc.basic_results.items()], key=lambda x: x[1])[0][0]
best_proba_labels = pqc.basic_results[best_solution_key]["proba_labels"]
proba_x_k = pqc.basic_results[best_solution_key]["proba_winner"]

sns.scatterplot(x=x_gen[:, 0], y=x_gen[:, 1], alpha=0.4, hue=best_proba_labels, size=proba_x_k, palette="deep")
plt.show()

knn_ratios = np.linspace(0.25, 0.35, 3)
pqc.scan_multiple_sigmas(knn_ratios=knn_ratios)

print("End of script!")
