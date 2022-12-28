import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from collections import Counter
# import sys
from main.pqc import PQC
from pqc_utils.generate_toy_datasets import get_2d_toy_data
from main.density_estimation import DensityEstimator

cm = plt.cm.get_cmap('RdYlBu')
D = 2
samples = 10
batch_manual = 100
scan_length = 3
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)
optimizer = tf.keras.optimizers.Nadam(learning_rate=0.05)

# X, y = get_2d_toy_data("original_paper_toy_data_1", n_samples=samples, noise=0.15)
X, y = get_2d_toy_data("blobs", n_samples=samples, noise=1.5)

scaler = StandardScaler()

x_gen = scaler.fit_transform(X)

scan_sigma = False
# find_best_sigma = True
preview = False
pqc_sigmas_check = False
if preview:
    sns.scatterplot(x=x_gen[:, 0], y=x_gen[:, 1], alpha=0.6, hue=y.flatten(), palette="deep")
    plt.show()


pqc = PQC(data_gen=x_gen, float_type=32, batch=batch_manual, force_cpu=False, merge_by_proba_clusters=False)

result_dict = {
    "sigmas": [],
    "clusters_sgd": [],
    "clusters_proba": [],
    "likelihood": [],
    "ll_trained": []
}

d_e = DensityEstimator(data_gen=pqc.data_gen, batch=int(0.25*len(X)), scale=pqc.scale, optimizer=optimizer)

if pqc_sigmas_check:
    knn_ratios = np.linspace(0.10, 0.4, 30)
    pqc.scan_multiple_sigmas(knn_ratios=knn_ratios, plot2d=True, plot3d=True)
    print("PQC_sigmas_check!")

for i in range(40):

    if i == 0:
        pqc.set_sigmas(knn_ratio=0.50)
        sigmas, log_sigmas = None, None
        ll = None
        sigma_dif = None
    else:
        if not scan_sigma:
            d_e.set_clusters(pqc.proba_labels)
            ll = d_e.fit(preset_init=pqc.sigmas, steps=25)
            # if len(np.unique(pqc.proba_labels)) > 1:
            #     d_e.set_clusters(pqc.proba_labels)
            #     ll = d_e.fit(preset_init=pqc.sigmas, steps=1)
            # else:
            #     d_e.reset_clusters()
            #     ll = d_e.fit(preset_init=pqc.sigmas, steps=100)

            log_sigmas = d_e.log_sigma.value()
            sigmas = np.exp(log_sigmas)
        else:
            sigmas = np.array(pqc.sigmas.value() * 1.2)
            log_sigmas = np.log(sigmas)
            ll = None

        print(pd.DataFrame(sigmas - pqc.sigmas.value()).describe())
        sigma_dif = (sigmas - pqc.sigmas.value()).numpy()
        pqc.set_sigmas(sigma_value=sigmas)

    pqc.cluster_allocation_by_sgd()
    pqc.cluster_allocation_by_probability()

    sgd_k = np.unique(pqc.sgd_labels).shape[0]
    proba_k = np.unique(pqc.proba_labels).shape[0]
    print("SGD clusters:\n", Counter(pqc.sgd_labels))
    print("Proba clusters:\n", Counter(pqc.proba_labels))

    result_dict["sigmas"].append(np.mean(pqc.sigmas))
    result_dict["clusters_sgd"].append(sgd_k)
    result_dict["clusters_proba"].append(proba_k)
    result_dict["likelihood"].append(pqc.ll)

    result_dict["ll_trained"].append(ll)

    if preview and sigmas is not None:

        fig, axs = plt.subplots(2, 1, figsize=(12, 8))
        axs[0].set_title("Log sigma")
        axs[1].set_title("Sigma")
        sc0 = axs[0].scatter(x_gen[:, 0], x_gen[:, 1], c=log_sigmas, vmin=min(log_sigmas), vmax=max(log_sigmas),
                             s=35, cmap=cm, alpha=0.4)
        sc1 = axs[1].scatter(x_gen[:, 0], x_gen[:, 1], c=sigmas, vmin=min(sigmas), vmax=max(sigmas),
                             s=35, cmap=cm, alpha=0.3)
        fig.colorbar(sc0, ax=axs[0])
        fig.colorbar(sc1, ax=axs[1])
        plt.show()

    if preview and sigma_dif is not None:
        _, ax = plt.subplots(figsize=(12, 12))
        ax.stem(np.arange(len(sigma_dif)), sigma_dif)
        ax.set_title(f"Iteration {i}")
        plt.show()

        fig, ax = plt.subplots(figsize=(12, 12))
        sc0 = ax.scatter(x_gen[:, 0], x_gen[:, 1], c=sigma_dif, s=35, cmap=cm, alpha=0.3)
        fig.colorbar(sc0, ax=ax)
        plt.show()

    import time
    print(f"Iteration {i}")
    time.sleep(1)
    print(" has ended.")

_, axs = plt.subplots(4, 1, figsize=(6, 12))
axs[0].plot(result_dict["sigmas"], '*-')
axs[1].plot(result_dict["clusters_sgd"], '*-')
axs[2].plot(result_dict["clusters_proba"], '*-')
axs[3].plot(result_dict["likelihood"], '*-')
axs[0].set_ylabel("sigmas")
axs[1].set_ylabel("clusters_sgd")
axs[2].set_ylabel("clusters_proba")
axs[3].set_ylabel("likelihood")
axs[3].set_xlabel("iteration")
plt.show()

plt.figure()
sns.scatterplot(x=x_gen[:, 0].flatten(), y=x_gen[:, 1].flatten(), alpha=0.4,
                hue=pqc.proba_labels.flatten(), size=pqc.k_proba.flatten(), palette="deep")
plt.show()

all_ll_trained = np.concatenate(result_dict["ll_trained"][1:])
plt.figure()
plt.plot(all_ll_trained)
plt.show()
print("End of script!")
