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
samples = 2000
batch_manual = 100
scan_length = 3
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

X, y = get_2d_toy_data("original_paper_toy_data_1", n_samples=samples, noise=0.15)

scaler = StandardScaler()

x_gen = scaler.fit_transform(X)

scan_sigma = True
find_best_sigma = True
preview = False
if preview:
    sns.scatterplot(x=x_gen[:, 0], y=x_gen[:, 1], alpha=0.6, hue=y.flatten(), palette="deep")
    plt.show()


pqc = PQC(data_gen=x_gen, float_type=32, batch=batch_manual, force_cpu=True)

result_dict = {
    "sigmas": [],
    "clusters_sgd": [],
    "clusters_proba": [],
    "likelihood": [],
    "ll_trained": []
}

d_e = DensityEstimator(data_gen=pqc.data_gen, batch=batch_manual, scale=pqc.scale, optimizer=optimizer)

for i in range(10):

    if i == 0:
        pqc.set_sigmas(knn_ratio=0.25)
        sigmas, log_sigmas = None, None
        ll = None
        sigma_dif = None
    else:
        if not scan_sigma:
            d_e.set_clusters(pqc.proba_labels)
            ll = d_e.fit(preset_init=pqc.sigmas, steps=1)
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

        sc0 = axs[0].scatter(x_gen[:, 0], x_gen[:, 1], c=log_sigmas, vmin=min(log_sigmas), vmax=max(log_sigmas),
                             s=35, cmap=cm, alpha=0.4)
        sc1 = axs[1].scatter(x_gen[:, 0], x_gen[:, 1], c=sigmas, vmin=min(sigmas), vmax=max(sigmas),
                             s=35, cmap=cm, alpha=0.3)
        fig.colorbar(sc0, ax=axs[0])
        fig.colorbar(sc1, ax=axs[1])
        plt.show()

    if preview and sigma_dif is not None:
        fig, ax = plt.subplots(figsize=(12, 12))
        sc0 = ax.scatter(x_gen[:, 0], x_gen[:, 1], c=sigma_dif, s=35, cmap=cm, alpha=0.3)
        fig.colorbar(sc0, ax=ax)
        plt.show()

plt.figure()
plt.plot(result_dict["sigmas"], result_dict["clusters_proba"], '*-')
plt.show()

plt.figure()
plt.plot(result_dict["sigmas"], result_dict["likelihood"], '*-')
plt.show()

plt.figure()
sns.scatterplot(x=x_gen[:, 0], y=x_gen[:, 1], alpha=0.4, hue=pqc.proba_labels, size=pqc.k_proba, palette="deep")
plt.show()

print("End of script!")
