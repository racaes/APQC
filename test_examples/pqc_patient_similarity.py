import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# import tensorflow as tf
from sklearn.preprocessing import StandardScaler
# import sys
from main.pqc import PQC
# from pqc_utils.generate_toy_datasets import get_2d_toy_data
from main.density_estimation import DensityEstimator


path_file = "../data/sensitive_data/ps_num_std.csv"
df = pd.read_csv(path_file)

already_normalized = True
y = None
find_best_sigma = True
preview = True
batch_manual = 5000
scan_length = 5

if not already_normalized:
    scaler = StandardScaler()
    x_gen = scaler.fit_transform(df)
else:
    x_gen = df.to_numpy()

if x_gen.shape[1] > 2:
    from sklearn.decomposition import PCA
    x_pca = PCA(n_components=2).fit_transform(x_gen)
else:
    x_pca = x_gen[:, :2]

if preview:
    sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], alpha=0.3, palette="deep")
    plt.show()

pqc = PQC(data_gen=x_gen, float_type=32, batch=batch_manual, force_cpu=True)

result_dict = {
    "sigma_type": [],
    "sigma_mean": [],
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

    sc0 = axs[0].scatter(x_pca[:, 0], x_pca[:, 1], c=log_sigmas, vmin=min(log_sigmas), vmax=max(log_sigmas),
                         s=35, cmap=cm, alpha=0.4)
    sc1 = axs[1].scatter(x_pca[:, 0], x_pca[:, 1], c=sigmas, vmin=min(sigmas), vmax=max(sigmas),
                         s=35, cmap=cm, alpha=0.3)
    fig.colorbar(sc0, ax=axs[0])
    fig.colorbar(sc1, ax=axs[1])
    plt.show()

    pqc.set_sigmas(sigma_value=sigmas)
    pqc.cluster_allocation_by_sgd()
    pqc.cluster_allocation_by_probability()
    sgd_k = np.unique(pqc.sgd_labels).shape[0]
    proba_k = np.unique(pqc.proba_labels).shape[0]

    result_dict["sigma_type"].append("trained_factor_00")
    result_dict["sigma_mean"].append(np.mean(sigmas))
    result_dict["clusters_sgd"].append(sgd_k)
    result_dict["clusters_proba"].append(proba_k)
    result_dict["likelihood"].append(pqc.ll)

    for i, factor_sigma in enumerate([15, 20, 25, 30, 35]):
        pqc.set_sigmas(sigma_value=sigmas*factor_sigma)
        pqc.cluster_allocation_by_sgd()
        pqc.cluster_allocation_by_probability()
        sgd_k = np.unique(pqc.sgd_labels).shape[0]
        proba_k = np.unique(pqc.proba_labels).shape[0]

        result_dict["sigma_type"].append(f"trained_factor_{factor_sigma}")
        result_dict["sigma_mean"].append(np.mean(sigmas*factor_sigma))
        result_dict["clusters_sgd"].append(sgd_k)
        result_dict["clusters_proba"].append(proba_k)
        result_dict["likelihood"].append(pqc.ll)

for i, sigma_i in enumerate([0.0050, 0.0075, 0.0100, 0.0150, 0.0200, 0.0250]):
    pqc.set_sigmas(knn_ratio=sigma_i)
    pqc.cluster_allocation_by_sgd()
    pqc.cluster_allocation_by_probability()
    sgd_k = np.unique(pqc.sgd_labels).shape[0]
    proba_k = np.unique(pqc.proba_labels).shape[0]

    result_dict["sigma_type"].append(f"knn_ratio_{sigma_i}")
    result_dict["sigma_mean"].append(np.mean(pqc.sigmas.value().numpy()))
    result_dict["clusters_sgd"].append(sgd_k)
    result_dict["clusters_proba"].append(proba_k)
    result_dict["likelihood"].append(pqc.ll)

results = pd.DataFrame(result_dict)

results = results.sort_values("sigma_mean")

plt.subplot(2, 1, 1)
plt.plot(results["sigma_mean"], results["likelihood"], "-*")
plt.grid(which="both")
plt.subplot(2, 1, 2)
plt.semilogy(results["sigma_mean"], results["clusters_proba"], "-+")
plt.grid(which="both")
plt.show()

best_solution_key = sorted([(k, v["loglikelihood"]) for k, v in pqc.basic_results.items()], key=lambda x: x[1])[0][0]
best_proba_labels = pqc.basic_results[best_solution_key]["proba_labels"]
proba_x_k = pqc.basic_results[best_solution_key]["proba_winner"]

plt.figure()
sns.scatterplot(x=x_pca[:, 0], y=x_pca[:, 1], alpha=0.4, hue=best_proba_labels, size=proba_x_k, palette="deep")
plt.show()

plt.figure(figsize=(20, 20))
sns.scatterplot(data=results, x="sigma_mean", y="likelihood", alpha=0.4, hue="sigma_type", size="clusters_proba")
plt.show()

print("End of script!")
