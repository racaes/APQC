import numpy as np
from sklearn import datasets


def get_2d_toy_data(data_type: str, n_samples=1500, noise=0.05, seed=170):
    """
    # ============
    # Generate datasets. We choose the size big enough to see the scalability
    # of the algorithms, but not too big to avoid too long running times
    # ============
    """
    if data_type == "noisy_circles":
        noisy_circles = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=noise)
        return noisy_circles

    elif data_type == "noisy_moons":
        noisy_moons = datasets.make_moons(n_samples=n_samples, noise=noise)
        return noisy_moons

    elif data_type == "blobs":
        blobs = datasets.make_blobs(n_samples=n_samples, random_state=seed)
        return blobs

    elif data_type == "no_structure":
        no_structure = np.random.rand(n_samples, 2), None
        return no_structure

    elif data_type == "aniso":
        # Anisotropicly distributed data
        x, y = datasets.make_blobs(n_samples=n_samples, random_state=seed)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        x_aniso = np.dot(x, transformation)
        aniso = (x_aniso, y)
        return aniso

    elif data_type == "varied":
        # blobs with varied variances
        varied = datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=seed)
        return varied

    elif "original_paper" in data_type:
        import scipy.io
        mat = scipy.io.loadmat('../data/datasets4.mat')
        if data_type == "original_paper_toy_data_1":
            return mat["data1"][0, 0]
        elif data_type == "original_paper_toy_data_2":
            return mat["data2"][0, 0]
        elif data_type == "original_paper_toy_data_3":
            return mat["data3"][0, 0]
        elif data_type == "original_paper_toy_data_4":
            return mat["data4"][0, 0]
        elif data_type == "original_paper_toy_data_5":
            return mat["data5"][0, 0]
        else:
            raise ValueError("Data type not recognized!")
    else:
        raise ValueError("Data type not recognized!")
