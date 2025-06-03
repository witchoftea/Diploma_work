import pickle
import scipy.sparse

from lightfm import LightFM


def load_lightfm_model(save_dir="data/models"):
    with open(f"{save_dir}/lightfm_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{save_dir}/lightfm_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    item_features = scipy.sparse.load_npz(f"{save_dir}/lightfm_item_features.npz")
    with open(f"{save_dir}/lightfm_mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
    return model, dataset, item_features, mappings


def load_svd_model(save_dir="data/models"):
    with open(f"{save_dir}/svd_model.pkl", "rb") as f:
        return pickle.load(f)


def load_knn_model(save_dir="data/models"):
    with open(f"{save_dir}/knn_model.pkl", "rb") as f:
        return pickle.load(f)