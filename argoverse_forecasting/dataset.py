import pickle
import numpy as np

from .utils.baseline_config import FEATURE_FORMAT

def pickle_features_to_numpy_trajectories(pickle_file_path: str):
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)

    # n_samples, horizon, n_features
    result = np.zeros((data.shape[0], data['FEATURES'][0].shape[0], 2))

    result[:, :, 0] = np.stack(data['FEATURES'].values)[:, :, FEATURE_FORMAT["X"]]
    result[:, :, 1] = np.stack(data['FEATURES'].values)[:, :, FEATURE_FORMAT["Y"]]

    return result
