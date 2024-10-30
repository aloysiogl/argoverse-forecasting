import numpy as np

from .utils.baseline_config import FEATURE_FORMAT


def xy_to_features(array):
    nb_features = len(FEATURE_FORMAT)
    output = np.zeros((array.shape[0], nb_features))

    output[:, FEATURE_FORMAT["X"]] = array[:, 0]
    output[:, FEATURE_FORMAT["Y"]] = array[:, 1]

    return output
