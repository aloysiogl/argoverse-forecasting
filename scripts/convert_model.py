import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

from argoverse_forecasting.dataset import pickle_features_to_numpy_trajectories
from argoverse_forecasting.model import LSTMForecaster

np_dataset = pickle_features_to_numpy_trajectories(
    "./feat/forecasting_features_val.pkl"
)

cpu_forecaster_path = "./saved_models/lstm/LSTM_rollout30_cpu.pth.tar"

path = "./saved_models/lstm/LSTM_rollout30.pth.tar"
forecaster_torch = torch.load(path, map_location=torch.device('cpu'))
torch.save(forecaster_torch, cpu_forecaster_path)

# forecaster = LSTMForecaster(cpu_forecaster_path)
