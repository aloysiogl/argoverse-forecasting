import pickle

from argoverse_forecasting.dataset import pickle_features_to_numpy_trajectories
from argoverse_forecasting.model import LSTMForecaster

np_dataset = pickle_features_to_numpy_trajectories(
    "./feat/forecasting_features_val.pkl"
)

path = "./saved_models/lstm/LSTM_rollout30.pth.tar"
forecaster = LSTMForecaster(path)

gts = np_dataset[:, 20:, :]

predictions = forecaster.predict_multiple(np_dataset)
print("Predictions shape:", predictions.shape)

with open("./predictions_calibration_data/argoverse_val_20_30.pkl", "wb") as f:
    pickle.dump((gts, predictions), f)
