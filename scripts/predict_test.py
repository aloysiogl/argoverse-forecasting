import numpy as np

import argoverse_forecasting.utils.baseline_utils as baseline_utils

from argoverse_forecasting.model import LSTMForecaster

path = "./saved_models/lstm/LSTM_rollout30.pth.tar"
forecaster = LSTMForecaster(path)

prediction = forecaster.predict(np.random.rand(50, 2))

print(prediction)