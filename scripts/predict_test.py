import os
import time
import numpy as np
import torch
import torch.nn as nn

from argoverse_forecasting.features import xy_to_features
from argoverse_forecasting.model import DecoderRNN, EncoderRNN, infer_single_none
import argoverse_forecasting.utils.baseline_utils as baseline_utils
from argoverse_forecasting.utils.lstm_utils import ModelUtils, LSTMDataset

ROLLOUT_LENS = [1, 10, 30]
obs_len = 20
pred_len = 30
tot_len = obs_len + pred_len

computed_sequence = xy_to_features(np.random.rand(tot_len, 2))

args = type(
    "",
    (),
    {
        "normalize": True,
        "use_map": False,
        "use_social": False,
        "use_delta": True,
        "obs_len": obs_len,
        "pred_len": pred_len,
        "test_features": True,
        "test_batch_size": 1,
        "train_features": False,
        "val_features": False,
        "model_path": "./saved_models/lstm/LSTM_rollout30.pth.tar",
        "lr": 0.001,
    },
)()

if not baseline_utils.validate_args(args):
    print("Invalid args")
    exit(1)

baseline_key = "none"

data_dict = baseline_utils.get_data(args, baseline_key, computed_sequence)

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if use_cuda:
    print(f"Using all ({torch.cuda.device_count()}) GPUs...")

criterion = nn.MSELoss()
encoder = EncoderRNN(
    input_size=len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key])
)
decoder = DecoderRNN(output_size=2)
if use_cuda:
    encoder = nn.DataParallel(encoder)
    decoder = nn.DataParallel(decoder)

encoder.to(device)
decoder.to(device)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

model_utils = ModelUtils()

# If model_path provided, resume from saved checkpoint
if args.model_path is not None and os.path.isfile(args.model_path):
    print("Loading model")
    epoch, rollout_len, _ = model_utils.load_checkpoint(
        args.model_path, encoder, decoder, encoder_optimizer,
        decoder_optimizer)
    start_epoch = epoch + 1
    start_rollout_idx = ROLLOUT_LENS.index(rollout_len) + 1

start_time = time.time()

test_size = data_dict["test_input"].shape[0]
# test_data_subsets = baseline_utils.get_test_data_dict_subset(
    # data_dict, args)

forecast = infer_single_none(data_dict, 0, encoder, decoder, model_utils, args)

print("Forecast")
print(forecast)

# test_batch_size should be lesser than joblib_batch_size
# Parallel(n_jobs=-2, verbose=2)(
#     delayed(infer_helper)(test_data_subsets[i], i, encoder, decoder,
#                             model_utils, temp_save_dir)
#     for i in range(0, test_size, args.joblib_batch_size))
#
# baseline_utils.merge_saved_traj(temp_save_dir, args.traj_save_path)
# shutil.rmtree(temp_save_dir)
#
# end = time.time()
# print(f"Test completed in {(end - start_time) / 60.0} mins")
# print(f"Forecasted Trajectories saved at {args.traj_save_path}")
