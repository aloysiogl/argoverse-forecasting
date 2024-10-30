import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argoverse_forecasting.utils.baseline_config as config

from .utils import baseline_utils
from .utils.lstm_utils import LSTMDataset, ModelUtils
from .features import xy_to_features


class EncoderRNN(nn.Module):
    """Encoder Network."""

    def __init__(
        self, input_size: int = 2, embedding_size: int = 8, hidden_size: int = 16
    ):
        """Initialize the encoder network.

        Args:
            input_size: number of features in the input
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM

        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(input_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)

    def forward(self, x: torch.FloatTensor, hidden: Any) -> Any:
        """Run forward propagation.

        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            hidden: final hidden

        """
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        return hidden


class DecoderRNN(nn.Module):
    """Decoder Network."""

    def __init__(self, embedding_size=8, hidden_size=16, output_size=2):
        """Initialize the decoder network.

        Args:
            embedding_size: Embedding size
            hidden_size: Hidden size of LSTM
            output_size: number of features in the output

        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(output_size, embedding_size)
        self.lstm1 = nn.LSTMCell(embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """Run forward propagation.

        Args:
            x: input to the network
            hidden: initial hidden state
        Returns:
            output: output from lstm
            hidden: final hidden state

        """
        embedded = F.relu(self.linear1(x))
        hidden = self.lstm1(embedded, hidden)
        output = self.linear2(hidden[0])
        return output, hidden


def infer_single_none(
    curr_data_dict: Dict[str, Any],
    start_idx: int,
    encoder: EncoderRNN,
    decoder: DecoderRNN,
    model_utils: ModelUtils,
    args: Any,
):
    """Inference on dataset of single trajectory.

    Args:
        curr_data_dict: Data dictionary for the current joblib batch
        start_idx: Start idx of the current joblib batch
        encoder: Encoder network instance
        decoder: Decoder network instance
        model_utils: ModelUtils instance
        forecasted_save_dir: Directory where forecasted trajectories are to be saved

    """
    curr_test_dataset = LSTMDataset(curr_data_dict, args, "test")
    curr_test_loader = torch.utils.data.DataLoader(
        curr_test_dataset,
        shuffle=False,
        batch_size=args.test_batch_size,
        collate_fn=model_utils.my_collate_fn,
    )

    print(
        f"#### LSTM+social inference at {start_idx} ####"
    ) if args.use_social else print(f"#### LSTM inference at {start_idx} ####")
    return infer_absolute(
        curr_test_loader,
        encoder,
        decoder,
        model_utils,
        args,
    )


def infer_absolute(
    test_loader: torch.utils.data.DataLoader,
    encoder: EncoderRNN,
    decoder: DecoderRNN,
    model_utils: ModelUtils,
    args: Any,
    single=True,
):
    """Infer function for non-map LSTM baselines and save the forecasted trajectories.

    Args:
        test_loader: DataLoader for the test set
        encoder: Encoder network instance
        decoder: Decoder network instance
        start_idx: start index for the current joblib batch
        forecasted_save_dir: Directory where forecasted trajectories are to be saved
        model_utils: ModelUtils instance

    """
    forecasted_trajectories = {}

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    for i, (_input, target, helpers) in enumerate(test_loader):
        _input = _input.to(device)

        batch_helpers = list(zip(*helpers))

        helpers_dict = {}
        for k, v in config.LSTM_HELPER_DICT_IDX.items():
            helpers_dict[k] = batch_helpers[v]

        # Set to eval mode
        encoder.eval()
        decoder.eval()

        # Encoder
        batch_size = _input.shape[0]
        input_length = _input.shape[1]
        input_shape = _input.shape[2]

        # Initialize encoder hidden state
        encoder_hidden = model_utils.init_hidden(
            batch_size, encoder.module.hidden_size if use_cuda else encoder.hidden_size
        )

        # Encode observed trajectory
        for ei in range(input_length):
            encoder_input = _input[:, ei, :]
            encoder_hidden = encoder(encoder_input, encoder_hidden)

        # Initialize decoder input with last coordinate in encoder
        decoder_input = encoder_input[:, :2]

        # Initialize decoder hidden state as encoder hidden state
        decoder_hidden = encoder_hidden

        decoder_outputs = torch.zeros((batch_size, args.pred_len, 2)).to(device)

        # Decode hidden state in future trajectory
        for di in range(args.pred_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            decoder_outputs[:, di, :] = decoder_output

            # Use own predictions as inputs at next step
            decoder_input = decoder_output

        # Get absolute trajectory
        abs_helpers = {}
        abs_helpers["REFERENCE"] = np.array(helpers_dict["DELTA_REFERENCE"])
        abs_helpers["TRANSLATION"] = np.array(helpers_dict["TRANSLATION"])
        abs_helpers["ROTATION"] = np.array(helpers_dict["ROTATION"])
        abs_inputs, abs_outputs = baseline_utils.get_abs_traj(
            _input.clone().cpu().numpy(),
            decoder_outputs.detach().clone().cpu().numpy(),
            args,
            abs_helpers,
        )

        # TODO very bad way to handle the single case
        if single:
            return abs_outputs[0]

        for i in range(abs_outputs.shape[0]):
            seq_id = int(helpers_dict["SEQ_PATHS"][i])
            forecasted_trajectories[seq_id] = [abs_outputs[i]]

    return forecasted_trajectories


class LSTMForecaster:
    def __init__(self, model_path: str, obs_len: int = 20, pred_len: int = 30):
        self.args = type(
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
                "lr": 0.001,
            },
        )()

        if not baseline_utils.validate_args(self.args):
            print("Invalid args")
            exit(1)

        self.baseline_key = "none"

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        if self.use_cuda:
            print(f"Using all ({torch.cuda.device_count()}) GPUs...")

        self.model_utils = ModelUtils()

        self.load_model(model_path)

    def load_model(self, model_path: str):
        criterion = nn.MSELoss()
        self.encoder = EncoderRNN(
            input_size=len(baseline_utils.BASELINE_INPUT_FEATURES[self.baseline_key])
        )
        self.decoder = DecoderRNN(output_size=2)
        if self.use_cuda:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder.to(self.device)
        self.decoder.to(self.device)

        encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.args.lr)
        decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=self.args.lr)

        # If model_path provided, resume from saved checkpoint
        if os.path.isfile(model_path):
            print("Loading model")
            self.model_utils.load_checkpoint(
                model_path,
                self.encoder,
                self.decoder,
                encoder_optimizer,
                decoder_optimizer,
            )
        else:
            raise ValueError("Model path not found")

    def predict(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Predict the future trajectory given x, y coordinates.
        """

        feature_sequence = xy_to_features(trajectory)

        data_dict = baseline_utils.get_data(self.args, self.baseline_key, feature_sequence)

        return infer_single_none(
            data_dict, 0, self.encoder, self.decoder, self.model_utils, self.args
        )

