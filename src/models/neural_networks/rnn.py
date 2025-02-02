import math
from copy import copy
from pathlib import Path

import torch
from torch import nn

from typing import Dict, List, Optional, Tuple

from .base import NNBase


class RecurrentNetwork(NNBase):

    model_name = 'rnn'

    def __init__(self, hidden_size: int,
                 dense_features: Optional[List[int]] = None,
                 rnn_dropout: float = 0.25,
                 data_folder: Path = Path('data'),
                 batch_size: int = 1,
                 experiment: str = 'one_month_forecast',
                 pred_months: Optional[List[int]] = None,
                 include_pred_month: bool = True,
                 include_latlons: bool = False,
                 include_monthly_aggs: bool = True,
                 include_yearly_aggs: bool = True,
                 surrounding_pixels: Optional[int] = None,
                 ignore_vars: Optional[List[str]] = None,
                 include_static: bool = True,
                 device: str = 'cuda:0') -> None:
        super().__init__(data_folder, batch_size, experiment, pred_months, include_pred_month,
                         include_latlons, include_monthly_aggs, include_yearly_aggs,
                         surrounding_pixels, ignore_vars, include_static, device)

        # to initialize and save the model
        self.hidden_size = hidden_size
        self.rnn_dropout = rnn_dropout
        self.input_dense_features = copy(dense_features)
        if dense_features is None: dense_features = []
        self.dense_features = dense_features

        self.features_per_month: Optional[int] = None
        self.current_size: Optional[int] = None
        self.yearly_agg_size: Optional[int] = None
        self.static_size: Optional[int] = None

    def save_model(self):

        assert self.model is not None, 'Model must be trained before it can be saved!'

        model_dict = {
            'model': {'state_dict': self.model.state_dict(),
                      'features_per_month': self.features_per_month,
                      'current_size': self.current_size,
                      'yearly_agg_size': self.yearly_agg_size,
                      'static_size': self.static_size},
            'batch_size': self.batch_size,
            'hidden_size': self.hidden_size,
            'rnn_dropout': self.rnn_dropout,
            'dense_features': self.input_dense_features,
            'include_pred_month': self.include_pred_month,
            'include_latlons': self.include_latlons,
            'surrounding_pixels': self.surrounding_pixels,
            'ignore_vars': self.ignore_vars,
            'include_monthly_aggs': self.include_monthly_aggs,
            'include_yearly_aggs': self.include_yearly_aggs,
            'experiment': self.experiment,
            'include_static': self.include_static,
            'device': self.device
        }

        torch.save(model_dict, self.model_dir / 'model.pt')

    def load(self, state_dict: Dict, features_per_month: int, current_size: Optional[int],
             yearly_agg_size: Optional[int], static_size: Optional[int]) -> None:
        self.features_per_month = features_per_month
        self.current_size = current_size
        self.yearly_agg_size = yearly_agg_size
        self.static_size = static_size

        self.model: RNN = RNN(features_per_month=self.features_per_month,
                              dense_features=self.dense_features,
                              hidden_size=self.hidden_size,
                              rnn_dropout=self.rnn_dropout,
                              include_pred_month=self.include_pred_month,
                              include_latlons=self.include_latlons,
                              experiment=self.experiment,
                              current_size=self.current_size,
                              yearly_agg_size=self.yearly_agg_size,
                              static_size=self.static_size)
        self.model.to(torch.device(self.device))
        self.model.load_state_dict(state_dict)

    def _initialize_model(self, x_ref: Optional[Tuple[torch.Tensor, ...]]) -> nn.Module:
        if self.features_per_month is None:
            assert x_ref is not None, \
                f"x_ref can't be None if features_per_month or current_size is not defined"
            self.features_per_month = x_ref[0].shape[-1]
        if self.experiment == 'nowcast':
            if self.current_size is None:
                assert x_ref is not None, \
                    f"x_ref can't be None if features_per_month or current_size is not defined"
                self.current_size = x_ref[3].shape[-1]

        if self.include_yearly_aggs:
            if self.yearly_agg_size is None:
                assert x_ref is not None, \
                    f"x_ref can't be None if features_per_month or current_size is not defined"
                self.yearly_agg_size = x_ref[4].shape[-1]

        if self.include_static:
            if self.static_size is None:
                assert x_ref is not None
                self.static_size = x_ref[5].shape[-1]

        model = RNN(features_per_month=self.features_per_month,
                    dense_features=self.dense_features,
                    hidden_size=self.hidden_size,
                    rnn_dropout=self.rnn_dropout,
                    include_pred_month=self.include_pred_month,
                    include_latlons=self.include_latlons,
                    experiment=self.experiment,
                    current_size=self.current_size,
                    yearly_agg_size=self.yearly_agg_size,
                    static_size=self.static_size)
        return model.to(torch.device(self.device))


class RNN(nn.Module):
    def __init__(self, features_per_month, dense_features, hidden_size,
                 rnn_dropout, include_pred_month,
                 include_latlons, experiment, current_size=None,
                 yearly_agg_size=None, static_size=None):
        super().__init__()

        self.experiment = experiment
        self.include_pred_month = include_pred_month
        self.include_latlons = include_latlons
        self.include_yearly_agg = False
        self.include_static = False

        self.dropout = nn.Dropout(rnn_dropout)
        self.rnn = UnrolledRNN(input_size=features_per_month,
                               hidden_size=hidden_size,
                               batch_first=True)
        self.hidden_size = hidden_size

        dense_input_size = hidden_size
        if include_pred_month:
            dense_input_size += 12
        if include_latlons:
            dense_input_size += 2
        if experiment == 'nowcast':
            assert current_size is not None
            dense_input_size += current_size
        if yearly_agg_size is not None:
            self.include_yearly_agg = True
            dense_input_size += yearly_agg_size
        if static_size is not None:
            self.include_static = True
            dense_input_size += static_size

        dense_features.insert(0, dense_input_size)
        if dense_features[-1] != 1:
            dense_features.append(1)

        self.dense_layers = nn.ModuleList([
            nn.Linear(in_features=dense_features[i - 1],
                      out_features=dense_features[i])
            for i in range(1, len(dense_features))
        ])

        self.initialize_weights()

    def initialize_weights(self):

        sqrt_k = math.sqrt(1 / self.hidden_size)
        for parameters in self.rnn.parameters():
            for pam in parameters:
                nn.init.uniform_(pam.data, -sqrt_k, sqrt_k)

        for dense_layer in self.dense_layers:
            nn.init.kaiming_uniform_(dense_layer.weight.data)
            nn.init.constant_(dense_layer.bias.data, 0)

    def forward(self, x, pred_month=None, latlons=None, current=None, yearly_aggs=None,
                static=None):

        sequence_length = x.shape[1]

        hidden_state = torch.zeros(1, x.shape[0], self.hidden_size)
        cell_state = torch.zeros(1, x.shape[0], self.hidden_size)

        if x.is_cuda:
            hidden_state = hidden_state.cuda()
            cell_state = cell_state.cuda()

        for i in range(sequence_length):
            # The reason the RNN is unrolled here is to apply dropout to each timestep;
            # The rnn_dropout argument only applies it after each layer.
            # https://www.tensorflow.org/api_docs/python/tf/nn/rnn_cell/DropoutWrapper
            input_x = x[:, i, :].unsqueeze(1)
            _, (hidden_state, cell_state) = self.rnn(input_x,
                                                     (hidden_state, cell_state))
            hidden_state = self.dropout(hidden_state)

        x = hidden_state.squeeze(0)

        if self.include_pred_month:
            x = torch.cat((x, pred_month), dim=-1)
        if self.include_latlons:
            x = torch.cat((x, latlons), dim=-1)
        if self.experiment == 'nowcast':
            assert current is not None
            x = torch.cat((x, current), dim=-1)
        if self.include_yearly_agg:
            x = torch.cat((x, yearly_aggs), dim=-1)
        if self.include_static:
            x = torch.cat((x, static), dim=-1)

        for layer_number, dense_layer in enumerate(self.dense_layers):
            x = dense_layer(x)
        return x


class UnrolledRNN(nn.Module):
    """An unrolled RNN. The motivation for this is mainly so that we can explain this model using
    the shap deep explainer, but also because we unroll the RNN anyway to apply dropout.
    """

    def __init__(self, input_size, hidden_size, batch_first=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.forget_gate = nn.Sequential(*[
            nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size,
                      bias=True), nn.Sigmoid()])

        self.update_gate = nn.Sequential(*[
            nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size,
                      bias=True), nn.Sigmoid()
        ])

        self.update_candidates = nn.Sequential(*[
            nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size,
                      bias=True), nn.Tanh()
        ])

        self.output_gate = nn.Sequential(*[
            nn.Linear(in_features=input_size + hidden_size, out_features=hidden_size,
                      bias=True), nn.Sigmoid()
        ])

        self.cell_state_activation = nn.Tanh()

    def forward(self, x, state):
        hidden, cell = state

        if self.batch_first:
            hidden, cell = torch.transpose(hidden, 0, 1), torch.transpose(cell, 0, 1)

        forget_state = self.forget_gate(torch.cat((x, hidden), dim=-1))
        update_state = self.update_gate(torch.cat((x, hidden), dim=-1))
        cell_candidates = self.update_candidates(torch.cat((x, hidden), dim=-1))

        updated_cell = (forget_state * cell) + (update_state * cell_candidates)

        output_state = self.output_gate(torch.cat((x, hidden), dim=-1))
        updated_hidden = output_state * self.cell_state_activation(updated_cell)

        if self.batch_first:
            updated_hidden = torch.transpose(updated_hidden, 0, 1)
            updated_cell = torch.transpose(updated_cell, 0, 1)

        return updated_hidden, (updated_hidden, updated_cell)
