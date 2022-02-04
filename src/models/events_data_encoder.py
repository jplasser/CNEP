import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import PackedSequence
from sklearn import metrics
import numpy as np
from tqdm import tqdm
from typing import Optional
from collections import OrderedDict


class LSTM_CNN2(nn.Module):

    def __init__(self, input_dim=390, hidden_dim=8, lstm_layers=1):

        # dim, batch_norm, dropout, rec_dropout, task,
        # target_repl = False, deep_supervision = False, num_classes = 1,
        # depth = 1, input_dim = 390, ** kwargs

        super(LSTM_CNN2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = lstm_layers
        self.bidirectional = True
        # self.dense = dense

        # some more parameters
        # self.output_dim = dim
        # self.batch_norm = batch_norm
        self.dropout = 0.3
        self.rec_dropout = 0.3
        self.depth = lstm_layers
        self.drop_conv = 0.5
        self.num_classes = 1

        # define the LSTM layer
        # in keras we have inputs: A 3D tensor with shape [batch, timesteps, feature]
        # units: Positive integer, dimensionality of the output space. = dim=num_units=hidden_size
        if self.layers >= 2:
            self.lstm1 = nn.LSTM(input_size=self.input_dim,
                                 hidden_size=self.hidden_dim,
                                 num_layers=self.layers - 1,
                                 dropout=self.rec_dropout,
                                 bidirectional=self.bidirectional,
                                 batch_first=True)
            self.do0 = nn.Dropout(self.dropout)

        # this is not in the original model
        # self.act1 = nn.ReLU()
        if self.layers >= 2:
            self.lstm2 = nn.LSTM(input_size=self.hidden_dim * 2,
                                 hidden_size=self.hidden_dim * 2,
                                 num_layers=1,
                                 dropout=self.rec_dropout,
                                 bidirectional=False,
                                 batch_first=True)
        else:
            self.lstm2 = nn.LSTM(input_size=self.input_dim,
                                 hidden_size=self.hidden_dim * 2,
                                 num_layers=1,
                                 dropout=self.rec_dropout,
                                 bidirectional=False,
                                 batch_first=True)

        self.do1 = nn.Dropout(self.dropout)
        # self.bn0 = nn.BatchNorm1d(48 * self.hidden_dim*2)

        # three Convolutional Neural Networks with different kernel sizes
        nfilters = [2, 3, 4]
        nb_filters = 100
        pooling_reps = []

        self.cnn1 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim * 2, out_channels=nb_filters, kernel_size=2,
                      stride=1, padding=0, dilation=1, groups=1, bias=True,
                      padding_mode='zeros'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        self.cnn2 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim * 2, out_channels=nb_filters, kernel_size=3,
                      stride=1, padding=0, dilation=1, groups=1, bias=True,
                      padding_mode='zeros'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        self.cnn3 = nn.Sequential(
            nn.Conv1d(in_channels=self.hidden_dim * 2, out_channels=nb_filters, kernel_size=4,
                      stride=1, padding=0, dilation=1, groups=1, bias=True,
                      padding_mode='zeros'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        self.do2 = nn.Dropout(self.drop_conv)
        self.final = nn.Linear(6800, self.num_classes)

    def forward(self, inputs, labels=None):
        out = inputs
        if self.layers >= 2:
            out, h = self.lstm1(out)
            out = self.do0(out)
        out, h = self.lstm2(out)
        out = self.do1(out)

        pooling_reps = []

        pool_vecs = self.cnn1(out.permute((0, 2, 1)))
        pooling_reps.append(pool_vecs)

        pool_vecs = self.cnn2(out.permute((0, 2, 1)))
        pooling_reps.append(pool_vecs)

        pool_vecs = self.cnn3(out.permute((0, 2, 1)))
        pooling_reps.append(pool_vecs)

        # concatenate all vectors
        representation = torch.cat(pooling_reps, dim=1).contiguous()
        out = self.do2(representation)
        out = self.final(out)

        return out


class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """

    def __init__(self, dropout: float, batch_first: Optional[bool] = False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x


class LSTMNew(nn.LSTM):
    def __init__(self, *args, dropouti: float = 0.,
                 dropoutw: float = 0., dropouto: float = 0.,
                 batch_first=True, unit_forget_bias=True, **kwargs):
        super().__init__(*args, **kwargs, batch_first=batch_first)
        self.unit_forget_bias = unit_forget_bias
        self.dropoutw = dropoutw
        self.input_drop = VariationalDropout(dropouti,
                                             batch_first=batch_first)
        self.output_drop = VariationalDropout(dropouto,
                                              batch_first=batch_first)
        self._init_weights()

    def _init_weights(self):
        """
        Use orthogonal init for recurrent layers, xavier uniform for input layers
        Bias is 0 except for forget gate
        """
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "bias" in name and self.unit_forget_bias:
                nn.init.zeros_(param.data)
                param.data[self.hidden_size:2 * self.hidden_size] = 1

    def _drop_weights(self):
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                getattr(self, name).data = \
                    torch.nn.functional.dropout(param.data, p=self.dropoutw,
                                                training=self.training).contiguous()

    def forward(self, input, hx=None):
        self._drop_weights()
        self.flatten_parameters()
        input = self.input_drop(input)
        seq, state = super().forward(input, hx=hx)
        return self.output_drop(seq), state


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class EventsDataEncoder(nn.Module):

    def __init__(self, input_dim=390, hidden_dim=512, lstm_layers=3,
                 filter_kernels=[2, 3, 4], filters=100, output_dim=1024,
                 add_embeds=True, embed_dim=700,
                 dropout=0.3, dropout_w=0.2, dropout_conv=0.2):

        # dim, batch_norm, dropout, rec_dropout, task,
        # target_repl = False, deep_supervision = False, num_classes = 1,
        # depth = 1, input_dim = 390, ** kwargs

        super(EventsDataEncoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layers = lstm_layers
        self.bidirectional = True

        # some more parameters
        self.dropout = dropout
        self.rec_dropout = dropout_w
        self.depth = lstm_layers
        self.drop_conv = dropout_conv
        self.num_classes = 1
        self.output_dim = output_dim
        self.add_embeds = add_embeds
        self.embed_dim = embed_dim if add_embeds else 0

        # define the LSTM layer
        # in keras we have inputs: A 3D tensor with shape [batch, timesteps, feature]
        # units: Positive integer, dimensionality of the output space. = dim=num_units=hidden_size
        if self.layers >= 2:
            self.lstm1 = LSTMNew(input_size=self.input_dim,
                                 hidden_size=self.hidden_dim,
                                 num_layers=self.layers - 1,
                                 dropoutw=self.rec_dropout,
                                 dropout=self.rec_dropout,
                                 bidirectional=self.bidirectional,
                                 batch_first=True)
            self.do0 = nn.Dropout(self.dropout)

        # this is not in the original model
        if self.layers >= 2:
            self.lstm2 = LSTMNew(input_size=self.hidden_dim * 2,
                                 hidden_size=self.hidden_dim * 2,
                                 num_layers=1,
                                 dropoutw=self.rec_dropout,
                                 dropout=self.rec_dropout,
                                 bidirectional=False,
                                 batch_first=True)
        else:
            self.lstm2 = LSTMNew(input_size=self.input_dim,
                                 hidden_size=self.hidden_dim * 2,
                                 num_layers=1,
                                 dropoutw=self.rec_dropout,
                                 dropout=self.rec_dropout,
                                 bidirectional=False,
                                 batch_first=True)

        # three Convolutional Neural Networks with different kernel sizes
        nfilters = filter_kernels
        nb_filters = filters

        # 48 hrs of events data
        L_out = [(48 - k) + 1 for k in nfilters]
        maxpool_padding, maxpool_dilation, maxpool_kernel_size, maxpool_stride = (0, 1, 2, 2)
        dim_ = self.embed_dim + \
               int(np.sum([100 * np.floor(
                   (l + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) / maxpool_stride + 1)
                           for l in
                           L_out]))

        self.cnn1 = nn.Sequential(OrderedDict([
            ("cnn1_conv1d", nn.Conv1d(in_channels=self.hidden_dim * 2, out_channels=nb_filters, kernel_size=nfilters[0],
                                      stride=1, padding=0, dilation=1, groups=1, bias=True,
                                      padding_mode='zeros')),
            ("cnn1_relu", nn.ReLU()),
            ("cnn1_maxpool1d", nn.MaxPool1d(kernel_size=2)),
            ("cnn1_flatten", nn.Flatten())
        ]))

        self.cnn2 = nn.Sequential(OrderedDict([
            ("cnn2_conv1d", nn.Conv1d(in_channels=self.hidden_dim * 2, out_channels=nb_filters, kernel_size=nfilters[1],
                                      stride=1, padding=0, dilation=1, groups=1, bias=True,
                                      padding_mode='zeros')),
            ("cnn2_relu", nn.ReLU()),
            ("cnn2_maxpool1d", nn.MaxPool1d(kernel_size=2)),
            ("cnn2_flatten", nn.Flatten())
        ]))

        self.cnn3 = nn.Sequential(OrderedDict([
            ("cnn3_conv1d", nn.Conv1d(in_channels=self.hidden_dim * 2, out_channels=nb_filters, kernel_size=nfilters[2],
                                      stride=1, padding=0, dilation=1, groups=1, bias=True,
                                      padding_mode='zeros')),
            ("cnn3_relu", nn.ReLU()),
            ("cnn3_maxpool1d", nn.MaxPool1d(kernel_size=2)),
            ("cnn3_flatten", nn.Flatten())
        ]))

        # dim_latent = int(np.sum([100 * np.floor(
        #            (l + 2 * maxpool_padding - maxpool_dilation * (maxpool_kernel_size - 1) - 1) / maxpool_stride + 1)
        #                    for l in
        #                    L_out]))
        # dim_encoder = embed_dim * 2

        # self.latent = nn.Sequential(OrderedDict([
        #     ("enc_fc1", nn.Linear(dim_latent, embed_dim)),
        #     # ("enc_fc1", nn.Linear(dim_, 1024)),
        #     # ("enc_fc1", nn.Linear(dim_, self.output_dim)),
        #     # ("enc_bn1", nn.BatchNorm1d(dim_ * 2)),    # new BN
        #     ("enc_relu", nn.ReLU())])
        # )

        self.encoder = nn.Sequential(OrderedDict([
            # ("enc_fc1", nn.Linear(dim_encoder, dim_ * 2)),
            ("enc_fc1", nn.Linear(dim_, dim_ * 2)),
            # ("enc_fc1", nn.Linear(dim_, 1024)),
            # ("enc_fc1", nn.Linear(dim_, self.output_dim)),
            # ("enc_bn1", nn.BatchNorm1d(dim_ * 2)),    # new BN
            # ("enc_dropout", nn.Dropout()),
            ("enc_relu", nn.ReLU()),
            ("enc_layernorm", nn.LayerNorm(dim_ * 2)),
            ("enc_fc2", nn.Linear(dim_ * 2, self.output_dim)),
            # ("enc_fc2", nn.Linear(1024, 2048)),
            # ("enc_bn2", nn.BatchNorm1d(self.output_dim)),  # new BN
            # ("enc_dropout2", nn.Dropout()),
            ("enc_relu2", nn.ReLU()),
            # ("enc_fc3", nn.Linear(2048, self.output_dim)),
            # ("enc_bn3", nn.BatchNorm1d(self.output_dim)),  # new BN
            # ("enc_relu3", nn.ReLU()),
            # ("enc_fc4", nn.Linear(4096, self.output_dim)),
            # ("enc_bn4", nn.BatchNorm1d(self.output_dim)),  # new BN
            # ("enc_relu4", nn.ReLU()),
            ("enc_layernorm2", nn.LayerNorm(self.output_dim))
        ]))

        self.do2 = nn.Dropout(self.drop_conv)
        # self.final = nn.Linear(dim_, self.num_classes)

    def forward(self, inputs, embeds=None):
        out = inputs
        if self.layers >= 2:
            out, h = self.lstm1(out)
            out = self.do0(out)
        out, h = self.lstm2(out)

        pooling_reps = []

        pool_vecs = self.cnn1(out.permute((0, 2, 1)))
        pooling_reps.append(pool_vecs)

        pool_vecs = self.cnn2(out.permute((0, 2, 1)))
        pooling_reps.append(pool_vecs)

        pool_vecs = self.cnn3(out.permute((0, 2, 1)))
        pooling_reps.append(pool_vecs)

        # concatenate all vectors
        representation = torch.cat(pooling_reps, dim=1).contiguous()
        # new model architecture
        # out = self.latent(representation)
        out = self.do2(representation)
        if embeds is not None:
            out = torch.cat([out, embeds], dim=1)
        encoding = self.encoder(out)
        # out = self.final(out)

        # return encoding in the shape of (output_dim)
        return encoding
