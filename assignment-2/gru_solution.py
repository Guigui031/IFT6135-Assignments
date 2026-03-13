import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size


        self.w_ir = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_iz = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_in = nn.Parameter(torch.empty(hidden_size, input_size))

        self.b_ir = nn.Parameter(torch.empty(hidden_size))
        self.b_iz = nn.Parameter(torch.empty(hidden_size))
        self.b_in = nn.Parameter(torch.empty(hidden_size))

        self.w_hr = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hn = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.b_hr = nn.Parameter(torch.empty(hidden_size))
        self.b_hz = nn.Parameter(torch.empty(hidden_size))
        self.b_hn = nn.Parameter(torch.empty(hidden_size))
        for param in self.parameters():
            nn.init.uniform_(param, a=-(1/hidden_size)**0.5, b=(1/hidden_size)**0.5)


    def forward(self, inputs, hidden_states):
        """GRU.

        This is a Gated Recurrent Unit
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, input_size)`)
          The input tensor containing the embedded sequences.

        hidden_states (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
          The (initial) hidden state.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
          A feature tensor encoding the input sentence.

        hidden_states (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
          The final hidden state.
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        batch_size, sequence_length, input_size = inputs.shape
        hidden_size = self.hidden_size
        hidden_states = hidden_states.squeeze(0)
        outputs = torch.zeros(batch_size, sequence_length, hidden_size, dtype=inputs.dtype, device=inputs.device)
        for t in range(sequence_length):
            x_t = inputs[:, t, :]
            r_t = torch.sigmoid(x_t @ self.w_ir.T + self.b_ir + hidden_states @ self.w_hr.T + self.b_hr)
            z_t = torch.sigmoid(x_t @ self.w_iz.T + self.b_iz + hidden_states @ self.w_hz.T + self.b_hz)
            n_t = torch.tanh(x_t @ self.w_in.T + self.b_in + r_t * (hidden_states @ self.w_hn.T + self.b_hn))
            hidden_states = (1 - z_t) * n_t + z_t * hidden_states
            outputs[:, t, :] = hidden_states
        return outputs, hidden_states.unsqueeze(0)

class Attn(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        dropout=0.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)

        self.W = nn.Linear(hidden_size*2, hidden_size)

        self.V = nn.Linear(hidden_size, hidden_size)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, inputs, hidden_states, mask = None):
        """Soft Attention mechanism.

        This is a one layer MLP network that implements Soft (i.e. Bahdanau) Attention with masking
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the embedded sequences.

        hidden_states (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
            The (initial) hidden state.

        mask ( optional `torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence with attention applied.

        x_attn (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`)
            The attention vector.
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        # hidden_states: (num_layers, batch, hidden) -> use last layer, expand to seq_len
        h = hidden_states[-1].unsqueeze(1).expand_as(inputs)  # (batch, seq_len, hidden)

        # 1. Concatenate encoder outputs and decoder hidden state
        combined = torch.cat((inputs, h), dim=2)  # (batch, seq_len, hidden*2)

        # 2. First linear + tanh
        energy = self.tanh(self.W(combined))  # (batch, seq_len, hidden)

        # 3. Second linear as multiply and sum: element-wise multiply with V weight, then sum
        # V.weight: (hidden, hidden), apply to energy then sum over last dim
        x_attn = (self.V.weight * energy).sum(dim=2, keepdim=True)  # (batch, seq_len, 1)

        # 4. Apply mask before softmax (set padding positions to -inf)
        if mask is not None:
            x_attn = x_attn.masked_fill(mask.unsqueeze(2) == 0, float('-inf'))

        # 5. Softmax over sequence dimension
        x_attn = self.softmax(x_attn)  # (batch, seq_len, 1)

        # 6. Elementwise multiply attention weights with encoder outputs
        outputs = inputs * x_attn  # (batch, seq_len, hidden)

        return outputs, x_attn


class Encoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            vocabulary_size, embedding_size, padding_idx=0,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, inputs, hidden_states):
        """GRU Encoder.

        This is a Bidirectional Gated Recurrent Unit Encoder network
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the token sequences.

        hidden_states
            The (initial) hidden state.
            - h (`torch.FloatTensor` of shape `(num_layers*2, batch_size, hidden_size)`)

        Returns
        -------
        x (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence.

        hidden_states
            The final hidden state.
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        # 1. Embed and apply dropout
        x = self.embedding(inputs)
        x = self.dropout(x)

        # 2. Run bidirectional GRU
        # x: (batch_size, seq_len, 2*hidden_size), h: (num_layers*2, batch_size, hidden_size)
        x, hidden_states = self.rnn(x, hidden_states)

        # 3. Sum forward and backward directions for outputs
        # Reshape (batch, seq, 2*hidden) -> (batch, seq, 2, hidden) then sum dim=2
        x = x.view(x.shape[0], x.shape[1], 2, self.hidden_size).sum(dim=2)

        # 4. Sum forward and backward directions for hidden states
        # (num_layers*2, batch, hidden) -> (num_layers, 2, batch, hidden) then sum dim=1
        hidden_states = hidden_states.view(self.num_layers, 2, -1, self.hidden_size).sum(dim=1)

        return x, hidden_states

    def initial_states(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        shape = (self.num_layers*2, batch_size, self.hidden_size)
        # The initial state is a constant here, and is not a learnable parameter
        h_0 = torch.zeros(shape, dtype=torch.float, device=device)
        return h_0

class DecoderAttn(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout=0.0,
        with_attn=True,
    ):

        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.GRU(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        if with_attn:
            self.mlp_attn = Attn(hidden_size, dropout)
        else:
            self.mlp_attn = None

    def forward(self, inputs, hidden_states, mask=None):
        """GRU Decoder network with Soft attention

        This is a Unidirectional Gated Recurrent Unit Encoder network

        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the encoded input sequence.

        hidden_states
            The (initial) hidden state.
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)

        Returns
        -------
        x (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor decoding the orginally encoded input sentence.

        hidden_states
            The final hidden state.
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        # 1. Apply dropout to encoder outputs
        x = self.dropout(inputs)

        # 2. Apply attention if enabled
        if self.mlp_attn is not None:
            x, _ = self.mlp_attn(x, hidden_states, mask)

        # 3. Feed attended input and hidden state into GRU
        x, hidden_states = self.rnn(x, hidden_states)

        return x, hidden_states


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout = 0.0,
        encoder_only=False,
        with_attn=True,
    ):
        super().__init__()
        self.encoder_only = encoder_only
        self.encoder = Encoder(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        if not encoder_only:
          self.decoder = DecoderAttn(
            vocabulary_size=vocabulary_size,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            with_attn=with_attn,
          )

    def forward(self, inputs, mask=None):
        """GRU Encoder-Decoder network with Soft attention.

        This is a Gated Recurrent Unit network for Sentiment Analysis. This
        module returns a decoded feature for classification.

        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
         Returns
        -------
        x (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor representing the input sentence for sentiment analysis

        hidden_states
            The final hidden state. This is a tuple containing
            - h (`torch.FloatTensor` of shape `(num_layers, batch_size, hidden_size)`)
        """
        hidden_states = self.encoder.initial_states(inputs.shape[0])
        x, hidden_states = self.encoder(inputs, hidden_states)
        if self.encoder_only:
            return x[:, 0], hidden_states
        x, hidden_states = self.decoder(x, hidden_states, mask)
        return x[:, 0], hidden_states
