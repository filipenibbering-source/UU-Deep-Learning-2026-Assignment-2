
import torch
from torch import nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        
    def forward(self, x):
        out, (last_h_state, last_c_state) = self.lstm(x)
        return last_h_state.squeeze(dim=0)
    
class LSTMDecoder(nn.Module):
    def __init__(self, output_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2, bidirectional: bool = True):
        super().__init__()
        
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, (last_h_state, last_c_state) = self.lstm(x)
        return self.fc(last_h_state.squeeze(dim=0))
    
    
## Adapted from https://github.com/matanle51/LSTM_AutoEncoder/blob/master/models/LSTMAE.py
# Encoder Class
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional=False):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        # width of the single context vector produced for the whole sequence
        self.context_size = hidden_size * self.num_directions

        self.lstm_enc = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        out, (last_h_state, last_c_state) = self.lstm_enc(x)
        # last_h_state: (num_layers * num_directions, B, H) -> take last layer's directions
        last_h_state = last_h_state.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        last_layer = last_h_state[-1]                       # (num_directions, B, H)
        context = last_layer.permute(1, 0, 2).reshape(batch_size, self.context_size)  # (B, H*num_directions)
        # repeat the context across every timestep so the decoder can unroll it
        x_enc = context.unsqueeze(1).repeat(1, seq_len, 1)  # (B, T, context_size)
        return x_enc


# Decoder Class
class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout, bidirectional=False, use_act=False):
        super(Decoder, self).__init__()
        self.input_size = output_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.use_act = use_act  # Parameter to control the last sigmoid activation - depends on the normalization used.
        self.act = nn.Sigmoid()

        # input feature dim must match the encoder's context_size (hidden_size * num_directions)
        context_size = hidden_size * self.num_directions
        self.lstm_dec = nn.LSTM(input_size=context_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * self.num_directions, output_size)

    def forward(self, z):
        dec_out, (hidden_state, cell_state) = self.lstm_dec(z)  # (B, T, hidden_size * num_directions)
        dec_out = self.fc(dec_out)                              # (B, T, output_size)
        if self.use_act:
            dec_out = self.act(dec_out)
        return dec_out
