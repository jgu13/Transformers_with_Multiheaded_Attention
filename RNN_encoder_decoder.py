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
    super(GRU, self).__init__()
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
    This is a Gated Recurrent Unit.
    
    Parameters
    ----------
    inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) input_size?
        The input tensor containing the embedded sequences.
    hidden_states 
        The (initial) hidden state.
        - h (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
    Returns
    -------
    x (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
        A feature tensor encoding the input sentence. 
    hidden_states 
        The final hidden state. 
        - h (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
    """

    seq_length = inputs.size(1)
    hidden_states = hidden_states.transpose(0, 1)
    h_t = hidden_states # B x 1 x H
    outputs = []
    for t in range(seq_length):
      x_t = inputs[:, t, :]
      h_t = h_t.squeeze(1) # B x H
      r = torch.mm(x_t, self.w_ir.t()) + self.b_ir + torch.mm(h_t, self.w_hr.t()) + self.b_hr
      r = F.sigmoid(r)

      z = torch.mm(x_t, self.w_iz.t()) + self.b_iz + torch.mm(h_t, self.w_hz.t()) + self.b_hz
      z = F.sigmoid(z) 

      nt = torch.mm(h_t, self.w_hn.t()) + self.b_hn 
      n = torch.mm(x_t, self.w_in.t()) + self.b_in + r * nt
      n = F.tanh(n)

      h_t = (1-z) * n + z * h_t
      h_t = h_t.unsqueeze(1) # B x 1 x H
      outputs.append(h_t)
    return torch.cat(outputs, dim=1), h_t.transpose(0,1)


class Attn(nn.Module):
    def __init__(
        self,
        hidden_size=256,
        dropout=0.0 # note, this is an extrenous argument
        ):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size

        self.W = nn.Linear(hidden_size*2, hidden_size)
        self.V = nn.Linear(hidden_size, hidden_size) # in the forwards, after multiplying
                                                     # do a torch.sum(..., keepdim=True), its a linear operation

        self.tanh = nn.Tanh()
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
        
        # For each batch, concatenate the last layer of hidden states with each time step (sequence) of the embedded new input
        hidden_states = hidden_states[-1:,:,:] # use the last layer, 1 x B x H
        hidden_states = hidden_states.transpose(0,1) # B x 1 x H
        hidden_states = hidden_states.repeat(1, inputs.size(1), 1) # repeat for sequence length, B x T x H
        concat = torch.cat((inputs, hidden_states), dim=2) # B x T x 2H

        attn_weights = self.W(concat) # B x T x H
        attn_weights = self.tanh(attn_weights)
        attn_weights = self.V(attn_weights) # B x T x H

        # new context representation
        x_attn = torch.sum(attn_weights, dim=2, keepdim=True) # B x T x 1

        if mask is not None:
            print(mask)
            x_attn = x_attn.masked_fill(mask.unsqueeze(-1)==0, -1e9)

        x_attn = self.softmax(x_attn) # B x T x 1, the masked weights will become zero

        # elementwise-multiply the input with attention weights
        x_out = x_attn * inputs # B x T x H

        return x_out, x_attn

class Encoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout=0.0,  
    ):
        super(Encoder, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            vocabulary_size, embedding_size, padding_idx=0,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)

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

        inputs = self.embedding(inputs)
        inputs = self.dropout(inputs)
        outputs, hidden_states = self.rnn(inputs, hidden_states) # outputs has shape (B, T, 2*hidden_size). Hidden_states has the shape (2*num_layers, B, hidden_size)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        hidden_states = hidden_states[:self.num_layers,:,:] + hidden_states[self.num_layers:,:,:] # sum the output of the forward and backward directions
        return outputs, hidden_states


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
    ):

        super(DecoderAttn, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=True)
        
        self.mlp_attn = Attn(hidden_size, dropout)

    def forward(self, inputs, hidden_states, mask=None):
        """GRU Decoder network with Soft attention

        This is a Unidirectional Gated Recurrent Unit Encoder network
        
        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
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
        
        # Encoder dropout
        dropout = self.dropout(inputs)
        attended_inputs, _ = self.mlp_attn(inputs, hidden_states)
        outputs = self.rnn(attended_inputs, hidden_states)
        return outputs
        
        
class EncoderDecoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        num_layers=1,
        dropout = 0.0,
        encoder_only=False
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder_only = encoder_only
        self.encoder = Encoder(vocabulary_size, embedding_size, hidden_size,
                num_layers, dropout=dropout)
        if not encoder_only:
          self.decoder = DecoderAttn(vocabulary_size, embedding_size, hidden_size, num_layers, dropout=dropout)
        
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
          x = x[:, 0]
          return x, hidden_states
        x, hidden_states = self.decoder(x, hidden_states, mask)
        x = x[:, 0]
        return x, hidden_states
