import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def forward(self, inputs):
        """Layer Normalization.

        This module applies Layer Normalization, with rescaling and shift,
        only on the last dimension. See Lecture 07 (I), slide 23.

        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The input tensor. This tensor can have an arbitrary number N of
            dimensions, as long as `inputs.shape[N-1] == hidden_size`. The
            leading N - 1 dimensions `dims` can be arbitrary.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The output tensor, having the same shape as `inputs`.
        """

        # ==========================
        # TODO: Write your code here
        # ==========================
        mean = torch.mean(inputs, dim=-1, keepdim=True)
        var = torch.var(inputs, dim=-1, keepdim=True, unbiased=False)
        outputs = (inputs - mean) / torch.sqrt(var + self.eps)
        outputs = outputs * self.weight + self.bias
        return outputs 


    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_size, num_heads, sequence_length):
        super(MultiHeadedAttention, self).__init__()
        self.head_size = head_size
        self.num_heads = num_heads
        self.sequence_length = sequence_length
        self.in_features = self.num_heads * self.head_size
        self.out_features = self.num_heads * self.head_size
        # ==========================
        # TODO: Write your code here
        # ==========================

        self.W_q = nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True)
        self.W_v = nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True)
        self.W_k = nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True)
        self.W_o = nn.Linear(in_features=self.in_features, out_features=self.out_features, bias=True)

    def get_attention_weights(self, queries, keys, mask=None):
        """Compute the attention weights.

        This computes the attention weights for all the sequences and all the
        heads in the batch. For a single sequence and a single head (for
        simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        and K are the keys (matrix of size `(sequence_length, head_size)`), then
        the attention weights are computed as

            weights = softmax(Q * K^{T} / sqrt(head_size))

        Here "*" is the matrix multiplication. See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads.

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. 
           
        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        attention_weights (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)`)
            Tensor containing the attention weights for all the heads and all
            the sequences in the batch.
        """
        # ==========================
        # TODO: Write your code here
        # ==========================
        # queries has shape (B, m, T, d)
        B = queries.size(0)
        attn_weights = []
        for b in range(B):
          head_weights = []
          for m in range(self.num_heads):
            x_t = queries[b,m] @ keys[b,m].t() # T x T
            x_t = x_t / np.sqrt(self.head_size)
            if mask is not None:
              batch_mask = mask[b,:] # (1, T)
              batch_mask = batch_mask.repeat(x_t.size(0), 1) # T x T
              # print("Batch mask size: {}".format(batch_mask.size()))
              # print("x_t size: {}".format(x_t.size()))
              x_t = x_t.masked_fill(batch_mask==0, float('-inf'))
            x_t = x_t.unsqueeze(0) # 1 x T x T
            head_weights.append(x_t)
          head_weights = torch.cat(head_weights, dim=0) # m x T x T
          head_weights = head_weights.unsqueeze(0) # 1 x m x T x T
          # print(head_weights.size())
          attn_weights.append(head_weights)
        attn_weights = torch.cat(attn_weights, dim=0) # B x m x T x T
        attn_weights = nn.Softmax(dim=3)(attn_weights) # softmax is applied to the entire sequence 
        return attn_weights
        
        
    def apply_attention(self, queries, keys, values, mask=None):
        """Apply the attention.

        This computes the output of the attention, for all the sequences and
        all the heads in the batch. For a single sequence and a single head
        (for simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        K are the keys (matrix of size `(sequence_length, head_size)`), and V are
        the values (matrix of size `(sequence_length, head_size)`), then the ouput
        of the attention is given by

            weights = softmax(Q * K^{T} / sqrt(head_size))
            attended_values = weights * V
            outputs = concat(attended_values)

        Here "*" is the matrix multiplication, and "concat" is the operation
        that concatenates the attended values of all the heads (see the
        `merge_heads` function). See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads. 

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. 

        values (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the values for all the positions in the sequences
            and all the heads. 
        
        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.
            
        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the concatenated outputs of the attention for all
            the sequences in the batch, and all positions in each sequence. 
        """

        # ==========================
        # TODO: Write your code here
        # ==========================
        # split queries, keys and values
        attn_weights = self.get_attention_weights(queries, keys, mask) # B x m x T x T
        Batch_size = values.size(0)
        num_heads = values.size(1)
        attended_values = []
        for b in range(Batch_size):
            head_values = []
            for m in range(num_heads):
                attn_weight = attn_weights[b, m] # T x T
                value = values[b, m] # T x b
                attended_v = torch.mm(attn_weight, value) # T x b
                attended_v = attended_v.unsqueeze(0) # 1 x T x b
                head_values.append(attended_v)
            head_values = torch.cat(head_values, dim=0) # m x T x b
            head_values = head_values.unsqueeze(0) # 1 x m x T x b
            attended_values.append(head_values)
        attended_values = torch.cat(attended_values, dim=0) # B x m x T x b
        # print("attended_values shape: {}".format(attended_values.shape))
        outputs = self.merge_heads(attended_values) # B x T x mb
        return outputs


    def split_heads(self, tensor):
        """Split the head vectors.

        This function splits the head vectors that have been concatenated (e.g.
        through the `merge_heads` function) into a separate dimension. This
        function also transposes the `sequence_length` and `num_heads` axes.
        It only reshapes and transposes the input tensor, and it does not
        apply any further transformation to the tensor. The function `split_heads`
        is the inverse of the function `merge_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Input tensor containing the concatenated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Reshaped and transposed tensor containing the separated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """

        # ==========================
        # TODO: Write your code here
        # ==========================
        tensor = torch.reshape(tensor, (tensor.size(0), tensor.size(1), self.num_heads, -1)) # B x T x m x d
        tensor = tensor.transpose(1,2) # B x m x T x d
        return tensor
        
    def merge_heads(self, tensor):
        """Merge the head vectors.

        This function concatenates the head vectors in a single vector. This
        function also transposes the `sequence_length` and the newly created
        "merged" dimension. It only reshapes and transposes the input tensor,
        and it does not apply any further transformation to the tensor. The
        function `merge_heads` is the inverse of the function `split_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Input tensor containing the separated head vectors (each having
            a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Reshaped and transposed tensor containing the concatenated head
            vectors. Here `dim` is the same dimension as the one in the
            definition of the input `tensor` above.
        """

        # ==========================
        # TODO: Write your code here
        # ==========================
        tensor = tensor.transpose(1, 2) # [0, 2, 1, 3]
        new_shape = (tensor.size(0), tensor.size(1), -1) # [0, 2, num_heads * 1 + 3]
        tensor = torch.reshape(tensor, new_shape)
        return tensor

    def forward(self, hidden_states, mask=None):
        """Multi-headed attention.

        This applies the multi-headed attention on the input tensors `hidden_states`.
        For a single sequence (for simplicity), if X are the hidden states from
        the previous layer (a matrix of size `(sequence_length, num_heads * head_size)`
        containing the concatenated head vectors), then the output of multi-headed
        attention is given by

            Q = X * W_{Q} + b_{Q}        # Queries
            K = X * W_{K} + b_{K}        # Keys
            V = X * W_{V} + b_{V}        # Values

            Y = attention(Q, K, V)       # Attended values (concatenated for all heads)
            outputs = Y * W_{Y} + b_{Y}  # Linear projection

        Here "*" is the matrix multiplication.

        Parameters
        ----------
        hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Input tensor containing the concatenated head vectors for all the
            sequences in the batch, and all positions in each sequence. This
            is, for example, the tensor returned by the previous layer.
            
        
        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the output of multi-headed attention for all the
            sequences in the batch, and all positions in each sequence.
        """
        # ==========================
        # TODO: Write your code here
        # ==========================

        # linear projection of hidden states into queries, keys and values
        queries = self.W_q(hidden_states) 
        keys = self.W_k(hidden_states)
        values = self.W_v(hidden_states)

        # split queries, keys and values into num_heads
        queries = self.split_heads(queries) # B x m x T x b
        keys = self.split_heads(keys)
        values = self.split_heads(values)
        # print(queries.size())

        output = self.apply_attention(queries, keys, values, mask)
        output = self.W_o(output) # linear projection of the output
        return output
        

class PostNormAttentionBlock(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_heads, sequence_length, dropout=0.30):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        
        
        self.layer_norm_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(embed_dim//num_heads, num_heads,sequence_length)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, x, mask=None):
       
        attention_outputs = self.attn(x, mask)
        attention_outputs = self.layer_norm_1(x + attention_outputs)
        outputs = self.linear(attention_outputs)

        outputs = self.layer_norm_2(outputs + attention_outputs)
        return outputs

class PreNormAttentionBlock(nn.Module):
    
    def __init__(self, embed_dim, hidden_dim, num_heads,sequence_length, dropout=0.0):
        """A decoder layer.

        This module combines a Multi-headed Attention module and an MLP to
        create a layer of the transformer, with normalization and skip-connections.
        See Lecture 06, slide 33.

        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network 
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            sequence_length - Length of the sequence
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()
        
        self.layer_norm_1 = LayerNorm(embed_dim)
        self.attn = MultiHeadedAttention(embed_dim//num_heads, num_heads,sequence_length)
        self.layer_norm_2 = LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        
    def forward(self, x, mask=None):
        # ==========================
        # TODO: Write your code here
        # ==========================
        x_norm = self.layer_norm_1(x)
        attention_outputs = self.attn(x_norm, mask)
        attention_outputs = x + attention_outputs # skip connection
        outputs = self.layer_norm_2(attention_outputs) 
        outputs = self.linear(outputs) 
        outputs = attention_outputs + outputs
        return outputs

class Transformer(nn.Module):
    
    def __init__(self, vocabulary_size=30522, embed_dim=256, hidden_dim=256, num_heads=1,
            num_layers=2, block='prenorm', dropout=0.3):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            block - Type of attention block
            dropout - Amount of dropout to apply in the feed-forward network and 
                      on the input encoding
        """
        super().__init__()
        
        #Adding the cls token to the sequnence 
        self.sequence_length= 1 + 256
        # Layers/Networks
        self.embedding = nn.Embedding(vocabulary_size, embed_dim)
        if block =='prenorm':
          self.transformer = nn.ModuleList([PreNormAttentionBlock(embed_dim, hidden_dim, num_heads,self.sequence_length, dropout=dropout) for _ in range(num_layers)])
        else:
          self.transformer = nn.ModuleList([PostNormAttentionBlock(embed_dim, hidden_dim, num_heads,self.sequence_length, dropout=dropout) for _ in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
        )
        self.dropout = nn.Dropout(dropout)
        
        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,self.sequence_length,embed_dim))
        # print(embed_dim) # 120
   
    def forward(self, x, mask=None):
        """Transformer
        This is a small version of  Transformer
        Parameters
        ----------
        x - (`torch.LongTensor` of shape `(batch_size, sequence length)`)
            The input tensor containing text.
        
        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.
        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, embed_dim)`)
            A tensor containing the output from the mlp_head.
        """
        # Preprocess input
        
        x = self.embedding(x) # B x T x embed_dim
        B, T, _ = x.shape

        # Add CLS token and positional encoding
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1) # B x T x embed_dim
        if mask is not None:
            mask = torch.cat((torch.ones((B,1)), mask), dim=1) # augment the mask by 1
        x = x + self.pos_embedding[:,:T+1]
        # Add dropout and then the transformer
        # ==========================
        # TODO: Write your code here
        # ==========================
        x = self.dropout(x)
        # Encoder
        for m in self.transformer:
            encoder_output = m(x, mask)
            x = encoder_output
            
        # Decoder
        x = self.dropout(x)
        for m in self.transformer:
            decoder_output = m(x, mask)
            x = decoder_output
        
        #Take the cls token representation and send it to mlp_head
        x = x[:,0,:] # take the first cls token
        output = self.mlp_head(x)
        return output
        