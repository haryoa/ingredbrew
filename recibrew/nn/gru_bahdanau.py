from torch.nn import Embedding, Module, GRU, Dropout, Linear
import torch


class Encoder(Module):

    def __init__(self, embedding_dim, hidden_dim, enc_gru_layers, dropout, enc_bidirectional, **kwargs):
        super(Encoder, self).__init__()
        self.gru = GRU(embedding_dim, hidden_dim, dropout=dropout, num_layers=enc_gru_layers,
                       bidirectional=enc_bidirectional)

    def forward(self, x):
        """

        :param x: result of shared embedding
        :return:
        """
        # x = [ seq_len, batch_size , num_embedding ]
        output_gru, hidden_state = self.gru(x)
        return output_gru, hidden_state


class BahdanauAttention(Module):

    def __init__(self, hidden_dim, enc_bidirectional, enc_gru_layers, **kwargs):
        super(BahdanauAttention, self).__init__()
        q1_fc_shape = hidden_dim * enc_gru_layers * 2 if enc_bidirectional else hidden_dim * enc_gru_layers
        q2_fc_shape = hidden_dim * 2 if enc_bidirectional else hidden_dim
        self.W1 = Linear(q1_fc_shape, hidden_dim)
        self.W2 = Linear(q2_fc_shape, hidden_dim)
        self.V = Linear(hidden_dim, 1)
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, query, values):
        # Input query = [1, batch_size (bs), hidden_dim]
        # Input values = [seq_len, bs, hidden_dim]
        score = self.V(torch.tanh(
                self.W1(query) + self.W2(values)))

        # score = [seq_len, bs, 1]
        attention_weights = self.softmax(score)

        # attention_weights : [seq_len, bs, 1]
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=0)

        # context_vector = [bs, hidden_dim]
        return context_vector, attention_weights


class Decoder(Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, enc_gru_layers, enc_bidirectional, **kwargs):
        super(Decoder, self).__init__()
        self.attention = BahdanauAttention(embedding_dim, enc_gru_layers=enc_gru_layers,
                                            enc_bidirectional=True)
        self.gru = torch.nn.GRU(embedding_dim +
                                (hidden_dim * 2 if enc_bidirectional else hidden_dim),
                                hidden_dim * enc_gru_layers * 2 if enc_bidirectional else hidden_dim * enc_gru_layers)

        self.linear = torch.nn.Linear(hidden_dim * enc_gru_layers * 2 if enc_bidirectional else hidden_dim *
                                                                                                enc_gru_layers,
                                      vocab_size)

    def forward(self, x, hidden, enc_out):
        # x : [ seq_len, batch_size , num_embedding ]
        # hidden : [enc_total_layer * bidirectional, bs, hidden_size * (2 or 1 according to bidirectional * gru_layers)]
        # enc_out : [ seq_len, bs, hidden_size * (2 or 1 according to bidirectional))]
        # context_vector_shape : [bs, hidden_dim]
        hidden = torch.cat([ x for x in hidden], axis=1)
        hidden = hidden.unsqueeze(0)
        context_vector, attention_weights = self.attention.forward(hidden, enc_out)
        # context_vector_shape : [bs, hidden_dim * bidirectional]

        x = torch.cat([context_vector.unsqueeze(0), x], axis=2)
        # x shape : [ seq_len, bs, num_embedding + hidden_encoder_unit (* 2 according to bidirectional)]

        output, state = self.gru(x)

        # output : [seq_len, bs, hidden_dim]
        out_linear = self.linear(output)

        # out_linear shape = [ seq_len, bs, vocab_size ]

        out_linear = out_linear.view(-1, out_linear.shape[2])
        # out_linear shape : [ bs * seq_len, vocab_size]

        return out_linear, state, attention_weights
