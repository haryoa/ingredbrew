from torch.nn import Embedding, Module, GRU, Dropout


class Encoder(Module):

    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=100, padding_idx=1,
                 gru_layers=1, dropout=0.2):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.gru = GRU(embedding_dim, hidden_dim, dropout=0.2, num_layers=gru_layers)

    def forward(self, x):
        x = self.embedding(x)
        output_gru, hidden_state = self.gru(x)
        return output_gru, hidden_state

