from torch.nn import Transformer, Embedding, Dropout, Module
import torch
import math


class PositionalEncoding(Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class FullTransformer(Module):

    def __init__(self, num_vocab, num_embedding=128, dim_feedforward=512, num_encoder_layer=4,
                 num_decoder_layer=4, dropout=0.3, padding_idx=1, max_seq_len=140, nhead=8):
        super(FullTransformer, self).__init__()

        self.padding_idx = padding_idx

        # [x : seq_len,  batch_size ]
        self.inp_embedding = Embedding(num_vocab , num_embedding, padding_idx=padding_idx)

        # [ x : seq_len, batch_size, num_embedding ]
        self.pos_embedding = PositionalEncoding(num_embedding, dropout, max_len=max_seq_len)

        self.trfm = Transformer(d_model=num_embedding, dim_feedforward=dim_feedforward,
                                num_encoder_layers=num_encoder_layer, num_decoder_layers=num_decoder_layer,
                                dropout=dropout, nhead=nhead)
        self.linear_out = torch.nn.Linear(num_embedding, num_vocab)

    def make_pad_mask(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Make mask attention that caused 'True' element will not be attended (ignored).
        Padding stated in self.padding_idx will not be attended at all.

        :param inp : input that to be masked in boolean Tensor
        """
        return (inp == self.padding_idx).transpose(0, 1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        forward!

        :param src : source tensor
        :param tgt : target tensor
        """
        # Generate mask for decoder attention
        tgt_mask = self.trfm.generate_square_subsequent_mask(len(tgt)).to(tgt.device)

        # trg_mask shape = [target_seq_len, target_seq_len]
        src_pad_mask = self.make_pad_mask(src)
        tgt_pad_mask = self.make_pad_mask(tgt)

        # [ src : seq_len, batch_size, num_embedding ]

        out_emb_enc = self.pos_embedding(self.inp_embedding(src))

        # [ src : seq_len, batch_size, num_embedding ]
        out_emb_dec = self.pos_embedding(self.inp_embedding(tgt))

        out_trf = self.trfm(out_emb_enc, out_emb_dec, src_mask=None, tgt_mask=tgt_mask, memory_mask=None,
                            src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask,
                            memory_key_padding_mask=src_pad_mask)

        # [ out_trf : seq_len, batch_size, num_embedding]

        out_to_logit = self.linear_out(out_trf)

        # final_out : [ seq_len, batch_size, vocab_size ]
        return out_to_logit
